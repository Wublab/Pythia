import pytorch_lightning as pl
import torch
import torchmetrics
from torch import nn, optim


class AMPNNLayer(nn.Module):
    def __init__(self, embed_dim=128, n_heads=8, dropout=0.2, neighbor_num=32) -> None:
        super().__init__()
        self.multihead_message = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout
        )
        self.feed_forwards_message = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.multihead_update = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, dropout=dropout
        )
        self.feed_forwards_update = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.message_transitions = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim))
        self.layer_norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(4)])
        self.neighbor_num = neighbor_num

    def forward(self, h0, e0):
        h1, _ = self.multihead_message(
            h0[0, :, :].unsqueeze(0).repeat(self.neighbor_num, 1, 1), h0, h0
        )
        h2 = self.layer_norms[1](
            self.feed_forwards_message(self.layer_norms[0](h1)) + h0
        )
        mess_t = self.message_transitions(torch.concat((h2, e0), dim=-1))
        h_3, _ = self.multihead_update(mess_t, h2, h2)
        h_4 = self.layer_norms[3](
            self.feed_forwards_update(self.layer_norms[2](h_3) + h2)
        )
        return h_4


class AMPNN(nn.Module):
    def __init__(
        self,
        embed_dim=128,
        edge_dim=27,
        node_dim=38,
        n_heads=8,
        layer_nums=3,
        token_num=33,
        dropout=0.2,
        neighbor_num=32,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.edge_dim = edge_dim
        self.node_dim = node_dim
        self.layer_nums = layer_nums
        self.token_num = token_num
        self.dropout = dropout
        self.neighbor_num = neighbor_num
        self.init_node_embed = nn.Linear(node_dim, embed_dim, bias=False)
        self.init_edge_embed = nn.Linear(edge_dim, embed_dim, bias=False)
        self.layer_list = nn.ModuleList(
            [
                AMPNNLayer(
                    embed_dim=embed_dim,
                    n_heads=n_heads,
                    dropout=dropout,
                    neighbor_num=self.neighbor_num,
                )
                for _ in range(layer_nums)
            ]
        )
        self.lm_heads = nn.Linear(embed_dim, token_num)
        self.layer_norms = nn.LayerNorm(embed_dim)

    def forward(self, node_features, edge_features):
        h0 = self.init_node_embed(node_features)
        e0 = self.init_edge_embed(edge_features)
        for layer in self.layer_list:
            h0 = layer(h0, e0)

        return self.lm_heads(h0.sum(0)), h0


class Liteampnn(pl.LightningModule):
    def __init__(
        self,
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
        learning_rate=1e-3,
        neighbor_num=32,
    ) -> None:
        super().__init__()
        self.ampnn = AMPNN(
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            node_dim=node_dim,
            token_num=token_num,
            layer_nums=layer_nums,
            dropout=dropout,
            neighbor_num=neighbor_num,
        )
        self.learning_rate = learning_rate

        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=21, top_k=1
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=21, top_k=1
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=21, top_k=1
        )

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Liteampnn")
        parser.add_argument("--embed_dim", type=int, default=128)
        parser.add_argument("--edge_dim", type=int, default=27)
        parser.add_argument("--node_dim", type=int, default=28)
        parser.add_argument("--dropout", type=float, default=0.2)
        parser.add_argument("--layer_nums", type=int, default=3)
        parser.add_argument("--token_num", type=int, default=21)
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        return parent_parser

    def training_step(self, batch, batch_idx):
        node, edge, y = batch
        y_hat, h = self.ampnn(node, edge)
        loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.train_acc(y_hat, y.squeeze(0))
        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc)
        return loss

    def validation_step(self, batch, batch_idx):
        node, edge, y = batch
        y_hat, h = self.ampnn(node, edge)
        val_loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.valid_acc(y_hat, y.squeeze(0))
        self.log("val_loss", val_loss, sync_dist=True)
        self.log("val_acc_step", self.valid_acc, sync_dist=True)

    def test_step(self, batch, batch_idx):
        node, edge, y = batch
        y_hat, h = self.ampnn(node, edge)
        test_loss = nn.functional.cross_entropy(y_hat, y.squeeze(0))
        self.test_acc(y_hat, y.squeeze(0))
        self.log("test_loss", test_loss)
        self.log("test_acc_step", self.test_acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        return optimizer


if __name__ == "__main__":
    model = Liteampnn()
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        max_epochs=100,
    )
