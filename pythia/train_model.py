import glob
import os
import pickle

import pytorch_lightning as pl
import torch
import torchmetrics
from joblib import Parallel, delayed
from model import AMPNN
from pdb_utils import myDataset, parallel_converter
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm


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
    ) -> None:
        super().__init__()
        self.ampnn = AMPNN(
            embed_dim=embed_dim,
            edge_dim=edge_dim,
            node_dim=node_dim,
            token_num=token_num,
            layer_nums=layer_nums,
            dropout=dropout,
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
        optimizer = optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-4)
        return optimizer


def get_dataset(
    path: str,
    list_file=None,
    file_type=["pdb", "pkl"],
    noise=0.0,
    neighbor=48,
    plus=False,
):
    if list_file == None:
        if file_type == "pkl":
            all_pkls = glob.glob(os.path.join(path, "*.pkl"))
            all_Protbb = []
            for pkl in tqdm(all_pkls):
                protbb = pickle.load(open(pkl, "rb"))
                all_Protbb.append(protbb)
        if file_type == "pdb":
            all_pdbs = glob.glob(os.path.join(path, "*.pdb"))
            all_Protbb = Parallel(n_jobs=-1)(
                delayed(parallel_converter)(pdb) for pdb in tqdm(all_pdbs)
            )
    else:
        all_files = open(list_file, "r").read().split("\n")[:-1]
        if file_type == "pkl":
            all_Protbb = []
            for pkl in tqdm(all_files):
                protbb = pickle.load(open(pkl, "rb"))
                all_Protbb.append(protbb)
        if file_type == "pdb":
            all_Protbb = Parallel(n_jobs=-1)(
                delayed(parallel_converter)(pdb) for pdb in tqdm(all_files)
            )

    if plus:
        # dataset = myDatasetPlus(
        #     all_Protbb, noise=noise, neighbor=neighbor, meta_batchsize=1400
        # )
        pass
    else:
        dataset = myDataset(
            all_Protbb, noise=noise, neighbor=neighbor, meta_batchsize=2000
        )

    return dataset


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser = Liteampnn.add_model_specific_args(parser)
    parser.add_argument("--train_data_dir", type=str, default="./train_data/")
    parser.add_argument("--test_data_dir", type=str, default="./test_data/")
    parser.add_argument("--train_list_file", type=str, default="./train_of_list.txt")
    parser.add_argument("--test_list_file", type=str, default="./test_list.txt")

    parser.add_argument("--file_type", type=str, default="pdb")
    parser.add_argument("--valid_num", type=int, default=512)
    args = parser.parse_args()

    train_data = get_dataset(
        "", file_type="pkl", list_file="mem_train_list2.txt", noise=0.50, neighbor=32
    )
    test_data = get_dataset(
        "", file_type="pkl", list_file="mem_test_list.txt", noise=0.00, neighbor=32
    )

    valid_data = get_dataset(
        "", file_type="pkl", list_file="mem_valid_list.txt", noise=0.00, neighbor=32
    )

    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        dirpath="./checkpoints/",
        filename="model-{epoch:02d}-{val_loss:.3f}",
    )

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        precision=32,
        max_epochs=40,
        callbacks=[checkpoint_callback],
    )

    # split the train set into two
    seed = torch.Generator().manual_seed(42)
    train_loader = DataLoader(
        train_data, batch_size=None, shuffle=True, num_workers=6, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_data, batch_size=None, shuffle=False, num_workers=6, pin_memory=True
    )
    test_loader = DataLoader(
        test_data, batch_size=None, shuffle=False, num_workers=6, pin_memory=True
    )

    litmodel = Liteampnn(
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
        learning_rate=0.0001,
    )

    trainer.fit(
        model=litmodel, train_dataloaders=train_loader, val_dataloaders=valid_loader
    )

    trainer.test(
        model=litmodel,
        ckpt_path="best",
        dataloaders=test_loader,
    )
