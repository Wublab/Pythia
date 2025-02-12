import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np
from model_utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

restypes_3letters = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
restypes = ["A","R","N","D","C","Q","E","G","H","I","L","K","M","F","P","S","T","W","Y","V"]
ALPHABET = "ARNDCQEGHILKMFPSTWYVX-"
one_letter_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
class ProtBB:
    def __init__(self, ca:list, cb:list, c:list, n:list, o:list, seq:list, resseq:list, chain:list) -> None:
        self.ca = Tensor(np.array(ca)).unsqueeze(1)  # L, 1, 3
        self.cb = Tensor(np.array(cb)).unsqueeze(1)  # L, 1, 3
        self.c = Tensor(np.array(c)).unsqueeze(1)  # L, 1, 3
        self.n = Tensor(np.array(n)).unsqueeze(1)  # L, 1, 3
        self.o = Tensor(np.array(o)).unsqueeze(1)  # L, 1, 3
        # seq = [x]
        self.seq = Tensor(np.array([ALPHABET.index(aa) for aa in seq])).unsqueeze(-1)  # L, 1
        self.resseq = Tensor(np.array(resseq)).unsqueeze(-1)  # L, 1
        self.resolve_chain(chain)
        self.chain_num = Tensor(np.array(self.chain_num)).unsqueeze(-1)  # L, 1

    def resolve_chain(self, chain:list):
        self.chain_num = []
        chain_k = list(set(chain))
        extra_chain_k = []
        for k in chain_k:
            if k not in one_letter_chains:
                extra_chain_k.append(k)
        for c in chain:
            try:
                self.chain_num.append(one_letter_chains.index(c))
            except:
                self.chain_num.append(extra_chain_k.index(c)+len(one_letter_chains))
        # return chain

def mk_zero_prot(l:int):
    protbb = ProtBB(
        ca=[[0.0,0.0,0.0] for _ in range(l)],
        cb=[[0.0,0.0,0.0] for _ in range(l)],
        c=[[0.0,0.0,0.0] for _ in range(l)],
        n=[[0.0,0.0,0.0] for _ in range(l)],
        o=[[0.0,0.0,0.0] for _ in range(l)],
        resseq=torch.zeros((l, 1), dtype=torch.long),
        seq=torch.zeros((l, 1), dtype=torch.long),
        chain=torch.zeros((l, 1), dtype=torch.long)
    )
    return protbb

def get_neighbor(protbb, neighbor: int = 32, noise_level=0.0):
    # L, 1, 3
    L = len(protbb.ca)
    # if L < neighbor:
    #     zero_prot = mk_zero_prot(neighbor)
    #     zero_prot.ca[:L, :, :] = protbb.ca
    #     zero_prot.cb[:L, :, :] = protbb.cb
    #     zero_prot.c[:L, :, :] = protbb.c
    #     zero_prot.o[:L, :, :] = protbb.o
    #     zero_prot.n[:L, :, :] = protbb.n
    #     zero_prot.seq[:L, :] = protbb.seq
    #     zero_prot.resseq[:L, :] = protbb.resseq
    #     zero_prot.chain_num[:L, :] = protbb.chain_num
    #     protbb = zero_prot
    #     L = neighbor
    #     # print(L, protbb.ca.shape)
    assert len(protbb.ca) == len(protbb.resseq) == len(
        protbb.seq) == len(protbb.chain_num)
    
    d = torch.sqrt(torch.sum((protbb.ca - protbb.ca.reshape(1, L, 3))**2, -1))
    values, indices = torch.topk(d, np.minimum(neighbor, L), largest=False)
    rel_pos = torch.clamp(
        (protbb.resseq.T - protbb.resseq), min=-32, max=32)  # L, L
    rel_chains = (protbb.chain_num.T - protbb.chain_num).int()  # L, L
    pos_num = torch.gather(rel_pos, 1, indices)  # L, N
    chain_id = torch.gather(rel_chains, 1, indices)
    bb_coords = torch.cat(
        (protbb.ca, protbb.cb, protbb.c, protbb.n, protbb.o), dim=-2)

    if noise_level > 0.0:
        bb_coords += torch.rand_like(bb_coords) * noise_level
    # print(protbb.ca.shape, bb_coords.shape)
    d_x = bb_coords.unsqueeze(0).unsqueeze(
        2) - bb_coords.unsqueeze(1).unsqueeze(3)
    # print(d_x.shape)
    d = torch.sqrt(torch.square(d_x).sum(dim=-1)).reshape(L, L, 25)
    dis = torch.gather(
        d, 1, indices.unsqueeze(-1).repeat(1, 1, 25))  # L, N, 25
    nodes = F.one_hot(protbb.seq.long(), 22)
    
    edge = torch.cat((dis, pos_num.unsqueeze(-1), chain_id.unsqueeze(-1)), dim=-1)

    return nodes.squeeze(1), edge, indices

class PositionWiseFeedForward(nn.Module):
    def __init__(self, num_hidden, num_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.W_in = nn.Linear(num_hidden, num_ff, bias=True)
        self.W_out = nn.Linear(num_ff, num_hidden, bias=True)
        self.act = torch.nn.GELU()
    def forward(self, h_V):
        h = self.act(self.W_in(h_V))
        h = self.W_out(h)
        return h

class EncLayer(nn.Module):
    def __init__(self, num_hidden, num_in, dropout=0.1, num_heads=None, scale=30):
        super(EncLayer, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.scale = scale
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(num_hidden)
        self.norm2 = nn.LayerNorm(num_hidden)
        self.norm3 = nn.LayerNorm(num_hidden)

        self.W1 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W2 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W3 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W11 = nn.Linear(num_hidden + num_in, num_hidden, bias=True)
        self.W12 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.W13 = nn.Linear(num_hidden, num_hidden, bias=True)
        self.act = torch.nn.GELU()
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, E_idx, mask_V=None, mask_attend=None):
        """ Parallel computation of full transformer layer """
        # print(h_V.shape, h_E.shape, E_idx.shape)
        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W3(self.act(self.W2(self.act(self.W1(h_EV)))))
        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale
        h_V = self.norm1(h_V + self.dropout1(dh))

        dh = self.dense(h_V)
        h_V = self.norm2(h_V + self.dropout2(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V

        h_EV = cat_neighbors_nodes(h_V, h_E, E_idx)
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_EV.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_EV], -1)
        h_message = self.W13(self.act(self.W12(self.act(self.W11(h_EV)))))
        h_E = self.norm3(h_E + self.dropout3(h_message))
        return h_V, h_E

class MPNNModel(nn.Module):
    def __init__(self, num_layers, num_hidden, node_num_in, edge_num_in, dropout=0.1, scale=30):
        super(MPNNModel, self).__init__()
        self.node_embedding = nn.Linear(node_num_in, num_hidden)
        self.edge_embedding = nn.Linear(edge_num_in, num_hidden)
        self.enc_layers = nn.ModuleList([EncLayer(num_hidden, num_hidden*2, dropout, scale) for _ in range(num_layers)])
        self.output_layer = nn.Linear(num_hidden, 1)


    def forward(self, node_feat, edge_feat, E_idx, mask, repr=False):
        h_V = self.node_embedding(node_feat)
        h_E = self.edge_embedding(edge_feat)
        for enc_layer in self.enc_layers:
            h_V, h_E = enc_layer(h_V, h_E, E_idx, mask)
        logits = self.output_layer(h_V)
        if repr:
            return {
                'logits': logits,
                'representation': h_V
            }
        return logits

class PocketDataset(Dataset):
    def __init__(self, names, noise_level=0):
        self.names = names
        self.noise_level = noise_level
    def __len__(self):
        return len(self.names)
    def __getitem__(self, idx):
        data = torch.load(self.names[idx])
        protbb = ProtBB(ca= data['coords'][:,1,:].tolist(), 
                        cb= data['coords'][:,4,:].tolist(), 
                        c = data['coords'][:,2,:].tolist(), 
                        n = data['coords'][:,0,:].tolist(), 
                        o = data['coords'][:,3,:].tolist(), 
                        seq = data['seq'].tolist(), 
                        resseq = data['resseq'].tolist(), 
                        chain = data['chain'].tolist())
        node_feat, edge_feat, E_idx = get_neighbor(protbb, noise_level=self.noise_level)
        target = torch.tensor(data['is_pocket']).float()
        mask = torch.tensor(data['residue_mask'])
        return node_feat.float(), edge_feat, E_idx, target, mask

### test
if __name__ == '__main__':
    import glob
    debug = False
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MPNNModel(6, 128, 22, 27)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    names = glob.glob("/data/jsun/database/pdb_bind/v2020-other-PL/pts/*.pt")
    train_data = names[:int(len(names)*0.8)]
    test_data = names[int(len(names)*0.8):]
    train_dataset = PocketDataset(train_data)
    test_dataset = PocketDataset(test_data)
    if debug:
        train_dataset = PocketDataset(train_data[:100])
        test_dataset = PocketDataset(train_data[:100])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    best_auc = 0
    for epoch in range(30):
        train_loss = []
        test_loss = []
        train_acc = []
        test_acc = []
        train_targets = []
        train_preds = []
        test_targets = []
        test_preds = []
        for data in tqdm(train_loader):
            optimizer.zero_grad()
            node_feat, edge_feat, E_idx, target, mask = data
            logits = model(node_feat.to(device), edge_feat.to(device), E_idx.to(device), mask=mask.to(device))
            logits = logits.squeeze(-1) * mask.to(device)
            loss = criterion(logits, target.to(device))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            train_targets += target.numpy().ravel().tolist()
            pred = logits.sigmoid().detach().cpu().numpy().ravel().tolist()
            train_preds += pred
            acc = (logits.sigmoid().round() == target.to(device)).float().mean()
            train_acc.append(acc.item())
        train_auc = roc_auc_score(train_targets, train_preds)
            # print(loss.item())
        for data in tqdm(test_loader):
            with torch.no_grad():
                node_feat, edge_feat, E_idx, target, mask = data
                logits = model(node_feat.to(device), edge_feat.to(device), E_idx.to(device), mask=mask.to(device))
                # logits = logits['logits']#.squeeze(-1) * mask.to(device)
                logits = logits.squeeze(-1) * mask.to(device)
                loss = criterion(logits, target.to(device))
                test_loss.append(loss.item())
                acc = (logits.sigmoid().round() == target.to(device)).float().mean()
                test_acc.append(acc.item())
                test_targets += target.numpy().ravel().tolist()
                pred = logits.sigmoid().detach().cpu().numpy().ravel().tolist()
                test_preds += pred
        test_auc = roc_auc_score(test_targets, test_preds)
            # print(loss.item())
        print("Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, train auc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}, test auc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc), train_auc, np.mean(test_loss), np.mean(test_acc), test_auc))
        if test_auc > best_auc:
            print("!!!Last Best Epoch: {}, train loss: {:.4f}, train acc: {:.4f}, train auc: {:.4f}, test loss: {:.4f}, test acc: {:.4f}, test auc: {:.4f}".format(epoch, np.mean(train_loss), np.mean(train_acc), train_auc, np.mean(test_loss), np.mean(test_acc), test_auc))
            best_auc = test_auc
            torch.save(model.state_dict(), f"model_v0.pt")