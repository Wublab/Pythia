import glob
import os
import warnings

import pandas as pd
from Bio import BiopythonDeprecationWarning
from Bio.PDB.Polypeptide import index_to_one, one_to_three, three_to_one
from model import *
from pdb_utils import *
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


def mk_res_to_index(pdb):
    res_to_index = {}
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("test", pdb)[0]
    i = 0
    for chain in structure.get_chains():
        for residue in chain.get_residues():
            if (
                "N" in residue and "C" in residue and "O" in residue and "CA" in residue
            ) and residue.id[0] == " ":
                res_name = three_to_one(residue.get_resname())
                res_key = f"{res_name}_{chain.id}_{residue.id[1]}"
                res_to_index[res_key] = i
                i += 1
    return res_to_index


def mutate_node(node_feature, mutation: str):
    wild_tok, pos, mut_tok = mutation.split("_")
    wild_feat = node_feature.clone()[:, int(pos) - 1, :]
    wild_feat[0, :22] = 0
    wild_feat[0, three_to_index(one_to_three(wild_tok))] = 1
    mut_feat = node_feature.clone()[:, int(pos) - 1, :]
    mut_feat[0, :22] = 0
    mut_feat[0, three_to_index(one_to_three(mut_tok))] = 1
    return wild_feat, mut_feat


class InfDDGDataset(Dataset):
    def __init__(self, probb: ProtBB, mutations: list, noise=0.0) -> None:
        super().__init__()
        self.probb = probb
        self.mutations = mutations
        self.noise = noise

    def __len__(self):
        return len(self.mutations)

    def __getitem__(self, index):
        mutation = self.mutations[index]
        seq_index = int(mutation.split("_")[1]) - 1
        node, edge, seq = get_neighbor(self.probb, noise_level=self.noise)
        wild_feat, mut_feat = mutate_node(node, mutation)
        return wild_feat, mut_feat, edge[:, seq_index, :]


def get_alphabet():
    alphabet = ""
    for i in range(20):
        alphabet += index_to_one(i)
    alphabet += "X"
    return alphabet


def run_predict(torch_model, dataloader):
    with torch.no_grad():
        all_preds = []
        for batch in tqdm(dataloader):
            wild_feat, mut_feat, edge = batch
            w_y_hat, _ = torch_model(
                wild_feat.transpose(1, 0).to(device), edge.transpose(1, 0).to(device)
            )
            m_y_hat, _ = torch_model(
                mut_feat.transpose(1, 0).to(device), edge.transpose(1, 0).to(device)
            )
            y_hat = torch.log(
                (
                    (nn.Softmax(dim=-1)(m_y_hat))
                    * mut_feat.to(device).transpose(1, 0)[0, :, :21]
                )
                .sum(-1)
                .unsqueeze(-1)
            ) - torch.log(
                (
                    (nn.Softmax(dim=-1)(w_y_hat))
                    * wild_feat.to(device).transpose(1, 0)[0, :, :21]
                )
                .sum(-1)
                .unsqueeze(-1)
            )
            all_preds = all_preds + y_hat.cpu().ravel().numpy().tolist()
    return all_preds


def get_torch_model(ckpt_path, device="cuda"):
    model = AMPNN(
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
    model.eval()
    model.to(device)
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_s669 = True

    if test_s669:
        path_to_pdb = "../s669_AF_PDBs/"
        data_df = pd.read_csv("../s669_data_chain_A.csv")
        ddgs = []
        all_preds = []
        names = []
        datasets = []
        for pdb in tqdm(data_df["pdb"].unique()):
            sub_df = data_df[data_df["pdb"] == pdb]
            # print(pdb)
            pdb_file_path = os.path.join(path_to_pdb, f"{pdb}.pdb")
            mutations = []
            res_to_index = mk_res_to_index(pdb_file_path)
            for index, row in sub_df.iterrows():
                res_key = f"{row['wildtype']}_{row['chain']}_{row['resseq']}"
                seq_index = res_to_index[res_key]
                mutations.append(f"{row['wildtype']}_{seq_index+1}_{row['mutation']}")
                names.append(
                    f"{row['pdb']}_{row['wildtype']}_{seq_index+1}_{row['mutation']}"
                )
                ddgs.append(row["ddg"])
            probb = read_pdb_to_protbb(pdb_file_path)
            dataset = InfDDGDataset(probb, mutations, noise=0.0)
            datasets.append(dataset)

        train_dev_sets = torch.utils.data.ConcatDataset(datasets)
        dataloader = DataLoader(train_dev_sets, batch_size=669, shuffle=False)

        models = ["../pythia-c.pt", "../pythia-p.pt"]
        data_df["pythiascore"] = 0
        for m in models:
            torch_model = get_torch_model(m, device=device)
            torch_model.eval()

            all_preds = run_predict(torch_model, dataloader)

            data_df["pythiascore"] += np.array(all_preds)
        data_df["name"] = names
        data_df["ddG"] = ddgs
        df_out = data_df[["name", "ddG", "pythiascore"]]
        df_out.to_csv(f"../pythia_s669.csv", index=None, sep=",")
        print(spearmanr(data_df["pythiascore"], ddgs))
        print(pearsonr(data_df["pythiascore"], ddgs))
