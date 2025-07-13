import argparse
import glob
import gzip
import os
import warnings

import numpy as np
import torch
import torch.nn.functional as F
from Bio import BiopythonDeprecationWarning
from Bio.PDB.Polypeptide import index_to_one
from joblib import Parallel, delayed
from tqdm import tqdm

from pythia.model import AMPNN
from pythia.pdb_utils import get_neighbor, read_pdb_to_protbb

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)
pythia_root_dpath = os.path.dirname(os.path.abspath(__file__))


def get_torch_model(ckpt_path, device="cuda"):
    model = AMPNN(
        embed_dim=128,
        edge_dim=27,
        node_dim=28,
        dropout=0.2,
        layer_nums=3,
        token_num=21,
    )
    model.load_state_dict(
        torch.load(ckpt_path, map_location=torch.device(device), weights_only=True)
    )
    model.eval()
    model.to(device)
    return model


def cal_plddt(pdb_file):
    bs = []
    if pdb_file.endswith(".pdb.gz"):
        with gzip.open(pdb_file, "rt") as f:
            for line in f:
                if line.startswith("ATOM"):
                    b = float(line[60:66])
                    bs.append(b)
    if pdb_file.endswith(".pdb"):
        with open(pdb_file, "r") as f:
            for line in f:
                if line.startswith("ATOM"):
                    b = float(line[60:66])
                    bs.append(b)
    return np.mean(bs)


def make_one_scan(pdb_file, torch_models: list, device, save_pt=False):
    protbb = read_pdb_to_protbb(pdb_file)
    node, edge, seq = get_neighbor(protbb, noise_level=0.0)
    probs = []
    with torch.no_grad():
        for torch_model in torch_models:
            logits, _ = torch_model(node.to(device), edge.to(device))
            prob = F.softmax(logits, dim=-1).detach().cpu().numpy()
            probs.append(prob)
    if save_pt:
        data_dict = {}
        for pos, aa in enumerate(protbb.seq):
            energy = np.zeros(21)
            aa_index = int(aa.item())
            one_letter_aa = index_to_one(aa_index)
            for prob in probs:
                energy += -np.log(prob[pos] / prob[pos][aa_index])
            data_dict[f"{one_letter_aa}_{pos+1}"] = np.float16(energy)
        torch.save(data_dict, f'{pdb_file.replace(".pdb", "")}_pred_mask.pt')
        print(f"save {pdb_file.replace('.pdb', '')}_pred_mask.pt")
    else:
        with open(f'{pdb_file.replace(".pdb", "")}_pred_mask.txt', "w") as f:
            for pos, aa in enumerate(protbb.seq):
                energy = np.zeros(21)
                for prob in probs:
                    energy += -np.log(prob[pos] / prob[pos][int(aa.item())])
                for i in range(20):
                    f.write(
                        f"{index_to_one(int(aa.item()))}{pos+1}{index_to_one(i)} {energy[i]}\n"
                    )


def main():
    args = parse_args()
    input_dir = args.input_dir
    pdb_filename = args.pdb_filename
    check_plddt = args.check_plddt
    plddt_cutoff = args.plddt_cutoff
    n_jobs = args.n_jobs
    device = args.device

    run_dir = bool(input_dir)

    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # thx to zhenglz
    torch_model_c = get_torch_model(
        os.path.join(pythia_root_dpath, "pythia-c.pt"), device
    )
    torch_model_p = get_torch_model(
        os.path.join(pythia_root_dpath, "pythia-p.pt"), device
    )

    if run_dir:
        files = glob.glob(f"{input_dir}*.pdb")
        print(len(files))
        if check_plddt:
            confident_list = []
            for pdb_file in tqdm(files):
                plddt = cal_plddt(pdb_file)
                if plddt > plddt_cutoff:
                    confident_list.append(pdb_file)
            files = confident_list
        Parallel(n_jobs=n_jobs)(
            delayed(make_one_scan)(pdb_file, [torch_model_c, torch_model_p], device)
            for pdb_file in tqdm(files)
        )

    if pdb_filename:
        make_one_scan(pdb_filename, [torch_model_c, torch_model_p], device)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Command line interface for the given code."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="../s669_AF_PDBs/",
        help="Input directory path.",
    )
    parser.add_argument(
        "--pdb_filename",
        type=str,
        default=None,
        help="Path to a specific PDB filename.",
    )
    parser.add_argument(
        "--check_plddt", action="store_true", help="Flag to check pLDDT value."
    )
    parser.add_argument(
        "--plddt_cutoff", type=float, default=95, help="pLDDT cutoff value."
    )
    parser.add_argument(
        "--n_jobs", type=int, default=2, help="Number of parallel jobs."
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Try to use gpu:0")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
