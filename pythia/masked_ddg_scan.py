import glob
import gzip
from model import *
from pdb_utils import *
from Bio.PDB.Polypeptide import index_to_one
from tqdm import tqdm

import warnings
from Bio import BiopythonDeprecationWarning

warnings.filterwarnings("ignore", category=BiopythonDeprecationWarning)


def get_torch_model(ckpt_path, device='cuda'):
    model = AMPNN(
        embed_dim = 128,
        edge_dim = 27,
        node_dim = 28,
        dropout = 0.2,
        layer_nums = 3,
        token_num = 21,
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=torch.device(device)))
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

def make_one_scan(pdb_file, torch_models:list, save_pt=False):
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
                energy += -np.log(prob[pos]/prob[pos][aa_index])
            data_dict[f"{one_letter_aa}_{pos+1}"] = np.float16(energy)
        torch.save(data_dict, f'{pdb_file.replace(".pdb", "")}_pred_mask.pt')
        print(f"save {pdb_file.replace('.pdb', '')}_pred_mask.pt")
    else:
        with open(f'{pdb_file.replace(".pdb", "")}_pred_mask.txt', 'w') as f:
            for pos, aa in enumerate(protbb.seq):
                energy = np.zeros(21)
                for prob in probs:
                    energy += -np.log(prob[pos]/prob[pos][int(aa.item())])
                for i in range(20):
                    f.write(f"{index_to_one(int(aa.item()))}{pos+1}{index_to_one(i)} {energy[i]}\n")

if __name__ == "__main__":
    from joblib import Parallel, delayed

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch_model_c = get_torch_model("../pythia-c.pt")
    torch_model_p = get_torch_model("../pythia-p.pt")

    run_dir = True
    check_plddt = False

    if run_dir:
        files = glob.glob('/root/autodl-tmp/fitness/output_unzipped/*/*.pdb')
        print(len(files))
        if check_plddt:
            confident_list = []
            for pdb_file in tqdm(files):
                plddt = cal_plddt(pdb_file)
                if plddt > 95:
                    confident_list.append(pdb_file)
            files = confident_list
        Parallel(n_jobs=2)(delayed(make_one_scan)(pdb_file, [torch_model_c, torch_model_p]) for pdb_file in tqdm(files))
        
    pdb_filename = None
    if pdb_filename:
        make_one_scan(pdb_filename, [torch_model_c, torch_model_p])