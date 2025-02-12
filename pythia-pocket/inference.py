import torch
import os
ABS_PATH = os.path.dirname(os.path.abspath(__file__))
from models import MPNNModel, ProtBB, get_neighbor
from parse_data import restype_name_to_atom14_names, bb_atoms
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import three_to_one, is_aa
import numpy as np
from torch.utils.data import Dataset, DataLoader

def process_pdb_files(protein_pdb_file):
    parser = PDBParser(QUIET=True)
    protein_structure = parser.get_structure("protein", protein_pdb_file)

    pocket_residues = set()

    seq, coords, residue_mask, chain_ids, residue_numbers, is_pocket = extract_data(
        protein_structure, pocket_residues)

    result = {
        "seq": np.array(seq),
        "coords": np.array(coords),
        "residue_mask": np.array(residue_mask),
        "chain": np.array(chain_ids),
        "is_pocket": np.array(is_pocket),
        'resseq': np.array(residue_numbers)
    }

    return result


def extract_data(structure, pocket_residues_set):
    seq = ""
    coords = []
    # chi_angles = []
    residue_mask = []
    chain_ids = []
    residue_numbers = []
    is_pocket = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname()
                    try:
                        one_letter = three_to_one(resname)
                    except:
                        one_letter = "X"
                    atom_names = restype_name_to_atom14_names.get(
                        resname, restype_name_to_atom14_names["UNK"])
                    seq += one_letter
                    coord = np.zeros((14, 3))
                    if "CA" in residue and "C" in residue and "N" in residue and "O" in residue:
                        if one_letter == "X":
                            if "CB" not in residue:
                                b = residue['CA'].get_coord() - residue['N'].get_coord()
                                c = residue['C'].get_coord() - residue['CA'].get_coord()
                                a = np.cross(b, c)
                                Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + residue['CA'].get_coord()
                                coord[4] = Cb
                            for i, atom_name in enumerate(bb_atoms):
                                if atom_name in residue:
                                    coord[i] = residue[atom_name].get_coord()
                        for i, atom_name in enumerate(atom_names):
                            if atom_name in residue:
                                coord[i] = residue[atom_name].get_coord()
                        if 'CB' not in residue:
                            b = residue['CA'].get_coord() - residue['N'].get_coord()
                            c = residue['C'].get_coord() - residue['CA'].get_coord()
                            a = np.cross(b, c)
                            Cb = -0.58273431*a + 0.56802827*b - 0.54067466*c + residue['CA'].get_coord()
                            coord[4] = Cb
                        residue_mask.append(1)
                    else:
                        residue_mask.append(0)
                    coords.append(coord)
                    chain_ids.append(chain.id)
                    residue_number = residue.get_id()[1]
                    residue_numbers.append(residue_number)

                    res_id = f"{resname}_{chain.id}_{residue_number}"
                    is_pocket.append(1 if res_id in pocket_residues_set else 0)

    return seq, np.array(coords), residue_mask, chain_ids, residue_numbers, is_pocket

class PocketDataset(Dataset):
    def __init__(self, data:list, noise_level=0):
        self.data = data
        self.noise_level = noise_level
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        data = self.data[idx]
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
    

def load_model(model, model_weights_path, device):
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    return model

def inference(model, data, device):
    node_feat, edge_feat, E_idx, mask = data
    with torch.no_grad():
        logits = model(node_feat.to(device), edge_feat.to(device), E_idx.to(device), mask=mask.to(device))
        logits = logits.squeeze(-1) * mask.to(device)
        probabilities = torch.sigmoid(logits).detach().cpu().numpy()
    return probabilities

# Load the model
if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MPNNModel(6, 128, 22, 27)
    path_to_model = os.path.join(ABS_PATH, "model-fl-b.pt")
    model = load_model(model, model_weights_path=path_to_model, device=device)
    input_pdb = "aes72_af3.pdb"
    data = process_pdb_files(protein_pdb_file=input_pdb)
    # breakpoint()
    dataset = PocketDataset([data], noise_level=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for node_feat, edge_feat, E_idx, target, mask in loader:
            probabilities = inference(model, (node_feat, edge_feat, E_idx, mask), device)

    # print("Probabilities:", probabilities[0])

    pocket = []
    for p, r in zip(probabilities[0], data['resseq']):
        if p >= 0.6:
            pocket.append(str(r))
        # print(f"{r} {p}")
    print("Pocket residues: ", "+".join(pocket))
