from Bio.PDB import PDBParser
import numpy as np
from Bio.PDB.Polypeptide import three_to_one, is_aa
import torch
restypes_3letters = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", "",],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}
bb_atoms = ["N", "CA", "C", "O", "CB"]


def process_pdb_files(pocket_pdb_file, protein_pdb_file):
    parser = PDBParser(QUIET=True)

    pocket_structure = parser.get_structure("pocket", pocket_pdb_file)
    protein_structure = parser.get_structure("protein", protein_pdb_file)

    pocket_residues = set()
    for model in pocket_structure:
        for chain in model:
            for residue in chain:
                if is_aa(residue):
                    resname = residue.get_resname()
                    residue_number = residue.get_id()[1]
                    pocket_residues.add(
                        f"{resname}_{chain.id}_{residue_number}")

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


if __name__ == "__main__":

    import glob
    from tqdm import tqdm
    pdb_bind_db = '/data/jsun/database/pdb_bind/v2020-other-PL/'
    pdbs = [x.split("/")[-1]
            for x in glob.glob(f'{pdb_bind_db}*')]
    # print(pdbs)
    for pdb in tqdm(pdbs):
        if len(pdb) != 4:
            continue
        result = process_pdb_files(
            f"{pdb_bind_db}{pdb}/{pdb}_pocket.pdb", f"{pdb_bind_db}{pdb}/{pdb}_protein.pdb")
        torch.save(result, f"{pdb_bind_db}pts/{pdb}.pt")