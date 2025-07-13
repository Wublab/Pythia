import gzip
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import FastMMCIFParser, PDBParser
from Bio.PDB.Polypeptide import three_to_index
from torch import Tensor
from torch.utils.data import Dataset


class ProtBB:
    def __init__(
        self,
        ca: list,
        cb: list,
        c: list,
        n: list,
        o: list,
        seq: list,
        resseq: list,
        chain: list,
        bb_ang: list,
        sasa: list,
    ) -> None:
        self.ca = Tensor(np.array(ca)).unsqueeze(1)  # L, 1, 3
        self.cb = Tensor(np.array(cb)).unsqueeze(1)  # L, 1, 3
        self.c = Tensor(np.array(c)).unsqueeze(1)  # L, 1, 3
        self.n = Tensor(np.array(n)).unsqueeze(1)  # L, 1, 3
        self.o = Tensor(np.array(o)).unsqueeze(1)  # L, 1, 3
        self.seq = Tensor(np.array(seq))  # L, 1
        self.resseq = Tensor(np.array(resseq))  # L, 1
        self.chain_num = Tensor(np.array(chain))  # L, 1
        self.bb_ang = Tensor(np.array(bb_ang))  # L, 6
        self.sasa = Tensor(np.array(sasa))  # L, 5


def mk_zero_prot(l: int):
    protbb = ProtBB(
        ca=[[0.0, 0.0, 0.0] for _ in range(l)],
        cb=[[0.0, 0.0, 0.0] for _ in range(l)],
        c=[[0.0, 0.0, 0.0] for _ in range(l)],
        n=[[0.0, 0.0, 0.0] for _ in range(l)],
        o=[[0.0, 0.0, 0.0] for _ in range(l)],
        resseq=torch.zeros((l, 1), dtype=torch.long),
        seq=torch.zeros((l, 1), dtype=torch.long),
        chain=torch.zeros((l, 1), dtype=torch.long),
        bb_ang=torch.zeros((l, 6)),
        sasa=torch.zeros((l, 5)),
    )
    return protbb


def read_pdb_to_protbb(pdb_file: str):
    cas = []
    cbs = []
    cs = []
    os = []
    ns = []
    resseqs = []
    seqs = []
    chains = []
    bb_ang = []
    sasa = []
    try:
        if pdb_file.endswith(".pdb"):
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("x", pdb_file)[0]
        if pdb_file.endswith(".cif"):
            cif_parser = FastMMCIFParser(QUIET=True)
            structure = cif_parser.get_structure("x", pdb_file)[0]
        if pdb_file.endswith(".pdb.gz"):
            with gzip.open(pdb_file, "rt") as f:
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("x", f)[0]
    except:
        print(pdb_file)
    structure.atom_to_internal_coordinates()
    chain_dict = {}
    for chain in structure.get_chains():
        if chain.id not in chain_dict:
            chain_dict[chain.id] = len(chain_dict)
        chain_id = chain_dict[chain.id]
        for residue in chain.get_residues():
            if (
                "N" in residue and "C" in residue and "O" in residue and "CA" in residue
            ) and residue.id[0] == " ":
                try:
                    cb = residue["CB"].coord
                except:
                    b = residue["CA"].coord - residue["N"].coord
                    c_ = residue["C"].coord - residue["CA"].coord
                    a = np.cross(b, c_)
                    cb = (
                        -0.58273431 * a
                        + 0.56802827 * b
                        - 0.54067466 * c_
                        + residue["CA"].coord
                    )
                chains.append([chain_id])
                cas.append(residue["CA"].coord)
                cbs.append(cb)
                cs.append(residue["C"].coord)
                os.append(residue["O"].coord)
                ns.append(residue["N"].coord)
                resseqs.append([residue.full_id[3][1]])
                try:
                    tok_id = three_to_index(residue.get_resname())
                except:
                    tok_id = 20
                seqs.append([tok_id])
                ric = residue.internal_coord
                phi = ric.get_angle("phi")
                psi = ric.get_angle("psi")
                omg = ric.get_angle("omg")
                if phi == None:
                    phi = 0
                if psi == None:
                    psi = 0
                if omg == None:
                    omg = 0
                bb_ang.append(
                    np.concatenate(
                        [
                            np.sin(np.deg2rad([phi, psi, omg])),
                            np.cos(np.deg2rad([phi, psi, omg])),
                        ]
                    )
                )
    return ProtBB(cas, cbs, cs, os, ns, seqs, resseqs, chains, bb_ang, sasa)


def get_neighbor_old(protbb, neighbor: int = 32, noise_level=0.0, train=False):
    # L, 1, 3
    L = len(protbb.ca)
    if L < neighbor:
        zero_prot = mk_zero_prot(neighbor)
        zero_prot.ca[:L, :, :] = protbb.ca
        zero_prot.cb[:L, :, :] = protbb.cb
        zero_prot.c[:L, :, :] = protbb.c
        zero_prot.o[:L, :, :] = protbb.o
        zero_prot.n[:L, :, :] = protbb.n
        zero_prot.seq[:L, :] = protbb.seq
        zero_prot.resseq[:L, :] = protbb.resseq
        zero_prot.chain_num[:L, :] = protbb.chain_num
        zero_prot.bb_ang[:L, :] = protbb.bb_ang
        protbb = zero_prot
        L = neighbor
    assert (
        len(protbb.ca) == len(protbb.resseq) == len(protbb.seq) == len(protbb.chain_num)
    )

    d = torch.sqrt(torch.sum((protbb.ca - protbb.ca.reshape(1, L, 3)) ** 2, -1))
    _, indices = torch.topk(d, neighbor, largest=False)
    rel_pos = torch.clamp((protbb.resseq.T - protbb.resseq), min=-32, max=32)  # L, L
    rel_chains = (protbb.chain_num.T - protbb.chain_num).int()  # L, L
    pos_num = torch.gather(rel_pos, 1, indices)  # L, N
    chain_id = torch.gather(rel_chains, 1, indices)
    seq_toks = torch.gather(protbb.seq.T.repeat(L, 1), 1, indices)  # L, N
    # seq_toks[:,0] = 21 # Y=19,X=20,<mask> = 21
    # seq_toks[0, 0] = 21 # make sure there is a <mask>
    if train:
        mask_prob = torch.rand(seq_toks.shape[0])
        seq_toks[:, 0] = (
            torch.tensor([21] * seq_toks.shape[0]) * (mask_prob < 0.85).int()
            + torch.randint(0, 20, (seq_toks.shape[0],)) * (mask_prob >= 0.85).int()
        )
    else:
        seq_toks[:, 0] = 21  # masked all for inference

    bb_ang = torch.gather(
        protbb.bb_ang.unsqueeze(1).repeat(1, L, 1),
        1,
        indices.unsqueeze(-1).repeat(1, 1, 6),
    )

    bb_coords = torch.cat((protbb.ca, protbb.cb, protbb.c, protbb.n, protbb.o), dim=-2)

    if noise_level > 0.0:
        bb_coords += torch.rand_like(bb_coords) * noise_level
    d_x = bb_coords.unsqueeze(0).unsqueeze(2) - bb_coords.unsqueeze(1).unsqueeze(3)
    d = torch.sqrt(torch.square(d_x).sum(dim=-1)).reshape(L, L, 25)
    dis = torch.gather(d, 1, indices.unsqueeze(-1).repeat(1, 1, 25))  # L, N, 25
    nodes = torch.cat((F.one_hot(seq_toks.long(), 22), bb_ang), dim=-1).transpose(1, 0)
    edge = torch.cat((dis, pos_num.unsqueeze(-1), chain_id.unsqueeze(-1)), dim=-1)

    return nodes, edge.transpose(1, 0), protbb.seq.squeeze(-1).long()


def get_neighbor(protbb, neighbor: int = 32, noise_level=0.0, train=False):
    L = len(protbb.ca)

    # 1. Pad sequence if it's shorter than the number of neighbors
    if L < neighbor:
        # The implementation of mk_zero_prot is assumed to work correctly
        zero_prot = mk_zero_prot(neighbor)
        (
            zero_prot.ca[:L],
            zero_prot.cb[:L],
            zero_prot.c[:L],
            zero_prot.o[:L],
            zero_prot.n[:L],
        ) = (protbb.ca, protbb.cb, protbb.c, protbb.o, protbb.n)
        (
            zero_prot.seq[:L],
            zero_prot.resseq[:L],
            zero_prot.chain_num[:L],
            zero_prot.bb_ang[:L],
        ) = (protbb.seq, protbb.resseq, protbb.chain_num, protbb.bb_ang)
        protbb = zero_prot
        L = neighbor

    assert len(protbb.ca) == L

    # 2. Efficiently compute C-alpha distances and find neighbor indices
    # Use torch.cdist for more efficient distance calculation
    ca_coords = protbb.ca.squeeze(1)  # Shape: [L, 3]
    d = torch.cdist(ca_coords, ca_coords)  # Shape: [L, L]
    _, indices = torch.topk(
        d, neighbor, largest=False
    )  # Shape: [L, k] (where k=neighbor)

    # 3. Compute edge features on-demand for neighbors only
    # Squeeze feature tensors from [L, 1] to [L] for easier indexing
    resseq_flat = protbb.resseq.squeeze(-1)
    chain_num_flat = protbb.chain_num.squeeze(-1)

    # Use advanced indexing to directly get features of neighbors
    neighbor_resseq = resseq_flat[indices]  # Shape: [L, k]
    neighbor_chains = chain_num_flat[indices]  # Shape: [L, k]

    # Calculate relative values by broadcasting [L, k] and [L, 1] tensors
    pos_num = torch.clamp(neighbor_resseq - resseq_flat.unsqueeze(1), min=-32, max=32)
    chain_id = (neighbor_chains - chain_num_flat.unsqueeze(1)).int()

    # 4. Compute node features on-demand for neighbors only
    seq_flat = protbb.seq.squeeze(-1)

    # Directly index to get neighbor sequence tokens and angles
    seq_toks = seq_flat[indices]  # Shape: [L, k]
    bb_ang = protbb.bb_ang[indices]  # Shape: [L, k, 6]

    # Masking logic (same as original)
    if train:
        mask_prob = torch.rand(L, device=protbb.ca.device)
        random_toks = torch.randint(0, 20, (L,), device=protbb.ca.device)
        # Use torch.where for a cleaner conditional assignment
        seq_toks[:, 0] = torch.where(mask_prob < 0.85, 21, random_toks)
    else:
        seq_toks[:, 0] = 21

    # 5. Efficiently compute backbone atom distances (most significant optimization)
    bb_coords = torch.cat(
        (protbb.ca, protbb.cb, protbb.c, protbb.n, protbb.o), dim=-2
    )  # Shape: [L, 5, 3]
    if noise_level > 0.0:
        bb_coords += torch.rand_like(bb_coords) * noise_level

    # Get neighbor backbone coordinates directly using indexing
    neighbor_coords = bb_coords[indices]  # Shape: [L, k, 5, 3]
    # Expand center residue coordinates for broadcasting
    center_coords = bb_coords.unsqueeze(1)  # Shape: [L, 1, 5, 3]

    # Compute distances only on the [L, k] subset
    # Broadcasting [L, 1, 5, 1, 3] and [L, k, 1, 5, 3] results in [L, k, 5, 5, 3]
    d_x = center_coords.unsqueeze(3) - neighbor_coords.unsqueeze(2)
    dis = torch.sqrt(torch.square(d_x).sum(dim=-1)).reshape(L, neighbor, 25)

    # 6. Combine final outputs (logic is same as original)
    nodes = torch.cat((F.one_hot(seq_toks.long(), 22), bb_ang), dim=-1)
    edge = torch.cat((dis, pos_num.unsqueeze(-1), chain_id.unsqueeze(-1)), dim=-1)

    # Return transposed results
    return nodes.transpose(1, 0), edge.transpose(1, 0), protbb.seq.squeeze(-1).long()


def parallel_converter(pdb):
    protbb = read_pdb_to_protbb(pdb)
    return protbb


def save_all(all_pdbs):
    all_Protbb = Parallel(n_jobs=22)(
        delayed(parallel_converter)(pdb) for pdb in tqdm(all_pdbs)
    )
    for pdb, protbb in zip(all_pdbs, all_Protbb):
        with open(f"{pdb}.pkl", "wb") as ofile:
            pickle.dump(protbb, ofile)


class myDataset(Dataset):
    def __init__(
        self,
        protbbs: list,
        meta_batchsize=2000,
        noise=0.0,
        neighbor=32,
        max_length=4000,
    ) -> None:
        super().__init__()
        self.protbbs = protbbs
        self.meta_batchsize = meta_batchsize
        self.make_metabatch()
        self.noise = noise
        self.neighbor = neighbor
        self.max_length = max_length

    def make_metabatch(self):
        self.batchs = []
        total_length = 0
        batch = []
        for _, protbb in enumerate(self.protbbs):
            if len(protbb.seq) > 2000:
                continue
            else:
                total_length += len(protbb.seq)
                if total_length < self.meta_batchsize:
                    batch.append(protbb)
                else:
                    self.batchs.append(batch)
                    total_length = len(protbb.seq)
                    batch = [protbb]
        self.batchs.append(batch)

    def __getitem__(self, idx):
        batch = self.batchs[idx]
        nodes = []
        edges = []
        targets = []
        for protbb in batch:
            node, edge, target = get_neighbor(
                protbb, neighbor=self.neighbor, noise_level=self.noise
            )
            nodes.append(node)
            edges.append(edge)
            targets.append(target)
        return (
            torch.cat(nodes, dim=1).float(),
            torch.cat(edges, dim=1).float(),
            torch.cat(targets).long(),
        )

    def __len__(self):
        return len(self.batchs)


if __name__ == "__main__":
    import glob

    from joblib import Parallel, delayed
    from model import AMPNN
    from tqdm import tqdm

    all_pdbs = glob.glob("/data3/jsun/dompdb/clean/*.pdb")
    save_all(all_pdbs)
