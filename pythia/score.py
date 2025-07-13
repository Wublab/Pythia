import os

import torch

from pythia.masked_ddg_scan import get_torch_model, pythia_root_dpath
from pythia.pdb_utils import get_neighbor, read_pdb_to_protbb


def sequence_score_stab(models: list, pdb_file: str, device):
    protbb = read_pdb_to_protbb(pdb_file)

    node, edge, native_seq = get_neighbor(protbb, noise_level=0.00)
    native_score = 0
    for model in models:
        with torch.no_grad():
            y_hat, h_ = model(node.to(device).float(), edge.to(device).float())
            y_hat = y_hat.cpu()
            native_score += -torch.mean(
                torch.log(torch.gather(y_hat.softmax(-1), 1, native_seq.unsqueeze(-1)))
            )
    native_score /= len(models)
    return native_score.item()


if __name__ == "__main__":
    device = "cuda"
    fpath_pdb = "../examples/1pga.pdb"
    torch_model_c = get_torch_model(
        os.path.join(pythia_root_dpath, "../pythia-c.pt"), device
    )
    torch_model_p = get_torch_model(
        os.path.join(pythia_root_dpath, "../pythia-p.pt"), device
    )
    models = [torch_model_c, torch_model_p]
    print(sequence_score_stab(models, fpath_pdb, device))
