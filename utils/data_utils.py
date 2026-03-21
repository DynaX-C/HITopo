import torch
import json
from typing import Dict, List

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_graph_dict(graph_pt_path: str):
    graphs = torch.load(graph_pt_path, weights_only=False)
    return {
        'atom': graphs['atom_graphs'],
        'nb_atom': graphs['nb_atom_graphs'],
        'bd_atom': graphs['bd_atom_graphs'],
        'bond': graphs['bond_graphs'],
        'angle': graphs['angle_graphs'],
        'dihedral': graphs['dihedral_graphs'],
    }

def split_graphs_by_pdbid(
    graph_pt_path: str,
    split_json_path: str,
) -> Dict[str, Dict[str, List]]:

    train_graphs = torch.load(graph_pt_path, weights_only=False)

    graph_dict = {
        'atom': train_graphs['atom_graphs'],
        'nb_atom': train_graphs['nb_atom_graphs'],
        'bd_atom': train_graphs['bd_atom_graphs'],
        'bond': train_graphs['bond_graphs'],
        'angle': train_graphs['angle_graphs'],
        'dihedral': train_graphs['dihedral_graphs'],
    }

    pdbid_list = [g.pdbid for g in train_graphs['atom_graphs']]
    pdbid2idx = {pdbid: i for i, pdbid in enumerate(pdbid_list)}

    with open(split_json_path) as f:
        split = json.load(f)

    train_idx = [pdbid2idx[p] for p in split['train'] if p in pdbid2idx]
    val_idx   = [pdbid2idx[p] for p in split['validation'] if p in pdbid2idx]

    def _split(graphs, indices):
        return [graphs[i] for i in indices]

    train_set = {k: _split(v, train_idx) for k, v in graph_dict.items()}
    val_set   = {k: _split(v, val_idx)   for k, v in graph_dict.items()}

    for k in graph_dict:
        assert len(train_set[k]) == len(train_idx)
        assert len(val_set[k]) == len(val_idx)

    return {
        'train': train_set,
        'val': val_set
    }


def get_test_sets(args):
    """
    Return:
        dict[name] = graph_dict
    """
    test_sets = {}

    if args.data_mode == "temporalsplit":
        test_sets["casf2013"] = load_graph_dict(args.casf13_graph_pt_path)
        test_sets["casf2016"] = load_graph_dict(args.casf16_graph_pt_path)
        test_sets["test2019"] = load_graph_dict(args.test19_graph_pt_path)

    elif args.data_mode in ["cleansplit", "pdbbind2020"]:
        test_sets["casf2016"] = load_graph_dict(args.casf16_graph_pt_path)
        test_sets["casf16_indep"] = load_graph_dict(args.casf16_indep_graph_pt_path)
        test_sets["casf2013"] = load_graph_dict(args.casf13_graph_pt_path)
        test_sets["casf13_indep"] = load_graph_dict(args.casf13_indep_graph_pt_path)

    else:
        raise ValueError(f"Unknown data_mode: {args.data_mode}")

    return test_sets