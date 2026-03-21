import os
import json
import argparse
import torch
import pandas as pd

from model import build_model
from model.variants import apply_variant
from utils.config import load_config, override_args

import warnings
warnings.filterwarnings("ignore")


def cut_residue_ligand_edges(data, residue_id: int, ligand_id: int = -1, res_ids: torch.Tensor = None):
    """
    Remove edges between one residue and ligand in a graph.
    Returns a new Data object and the number of removed residue-ligand atom pairs.
    """
    edge_index = data.edge_index
    edge_attr = getattr(data, "edge_attr", None)

    is_res = (res_ids == residue_id)
    is_lig = (res_ids == ligand_id)

    u = edge_index[0]
    v = edge_index[1]

    is_res_lig_edge = (is_res[u] & is_lig[v]) | (is_lig[u] & is_res[v])

    keep = ~is_res_lig_edge

    new_edge_index = edge_index[:, keep]
    new_edge_attr = edge_attr[keep] if edge_attr is not None else None

    new_data = data.clone()
    new_data.edge_index = new_edge_index
    if edge_attr is not None:
        new_data.edge_attr = new_edge_attr

    return new_data


def get_residue_ids(ligand_id: int = -1, res_ids: torch.Tensor = None):
    res = torch.unique(res_ids)
    res = res[res != ligand_id]
    return res.tolist()


def build_residue_ablations(data, ligand_id: int = -1, res_ids: torch.Tensor = None):
    residue_ids = get_residue_ids(ligand_id=ligand_id, res_ids=res_ids)
    ablated_graphs = {}

    for rid in residue_ids:
        cut_graph = cut_residue_ligand_edges(
            data,
            residue_id=int(rid),
            ligand_id=ligand_id,
            res_ids=res_ids
        )
        ablated_graphs[int(rid)] = cut_graph

    return ablated_graphs


@torch.no_grad()
def residue_attribution_inference(
    model,
    model_paths,
    atom_graphs,
    nb_atom_graphs,
    bd_atom_graphs,
    bond_graphs,
    angle_graphs,
    dihedral_graphs,
    variant="full",
    device="cuda",
    save_dir="./results",
):
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device(device)
    model = model.to(device)
    model.eval()

    # base: unified atom graph -> cut atom_graph
    # others: cut nb_atom_graph
    cut_on_atom_graph = (variant in ["base", "nb"])

    print(f"[INFO] variant = {variant}, cut_graph = {'atom' if cut_on_atom_graph else 'nb_atom'}")

    for atom_g, nb_atom_g, bd_atom_g, bond_g, angle_g, dihedral_g in zip(
        atom_graphs, nb_atom_graphs, bd_atom_graphs,
        bond_graphs, angle_graphs, dihedral_graphs
    ):
        atom_g = atom_g.to(device)
        nb_atom_g = nb_atom_g.to(device)
        bd_atom_g = bd_atom_g.to(device)
        bond_g = bond_g.to(device)
        angle_g = angle_g.to(device)
        dihedral_g = dihedral_g.to(device)

        label = float(atom_g.y) if not isinstance(atom_g.y, torch.Tensor) else float(atom_g.y.view(-1)[0].detach().cpu().item())
        pdbid = atom_g.pdbid if isinstance(atom_g.pdbid, str) else str(atom_g.pdbid)

        g_base = {
            "atom_graph": atom_g,
            "nb_atom_graph": nb_atom_g,
            "bd_atom_graph": bd_atom_g,
            "bond_graph": bond_g,
            "angle_graph": angle_g,
            "dihedral_graph": dihedral_g,
        }

        # 1. baseline prediction
        base_sum = None
        for model_path in model_paths:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
            model.load_state_dict(state_dict)
            model = model.to(device)
            pred = model(g_base).detach().cpu().view(-1)
            base_sum = pred if base_sum is None else base_sum + pred

        base_pred = (base_sum / len(model_paths)).item()

        # 2. build residue perturbations
        res_ids = atom_g.res_ids
        if cut_on_atom_graph:
            ablations = build_residue_ablations(atom_g, ligand_id=-1, res_ids=res_ids)
        else:
            ablations = build_residue_ablations(nb_atom_g, ligand_id=-1, res_ids=res_ids)

        rows = []
        RT_ln10 = 1.363

        for resid, cut_graph in ablations.items():
            cut_graph = cut_graph.to(device)

            if cut_on_atom_graph:
                g_cut = {
                    "atom_graph": cut_graph,
                    "nb_atom_graph": nb_atom_g,
                    "bd_atom_graph": bd_atom_g,
                    "bond_graph": bond_g,
                    "angle_graph": angle_g,
                    "dihedral_graph": dihedral_g,
                }
            else:
                g_cut = {
                    "atom_graph": atom_g,
                    "nb_atom_graph": cut_graph,
                    "bd_atom_graph": bd_atom_g,
                    "bond_graph": bond_g,
                    "angle_graph": angle_g,
                    "dihedral_graph": dihedral_g,
                }

            preds_sum = None
            for model_path in model_paths:
                state_dict = torch.load(model_path, map_location="cpu", weights_only=False)
                model.load_state_dict(state_dict)
                model = model.to(device)
                preds = model(g_cut).detach().cpu().view(-1)
                preds_sum = preds if preds_sum is None else preds_sum + preds

            pred_mean = (preds_sum / len(model_paths)).item()

            # unified definition for all variants
            delta = base_pred - pred_mean
            ddg = -RT_ln10 * delta

            rows.append([
                int(resid),
                round(label, 4),
                round(base_pred, 4),
                round(pred_mean, 4),
                round(delta, 4),
                round(ddg, 4),
            ])

        df = pd.DataFrame(
            rows,
            columns=["resid", "label", "base_pred", "pred_cut", "delta", "ddG(kcal/mol)"]
        )

        csv_path = os.path.join(save_dir, f"{pdbid}_residue_attribution.csv")
        df.to_csv(csv_path, index=False)

        print(f"Saved: {csv_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--graph_pt_path", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)

    # optional overrides
    parser.add_argument("--device", type=str)
    parser.add_argument("--model_paths", type=str)

    known_args, unknown_args = parser.parse_known_args()

    args = load_config(known_args.config)
    args = override_args(args, unknown_args)
    args = apply_variant(args)

    args.graph_pt_path = known_args.graph_pt_path
    args.save_dir = known_args.save_dir

    if known_args.device is not None:
        args.device = known_args.device
    if known_args.model_paths is not None:
        args.model_paths = json.loads(known_args.model_paths)

    return args


if __name__ == "__main__":
    args = parse_args()

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    graphs = torch.load(args.graph_pt_path, map_location="cpu", weights_only=False)

    atom_graphs = graphs["atom_graphs"]
    nb_atom_graphs = graphs["nb_atom_graphs"]
    bd_atom_graphs = graphs["bd_atom_graphs"]
    bond_graphs = graphs["bond_graphs"]
    angle_graphs = graphs["angle_graphs"]
    dihedral_graphs = graphs["dihedral_graphs"]

    model = build_model(args)

    model_paths = args.model_paths
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    residue_attribution_inference(
        model=model,
        model_paths=model_paths,
        atom_graphs=atom_graphs,
        nb_atom_graphs=nb_atom_graphs,
        bd_atom_graphs=bd_atom_graphs,
        bond_graphs=bond_graphs,
        angle_graphs=angle_graphs,
        dihedral_graphs=dihedral_graphs,
        variant=args.variant,
        device=args.device,
        save_dir=args.save_dir,
    )