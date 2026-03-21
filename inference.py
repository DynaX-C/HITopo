import os
import json
import torch
import numpy as np
import argparse

from model import build_model
from model.variants import apply_variant
from utils.config import load_config, override_args
from data.dataloader import data_loader
from dataprep_workflow import prepare_and_build_graphs

import warnings
warnings.filterwarnings("ignore")


@torch.no_grad()
def predict_single(
    model,
    atom_loader,
    nb_atom_loader,
    bd_atom_loader,
    bond_loader,
    angle_loader,
    dihedral_loader,
    model_paths,
    device="cuda",
):
    model = model.to(device)
    model.eval()

    preds_all = []
    for atom_batch, nb_atom_batch, bd_atom_batch, bond_batch, angle_batch, dihedral_batch in zip(
        atom_loader, nb_atom_loader, bd_atom_loader,
        bond_loader, angle_loader, dihedral_loader
    ):
        atom_batch = atom_batch.to(device)
        nb_atom_batch = nb_atom_batch.to(device)
        bd_atom_batch = bd_atom_batch.to(device)
        bond_batch = bond_batch.to(device)
        angle_batch = angle_batch.to(device)
        dihedral_batch = dihedral_batch.to(device)

        batch = {
            "atom_graph": atom_batch,
            "nb_atom_graph": nb_atom_batch,
            "bd_atom_graph": bd_atom_batch,
            "bond_graph": bond_batch,
            "angle_graph": angle_batch,
            "dihedral_graph": dihedral_batch
        }

        preds_sum = None
        for model_path in model_paths:
            state_dict = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state_dict)
            model = model.to(device)
            preds = model(batch).detach().cpu().view(-1)

            if preds_sum is None:
                preds_sum = preds
            else:
                preds_sum += preds

        preds_mean = preds_sum / len(model_paths)
        preds_all.append(preds_mean)

    return preds_all


def parse_inference_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--protein_pdb", type=str, required=True)
    parser.add_argument("--ligand_mol2", type=str, required=True)

    known_args, unknown_args = parser.parse_known_args()

    args = load_config(known_args.config)
    args = override_args(args, unknown_args)

    args.protein_pdb = known_args.protein_pdb
    args.ligand_mol2 = known_args.ligand_mol2

    return args


def get_or_build_graph_dict(args):
    protein_dir = os.path.dirname(os.path.abspath(args.protein_pdb))
    graph_pt_path = os.path.join(protein_dir, "inference_graph.pt")

    if os.path.exists(graph_pt_path):
        print(f"Found existing graph file: {graph_pt_path}")
        graph_dict = torch.load(graph_pt_path, map_location="cpu", weights_only=False)
    else:
        print(f"Graph file not found. Building graphs and saving to: {graph_pt_path}")
        graph_dict = prepare_and_build_graphs(
            protein_pdb=args.protein_pdb,
            ligand_mol2=args.ligand_mol2,
            pocket_distance=getattr(args, "pocket_distance", 5),
            cutoff=getattr(args, "cutoff", 6),
        )
        torch.save(graph_dict, graph_pt_path)

    return graph_dict, graph_pt_path


if __name__ == "__main__":
    args = parse_inference_args()
    args = apply_variant(args)

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        args.device = "cpu"

    graph_dict, graph_pt_path = get_or_build_graph_dict(args)

    atom_loader, nb_atom_loader, bd_atom_loader, bond_loader, angle_loader, dihedral_loader = data_loader(
        [graph_dict["atom"]],
        [graph_dict["nb_atom"]],
        [graph_dict["bd_atom"]],
        [graph_dict["bond"]],
        [graph_dict["angle"]],
        [graph_dict["dihedral"]],
        batch_size=1,
        shuffle=False,
        seed=42,
    )

    model = build_model(args)

    model_paths = args.model_paths
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    preds = predict_single(
        model,
        atom_loader, nb_atom_loader, bd_atom_loader,
        bond_loader, angle_loader, dihedral_loader,
        model_paths=model_paths,
        device=args.device,
    )

    pred_value = float(torch.cat(preds).mean().item())

    print("\n===== Inference Result =====")
    print(f"Protein : {args.protein_pdb}")
    print(f"Ligand  : {args.ligand_mol2}")
    print(f"Graph   : {graph_pt_path}")
    print(f"Prediction (affinity): {pred_value:.4f}")
    print(f"Models used: {model_paths}")