import os
import json
import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

from model import build_model
from model.variants import apply_variant
from utils.config import parse_config_and_override
from utils.data_utils import load_graph_dict, set_seed
from data.dataloader import data_loader

import warnings
warnings.filterwarnings("ignore")


def build_loader_from_graph_dict(graph_dict, batch_size=128, seed=42):
    return data_loader(
        graph_dict["atom"],
        graph_dict["nb_atom"],
        graph_dict["bd_atom"],
        graph_dict["bond"],
        graph_dict["angle"],
        graph_dict["dihedral"],
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
    )


def get_test_sets(args):
    """
    Return:
        dict[str, dict] : {test_name: graph_dict}
    """
    test_sets = {}

    if args.data_mode == "temporalsplit":
        test_sets["casf2013"] = load_graph_dict(args.casf13_graph_pt_path)
        test_sets["casf2016"] = load_graph_dict(args.casf16_graph_pt_path)
        test_sets["pdbbind2019"] = load_graph_dict(args.pdbbind19_graph_pt_path)

    elif args.data_mode in ["cleansplit", "pdbbind2020"]:
        test_sets["casf2016"] = load_graph_dict(args.casf16_graph_pt_path)
        test_sets["casf16_indep"] = load_graph_dict(args.casf16_indep_graph_pt_path)
        test_sets["casf2013"] = load_graph_dict(args.casf13_graph_pt_path)
        test_sets["casf13_indep"] = load_graph_dict(args.casf13_indep_graph_pt_path)

    else:
        raise ValueError(f"Unknown data_mode: {args.data_mode}")

    return test_sets


@torch.no_grad()
def test_and_save(
    model,
    test_atom_loader,
    test_nb_atom_loader,
    test_bd_atom_loader,
    test_bond_loader,
    test_angle_loader,
    test_dihedral_loader,
    model_paths,
    device="cuda",
    save_csv_path="predictions.csv",
):
    model = model.to(device)
    model.eval()

    all_pdbids = []
    all_predictions = []
    all_labels = []

    for atom_batch, nb_atom_batch, bd_atom_batch, bond_batch, angle_batch, dihedral_batch in zip(
        test_atom_loader, test_nb_atom_loader, test_bd_atom_loader,
        test_bond_loader, test_angle_loader, test_dihedral_loader
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

        labels = nb_atom_batch.y.detach().cpu().view(-1)

        preds_sum = None
        for model_path in model_paths:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
            preds = model(batch).detach().cpu().view(-1)

            if preds_sum is None:
                preds_sum = preds
            else:
                preds_sum += preds

        preds_mean = preds_sum / len(model_paths)

        pdbids = atom_batch.pdbid
        if isinstance(pdbids[0], torch.Tensor):
            pdbids = [p.item() if isinstance(p, torch.Tensor) else str(p) for p in pdbids]

        all_pdbids.extend(pdbids)
        all_predictions.extend(preds_mean.tolist())
        all_labels.extend(labels.tolist())

    pear_corr, _ = pearsonr(all_labels, all_predictions)
    mse = mean_squared_error(all_labels, all_predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_labels, all_predictions)

    df = pd.DataFrame({
        "pdbid": all_pdbids,
        "label": all_labels,
        "prediction": all_predictions
    })
    df.to_csv(save_csv_path, index=False)

    print(f"Saved: {save_csv_path}")
    print(f"Pearson r: {pear_corr:.4f}")
    print(f"MSE      : {mse:.4f}")
    print(f"RMSE     : {rmse:.4f}")
    
    return {
        "pearson_r": pear_corr,
        "mse": mse,
        "rmse": rmse,
    }


if __name__ == "__main__":
    args = parse_config_and_override()
    args = apply_variant(args)

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    set_seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    config_path = os.path.join(
        args.out_dir,
        f"test_config_{args.variant}_{args.model}_{args.data_mode}.json"
    )
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    model = build_model(args)

    model_paths = args.model_paths
    if isinstance(model_paths, str):
        model_paths = [model_paths]

    test_sets = get_test_sets(args)

    summary_rows = []

    for test_name, graph_dict in test_sets.items():
        print(f"\n===== {test_name} =====")

        atom_loader, nb_atom_loader, bd_atom_loader, bond_loader, angle_loader, dihedral_loader = \
            build_loader_from_graph_dict(
                graph_dict,
                batch_size=args.batch_size,
            )

        save_csv_path = os.path.join(
            args.out_dir,
            f"predictions_{args.variant}_{args.model}_{args.data_mode}_{test_name}.csv"
        )

        metrics = test_and_save(
            model,
            atom_loader,
            nb_atom_loader,
            bd_atom_loader,
            bond_loader,
            angle_loader,
            dihedral_loader,
            model_paths=model_paths,
            device=args.device,
            save_csv_path=save_csv_path
        )

        summary_rows.append({
            "test_set": test_name,
            **metrics
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(
        args.out_dir,
        f"summary_{args.variant}_{args.model}_{args.data_mode}.csv"
    )
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")
