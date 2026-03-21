import os
import glob
import argparse
import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def safe_spearman(x, y):
    if len(x) < 3:
        return np.nan, np.nan
    if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
        return np.nan, np.nan
    return spearmanr(x, y)


def main(args):
    ATTR_DIR = args.attr_dir
    ENERGY_DIR = args.energy_dir
    OUT_CSV = args.out_csv

    rows = []

    attr_files = sorted(glob.glob(os.path.join(ATTR_DIR, "*_residue_attribution.csv")))

    if not attr_files:
        raise FileNotFoundError(f"No attribution files found in {ATTR_DIR}")

    for attr_path in attr_files:
        pdbid = os.path.basename(attr_path).split("_")[0].lower()
        energy_path = os.path.join(ENERGY_DIR, f"{pdbid}_{args.energy_suffix}.csv")

        if not os.path.exists(energy_path):
            print(f"[SKIP] {pdbid}: no energy file")
            continue

        df_attr = pd.read_csv(attr_path)
        df_energy = pd.read_csv(energy_path)

        df = pd.merge(df_attr, df_energy, on="resid", how="inner")

        if len(df) < 3:
            print(f"[SKIP] {pdbid}: too few residues ({len(df)})")
            continue

        ddg = df["ddG(kcal/mol)"].values
        e_total = df["ETOTAL"].values

        rho_total, p_total = safe_spearman(ddg, e_total)

        rows.append({
            "pdbid": pdbid,
            "rho_total": rho_total,
            "p_total": p_total,
            "n_res": len(df)
        })

        print(
            f"{pdbid}: "
            f"ETOTAL rho={rho_total:.3f} | "
            f"(n={len(df)})"
        )

    if not rows:
        print("[WARN] No valid complexes processed.")
        return

    df_out = pd.DataFrame(rows).sort_values("rho_total", ascending=False)
    df_out.to_csv(OUT_CSV, index=False)

    print("\nSaved:", OUT_CSV)

    print("\nSummary:")
    print(df_out[["rho_total"]].describe())

def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-complex Spearman correlation between model attribution and GBSA decomposition."
    )

    parser.add_argument(
        "--attr_dir",
        type=str,
        required=True,
        help="Directory containing *_residue_attribution.csv files (model outputs)."
    )

    parser.add_argument(
        "--energy_dir",
        type=str,
        required=True,
        help="Directory containing GBSA decomposition CSV files."
    )

    parser.add_argument(
        "--energy_suffix",
        type=str,
        default="pocket_decomp",
        help="Suffix of energy file (default: pdbid_pocket_decomp.csv)."
    )

    parser.add_argument(
        "--out_csv",
        type=str,
        default="spearman_per_complex.csv",
        help="Output CSV file for correlation results."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)