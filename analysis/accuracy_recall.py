import os
import glob
import argparse
import pandas as pd
import numpy as np


def main(args):
    ATTR_DIR = args.attr_dir
    ENERGY_DIR = args.energy_dir
    OUT_CSV = args.out_csv

    rows = []

    attr_files = sorted(glob.glob(os.path.join(ATTR_DIR, "*_residue_attribution.csv")))

    for attr_path in attr_files:
        pdbid = os.path.basename(attr_path).split("_")[0].lower()
        energy_path = os.path.join(ENERGY_DIR, f"{pdbid}_pocket_decomp.csv")

        if not os.path.exists(energy_path):
            continue

        df_attr = pd.read_csv(attr_path)
        df_energy = pd.read_csv(energy_path)

        df = pd.merge(df_attr, df_energy, on="resid", how="inner")

        if len(df) < 5:
            continue

        df["phys_favorable"] = df["ETOTAL"] < 0
        df["model_favorable"] = df["ddG(kcal/mol)"] < 0

        correct = (df["phys_favorable"] == df["model_favorable"])

        overall_acc = correct.mean()

        fav_mask = df["phys_favorable"]

        fav_recall = (df.loc[fav_mask, "model_favorable"] == True).mean() if fav_mask.any() else np.nan

        rows.append({
            "pdbid": pdbid,
            "n_residues": len(df),
            "overall_accuracy": overall_acc,
            "favorable_recall": fav_recall,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUT_CSV, index=False)

    print("Saved:", OUT_CSV)
    print(df_out.describe())


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attr_dir", type=str, required=True)
    parser.add_argument("--energy_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)