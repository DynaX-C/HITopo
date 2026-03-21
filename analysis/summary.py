from pathlib import Path
import pandas as pd
import re
import argparse


def parse_decomp_file(decomp_path: Path) -> pd.DataFrame:
    rows = []
    pat = re.compile(r"^([A-Za-z]{3})\s*(\d+)$")

    with decomp_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("|") or "," not in line:
                continue

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 20:
                continue

            key = parts[0].replace(" ", "")
            m = pat.match(key)
            if not m:
                continue

            resname = m.group(1).upper()
            resid = int(m.group(2))

            try:
                etotal = float(parts[17])
            except ValueError:
                continue

            rows.append({
                "resid": resid,
                "resname": resname,
                "ETOTAL": etotal,
            })

    return pd.DataFrame(rows)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and align GBSA residue-wise decomposition with model attribution results."
    )
    
    parser.add_argument(
        "--model_attr_dir",
        type=str,
        required=True,
        help=(
            "Directory containing model attribution CSV files "
            "(e.g., *_residue_attribution.csv), where each file corresponds to one complex."
            )
    )

    parser.add_argument(
        "--work_dir",
        type=str,
        required=True,
        help=(
            "Root directory of prepared complexes (e.g., CASF dataset), "
            "organized as <work_dir>/<pdbid>/..., each containing GBSA outputs."
        )
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help=(
            "Output directory to save extracted and filtered GBSA decomposition results "
            "(per-complex CSV and combined summary file)."
        )
    )

    parser.add_argument(
        "--decomp_path",
        type=str,
        default="gbsa/FINAL_DECOMP_MMPBSA.dat",
        help=(
            "Relative path (from each pdbid folder) to the GBSA decomposition file "
            "(default: gbsa/FINAL_DECOMP_MMPBSA.dat)."
        )
    )

    return parser.parse_args()

def main(args):
    MODEL_ATTR_DIR = Path(args.model_attr_dir)
    WORK_DIR = Path(args.work_dir)
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DECOMP_PATH = Path(args.decomp_path)

    attrib_files = sorted(MODEL_ATTR_DIR.glob("*_residue_attribution.csv"))
    if not attrib_files:
        raise FileNotFoundError(f"No *_residue_attribution.csv found in: {MODEL_ATTR_DIR.resolve()}")

    combined_rows = []

    for af in attrib_files:
        pdbid = af.name.split("_")[0].lower()
        df_attr = pd.read_csv(af)

        if "resid" not in df_attr.columns:
            print(f"[SKIP] {af.name}: no 'resid' column")
            continue

        target_resids = sorted(set(int(x) for x in df_attr["resid"].dropna().tolist()))
        if not target_resids:
            print(f"[SKIP] {af.name}: empty resid list")
            continue

        decomp_path = WORK_DIR / pdbid / DECOMP_PATH
        if not decomp_path.exists():
            print(f"[MISS] {pdbid}: decomp file not found -> {decomp_path}")
            continue

        df_decomp = parse_decomp_file(decomp_path)
        if df_decomp.empty:
            print(f"[ERR ] {pdbid}: parsed decomp is empty -> {decomp_path}")
            continue

        df_sel = df_decomp[df_decomp["resid"].isin(target_resids)].copy()
        df_sel = df_sel.sort_values("resid").reset_index(drop=True)

        found = set(df_sel["resid"].tolist())
        missing = [r for r in target_resids if r not in found]
        if missing:
            print(f"[WARN] {pdbid}: missing {len(missing)} residues in decomp (e.g. {missing[:10]})")

        out_path = OUT_DIR / f"{pdbid}_pocket_decomp.csv"
        df_sel.to_csv(out_path, index=False)

        print(f"[OK  ] {pdbid}: wrote {out_path} ({len(df_sel)} residues)")

        if not df_sel.empty:
            df_tmp = df_sel.copy()
            df_tmp.insert(0, "pdbid", pdbid)
            combined_rows.append(df_tmp)

    if combined_rows:
        df_all = pd.concat(combined_rows, ignore_index=True)
        all_path = OUT_DIR / "ALL_pocket_decomp.csv"
        df_all.to_csv(all_path, index=False)
        print(f"\n[DONE] combined -> {all_path} (rows={len(df_all)})")
    else:
        print("\n[DONE] no outputs generated (all missing/empty).")


if __name__ == "__main__":
    args = parse_args()
    main(args)