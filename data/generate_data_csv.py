import json
import csv
import argparse


def load_index(index_file):
    data = {}

    with open(index_file, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("#") or line == "":
                continue

            parts = line.split()

            pdbid = parts[0].lower()
            neglog = float(parts[3])

            data[pdbid] = neglog

    return data


def main(args):

    index_data = load_index(args.index_file)

    if args.split_json is not None and args.split_name is not None:
        with open(args.split_json, "r") as f:
            splits = json.load(f)

        if args.split_name not in splits:
            raise ValueError(f"split {args.split_name} not found in json")

        target_pdbids = [p.lower() for p in splits[args.split_name]]
    else:
        target_pdbids = list(index_data.keys())

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pdbid", "-logKd/Ki"])

        rows = []
        missing = []

        for pdbid in target_pdbids:
            if pdbid in index_data:
                rows.append((pdbid, index_data[pdbid]))
            else:
                missing.append(pdbid)

        rows.sort(key=lambda x: x[1])

        for pdbid, neglog in rows:
            writer.writerow([pdbid, neglog])

    print(f"Saved CSV to {args.output}")

    if missing:
        print("Missing entries:")
        for m in missing:
            print(m)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Generate CSV from PDBbind index and split JSON"
    )

    parser.add_argument(
        "--index_file",
        type=str,
        required=True,
        help="INDEX_general_PL_data.2020 file"
    )

    parser.add_argument(
        "--split_json",
        type=str,
        required=False,
        help="PDBbind_data_split_cleansplit.json"
    )

    parser.add_argument(
        "--split_name",
        type=str,
        required=False,
        help="target split name (e.g., casf2016)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default="output.csv",
        help="output csv file"
    )

    args = parser.parse_args()

    if args.split_json is not None and args.split_name is None:
        parser.error("--split_name is required when --split_json is provided")

    main(args)