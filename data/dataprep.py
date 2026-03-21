import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import MDAnalysis as mda
import parmed as pmd
from tqdm import tqdm
import pandas as pd
from pdbfixer import PDBFixer
from openmm.app import PDBFile
import warnings
from multiprocessing import Pool
from rdkit import RDLogger

warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def reorder(input_pdb, remove_hydrogens=False):
    output_pdb = input_pdb.replace(".pdb", "_reorder.pdb")

    structure = pmd.load_file(input_pdb)
    if remove_hydrogens:
        structure.strip("@H=")
    structure.save(output_pdb, overwrite=True)

    return output_pdb


def fix_pdb(input_pdb, cover=True, remove_hydrogens=False):
    output_pdb = input_pdb.replace(".pdb", "_fixed.pdb")

    if not cover and os.path.exists(output_pdb):
        print(f"File '{output_pdb}' already exists. Skipping fixing.")
        return output_pdb
    
    fixer = PDBFixer(filename=input_pdb)

    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    keys = fixer.missingResidues.keys()
    for key in list(keys):
        chain = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain.residues())):
            del fixer.missingResidues[key]

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()

    fixer.removeHeterogens(keepWater=False)

    fixer.findMissingAtoms()
    fixer.addMissingAtoms()

    PDBFile.writeFile(
        fixer.topology,
        fixer.positions,
        open(output_pdb,"w"),keepIds=True)
    reordered_pdb = reorder(output_pdb, remove_hydrogens=remove_hydrogens)
    return reordered_pdb


def get_ligand_pdb(ligand_mol2_file, remove_hydrogens=False):
    output_pdb = ligand_mol2_file.replace(".mol2", ".pdb")

    cmd = f"obabel -imol2 {ligand_mol2_file} -O {output_pdb}"
    if remove_hydrogens:
        cmd += " -d"
    cmd += " > /dev/null 2>&1"

    ret = os.system(cmd)
    if ret != 0 or not os.path.exists(output_pdb):
        raise RuntimeError(f"OpenBabel conversion failed for {ligand_mol2_file}")

    return output_pdb


def combine_structure(protein_pdb, ligand_pdb):
    base_path = os.path.dirname(protein_pdb)
    output_pdb = os.path.join(base_path, 'complex.pdb')
    
    atom_serial = 1
    output_lines = []
    
    # ---------------------------
    # Protein (Chain A)
    # ---------------------------
    last_res_name = "UNK"
    last_res_seq = "   1"
    
    with open(protein_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                res_name = line[17:20]
                res_seq = line[22:26]
                
                last_res_name = res_name
                last_res_seq = res_seq

                new_serial = f"{atom_serial:5d}"
                
                new_line = line[:6] + new_serial + line[11:21] + 'A' + line[22:]
                output_lines.append(new_line)
                atom_serial += 1
    
    ter_line = f"TER   {atom_serial:5d}      {last_res_name} A{last_res_seq}\n"
    output_lines.append(ter_line)
    atom_serial += 1

    # ---------------------------
    # Ligand (Chain B, Name MOL)
    # ---------------------------
    last_lig_res_name = "MOL"
    last_lig_res_seq = "   1"
    with open(ligand_pdb, 'r') as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                new_serial = f"{atom_serial:5d}"
                replaced_line = line[:6] + new_serial + line[11:17] + last_lig_res_name + " B" + line[22:]
                last_lig_res_seq = line[22:26]
                
                output_lines.append(replaced_line)
                atom_serial += 1
                
    ter_line_ligand = f"TER   {atom_serial:5d}      MOL B{last_lig_res_seq}\n"
    output_lines.append(ter_line_ligand)
    output_lines.append("END\n")

    with open(output_pdb, 'w') as f:
        f.writelines(output_lines)

    return output_pdb

def extract_pocket(com_pdb='complex.pdb', ligand_name='MOL', distance=5, remove_hydrogens=False):
    u = mda.Universe(com_pdb)
    selection = u.select_atoms(f'(resname {ligand_name}) or (around {distance} (resname {ligand_name}))').residues.atoms

    if remove_hydrogens:
        selection = selection.select_atoms('not name H*')

    pocket_pdb_file = f'complex_{distance}A.pdb'
    base_path = os.path.dirname(com_pdb)
    output_pdb = os.path.join(base_path, pocket_pdb_file)
    selection.write(output_pdb)
    
    return output_pdb

def process_row(row):
    pdbid = str(row['pdbid']).lower()
    work_dir = row['work_dir']

    protein_file = f"{work_dir}/{pdbid}/{pdbid}_protein.pdb"
    ligand_file = f"{work_dir}/{pdbid}/{pdbid}_ligand.mol2"
    pocket_file = f"{work_dir}/{pdbid}/complex_5A.pdb"

    if os.path.exists(pocket_file):
        return pdbid, True, pocket_file, None

    try:
        fixed_pdb = fix_pdb(protein_file, remove_hydrogens=False)
        ligand_pdb = get_ligand_pdb(ligand_file, remove_hydrogens=False)
        complex_pdb = combine_structure(fixed_pdb, ligand_pdb)
        pocket_pdb = extract_pocket(complex_pdb, ligand_name='MOL', distance=5, remove_hydrogens=True)
        return pdbid, True, pocket_pdb, None
    except Exception as e:
        return pdbid, False, None, str(e)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    index_file = args.index_file
    work_dir = args.work_dir
    num_workers = args.num_workers

    df_index = pd.read_csv(index_file)
    df_index["work_dir"] = work_dir

    rows = [row for _, row in df_index.iterrows()]

    total = len(rows)
    success = 0
    failed_list = []
    results = []

    with Pool(processes=num_workers) as pool:
        for pdbid, ok, pocket_pdb, err in tqdm(
            pool.imap_unordered(process_row, rows),
            total=total,
            desc="Processing PDB files"
        ):
            results.append((pdbid, pocket_pdb))
            if ok:
                success += 1
            else:
                failed_list.append(pdbid)
                print(f"Error processing {pdbid}: {err}")

    failed = len(failed_list)

    print("\n========== SUMMARY ==========")
    print(f"Total entries : {total}")
    print(f"Success       : {success}")
    print(f"Failed        : {failed}")

    if failed_list:
        print("\nFailed PDBIDs:")
        for pdbid in failed_list:
            print(pdbid)

    df_results = pd.DataFrame(results, columns=["pdbid", "pocket_pdb"])
    # df_results.to_csv("pocket_results.csv", index=False)

    if failed_list:
        pd.DataFrame({"failed_pdbid": failed_list}).to_csv("failed_pdbids.csv", index=False)