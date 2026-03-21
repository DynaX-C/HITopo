import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import subprocess
import argparse
import MDAnalysis as mda
import parmed as pmd
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
from .dataprep import fix_pdb
import warnings
from rdkit import RDLogger
warnings.filterwarnings('ignore')
RDLogger.DisableLog('rdApp.*')


def mol2_am1_bcc(mol2_file, charge_sum=None, multiplicity=None):
    ligand_sdf_file = mol2_file.replace(".mol2", ".sdf")

    mol = Chem.MolFromMol2File(mol2_file, removeHs=False, sanitize=False)
    if mol is None:
        return False

    # calculate charge sum
    if charge_sum is None:
        charge_sum = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())

    # calculate multiplicity
    if multiplicity is None:
        multiplicity = 1 + sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())

    subprocess.run(['antechamber', '-i', mol2_file, '-fi', 'mol2', '-o', 'ligand_bcc.mol2',
               '-fo', 'mol2', '-c', 'bcc', '-nc', str(charge_sum), '-m', str(multiplicity),
               '-at', 'gaff2', '-pf', 'y', '-dr', 'no', '-rn', 'MOL'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not os.path.exists('ligand_bcc.mol2'):
        suppl = Chem.SDMolSupplier(ligand_sdf_file, removeHs=False, sanitize=False, strictParsing=False)
        mol_list = [m for m in suppl if m is not None]
        if len(mol_list) == 1:
            mol = mol_list[0]
            if charge_sum is None:
                charge_sum = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
            if multiplicity is None:
                multiplicity = 1 + sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            subprocess.run(['antechamber', '-i', ligand_sdf_file, '-fi', 'sdf', '-o', 'ligand_bcc.mol2',
                          '-fo', 'mol2', '-c', 'bcc', '-nc', str(charge_sum), '-m', str(multiplicity),
                          '-at', 'gaff2', '-pf', 'y', '-dr', 'no', '-rn', 'MOL'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(['parmchk2', '-i', 'ligand_bcc.mol2', '-f', 'mol2', '-o', 'ligand_bcc.frcmod',
                    '-s', 'gaff2'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists('ligand_bcc.mol2') and os.path.exists('ligand_bcc.frcmod'):
        return True
    else:
        return False

def extract_ssbonds(protein_pdb):
    structure = pmd.load_file(protein_pdb)
    out_pdb = protein_pdb.replace(".pdb", "_ssbonds.pdb")
    ssbond_list = []
    for bond in structure.bonds:
        if bond.atom1.name=='SG' and  bond.atom2.name=='SG' and bond.atom1.residue.name=='CYS' and bond.atom2.residue.name=='CYS':
            bond.atom1.residue.name = 'CYX'
            bond.atom2.residue.name = 'CYX'
            ssbond_list.append((bond.atom1.residue.idx+1, bond.atom2.residue.idx+1))
    structure.save(out_pdb, format="pdb", overwrite=True)
    return ssbond_list, out_pdb

def leap(protein_pdb, ssbond_list):
    if len(ssbond_list) > 0:
        ssbond_leap = "\n".join([f"bond p.{a}.SG p.{b}.SG" for a, b in ssbond_list])
    else:
        ssbond_leap = ""

    leap_input = """\
source leaprc.protein.ff14SB
source leaprc.gaff2
set default PBradii mbondi2
loadamberparams ligand_bcc.frcmod

p = loadpdb {protein_pdb}
l = loadmol2 ligand_bcc.mol2
c = combine {{ p l }}
{ssbond_leap}

savepdb p pro.pdb
savepdb l lig.pdb
savepdb c com.pdb
saveamberparm p pro.top pro.crd
saveamberparm l lig.top lig.crd
saveamberparm c com.top com.crd
quit
""".format(protein_pdb=protein_pdb, ssbond_leap=ssbond_leap)

    with open("leap.in", "w") as f:
        f.write(leap_input)

    try:
        result = subprocess.run(["tleap", "-sf", "leap.in"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except FileNotFoundError:
        print("Error: tleap not found. Make sure AmberTools is installed and in your PATH.")
        return False

    if result.returncode != 0:
        print("tleap execution failed!")
        print(result.stderr)
        return False

    if not os.path.exists("leap.log"):
        print("Error: leap.log not found. tleap might have failed.")
        return False

    with open("leap.log", "r") as f:
        lines = f.readlines()

    if not lines:
        print("Error: leap.log is empty.")
        return False

    if "Errors = 0" in lines[-1]:
        return True
    else:
        error_lines = [line.strip() for line in lines if "Error" in line or "error" in line]
        print("Errors detected in leap.log:")
        for err in error_lines:
            print(err)
        return False

def minimize_energy(
    crd_file,
    top_file,
    output_pdb='min.pdb',
    stage1_maxcyc=500,
    stage1_ncyc=250,
    stage1_restraint_wt=500.0,
    stage1_restraint_mask='!@H=',
    stage2_maxcyc=1000,
    stage2_ncyc=500,
    cut=999.0,
    keep_intermediate=True,
):
    """
    Performs a two-stage energy minimization using Amber's sander module.
    """

    stage1_in = 'min_stage1.in'
    stage1_out = 'min_stage1.out'
    stage1_rst = 'min_stage1.rst'

    stage2_in = 'min_stage2.in'
    stage2_out = 'min_stage2.out'
    stage2_rst = 'min_stage2.rst'

    # -------------------------
    # Stage 1: Optimize only hydrogens with heavy atom restraints
    # -------------------------
    amber_input_stage1 = f"""Two-stage minimization: Stage 1 (H-only optimization)
&cntrl
  imin=1,
  maxcyc={stage1_maxcyc},
  ncyc={stage1_ncyc},
  ntpr=100,
  ntb=0,
  igb=5,
  gbsa=1,
  cut={cut},
  ntr=1,
  restraint_wt={stage1_restraint_wt},
  restraintmask='{stage1_restraint_mask}',
/
"""

    # -------------------------
    # Stage 2: Full minimization without restraints
    # -------------------------
    amber_input_stage2 = f"""Two-stage minimization: Stage 2 (full minimization)
&cntrl
  imin=1,
  maxcyc={stage2_maxcyc},
  ncyc={stage2_ncyc},
  ntpr=100,
  ntb=0,
  igb=5,
  gbsa=1,
  cut={cut},
  ntr=0,
/
"""

    try:
        with open(stage1_in, 'w') as f:
            f.write(amber_input_stage1)

        with open(stage2_in, 'w') as f:
            f.write(amber_input_stage2)

        # -------------------------
        # run stage 1
        # -------------------------
        cmd1 = [
            "sander", "-O",
            "-i", stage1_in,
            "-o", stage1_out,
            "-p", top_file,
            "-c", crd_file,
            "-r", stage1_rst,
            "-ref", crd_file
        ]
        subprocess.run(cmd1, check=True, capture_output=True, text=True)

        # -------------------------
        # run stage 2
        # -------------------------
        cmd2 = [
            "sander", "-O",
            "-i", stage2_in,
            "-o", stage2_out,
            "-p", top_file,
            "-c", stage1_rst,
            "-r", stage2_rst
        ]
        subprocess.run(cmd2, check=True, capture_output=True, text=True)

        # -------------------------
        # output
        # -------------------------
        with open(output_pdb, 'w') as fout:
            subprocess.run(
                ["ambpdb", "-p", top_file, "-c", stage2_rst],
                check=True,
                stdout=fout,
                text=True
            )

        if not keep_intermediate:
            for fn in [stage1_in, stage1_out, stage1_rst,
                       stage2_in, stage2_out, stage2_rst]:
                if os.path.exists(fn):
                    os.remove(fn)

        return output_pdb

    except subprocess.CalledProcessError as e:
        print("Amber sander failed!")
        print(e)
        if e.stderr:
            print(e.stderr)
        return None


def extract_pocket(com_pdb='com.pdb', com_top='com.top', lig_pdb='lig.pdb', distance=5):
    '''
    Extracts the complex from a pdb file and saves it as a new pdb file and a new prmtop file.
    The complex is defined as the ligand and all residues within a certain distance from the ligand.
    com_pdb: str, path to the pdb file of the complex
    com_top: str, path to the top file of the complex
    lig_pdb: str, path to the pdb file of the ligand
    distance: float, distance in Angstrom to define the complex
    '''
    ligand_u = mda.Universe(lig_pdb)
    ligand_name = ligand_u.atoms.residues[0].resname

    u = mda.Universe(com_top, com_pdb)

    selection = u.select_atoms(f'(resname {ligand_name}) or (around {distance} (resname {ligand_name}))').residues.atoms
    pocket_pdb_file = f'complex_{distance}A_em.pdb'
    selection.write(pocket_pdb_file)

    structure = pmd.load_file(com_top, structure=True)
    selected_structure = structure[selection.indices]
    pocket_top_file = f'complex_{distance}A_em.prmtop'
    selected_structure.save(pocket_top_file, overwrite=True)

    return ligand_name, pocket_pdb_file, pocket_top_file

def process_one_pdb(row, base_dir, orig_cwd):
    pdbid = row["pdbid"]
    work_dir = os.path.join(base_dir, pdbid)

    final_pdb_file = os.path.join(work_dir, 'complex_5A_em.pdb')
    final_top_file = os.path.join(work_dir, 'complex_5A_em.prmtop')

    if os.path.exists(final_pdb_file) and os.path.exists(final_top_file):
        return pdbid, True, "already exists"

    try:
        os.chdir(work_dir)

        protein_file = f"{pdbid}_protein.pdb"
        ligand_file = f"{pdbid}_ligand.mol2"

        fixed_pdb = fix_pdb(protein_file, remove_hydrogens=True)
        ssbond_list, ssbond_protein = extract_ssbonds(fixed_pdb)

        if not leap(ssbond_protein, ssbond_list):
            return pdbid, False, "leap failed"

        min_pdb = minimize_energy("com.crd", "com.top")
        if min_pdb is None or (not os.path.exists(min_pdb)):
            return pdbid, False, "minimize failed"

        ligand_name, pocket_pdb_file, pocket_top_file = extract_pocket(
            com_pdb=min_pdb,
            com_top='com.top',
            lig_pdb='lig.pdb',
            distance=5
        )

        if os.path.exists(pocket_pdb_file) and os.path.exists(pocket_top_file):
            return pdbid, True, "success"
        else:
            return pdbid, False, "extract_pocket failed"

    except Exception as e:
        return pdbid, False, f"exception: {e}"

    finally:
        os.chdir(orig_cwd)


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

    orig_cwd = os.getcwd()

    success_num = 0
    failed_list = []

    rows = [row for _, row in df_index.iterrows()]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_one_pdb, row, work_dir, orig_cwd)
            for row in rows
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing PDB files"):
            pdbid, success, msg = future.result()
            if success:
                success_num += 1
            else:
                failed_list.append((pdbid, msg))

    print(f"Successfully processed {success_num} out of {len(df_index)} PDB files.")
    print(f"Failed PDB files: {failed_list}")