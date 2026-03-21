import os
import parmed as pmd
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import pandas as pd
import argparse
from .dataset import *
import warnings
from rdkit import RDLogger
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def get_resids(
    structure: pmd.Structure,
    lig_resname: str = "MOL",
):
    res_ids = []
    for a in structure.atoms:
        res_name = a.residue.name
        res_id = a.residue.number
        if res_name != lig_resname:
            res_ids.append(res_id)
        if res_name == lig_resname:
            res_ids.append(-1)
    
    return torch.tensor(res_ids, dtype=torch.long)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, required=True)
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    index_file = args.index_file
    work_dir = args.work_dir
    output = args.output

    df_index = pd.read_csv(index_file)

    atom_graphs = []
    nb_atom_graphs = []
    bd_atom_graphs = []
    bond_graphs = []
    angle_graphs = []
    dihedral_graphs = []
    for _, row in tqdm(df_index.iterrows(), desc="Processing PDB files", total=len(df_index)):
        pdbid = row['pdbid']
        label = row['-logKd/Ki']
        pdb_dir = f"{work_dir}/{pdbid}/"
        pdb_file = os.path.join(pdb_dir, "complex_5A.pdb")

        if not os.path.exists(pdb_file):
            print(f"Files for {pdbid} not found, skipping...")
            continue


        # load structure
        structure = pmd.load_file(pdb_file)
        structure = structure['!@H=']   # remove hydrogens
        noh_pdb_file = os.path.join(work_dir, "complex_5A_em_noH.pdb")
        structure.save(noh_pdb_file, overwrite=True)

        res_ids = get_resids(structure, "MOL")
        atom_features = extract_atom_features(noh_pdb_file, "MOL")

        # extract coordinates
        coords = coordinate_extraction(structure)

        # nonbond extraction
        cutoff = 6  # you can adjust this value
        nonbond_index_tensor, nonbond_features_tensor = nonbond_extraction(structure, cutoff=cutoff)
        nonbond_index_tensor_d = torch.cat([nonbond_index_tensor, nonbond_index_tensor.flip([0])], dim=1)
        nonbond_features_tensor = torch.cat([nonbond_features_tensor, nonbond_features_tensor], dim=0)

        # bond extraction
        bond_index_tensor, bond_features_tensor, neighbors = bond_extraction(structure)
        bond_idx = []
        for idx, (src, dst) in enumerate(bond_index_tensor.t().tolist()):
            bond_idx.append(idx)
        bond_index_tensor_d = torch.cat([bond_index_tensor, bond_index_tensor.flip([0])], dim=1)
        bond_idx = bond_idx + bond_idx
        bond_to_node = {}
        for idx, (src, dst) in zip(bond_idx, bond_index_tensor_d.t().tolist()):
            key = tuple([src, dst])
            bond_to_node[key] = idx
        bond_features_tensor = torch.cat([bond_features_tensor, bond_features_tensor], dim=0)

        # angle extraction
        angle_index_tensor, angle_feature_tensor, angle_three_index_tensor = angle_extraction(bond_to_node, neighbors, coords)
        angle_idx = []
        for idx, (src, dst) in enumerate(angle_index_tensor.t().tolist()):
            angle_idx.append(idx)
        angle_index_tensor = torch.cat([angle_index_tensor, angle_index_tensor.flip([0])], dim=1)
        angle_feature_tensor = torch.cat([angle_feature_tensor, angle_feature_tensor], dim=0)
        angle_three_index_tensor = torch.tensor(angle_three_index_tensor, dtype=torch.long)
        angle_three_index_tensor_d = torch.cat([angle_three_index_tensor, angle_three_index_tensor.flip([0])], dim=1)
        angle_idx = angle_idx + angle_idx
        angle_to_node = {}
        for idx, (src, mid, dst) in zip(angle_idx, angle_three_index_tensor_d.t().tolist()):
            key = tuple([src, mid, dst])
            angle_to_node[key] = idx

        # dihedral extraction
        dihedral_index_tensor, dihedral_feature_tensor, dihedral_four_index, improper_index_tensor, improper_feature_tensor, improper_four_index = dihedral_extraction(structure, angle_to_node, neighbors, coords)
        dihedral_index_tensor = torch.cat([dihedral_index_tensor, dihedral_index_tensor.flip([0])], dim=1)
        dihedral_feature_tensor = torch.cat([dihedral_feature_tensor, dihedral_feature_tensor], dim=0)
        dihedral_four_index = torch.cat([dihedral_four_index, dihedral_four_index.flip([0])], dim=1)
        improper_index_tensor = torch.cat([improper_index_tensor, improper_index_tensor.flip([0])], dim=1)
        improper_feature_tensor = torch.cat([improper_feature_tensor, improper_feature_tensor], dim=0)
        a, b, c, d = improper_four_index
        improper_reverse = torch.stack([d, b, c, a], dim=0)
        improper_four_index = torch.cat([improper_four_index, improper_reverse], dim=1)

        # create Data objects
        atom_total_features = torch.tensor(atom_features, dtype=torch.float)
        edge_total_index = torch.cat([bond_index_tensor_d, nonbond_index_tensor_d], dim=1)
        edge_total_features = torch.cat([bond_features_tensor, nonbond_features_tensor], dim=0)
        bond_two_index = torch.cat([bond_index_tensor, bond_index_tensor, nonbond_index_tensor, nonbond_index_tensor], dim=1)
        angle_total_index = angle_index_tensor
        angle_total_features = angle_feature_tensor
        angle_three_index = torch.cat([angle_three_index_tensor, angle_three_index_tensor], dim=1)  
        dihedral_total_index = torch.cat([dihedral_index_tensor, improper_index_tensor], dim=1)
        dihedral_total_features = torch.cat([dihedral_feature_tensor, improper_feature_tensor], dim=0)
        dihedral_total_four_index = torch.cat([dihedral_four_index, improper_four_index], dim=1)

        atom_data = Data(x=atom_total_features, edge_index=edge_total_index, y=label, edge_attr=edge_total_features, coord=coords, face=angle_three_index, pdbid=pdbid, res_ids=res_ids)
        nb_atom_data = Data(x=atom_total_features, edge_index=nonbond_index_tensor_d, y=label, edge_attr=nonbond_features_tensor, coord=coords, pdbid=pdbid)
        bd_atom_data = Data(x=atom_total_features, edge_index=bond_index_tensor_d, y=label, edge_attr=bond_features_tensor, coord=coords, face=angle_three_index, pdbid=pdbid)
        bond_data = Data(x=bond_features_tensor, edge_index=angle_total_index, edge_attr=angle_total_features, pdbid=pdbid)
        angle_data = Data(x=angle_total_features, edge_index=dihedral_total_index, edge_attr=dihedral_total_features, pdbid=pdbid)
        dihedral_data = Data(x=atom_total_features, face=dihedral_total_four_index, pdbid=pdbid)

        atom_graphs.append(atom_data)
        nb_atom_graphs.append(nb_atom_data)
        bd_atom_graphs.append(bd_atom_data)
        bond_graphs.append(bond_data)
        angle_graphs.append(angle_data)
        dihedral_graphs.append(dihedral_data)

    graphs = {
        "atom_graphs": atom_graphs,
        "nb_atom_graphs": nb_atom_graphs,
        "bd_atom_graphs": bd_atom_graphs,
        "bond_graphs": bond_graphs,
        "angle_graphs": angle_graphs,
        "dihedral_graphs": dihedral_graphs
    }

    torch.save(graphs, output)

if __name__ == "__main__":
    main()