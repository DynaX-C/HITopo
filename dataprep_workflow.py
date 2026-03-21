import os
import torch
import parmed as pmd
from torch_geometric.data import Data
from data.dataprep import fix_pdb, get_ligand_pdb, combine_structure, extract_pocket
from data.dataset import extract_atom_features, coordinate_extraction, nonbond_extraction, bond_extraction, angle_extraction, dihedral_extraction

def prepare_and_build_graphs(
    protein_pdb,
    ligand_mol2,
    pocket_distance=5,
    cutoff=6,
):

    fixed_pdb = fix_pdb(protein_pdb, remove_hydrogens=False)
    ligand_pdb = get_ligand_pdb(ligand_mol2, remove_hydrogens=False)
    complex_pdb = combine_structure(fixed_pdb, ligand_pdb)
    pocket_pdb = extract_pocket(complex_pdb, ligand_name='MOL', distance=pocket_distance, remove_hydrogens=True)

    structure = pmd.load_file(pocket_pdb)
    atom_features = extract_atom_features(pocket_pdb, "MOL")

    # extract coordinates
    coords = coordinate_extraction(structure)

    # nonbond extraction
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

    atom_data = Data(x=atom_total_features, edge_index=edge_total_index, edge_attr=edge_total_features, coord=coords, face=angle_three_index)
    nb_atom_data = Data(x=atom_total_features, edge_index=nonbond_index_tensor_d, edge_attr=nonbond_features_tensor, coord=coords)
    bd_atom_data = Data(x=atom_total_features, edge_index=bond_index_tensor_d, edge_attr=bond_features_tensor, coord=coords, face=angle_three_index)
    bond_data = Data(x=bond_features_tensor, edge_index=angle_total_index, edge_attr=angle_total_features)
    angle_data = Data(x=angle_total_features, edge_index=dihedral_total_index, edge_attr=dihedral_total_features)
    dihedral_data = Data(x=atom_total_features, face=dihedral_total_four_index)

    return {
        "atom": atom_data,
        "nb_atom": nb_atom_data,
        "bd_atom": bd_atom_data,
        "bond": bond_data,
        "angle": angle_data,
        "dihedral": dihedral_data,
        "pocket_pdb": pocket_pdb,
    }
