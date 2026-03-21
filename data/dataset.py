import numpy as np
import os
import parmed as pmd
import torch
from torch_geometric.data import Data
from rdkit import Chem
from tqdm import tqdm
from itertools import combinations
import pandas as pd
import argparse
import warnings
from rdkit import RDLogger
warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def _cross(v1, v2):
    x = v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1]
    y = v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2]
    z = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]
    return torch.stack([x, y, z], dim=-1)

def calc_dihedral(a0, a1, a2, a3, eps=1e-8):
    v1 = a1 - a0
    v2 = a1 - a2
    v3 = a3 - a2

    v1xv2 = _cross(v1, v2)
    v2xv3 = _cross(v2, v3)
    l1 = torch.sqrt(torch.sum(v1xv2 * v1xv2, dim=-1) + eps)
    l2 = torch.sqrt(torch.sum(v2xv3 * v2xv3, dim=-1) + eps)
    cosa = torch.sum(v1xv2 * v2xv3, dim=-1) / (l1 * l2 + eps)
    cosa = torch.clamp(cosa, -1.0, 1.0)

    angle = torch.acos(cosa)
    sign = torch.where(torch.sum(v3 * v1xv2, dim=-1) <= 0, 1.0, -1.0)
    return angle * sign

def calc_angle(a1, a2, a3, eps=1e-8):
    v1 = a2 - a1
    v2 = a2 - a3

    l1 = torch.sqrt(torch.sum(v1 * v1, dim=-1) + eps)
    l2 = torch.sqrt(torch.sum(v2 * v2, dim=-1) + eps)

    cos_angle = torch.sum(v1 * v2, dim=-1) / (l1 * l2 + eps)
    cos_angle = torch.clamp(cos_angle, -1.0, 1.0)

    angle = torch.acos(cos_angle)
    return angle

def one_hot_encoding(value: str, choices: list):
    if value not in choices:
        value = choices[-1]
    return list(map(lambda s: value == s, choices))

def extract_atom_features(pdb_file, ligand_name):
    '''
    get atom features via RDKit
    including: atom symbol, atom degree, number of Hs, is in ring,
               chiral tag, aromatic, hybridization, lig or pro
    '''
    atom_feats = []
    mol = Chem.MolFromPDBFile(pdb_file, removeHs=False, sanitize=False)

    if mol is None:
        print(f"Error: RDKit failed to read {pdb_file}.")
    for atom in mol.GetAtoms():
        symbol = one_hot_encoding(atom.GetSymbol(), ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'I', 'Unknown']) + \
                one_hot_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
                one_hot_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
                one_hot_encoding(atom.GetHybridization(), [
                    Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                    Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D, 
                    Chem.rdchem.HybridizationType.SP3D2
                    ]) + [atom.GetIsAromatic()]
        symbol = symbol + one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
        atom_feats.append(symbol)
        
    return atom_feats

def coordinate_extraction(structure):
    '''
    extract coordinates from parmed structure
    '''
    coord = []
    for atom in structure.atoms:
        atom_coord = [float(atom.xx), float(atom.xy), float(atom.xz)]
        # print(f"Atom {atom} coordinates: {atom_coord}")
        coord.append(atom_coord)
    return torch.tensor(coord, dtype=torch.float)

def nonbond_extraction(structure, cutoff=6):
    nonbond_index = [[], []]
    nonbond_features = []
    for i in range(len(structure.atoms)):
        for j in range(i+1, len(structure.atoms)):
            if structure.atoms[i].residue.name == "MOL" and structure.atoms[j].residue.name == "MOL":
                continue
            elif structure.atoms[i].residue.name != "MOL" and structure.atoms[j].residue.name != "MOL":
                continue
            src_coord = np.array([structure.atoms[i].xx, structure.atoms[i].xy, structure.atoms[i].xz])
            dst_coord = np.array([structure.atoms[j].xx, structure.atoms[j].xy, structure.atoms[j].xz])
            nonbond_dist = np.linalg.norm(src_coord - dst_coord)
            if nonbond_dist <= cutoff:
                nonbond_index[0].append(structure.atoms[i].idx)
                nonbond_index[1].append(structure.atoms[j].idx)
                nonbond_features.append([nonbond_dist])

    return torch.tensor(nonbond_index, dtype=torch.long), torch.tensor(nonbond_features, dtype=torch.float)

def bond_extraction(structure):
    bond_index = [[], []]
    bond_features = []
    neighbors = {atom.idx: [] for atom in structure.atoms}
    for bond in structure.bonds:
        bond_index[0].append(bond.atom1.idx)
        bond_index[1].append(bond.atom2.idx)
        neighbors[bond.atom1.idx].append(bond.atom2.idx)
        neighbors[bond.atom2.idx].append(bond.atom1.idx)
        src_coord = np.array([bond.atom1.xx, bond.atom1.xy, bond.atom1.xz])
        dst_coord = np.array([bond.atom2.xx, bond.atom2.xy, bond.atom2.xz])
        bond_dist = np.linalg.norm(src_coord - dst_coord)
        bond_features.append([bond_dist])

    return torch.tensor(bond_index, dtype=torch.long), torch.tensor(bond_features, dtype=torch.float), neighbors

def angle_extraction(bond_to_node, neighbors, coords):
    '''
    extract angle indices and features from parmed structure
    '''
    angle_index = [[], []]
    angle_features = []
    angle_three_index = [[], [], []]
    for center_idx, neighs in neighbors.items():
        if len(neighs) < 2:
            continue
        for i in range(len(neighs)):
            for j in range(i+1, len(neighs)):
                a = neighs[i]
                c = neighs[j]
                bond1 = tuple([a, center_idx])
                bond2 = tuple([c, center_idx])
                if bond1 in bond_to_node.keys() and bond2 in bond_to_node.keys():
                    angle_index[0].append(bond_to_node[bond1])
                    angle_index[1].append(bond_to_node[bond2])
                    angle_three_index[0].append(a)
                    angle_three_index[1].append(center_idx)
                    angle_three_index[2].append(c)
                    angle = calc_angle(coords[a], coords[center_idx], coords[c])
                    angle_features.append([angle])
                else:
                    print(f"Bond {bond1} or {bond2} not found in bond_to_node, skipping angle {a}-{center_idx}-{c}")
                    continue
    return torch.tensor(angle_index, dtype=torch.long), torch.tensor(angle_features, dtype=torch.float), torch.tensor(angle_three_index, dtype=torch.long)


def dihedral_extraction(structure, angle_to_node, neighbors, coords):
    '''
    extract dihedral indices and features from parmed structure
    '''
    dihedral_index = [[], []]
    dihedral_features = []
    dihedral_four_index = [[], [], [], []]
    improper_index = [[], []]
    improper_features = []
    improper_four_index = [[], [], [], []]

    for bond in structure.bonds:
        b = bond.atom1.idx
        c = bond.atom2.idx
        for a in neighbors[b]:
            if a == c: continue
            for d in neighbors[c]:
                if d == b: continue
                ange1 = tuple([a, b, c])
                angle2 = tuple([d, c, b])
                if ange1 in angle_to_node.keys() and angle2 in angle_to_node.keys():
                    dihedral_index[0].append(angle_to_node[ange1])
                    dihedral_index[1].append(angle_to_node[angle2])
                    dihedral_four_index[0].append(a)
                    dihedral_four_index[1].append(b)
                    dihedral_four_index[2].append(c)
                    dihedral_four_index[3].append(d)
                    dihedral = calc_dihedral(coords[a], coords[b], coords[c], coords[d])
                    dihedral_features.append([dihedral])
                else:
                    print(f"Angle {ange1} or {angle2} not found in angle_to_node, skipping dihedral {a}-{b}-{c}-{d}")
                    continue

    for c_idx, neighs in neighbors.items():
        if len(neighs) < 3:
            continue
        for a,b,d in combinations(neighs, 3):
            ange1 = tuple([a, c_idx, b])
            angle2 = tuple([d, c_idx, b])
            if ange1 in angle_to_node.keys() and angle2 in angle_to_node.keys():
                improper_index[0].append(angle_to_node[ange1])
                improper_index[1].append(angle_to_node[angle2])
                improper_four_index[0].append(a)
                improper_four_index[1].append(b)
                improper_four_index[2].append(c_idx)
                improper_four_index[3].append(d)
                improper = calc_dihedral(coords[a], coords[b], coords[c_idx], coords[d])
                improper_features.append([improper])
            else:
                print(f"Angle {ange1} or {angle2} not found in angle_to_node, skipping improper {a}-{b}-{c_idx}-{d}")
                continue

    return (torch.tensor(dihedral_index, dtype=torch.long), 
            torch.tensor(dihedral_features, dtype=torch.float),
            torch.tensor(dihedral_four_index, dtype=torch.long),
            torch.tensor(improper_index, dtype=torch.long),
            torch.tensor(improper_features, dtype=torch.float),
            torch.tensor(improper_four_index, dtype=torch.long))

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
        atom_features = extract_atom_features(pdb_file, "MOL")

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

        atom_data = Data(x=atom_total_features, edge_index=edge_total_index, y=label ,edge_attr=edge_total_features, coord=coords, face=angle_three_index, pdbid=pdbid)
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
