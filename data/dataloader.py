import numpy as np
from torch_geometric.data import DataLoader

def data_loader(atom_graphs, nb_atom_graphs, bd_atom_graphs, bond_graphs, angle_graphs, dihedral_graphs, batch_size=32, shuffle=True, seed=42):
    '''
    Create DataLoaders for each subgraph, keeping the order consistent.
    '''
    assert len(atom_graphs) == len(nb_atom_graphs) == len(bd_atom_graphs) == len(bond_graphs) == len(angle_graphs) == len(dihedral_graphs)
    if shuffle:
        np.random.seed(seed)
        num_samples = len(nb_atom_graphs)
        indices = list(range(num_samples))
        np.random.shuffle(indices)
        atom_graphs = [atom_graphs[i] for i in indices]
        nb_atom_graphs =  [nb_atom_graphs[i] for i in indices]
        bd_atom_graphs = [bd_atom_graphs[i] for i in indices]
        bond_graphs = [bond_graphs[i] for i in indices]
        angle_graphs = [angle_graphs[i] for i in indices]
        dihedral_graphs = [dihedral_graphs[i] for i in indices]
    atom_loader = DataLoader(atom_graphs, batch_size=batch_size, shuffle=False)
    nb_atom_loader = DataLoader(nb_atom_graphs, batch_size=batch_size, shuffle=False)
    bd_atom_loader = DataLoader(bd_atom_graphs, batch_size=batch_size, shuffle=False)
    bond_loader = DataLoader(bond_graphs, batch_size=batch_size, shuffle=False)
    angle_loader = DataLoader(angle_graphs, batch_size=batch_size, shuffle=False)
    dihedral_loader = DataLoader(dihedral_graphs, batch_size=batch_size, shuffle=False)
    return atom_loader, nb_atom_loader, bd_atom_loader, bond_loader, angle_loader, dihedral_loader