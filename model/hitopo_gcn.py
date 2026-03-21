from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, Linear

class HITopo_GCN(torch.nn.Module):
    """
    Hierarchical interaction topology with GCN backbone.

    Args:
        node_features_dim (int): input node feature dimension.
        hidden_channels (int): hidden dimension of all graph representations.
        dropout (float): dropout rate.
        improved, cached, add_self_loops, normalize, bias:
            standard GCNConv options from PyG.
        edge_dim (int): dimension of edge features.
        num_layers (int): number of GCN layers.
        num_classes (int): output dimension (default: 1 for affinity).
        use_deg_norm (bool): whether to apply degree normalization in cross-level aggregation.

    Ablation-related switches:
        use_cross_interaction (bool): enable cross-level interaction block.
        use_high_order_graph (bool): include bond2bond and angle2angle graphs.
        separate_nb_bd (bool): separate non-bonded and bonded atom graphs.
        nb_only (bool): use only non-bonded atom graph (disables other components).
    """

    def __init__(self, 
                 node_features_dim: int, 
                 hidden_channels: int,
                 dropout: float = 0.1,
                 improved: bool = False,
                 cached: bool = False,
                 add_self_loops: Optional[int] = None,
                 normalize: bool = True,
                 bias: bool = True,
                 edge_dim: Optional[int] = None,
                 num_layers: int = 3,                 
                 num_classes: int = 1,
                 use_deg_norm: bool = True,
                 use_cross_interaction: bool = True,
                 use_high_order_graph: bool = True,
                 separate_nb_bd: bool = True,
                 nb_only: bool = False
                 ):
                 
        super().__init__()
        
        self.use_deg_norm = use_deg_norm
        self.use_cross_interaction = use_cross_interaction
        self.use_high_order_graph = use_high_order_graph
        self.separate_nb_bd = separate_nb_bd
        self.nb_only = nb_only

        if self.nb_only:
            self.separate_nb_bd = True
            self.use_high_order_graph = False
            self.use_cross_interaction = False

        if self.use_cross_interaction:
            self.use_high_order_graph = True

        if not self.use_high_order_graph:
            self.use_cross_interaction = False

        assert not (self.nb_only and not self.separate_nb_bd), "When `nb_only=True`, `separate_nb_bd=True` must be set."
        assert not (self.use_cross_interaction and not self.use_high_order_graph), "cross_interaction requires high_order_graph=True"


        self.edge_dim = edge_dim
        self.lin_node = nn.Sequential(Linear(node_features_dim, hidden_channels), nn.SiLU())
        self.lin_bond = nn.Sequential(Linear(node_features_dim + edge_dim, hidden_channels), nn.SiLU())
        self.lin_angle = nn.Sequential(Linear(node_features_dim + edge_dim * 2, hidden_channels), nn.SiLU())

        self.nb2weight = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        self.bd2weight = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        self.ang2weight = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )
        self.dih2weight = nn.Sequential(
            nn.Linear(edge_dim, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, 1),
        )

        self.nb_scale = nn.Parameter(torch.tensor(-2.2))
        self.bd_scale = nn.Parameter(torch.tensor(-2.2))
        self.bond_scale = nn.Parameter(torch.tensor(-2.2))
        self.angle_scale = nn.Parameter(torch.tensor(-2.2))

        self.nb_convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias
            )
            for _ in range(num_layers)
        ])

        self.nb_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.bd_convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias
            )
            for _ in range(num_layers)
        ])

        self.bd_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.bond_convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias
            )
            for _ in range(num_layers)
        ])

        self.bond_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.angle_convs = nn.ModuleList([
            GCNConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                improved=improved,
                cached=cached,
                add_self_loops=add_self_loops,
                normalize=normalize,
                bias=bias
            )
            for _ in range(num_layers)
        ])

        self.angle_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
            )
            for _ in range(num_layers)
        ])

        self.atom_from_bond = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])
        self.atom_from_angle = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])

        self.bond_from_atom = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])
        self.bond_from_angle = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])

        self.angle_from_atom = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])
        self.angle_from_bond = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels), 
                nn.SiLU()
                )
            for _ in range(num_layers)
        ])

        if self.use_high_order_graph:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_channels * 3, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes)
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels),
                nn.BatchNorm1d(hidden_channels),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_channels, num_classes)
            )

    def forward(self, data):
        x, edge_index, edge_index_intra, edge_index_inter, pos, bond_edge_index, angle_edge_index, angle_face, dihedral_face = \
            data['atom_graph'].x, data['atom_graph'].edge_index, data['bd_atom_graph'].edge_index, data['nb_atom_graph'].edge_index, data['nb_atom_graph'].coord, \
            data['bond_graph'].edge_index, data['angle_graph'].edge_index, data['bd_atom_graph'].face, data['dihedral_graph'].face

        dis, nb_dis, bd_dis, angle, dihedral = \
            data['atom_graph'].edge_attr.view(-1), data['nb_atom_graph'].edge_attr.view(-1), \
            data['bd_atom_graph'].edge_attr.view(-1), data['bond_graph'].edge_attr.view(-1), data['angle_graph'].edge_attr.view(-1)

        edge_bd_attr = self._rbf(bd_dis, D_min=0., D_max=6., D_count=self.edge_dim, device=x.device)
        edge_nb_attr = self._rbf(nb_dis, D_min=0., D_max=6., D_count=self.edge_dim, device=x.device)
        atom_edge_attr = self._rbf(dis, D_min=0., D_max=6., D_count=self.edge_dim, device=x.device)
        
        if self.use_high_order_graph:
            row_cov, col_cov = edge_index_intra
            bond_x_init = torch.cat([
                    x[row_cov] + x[col_cov],
                    edge_bd_attr,
                ], dim=-1)
            bond_x = self.lin_bond(bond_x_init)

            features = []
            for n in range(1, (self.edge_dim // 2)+1):
                features.append(torch.cos(n*angle))
                features.append(torch.sin(n*angle))
            bond_edge_attr = torch.stack(features, dim=-1)

            row_bcov, col_bcov = bond_edge_index
            angle_x = torch.cat([
                    bond_x_init[row_bcov] + bond_x_init[col_bcov],
                    bond_edge_attr,
                ], dim=-1)
            angle_x = self.lin_angle(angle_x)

            features = []
            for n in range(1, (self.edge_dim // 2)+1):
                features.append(torch.cos(n*dihedral))
                features.append(torch.sin(n*dihedral))
            angle_edge_attr = torch.stack(features, dim=-1)


        atom_x = self.lin_node(x)
        if self.separate_nb_bd:
            if self.nb_only:
                edge_nb_weight = self.nb2weight(edge_nb_attr).squeeze(-1)
                s_nb = torch.sigmoid(self.nb_scale) * 0.9
                edge_nb_weight = 1.0 - s_nb * torch.sigmoid(edge_nb_weight)
            else:
                edge_nb_weight = self.nb2weight(edge_nb_attr).squeeze(-1)
                s_nb = torch.sigmoid(self.nb_scale) * 0.9
                edge_nb_weight = 1.0 - s_nb * torch.sigmoid(edge_nb_weight)

                edge_bd_weight = self.bd2weight(edge_bd_attr).squeeze(-1)
                s_bd = torch.sigmoid(self.bd_scale) * 0.9
                edge_bd_weight = 1.0 - s_bd * torch.sigmoid(edge_bd_weight)
        else:
            edge_atom_weight = self.nb2weight(atom_edge_attr).squeeze(-1)
            s_atom = torch.sigmoid(self.nb_scale) * 0.9
            edge_atom_weight = 1.0 - s_atom * torch.sigmoid(edge_atom_weight)

        if self.use_high_order_graph:
            edge_bond_weight = self.ang2weight(bond_edge_attr).squeeze(-1)
            s_bond = torch.sigmoid(self.bond_scale) * 0.9
            edge_bond_weight = 1.0 - s_bond * torch.sigmoid(edge_bond_weight)

            edge_angle_weight = self.dih2weight(angle_edge_attr).squeeze(-1)
            s_angle = torch.sigmoid(self.angle_scale) * 0.9
            edge_angle_weight = 1.0 - s_angle * torch.sigmoid(edge_angle_weight)


        for l, (nb_conv, bd_conv, bond_conv, angle_conv, nb_mlp, bd_mlp, bond_mlp, angle_mlp) in enumerate(
                zip(self.nb_convs, self.bd_convs, self.bond_convs, self.angle_convs, self.nb_mlps, self.bd_mlps, self.bond_mlps, self.angle_mlps)):
            atom_x_prev = atom_x
            if self.separate_nb_bd:
                if self.nb_only:
                    x_nb = nb_conv(atom_x_prev, edge_index_inter, edge_nb_weight)
                    x_nb = nb_mlp(x_nb)
                    atom_x_self = x_nb + atom_x_prev
                else:
                    x_nb = nb_conv(atom_x_prev, edge_index_inter, edge_nb_weight)
                    x_nb = nb_mlp(x_nb)
                    x_bd = bd_conv(atom_x_prev, edge_index_intra, edge_bd_weight)
                    x_bd = bd_mlp(x_bd)
                    atom_x_self = x_nb + x_bd + atom_x_prev
            else:
                x_all = nb_conv(atom_x_prev, edge_index, edge_atom_weight)
                x_all = nb_mlp(x_all)
                atom_x_self = x_all + atom_x_prev
            
            if self.use_high_order_graph:
                bond_x_prev = bond_x
                angle_x_prev = angle_x
                bond_x_self = bond_conv(bond_x_prev, bond_edge_index, edge_bond_weight)
                bond_x_self = bond_mlp(bond_x_self) + bond_x_prev
                angle_x_self = angle_conv(angle_x_prev, angle_edge_index, edge_angle_weight)
                angle_x_self = angle_mlp(angle_x_self) + angle_x_prev

                # ====== Cross-level interaction block ======
                if self.use_cross_interaction:
                    num_nodes = atom_x_prev.size(0)
                    num_bonds = bond_x_prev.size(0)
                    bond_msg_to_atom = self.bond_to_atom_message(bond_x_self, edge_index_intra, num_nodes, use_deg_norm=self.use_deg_norm)
                    angle_msg_to_atom = self.angle_to_atom_message(angle_x_self, angle_face, num_nodes, use_deg_norm=self.use_deg_norm)
                    angle_msg_to_bond = self.angle_to_bond_message(angle_x_self, bond_edge_index, num_bonds, use_deg_norm=self.use_deg_norm)

                    atom_from_bond = self.atom_from_bond[l](bond_msg_to_atom)
                    atom_from_angle = self.atom_from_angle[l](angle_msg_to_atom)
                    bond_from_angle = self.bond_from_angle[l](angle_msg_to_bond)

                    # atom -> bond
                    row_cov, col_cov = edge_index_intra
                    atom_pair_for_bond = atom_x_self[row_cov] + atom_x_self[col_cov]
                    bond_from_atom = self.bond_from_atom[l](atom_pair_for_bond)

                    # bond -> angle
                    row_bcov, col_bcov = bond_edge_index
                    bond_pair_for_angle = bond_x_self[row_bcov] + bond_x_self[col_bcov]
                    angle_from_bond = self.angle_from_bond[l](bond_pair_for_angle)

                    # atom -> angle
                    a, b, c = angle_face
                    atom_triplet = atom_x_self[a] + atom_x_self[b] + atom_x_self[c]
                    angle_from_atom = self.angle_from_atom[l](atom_triplet)

                    atom_x = atom_x_self + atom_from_bond + atom_from_angle
                    bond_x = bond_x_self + bond_from_atom + bond_from_angle
                    angle_x = angle_x_self + angle_from_atom + angle_from_bond
                else:
                    atom_x = atom_x_self
                    bond_x = bond_x_self
                    angle_x = angle_x_self
            else:
                atom_x = atom_x_self

        atom_batch = data['nb_atom_graph'].batch if self.separate_nb_bd else data['atom_graph'].batch
        if self.use_high_order_graph:
            x = global_add_pool(atom_x, atom_batch)
            bond_x = global_add_pool(bond_x, data['bond_graph'].batch)
            angle_x = global_add_pool(angle_x, data['angle_graph'].batch)
            global_feat  = torch.cat([x, bond_x, angle_x], dim=-1)
        else:
            x = global_add_pool(atom_x, atom_batch)
            global_feat = x
        x = self.predictor(global_feat)

        return x.squeeze(-1) 


    def bond_to_atom_message(self, bond_x, edge_index_intra, num_nodes, use_deg_norm=True):
        row, col = edge_index_intra
        msg = torch.zeros(num_nodes, bond_x.size(-1), device=bond_x.device)
        msg.index_add_(0, row, bond_x)
        msg.index_add_(0, col, bond_x)

        if use_deg_norm:
            deg = torch.zeros(num_nodes, device=bond_x.device)
            deg.index_add_(0, row, torch.ones_like(row, dtype=deg.dtype))
            deg.index_add_(0, col, torch.ones_like(col, dtype=deg.dtype))
            msg = msg / deg.clamp(min=1).unsqueeze(-1)
        return msg
    
    def angle_to_atom_message(self, angle_x, angle_face, num_nodes, use_deg_norm=True):
        a, b, c = angle_face
        msg = torch.zeros(num_nodes, angle_x.size(-1), device=angle_x.device)
        msg.index_add_(0, a, angle_x)
        msg.index_add_(0, b, angle_x)
        msg.index_add_(0, c, angle_x)

        if use_deg_norm:
            deg = torch.zeros(num_nodes, device=angle_x.device)
            one = torch.ones_like(a, dtype=deg.dtype)
            deg.index_add_(0, a, one)
            deg.index_add_(0, b, one)
            deg.index_add_(0, c, one)
            msg = msg / deg.clamp(min=1).unsqueeze(-1)
        return msg
    
    def angle_to_bond_message(self, angle_x, angle_to_bond_index, num_bonds, use_deg_norm=True):
        row, col = angle_to_bond_index
        msg = torch.zeros(num_bonds, angle_x.size(-1), device=angle_x.device)
        msg.index_add_(0, row, angle_x)
        msg.index_add_(0, col, angle_x)

        if use_deg_norm:
            deg = torch.zeros(num_bonds, device=angle_x.device)
            one = torch.ones_like(row, dtype=deg.dtype)
            deg.index_add_(0, row, one)
            deg.index_add_(0, col, one)
            msg = msg / deg.clamp(min=1).unsqueeze(-1)
        return msg

    def _rbf(self, D, D_min=0., D_max=6., D_count=8, device='cpu'):
        '''
        From https://github.com/jingraham/neurips19-graph-protein-design
        '''
        D_mu = torch.linspace(D_min, D_max, D_count).to(device)
        D_mu = D_mu.view([1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(D, -1)

        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
        return RBF
