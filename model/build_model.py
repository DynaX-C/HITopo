from .hitopo_gcn import HITopo_GCN
from .hitopo_gat import HITopo_GAT
from .hitopo_gine import HITopo_GINE


def build_model(args):
    common_kwargs = dict(
        node_features_dim=args.node_features_dim,
        hidden_channels=args.hidden_channels,
        dropout=args.dropout,
        num_layers=args.num_layers,
        edge_dim=args.edge_dim,
        num_classes=args.num_classes,
        use_deg_norm=args.use_deg_norm,
        use_cross_interaction=args.use_cross_interaction,
        use_high_order_graph=args.use_high_order_graph,
        separate_nb_bd=args.separate_nb_bd,
        nb_only=args.nb_only,
    )

    if args.model == "gcn":
        return HITopo_GCN(
            **common_kwargs,
            improved=args.improved,
            cached=args.cached,
            add_self_loops=args.add_self_loops,
            normalize=args.normalize,
            bias=args.bias,
        )

    elif args.model == "gat":
        return HITopo_GAT(
            **common_kwargs,
            heads=args.heads,
            concat=args.concat,
            negative_slope=args.negative_slope,
            add_self_loops=args.add_self_loops,
            fill_value=args.fill_value,
            bias=args.bias,
            residual=args.residual,
        )

    elif args.model == "gine":
        return HITopo_GINE(
            **common_kwargs,
            eps=args.eps,
            train_eps=args.train_eps,
        )

    else:
        raise ValueError(f"Unknown model: {args.model}")