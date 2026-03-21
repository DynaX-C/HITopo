"""
Predefined ablation variants for HITopo models.

Each variant corresponds to a fixed combination of:
    - use_cross_interaction
    - use_high_order_graph
    - separate_nb_bd
    - nb_only

These variants are shared across GCN / GAT / GINE backbones.
"""


VARIANT_CONFIGS = {
    # Full model (default)
    "full": {
        "use_cross_interaction": True,
        "use_high_order_graph": True,
        "separate_nb_bd": True,
        "nb_only": False,
    },

    # Remove cross-level interaction
    "noci": {
        "use_cross_interaction": False,
        "use_high_order_graph": True,
        "separate_nb_bd": True,
        "nb_only": False,
    },

    # Remove high-order graph (only atom-level)
    "noho": {
        "use_cross_interaction": False,
        "use_high_order_graph": False,
        "separate_nb_bd": True,
        "nb_only": False,
    },

    # Unified atom graph (no NB/BD split)
    "unig": {
        "use_cross_interaction": True,
        "use_high_order_graph": True,
        "separate_nb_bd": False,
        "nb_only": False,
    },

    # Only atom-level interactions (no high-order, no cross interaction)
    "base": {
        "use_cross_interaction": False,
        "use_high_order_graph": False,
        "separate_nb_bd": False,
        "nb_only": False,
    },

    # Only non-bonded interactions
    "nb": {
        "use_cross_interaction": False,
        "use_high_order_graph": False,
        "separate_nb_bd": False,
        "nb_only": True,
    },
}


def apply_variant(args):
    """
    Apply variant settings to args.

    Args:
        args: config object with attribute `variant`

    Returns:
        args with updated attributes
    """
    if not hasattr(args, "variant"):
        raise ValueError("args must contain 'variant'")

    variant = args.variant

    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown variant: {variant}. Available: {list(VARIANT_CONFIGS.keys())}")

    # apply preset
    for k, v in VARIANT_CONFIGS[variant].items():
        setattr(args, k, v)

    # ===== Safety constraints =====
    # nb_only implies no high-order and no cross interaction
    if args.nb_only:
        args.separate_nb_bd = True
        args.use_high_order_graph = False
        args.use_cross_interaction = False

    # cross interaction requires high-order graph
    if args.use_cross_interaction:
        args.use_high_order_graph = True

    # if no high-order, disable cross interaction
    if not args.use_high_order_graph:
        args.use_cross_interaction = False

    return args