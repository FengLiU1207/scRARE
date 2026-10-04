import torch

from opt import args


# =============================================================================
# Dataset-specific parameter profiles
# =============================================================================
# Profiles override command-line defaults for known datasets. Dataset names are
# matched exactly, and relation/topology sizes are derived from ``topo_k``.


_SCRNA_DEFAULT_PROFILE = {
    # -------------------------------------------------------------------------
    # Single-cell preprocessing
    # -------------------------------------------------------------------------
    "target_sum": 1e4,
    "gene_selection": "seurat",
    "hvg_num": 2000,
    "n_input": 256,               # PCA dimension; -1 disables PCA

    # -------------------------------------------------------------------------
    # Cell-graph construction
    # -------------------------------------------------------------------------
    "topo_k": 15,
    "topo_metric": "euclidean",

    "knn_k": 10,
    "knn_metric": "cosine",

    "structural_normalization": "sym",

    # -------------------------------------------------------------------------
    # Representation learning
    # -------------------------------------------------------------------------
    "t": 2,                       # graph-filtering order
    "dims": 512,                  # embedding dimension
    "activate": "ident",

    # -------------------------------------------------------------------------
    # Clustering / hard-sample learning
    # -------------------------------------------------------------------------
    "beta": 1.0,
    "cluster_gamma": 1.0,

    # -------------------------------------------------------------------------
    # Optimization
    # -------------------------------------------------------------------------
    "lr": 1e-3,

    # -------------------------------------------------------------------------
    # Relation purification
    # -------------------------------------------------------------------------
    "lambda_r": 0.1,
    "relation_sparse_threshold": 0.1,
}


# =============================================================================
# Per-dataset overrides
# =============================================================================
# Each entry changes only the values listed for that dataset.
 
_SCRNA_PROFILES = {
    "Muraro": {
       "lr":0.003, "topo_k":5, "knn_k":10, "lambda_r":0.001, "relation_sparse_threshold":0.1, "cluster_gamma":1, "beta":0, "t":50, "dims":512, "n_input":256
    },
    
    "Quake_10x_Limb_Muscle": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.0003,
        "topo_k": 15,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 1.0,
        "beta": 1.0,
        "t": 2,
        "dims": 512,
        "n_input": 256,
    },

    "Quake_10x_Spleen": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.00008,
        "topo_k": 200,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.5,
        "cluster_gamma": 0.1,
        "beta": 1.0,
        "t": 14,
        "dims": 256,
        "n_input": 256,
    },

    "Quake_10x_Bladder": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.0015,
        "topo_k": 15,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 1.0,
        "beta": 1.0,
        "t": 150,
        "dims": 512,
        "n_input": 256,
    },

    "Romanov": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.0003,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 2.0,
        "topo_k": 200,
        "knn_k": 10,
        "cluster_gamma": 0.01,
        "beta": 1.0,
        "t": 6,
        "dims": 256,
        "n_input": 28,
    },

    "Wang_Lung": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.005,
        "topo_k": 2,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 1.0,
        "beta": 1.0,
        "t": 2,
        "dims": 512,
        "n_input": 256,
    },

    "Klein": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.01,
        "topo_k": 1,
        "knn_k": 10,
        "lambda_r": 1.0,
        "relation_sparse_threshold": 0.2,
        "cluster_gamma": 0.2,
        "beta": 20.0,
        "t": 2,
        "dims": 512,
        "n_input": 32,
    },
    "Plasschaert": {
    **_SCRNA_DEFAULT_PROFILE,
    "lr": 0.00003,
    "topo_k": 30,
    "knn_k": 10,
    "lambda_r": 0.1,
    "relation_sparse_threshold": 0.1,
    "cluster_gamma": 1.0,
    "beta": 1.0,
    "t": 78,
    "dims": 512,
    "n_input": 26,
},

    "Chen": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.001,
        "topo_k": 15,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 1.0,
        "beta": 1.0,
        "t": 50,
        "dims": 512,
        "n_input": 256,
    },
    "Adam": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.001,
        "topo_k": 5,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 0.1,
        "beta": 1.0,
        "t": 1,
        "dims": 512,
        "n_input": 48,
    },

    "10X_PBMC": {
        **_SCRNA_DEFAULT_PROFILE,
        "lr": 0.001,
        "topo_k": 150,
        "knn_k": 10,
        "lambda_r": 0.1,
        "relation_sparse_threshold": 0.1,
        "cluster_gamma": 0.2,
        "beta": 1.0,
        "t": 3,
        "dims": 1024,
        "n_input": 128,
    },
}


def _validate_args():
    """Validate parameters used by the current scRNA-seq pipeline."""

    positive = {
        # training / runtime
        "epochs": args.epochs,
        "runs": args.runs,
        "clustering_update_interval": args.clustering_update_interval,
        "log_interval": args.log_interval,
        "cluster_gamma": args.cluster_gamma,
        "topology_chunk_size": args.topology_chunk_size,

        # scRNA-seq preprocessing
        "target_sum": args.target_sum,
        "hvg_num": args.hvg_num,

        # graph construction
        "topo_k": args.topo_k,
        "knn_k": args.knn_k,

        # representation
        "dims": args.dims,
    }

    invalid = [
        name
        for name, value in positive.items()
        if value <= 0
    ]

    if invalid:
        raise ValueError(
            "These arguments must be positive: "
            + ", ".join(invalid)
        )

    # PCA dimension: -1 disables PCA; otherwise it must be positive.
    if args.n_input == 0 or args.n_input < -1:
        raise ValueError(
            "n_input must be -1 or a positive integer"
        )

    # Graph filtering order can be zero or positive.
    if args.t < 0:
        raise ValueError(
            "graph-filtering order t must be non-negative"
        )

    if args.beta < 0:
        raise ValueError(
            "beta must be non-negative"
        )

    if args.lambda_r < 0.0:
        raise ValueError(
            "lambda_r must be non-negative"
        )

    if args.relation_sparse_threshold < 0.0:
        raise ValueError(
            "relation_sparse_threshold (tau_e) must be non-negative"
        )

    if args.kmeans_restarts < 1:
        raise ValueError(
            "kmeans_restarts must be at least 1"
        )


def setup_args(dataset_name=None):
    """Apply a dataset profile, resolve the device, and derive dependent settings."""

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------
    if dataset_name is not None:
        args.dataset = dataset_name

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------
    if args.device == "auto":
        args.device = (
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif (
        str(args.device).startswith("cuda")
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            f"CUDA device '{args.device}' was requested, "
            "but CUDA is unavailable. "
            "Use --device cpu or --device auto."
        )

    # -------------------------------------------------------------------------
    # Apply the profile associated with the exact dataset name.
    # -------------------------------------------------------------------------
    profile = _SCRNA_PROFILES.get(args.dataset)

    if profile is not None:
        for name, value in profile.items():
            setattr(args, name, value)
    else:
        print(
            f"[setup] No scRNA-seq profile found for dataset "
            f"'{args.dataset}'. Using opt.py defaults."
        )

    # -------------------------------------------------------------------------
    # Derived quantities
    # -------------------------------------------------------------------------
    # Reuse the base-graph neighborhood size for relation, candidate, and anchor counts.
    args.relation_k = int(args.topo_k)
    args.semantic_candidate_k = int(args.relation_k)
    args.anchors_per_cluster = int(args.relation_k)

    # -------------------------------------------------------------------------
    # Metric accumulators
    # -------------------------------------------------------------------------
    args.acc = 0.0
    args.nmi = 0.0
    args.ari = 0.0
    args.ami = 0.0

    if args.method == "legacy":
        args.legacy_weighting = True

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    _validate_args()

    # -------------------------------------------------------------------------
    # Print the resolved configuration used by the training script
    # -------------------------------------------------------------------------
    print("=" * 72)
    print("scRNA-seq clustering configuration")
    print("=" * 72)

    print(f"method                  : {args.method}")
    print(f"device                  : {args.device}")
    print(f"dataset                 : {args.dataset}")
    print(f"runs                    : {args.runs}")
    print(f"epochs                  : {args.epochs}")
    print(f"seed                    : {args.seed}")

    print("-" * 72)
    print("Single-cell preprocessing")
    print("-" * 72)

    print(f"target_sum               : {args.target_sum}")
    print(f"gene_selection           : {args.gene_selection}")
    print(f"hvg_num                  : {args.hvg_num}")
    print(f"n_input (PCA dim)        : {args.n_input}")

    print("-" * 72)
    print("Cell-graph construction")
    print("-" * 72)

    print(
        f"base topology            : "
        f"k={args.topo_k}, metric={args.topo_metric}"
    )
    print(
        f"attribute kNN            : "
        f"k={args.knn_k}, metric={args.knn_metric}"
    )
    print(
        f"structural normalization : "
        f"{args.structural_normalization}"
    )

    print("-" * 72)
    print("Representation / clustering")
    print("-" * 72)

    print(f"graph filter order t     : {args.t}")
    print(f"embedding dimension      : {args.dims}")
    print(f"activation               : {args.activate}")
    print(f"beta                     : {args.beta}")
    print(f"cluster_gamma            : {args.cluster_gamma}")
    print(f"learning rate            : {args.lr}")

    print("-" * 72)

    if args.method == "proposed":
        print("Proposed relation learning")
        print("-" * 72)

        print(f"lambda_R                 : {args.lambda_r}")
        print(
            f"tau_e                    : "
            f"{args.relation_sparse_threshold}"
        )
        print(
            f"relation_k               : "
            f"{args.relation_k} "
            "(derived from topo_k)"
        )
        print(
            f"semantic_candidate_k     : "
            f"{args.semantic_candidate_k} "
            "(derived from topo_k)"
        )
        print(
            f"anchors_per_cluster      : "
            f"{args.anchors_per_cluster} "
            "(derived from topo_k)"
        )
        print(
            "objective                 : "
            "L_con + L_clu + lambda_R * L_pur"
        )

    else:
        print("Legacy configuration")
        print("-" * 72)

        print(
            f"lambda_cluster           : "
            f"{args.lambda_cluster}"
        )
        print(
            f"lambda_hard_self         : "
            f"{args.lambda_hard_self}"
        )

    print("=" * 72)

    return args
