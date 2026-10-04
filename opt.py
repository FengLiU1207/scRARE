import argparse


# -----------------------------------------------------------------------------
# Global constants and command-line configuration
# -----------------------------------------------------------------------------
# Contrastive temperature is fixed here so every run uses the same value.
FIXED_TEMPERATURE = 1.0


def str2bool(value):
    """Argparse-compatible boolean that also works on Python < 3.9."""
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot interpret '{value}' as a boolean.")


def add_bool_argument(parser, name, default, help_text):
    parser.add_argument(
        f"--{name}", type=str2bool, nargs="?", const=True, default=default,
        help=help_text,
    )


parser = argparse.ArgumentParser(
    description="scRARE: Reliability-Aware Relation Evolution for scRNA-seq Clustering"
)

# Dataset, runtime, and result-output settings.
parser.add_argument("--device", type=str, default="auto", help="auto / cpu / cuda:0")
parser.add_argument("--dataset", type=str, default="Romanov")
parser.add_argument(
    "--cluster_num", type=int, default=-1,
    help="Number of clusters; -1 infers K from the available evaluation labels",
)
parser.add_argument("--data_root", type=str, default="dataset")
parser.add_argument("--label_key", type=str, default="auto")
parser.add_argument("--method", choices=["proposed", "legacy"], default="proposed")
parser.add_argument("--result_file", type=str, default="result.csv")

# Preprocessing, graph construction, representation, and clustering settings.
parser.add_argument("--target_sum", type=float, default=1e4)
parser.add_argument("--gene_selection", choices=["seurat", "variance", "all"], default="seurat")
parser.add_argument("--hvg_num", type=int, default=2000)
parser.add_argument("--n_input", type=int, default=256, help="PCA dimension; -1 disables PCA")
parser.add_argument("--t", type=int, default=2, help="number of graph-smoothing propagation steps")
parser.add_argument("--topo_k", type=int, default=15, help="base-graph neighborhood size k")
parser.add_argument("--topo_metric", choices=["cosine", "euclidean"], default="euclidean")
parser.add_argument("--knn_k", type=int, default=10, help="number of neighbors used to construct the auxiliary kNN graph")
parser.add_argument( "--knn_metric", choices=["cosine", "euclidean"], default="cosine")
parser.add_argument("--structural_normalization", choices=["none", "row", "sym"], default="sym",help="normalization before structural encoders",
)
parser.add_argument("--beta", type=float, default=1.0, help="exponent controlling hard-pair reweighting")
parser.add_argument("--dims", type=int, default=512, help="cell embedding dimension d")
parser.add_argument("--activate", choices=["ident", "sigmoid"], default="ident")
parser.add_argument("--tao", type=float, default=0.6, help="legacy-only top-H filtering rate")
parser.add_argument("--cluster_gamma", type=float, default=1.0, help="scale parameter of the Student-t soft assignment",
)

parser.add_argument("--runs", type=int, default=5)
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--seed", type=int, default=1206)
parser.add_argument("--clustering_update_interval", type=int, default=10)
parser.add_argument("--log_interval", type=int, default=1)
parser.add_argument("--kmeans_restarts", type=int, default=1)
parser.add_argument( "--max_grad_norm", type=float, default=5.0,help="gradient-clipping safeguard; <=0 disables",
)
parser.add_argument("--eps", type=float, default=1e-8)

# Relation-purification settings.
parser.add_argument(
    "--lambda_r", type=float, default=0.1,
    help="weight of the relation-purification loss",
)
parser.add_argument(
    "--tau_e", "--relation_sparse_threshold",
    dest="relation_sparse_threshold", type=float, default=0.1,
    help="soft-threshold value used to compute sparse relation residuals",
)

# Topology-construction settings. Derived neighbor counts are assigned in setup.py.
parser.add_argument("--topology_chunk_size", type=int, default=512)
parser.add_argument(
    "--topology_mode", choices=["full_chunked", "candidate_sparse", "auto"],
    default="full_chunked",
    help=(
        "full_chunked scores all cell pairs in chunks; candidate_sparse "
        "restricts scoring to a sparse candidate set"
    ),
)
parser.add_argument("--topology_full_threshold", type=int, default=8000)

# Feature switches for the relation-learning branch.
add_bool_argument(parser, "use_dynamic_topology", True, "False keeps A_dyn=A")
add_bool_argument(
    parser, "use_lowrank_relation", True,
    "False builds topology directly from the reliable target R",
)
add_bool_argument(parser, "use_pair_reliability", True, "False sets c_ij=1")
add_bool_argument(
    parser, "use_soft_cluster_relation", True,
    "False uses binary pseudo relations",
)
add_bool_argument(
    parser, "use_continuous_confidence", True,
    "False uses the legacy top-H binary confidence",
)
add_bool_argument(
    parser, "legacy_weighting", False,
    "Use the original H/M hard-sample weighting",
)

# Parameters used only by the legacy training branch.
parser.add_argument("--warmup_epochs", type=int, default=50)
parser.add_argument("--lambda_cluster", type=float, default=0.0)
parser.add_argument("--lambda_hard_self", type=float, default=0.0)

# Metric fields used by existing result-processing scripts.
parser.add_argument("--acc", type=float, default=0.0)
parser.add_argument("--nmi", type=float, default=0.0)
parser.add_argument("--ari", type=float, default=0.0)
parser.add_argument("--ami", type=float, default=0.0)

args = parser.parse_args()

# Fixed runtime quantities that are not exposed as command-line options.
args.temperature = FIXED_TEMPERATURE
