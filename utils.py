import os
import random

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_mutual_info_score as ami_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.neighbors import kneighbors_graph

from kmeans_gpu import kmeans
from opt import args
from relation import normalize_adjacency, remove_self_loops, to_sparse_coo
from reliability import (
    cluster_relation,
    pair_reliability,
    similarity_to_unit_interval,
    square_euclidean_distance,
    student_t_soft_assignment,
)


def cluster_acc(y_true, y_pred):
    """Permutation-invariant clustering ACC via Hungarian label matching."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    true_values, y_true_idx = np.unique(y_true, return_inverse=True)
    pred_values, y_pred_idx = np.unique(y_pred, return_inverse=True)
    dim = max(len(true_values), len(pred_values))
    contingency = np.zeros((dim, dim), dtype=np.int64)
    for index in range(y_true_idx.size):
        contingency[y_true_idx[index], y_pred_idx[index]] += 1
    row_ind, col_ind = linear_sum_assignment(contingency.max() - contingency)
    mapping = {column: row for row, column in zip(row_ind, col_ind)}
    mapped = np.array(
        [mapping.get(value, value) for value in y_pred_idx], dtype=np.int64
    )
    accuracy = metrics.accuracy_score(y_true_idx, mapped)
    # Keep the second return slot for backward compatibility; it is no longer
    # an evaluation metric and is ignored by eva().
    return accuracy, mapped


def eva(y_true, y_pred, show_details=True):
    # ACC still requires Hungarian label alignment.
    acc, _ = cluster_acc(y_true.copy(), y_pred.copy())

    # Permutation-invariant clustering metrics.
    nmi = nmi_score(
        y_true,
        y_pred,
        average_method="arithmetic"
    )
    ari = ari_score(y_true, y_pred)
    ami = ami_score(
        y_true,
        y_pred,
        average_method="arithmetic"
    )

    if show_details:
        print(
            ':acc {:.4f}'.format(acc),
            ', nmi {:.4f}'.format(nmi),
            ', ari {:.4f}'.format(ari),
            ', ami {:.4f}'.format(ami)
        )

    return acc, nmi, ari, ami


def scipy_to_torch_sparse(matrix):
    matrix = matrix.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack([matrix.row, matrix.col]).astype(np.int64))
    values = torch.from_numpy(matrix.data)
    return torch.sparse_coo_tensor(indices, values, matrix.shape).coalesce()


def construct_knn_graph(x, k=10, metric="cosine"):
    """Construct a symmetric binary sparse COO KNN graph."""
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=np.float32)
    node_num = x.shape[0]
    if node_num <= 1:
        empty_indices = torch.empty((2, 0), dtype=torch.long)
        return torch.sparse_coo_tensor(
            empty_indices, torch.empty(0), (node_num, node_num)
        ).coalesce()
    effective_k = min(max(int(k), 1), node_num - 1)
    adjacency = kneighbors_graph(
        x,
        n_neighbors=effective_k,
        mode="connectivity",
        metric=metric,
        include_self=False,
        n_jobs=-1,
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return scipy_to_torch_sparse(adjacency)


def _decode_array(array):
    array = np.asarray(array).reshape(-1)
    if array.dtype.kind in {"S", "O", "U"}:
        return np.asarray([
            item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item)
            for item in array
        ])
    return array


def _read_h5_matrix(group, h5py_module):
    if isinstance(group, h5py_module.Dataset):
        return np.asarray(group)
    keys = set(group.keys())
    if {"data", "indices", "indptr"} <= keys:
        data = np.asarray(group["data"])
        indices = np.asarray(group["indices"])
        indptr = np.asarray(group["indptr"])
        shape = (
            tuple(int(value) for value in np.asarray(group["shape"]))
            if "shape" in group
            else tuple(int(value) for value in group.attrs["shape"])
        )
        encoding = group.attrs.get("encoding-type", "csr_matrix")
        if isinstance(encoding, bytes):
            encoding = encoding.decode("utf-8")
        if "csc" in str(encoding).lower():
            return sp.csc_matrix((data, indices, indptr), shape=shape)
        return sp.csr_matrix((data, indices, indptr), shape=shape)
    dense_keys = [
        key for key in group.keys()
        if isinstance(group[key], h5py_module.Dataset)
    ]
    if len(dense_keys) == 1:
        return np.asarray(group[dense_keys[0]])
    raise KeyError("Unsupported expression-matrix layout in H5 file.")


def _choose_label(file_handle, requested="auto"):

    # ============================================================
    # 1. User explicitly specifies a label key
    # ============================================================
    if requested is not None and str(requested).lower() != "auto":
        requested = str(requested)

        # First look at H5 root
        if requested in file_handle:
            obj = file_handle[requested]

            # Label must be a Dataset, not a Group
            if hasattr(obj, "shape"):
                raw = _decode_array(obj[...])

                if raw.size == 0:
                    raise ValueError(
                        f"Requested label field '{requested}' is empty."
                    )

                _, labels = np.unique(raw, return_inverse=True)

                return (
                    labels.astype(np.int64),
                    requested,
                    raw,
                )

        # Then look inside obs/
        if "obs" in file_handle:
            observations = file_handle["obs"]

            if requested in observations:
                raw = _decode_array(observations[requested][...])

                if raw.size == 0:
                    raise ValueError(
                        f"Requested label field 'obs/{requested}' is empty."
                    )

                _, labels = np.unique(raw, return_inverse=True)

                return (
                    labels.astype(np.int64),
                    f"obs/{requested}",
                    raw,
                )

        raise KeyError(
            f"Requested label key '{requested}' was not found. "
            f"Root keys: {list(file_handle.keys())}; "
            f"obs keys: "
            f"{list(file_handle['obs'].keys()) if 'obs' in file_handle else 'N/A'}"
        )

    # ============================================================
    # 2. Auto mode: first search labels stored at H5 root
    # ============================================================
    root_candidates = [
        "Y",
        "y",
        "labels",
        "label",
        "cell_type1",
        "cell_type",
        "cell_ontology_class",
        "cluster",
        "clusters",
        "Group",
        "group",
    ]

    for key in root_candidates:
        if key not in file_handle:
            continue

        obj = file_handle[key]

        # Avoid accidentally treating a Group as label data
        if not hasattr(obj, "shape"):
            continue

        raw = _decode_array(obj[...])

        if raw.size == 0:
            continue

        _, labels = np.unique(raw, return_inverse=True)

        return (
            labels.astype(np.int64),
            key,
            raw,
        )

    # ============================================================
    # 3. AnnData-style: search labels under obs/
    # ============================================================
    if "obs" in file_handle:
        observations = file_handle["obs"]

        obs_candidates = [
            "cell_type1",
            "cell_type",
            "cell_ontology_class",
            "labels",
            "label",
            "cluster",
            "clusters",
            "Group",
            "group",
            "Y",
            "y",
        ]

        for key in obs_candidates:
            if key not in observations:
                continue

            obj = observations[key]

            if not hasattr(obj, "shape"):
                continue

            raw = _decode_array(obj[...])

            if raw.size == 0:
                continue

            _, labels = np.unique(raw, return_inverse=True)

            return (
                labels.astype(np.int64),
                f"obs/{key}",
                raw,
            )

    # ============================================================
    # 4. Nothing usable was found
    # ============================================================
    root_keys = list(file_handle.keys())

    if "obs" in file_handle:
        obs_keys = list(file_handle["obs"].keys())
    else:
        obs_keys = []

    raise KeyError(
        "No usable label field was found in the H5 file.\n"
        f"Root keys: {root_keys}\n"
        f"obs keys : {obs_keys}\n"
        "Supported label names include: "
        "Y, y, labels, label, cell_type, cluster, Group."
    )

def _library_log_normalize(x, target_sum=1e4):
    if sp.issparse(x):
        x = x.tocsr().astype(np.float32)
        library = np.asarray(x.sum(axis=1)).reshape(-1)
        scale = float(target_sum) / np.maximum(library, 1e-12)
        x = sp.diags(scale.astype(np.float32)) @ x
        x.data = np.log1p(x.data)
        return x.tocsr()
    x = np.asarray(x, dtype=np.float32)
    library = x.sum(axis=1, keepdims=True)
    x = x * (float(target_sum) / np.maximum(library, 1e-12))
    return np.log1p(x)


def _select_genes(x, file_handle, strategy="seurat", n_genes=2000):
    total = x.shape[1]
    if strategy == "all" or total <= n_genes:
        return x, np.arange(total)
    if (
        strategy == "seurat"
        and "uns" in file_handle
        and "seurat_genes" in file_handle["uns"]
        and "var_names" in file_handle
    ):
        names = _decode_array(file_handle["var_names"][...])
        selected_names = set(
            _decode_array(file_handle["uns"]["seurat_genes"][...]).tolist()
        )
        selected = np.asarray(
            [index for index, gene in enumerate(names) if gene in selected_names],
            dtype=np.int64,
        )
        if selected.size >= 50:
            return x[:, selected], selected

    keep = min(int(n_genes), total)
    if sp.issparse(x):
        mean = np.asarray(x.mean(axis=0)).reshape(-1)
        mean2 = np.asarray(x.multiply(x).mean(axis=0)).reshape(-1)
        variance = mean2 - mean * mean
    else:
        variance = np.var(x, axis=0)
    selected = np.argpartition(variance, -keep)[-keep:]
    selected = selected[np.argsort(variance[selected])[::-1]]
    return x[:, selected], selected


def load_scrna_h5(dataset_name, show_details=False):
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to load scRNA data.h5 files.") from error

    h5_path = os.path.join(args.data_root, dataset_name, "data.h5")
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"Cannot find {h5_path}")
    with h5py.File(h5_path, "r") as file_handle:
        expression_key = next(
            (key for key in ["exprs", "X", "x", "data", "matrix"] if key in file_handle),
            None,
        )
        if expression_key is None:
            raise KeyError(
                f"No expression matrix found in {h5_path}. "
                f"H5 keys: {list(file_handle.keys())}"
            )
        x = _read_h5_matrix(file_handle[expression_key], h5py)
        labels, label_key, raw_labels = _choose_label(file_handle, args.label_key)
        if x.shape[0] != len(labels) and x.shape[1] == len(labels):
            x = x.T
        if x.shape[0] != len(labels):
            raise ValueError(
                f"Expression shape {x.shape} does not match label length {len(labels)}."
            )
        x = _library_log_normalize(x, target_sum=args.target_sum)
        x, _ = _select_genes(
            x, file_handle, strategy=args.gene_selection, n_genes=args.hvg_num
        )

    x_dense = (
        x.toarray().astype(np.float32, copy=False)
        if sp.issparse(x)
        else np.asarray(x, dtype=np.float32)
    )
    if args.n_input != -1:
        components = min(int(args.n_input), x_dense.shape[0] - 1, x_dense.shape[1])
        if components < 2:
            raise ValueError("PCA dimension is too small for this dataset.")
        feature = PCA(
            n_components=components, random_state=args.seed
        ).fit_transform(x_dense).astype(np.float32)
    else:
        feature = x_dense

    A_original = construct_knn_graph(feature, k=args.topo_k, metric=args.topo_metric)
    A_knn = construct_knn_graph(feature, k=args.knn_k, metric=args.knn_metric)
    node_num = feature.shape[0]
    # Infer the cluster count from available labels when no explicit value is supplied.
    cluster_num = int(args.cluster_num) if args.cluster_num > 0 else len(np.unique(labels))

    print(f"[scRNA] dataset        : {dataset_name}")
    print(f"[scRNA] H5             : {h5_path}")
    print(f"[scRNA] cells x genes  : {x_dense.shape[0]} x {x_dense.shape[1]}")
    print(f"[scRNA] feature dim    : {feature.shape[1]}")
    print(f"[scRNA] label key      : {label_key}")
    print(f"[scRNA] clusters (K)   : {cluster_num}")
    print(f"[scRNA] topology graph : k={args.topo_k}, metric={args.topo_metric}")
    print(f"[scRNA] attribute graph: k={args.knn_k}, metric={args.knn_metric}")
    if show_details:
        names, counts = np.unique(raw_labels, return_counts=True)
        print("[scRNA] cell-type distribution:")
        for name, count in zip(names, counts):
            print(f"  {name}: {count}")
    return (
        torch.from_numpy(feature), labels, A_original, A_knn,
        node_num, cluster_num,
    )


def load_graph_data(dataset_name, show_details=False):
    """Load scRNA-seq features and construct the base and auxiliary kNN graphs.

    H5 input uses the full preprocessing pipeline. NPY input is treated as an
    already processed feature matrix, optionally reduced with PCA before graph
    construction.
    """
    h5_path = os.path.join(args.data_root, dataset_name, "data.h5")
    if os.path.exists(h5_path):
        return load_scrna_h5(dataset_name, show_details=show_details)

    load_path = os.path.join(args.data_root, dataset_name, dataset_name)
    feature = np.load(load_path + "_feat.npy", allow_pickle=True).astype(np.float32)
    labels = np.load(load_path + "_label.npy", allow_pickle=True).reshape(-1)

    if args.n_input != -1:
        components = min(int(args.n_input), feature.shape[0] - 1, feature.shape[1])
        if components < 2:
            raise ValueError("PCA dimension is too small for this dataset.")
        feature = PCA(
            n_components=components, random_state=args.seed
        ).fit_transform(feature).astype(np.float32)

    # Construct both neighborhood graphs from the final processed feature matrix.
    A_original = construct_knn_graph(
        feature, k=args.topo_k, metric=args.topo_metric
    )
    A_knn = construct_knn_graph(
        feature, k=args.knn_k, metric=args.knn_metric
    )
    cluster_num = int(args.cluster_num) if args.cluster_num > 0 else len(np.unique(labels))
    node_num = feature.shape[0]
    return (
        torch.from_numpy(feature), labels.astype(np.int64),
        A_original, A_knn, node_num, cluster_num,
    )


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def run_kmeans(feature, cluster_num, restarts=None, device=None):
    restarts = args.kmeans_restarts if restarts is None else int(restarts)
    device = args.device if device is None else device
    if restarts < 1:
        raise ValueError("kmeans_restarts must be at least 1")
    best_inertia = float("inf")
    best_labels, best_centers = None, None
    feature_device = feature.detach().float().to(device)
    for _ in range(restarts):
        labels, centers = kmeans(
            X=feature_device,
            num_clusters=cluster_num,
            distance="euclidean",
            device=device,
        )
        labels_device = labels.to(device=feature_device.device, dtype=torch.long)
        inertia = torch.sum(
            (feature_device - centers.index_select(0, labels_device)).square()
        ).item()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers.detach()
    return best_labels, best_centers, best_inertia


def phi(feature, true_labels, cluster_num):
    predicted, centers, _ = run_kmeans(feature, cluster_num)
    accuracy, nmi, ari, ami = eva(
        true_labels, predicted.numpy(), show_details=False
    )
    return (
        100.0 * accuracy, 100.0 * nmi, 100.0 * ari, 100.0 * ami,
        predicted.numpy(), centers,
    )


def laplacian_filtering(adjacency, feature, steps, eps=1e-8):
    """X_tilde=(I-L_tilde)^t X using sparse symmetric normalization."""
    adjacency = remove_self_loops(to_sparse_coo(adjacency))
    normalized = normalize_adjacency(
        adjacency, mode="sym", add_self_loops=True, eps=eps
    )
    output = feature.float().to(normalized.device)
    for _ in range(int(steps)):
        output = torch.sparse.mm(normalized, output)
    if not torch.isfinite(output).all():
        raise FloatingPointError("Graph filtering produced NaN or Inf.")
    return output.float()


def comprehensive_similarity(z1, z2, h1, h2, alpha):
    """Compute the dense mixed similarity matrix used by the legacy branch."""
    attributes = torch.cat([z1, z2], dim=0)
    structures = torch.cat([h1, h2], dim=0)
    return alpha * (attributes @ attributes.t()) + (1.0 - alpha) * (
        structures @ structures.t()
    )


def stable_hard_sample_contrastive_loss(
    z1,
    z2,
    h1,
    h2,
    mix,
    node_num,
    temperature=1.0,
    chunk_size=512,
    q=None,
    confidence=None,
    beta=1.0,
    use_pair_reliability=True,
    use_soft_cluster_relation=True,
    legacy_pos_neg_weight=None,
    legacy_pos_weight=None,
):
    """Compute chunked contrastive loss with optional reliability-based reweighting."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    attributes = torch.cat([z1, z2], dim=0)
    structures = torch.cat([h1, h2], dim=0)
    total_nodes = 2 * node_num
    all_node_ids = torch.arange(total_nodes, device=z1.device) % node_num
    losses = []
    weight_means = []

    for start in range(0, total_nodes, chunk_size):
        end = min(start + chunk_size, total_nodes)
        logits = (
            mix * (attributes[start:end] @ attributes.t())
            + (1.0 - mix) * (structures[start:end] @ structures.t())
        )
        global_rows = torch.arange(start, end, device=z1.device)
        row_node_ids = global_rows % node_num
        positive_columns = (global_rows + node_num) % total_nodes

        if legacy_pos_neg_weight is not None:
            weights = legacy_pos_neg_weight[start:end]
            weighted_logits = logits * weights / temperature
            positive_logits = logits[
                torch.arange(end - start, device=z1.device), positive_columns
            ]
            positive_logits = (
                positive_logits * legacy_pos_weight[start:end] / temperature
            )
        else:
            if q is None or confidence is None:
                raise ValueError("Q and node confidence are required in proposed mode.")
            cluster_probability = cluster_relation(
                q.index_select(0, row_node_ids),
                q.index_select(0, all_node_ids),
                soft=use_soft_cluster_relation,
            )
            pair_confidence = pair_reliability(
                z1.index_select(0, row_node_ids),
                z2.index_select(0, row_node_ids),
                z1.index_select(0, all_node_ids),
                z2.index_select(0, all_node_ids),
                confidence.index_select(0, row_node_ids),
                confidence.index_select(0, all_node_ids),
                enabled=use_pair_reliability,
            )
            hardness = torch.abs(
                cluster_probability - similarity_to_unit_interval(logits)
            ).pow(beta)
            weights = (1.0 + pair_confidence * hardness).clamp(1.0, 2.0).detach()
            weighted_logits = logits * weights / temperature
            positive_logits = weighted_logits[
                torch.arange(end - start, device=z1.device), positive_columns
            ]

        # Exclude only the same-view self pair while retaining the cross-view positive.
        weighted_logits[
            torch.arange(end - start, device=z1.device), global_rows
        ] = float("-inf")
        denominator = torch.logsumexp(weighted_logits, dim=1)
        chunk_loss = -positive_logits + denominator
        if not torch.isfinite(chunk_loss).all():
            raise FloatingPointError("L_con contains NaN or Inf.")
        losses.append(chunk_loss)
        weight_means.append(weights.detach().mean())
    return torch.cat(losses).mean(), torch.stack(weight_means).mean()


def hard_sample_aware_infoNCE(S, mask, pos_neg_weight, pos_weight, node_num):
    """Compute the legacy weighted InfoNCE loss in the log domain."""
    del mask
    total = 2 * node_num
    row = torch.arange(total, device=S.device)
    positive_col = (row + node_num) % total
    weighted = S * pos_neg_weight
    weighted[row, row] = float("-inf")
    positive = S[row, positive_col] * pos_weight
    return (-positive + torch.logsumexp(weighted, dim=1)).mean()


def continuous_reliable_cluster_loss(z, centers, q, confidence, eps=1e-8):
    if centers is None or q is None or confidence is None:
        return z.new_zeros(())
    centers = centers.detach().to(device=z.device, dtype=z.dtype)
    distance = square_euclidean_distance(z, centers)
    per_node = torch.sum(q.detach() * distance, dim=1)
    weights = confidence.detach().clamp(0.0, 1.0)
    loss = torch.sum(weights * per_node) / weights.sum().clamp_min(eps)
    if not torch.isfinite(loss):
        raise FloatingPointError("L_clu contains NaN or Inf.")
    return loss


# Descriptive aliases used by the current training loop.
reliability_aware_contrastive_loss = stable_hard_sample_contrastive_loss
confidence_weighted_clustering_loss = continuous_reliable_cluster_loss


def high_confidence_cluster_loss(z, centers, high_conf_indices, gamma=1.0):
    if centers is None or high_conf_indices is None or high_conf_indices.numel() == 0:
        return z.new_zeros(())
    centers = centers.detach().to(device=z.device, dtype=z.dtype)
    indices = high_conf_indices.detach().to(device=z.device, dtype=torch.long)
    distance = square_euclidean_distance(z, centers)
    q = student_t_soft_assignment(distance, gamma=gamma, eps=args.eps)
    per_node = torch.sum(q.detach() * distance, dim=1)
    return per_node.index_select(0, indices).mean()


def hard_self_supervision_loss(logits1, logits2, pseudo_labels, high_conf_indices):
    if pseudo_labels is None or high_conf_indices is None or high_conf_indices.numel() == 0:
        return logits1.new_zeros(())
    indices = high_conf_indices.detach().to(device=logits1.device, dtype=torch.long)
    pseudo_labels = torch.as_tensor(
        pseudo_labels, device=logits1.device, dtype=torch.long
    )
    targets = pseudo_labels.index_select(0, indices)
    return (
        F.cross_entropy(logits1.index_select(0, indices), targets)
        + F.cross_entropy(logits2.index_select(0, indices), targets)
    )


def legacy_pseudo_matrix(predicted, similarity, node_num, beta=1.0, eps=1e-8):
    predicted = torch.as_tensor(predicted, device=similarity.device, dtype=torch.long)
    predicted = torch.cat([predicted, predicted], dim=0)
    binary_relation = (predicted[:, None] == predicted[None, :]).to(similarity.dtype)
    minimum, maximum = similarity.min(), similarity.max()
    spread = maximum - minimum
    if spread <= eps:
        normalized = torch.full_like(similarity, 0.5)
    else:
        normalized = (similarity - minimum) / spread.clamp_min(eps)
    matrix = torch.abs(binary_relation - normalized).pow(beta)
    positive = torch.cat([
        torch.diag(matrix, node_num), torch.diag(matrix, -node_num)
    ])
    return positive.detach(), matrix.detach()


def legacy_high_confidence_indices(binary_confidence):
    indices = torch.nonzero(binary_confidence > 0, as_tuple=False).reshape(-1)
    return indices.detach(), torch.cat(
        [indices, indices + binary_confidence.numel()], dim=0
    ).detach()
