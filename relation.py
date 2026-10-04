import numpy as np
import torch

from reliability import (
    cluster_relation,
    cluster_relation_edges,
    pair_reliability,
    pair_reliability_edges,
    similarity_to_unit_interval,
)


def to_sparse_coo(adjacency, device=None, dtype=torch.float32):
    if not torch.is_tensor(adjacency):
        adjacency = torch.as_tensor(adjacency, dtype=dtype)
    adjacency = adjacency.to(device=device, dtype=dtype)
    if adjacency.layout == torch.sparse_coo:
        return adjacency.coalesce()
    sparse_csr_layout = getattr(torch, "sparse_csr", None)
    if sparse_csr_layout is not None and adjacency.layout == sparse_csr_layout:
        return adjacency.to_sparse_coo().coalesce()
    if hasattr(adjacency, "to_sparse_coo"):
        return adjacency.to_sparse_coo().coalesce()
    return adjacency.to_sparse().coalesce()


def remove_self_loops(adjacency):
    adjacency = to_sparse_coo(adjacency)
    indices = adjacency.indices()
    keep = indices[0] != indices[1]
    return torch.sparse_coo_tensor(
        indices[:, keep], adjacency.values()[keep], adjacency.shape,
        device=adjacency.device, dtype=adjacency.dtype,
    ).coalesce()


def symmetrize_sparse(adjacency):
    """Average reciprocal weights while preserving unilateral edges."""
    adjacency = remove_self_loops(adjacency)
    indices = adjacency.indices()
    reverse = torch.stack([indices[1], indices[0]], dim=0)
    both_indices = torch.cat([indices, reverse], dim=1)
    both_values = torch.cat([adjacency.values(), adjacency.values()], dim=0)
    summed = torch.sparse_coo_tensor(
        both_indices, both_values, adjacency.shape,
        device=adjacency.device, dtype=adjacency.dtype,
    ).coalesce()
    counts = torch.sparse_coo_tensor(
        both_indices, torch.ones_like(both_values), adjacency.shape,
        device=adjacency.device, dtype=adjacency.dtype,
    ).coalesce()
    values = summed.values() / counts.values().clamp_min(1.0)
    return torch.sparse_coo_tensor(
        summed.indices(), values.clamp(0.0, 1.0), adjacency.shape,
        device=adjacency.device, dtype=adjacency.dtype,
    ).coalesce()


def sparse_edge_count(adjacency):
    return int(to_sparse_coo(adjacency)._nnz())


def normalize_adjacency(adjacency, mode="sym", add_self_loops=True, eps=1e-8):
    """Normalize sparse raw relation values only at structural-encoder input."""
    adjacency = to_sparse_coo(adjacency)
    if mode == "none":
        return adjacency
    if mode not in {"row", "sym"}:
        raise ValueError(f"Unknown adjacency normalization: {mode}")

    n = adjacency.shape[0]
    indices = adjacency.indices()
    values = adjacency.values()
    if add_self_loops:
        diagonal = torch.arange(n, device=adjacency.device, dtype=torch.long)
        loop_indices = torch.stack([diagonal, diagonal], dim=0)
        indices = torch.cat([indices, loop_indices], dim=1)
        values = torch.cat([values, torch.ones(n, device=values.device, dtype=values.dtype)])
    adjacency = torch.sparse_coo_tensor(
        indices, values, adjacency.shape, device=values.device, dtype=values.dtype
    ).coalesce()

    row, col = adjacency.indices()
    values = adjacency.values()
    degree = torch.zeros(n, device=values.device, dtype=values.dtype)
    degree.index_add_(0, row, values)
    if mode == "row":
        normalized_values = values / degree.index_select(0, row).clamp_min(eps)
    else:
        inv_sqrt = degree.clamp_min(eps).pow(-0.5)
        normalized_values = (
            values * inv_sqrt.index_select(0, row) * inv_sqrt.index_select(0, col)
        )
    if not torch.isfinite(normalized_values).all():
        raise FloatingPointError("Adjacency normalization produced NaN or Inf.")
    return torch.sparse_coo_tensor(
        adjacency.indices(), normalized_values, adjacency.shape,
        device=values.device, dtype=values.dtype,
    ).coalesce()


def prune_sparse_topk(adjacency, k):
    """Keep at most k outgoing non-self edges per row without densifying."""
    adjacency = remove_self_loops(adjacency).coalesce()
    n = adjacency.shape[0]
    k = min(max(int(k), 1), max(n - 1, 1))
    row, col = adjacency.indices()
    values = adjacency.values()
    kept_positions = []
    # coalesce() orders COO entries lexicographically, hence rows are contiguous.
    # Using row==node inside the loop would be O(N*E); bincount gives O(N+E).
    row_counts = torch.bincount(row, minlength=n).detach().cpu().tolist()
    cursor = 0
    for count in row_counts:
        if count == 0:
            continue
        positions = torch.arange(
            cursor, cursor + count, device=row.device, dtype=torch.long
        )
        cursor += count
        if count > k:
            local = torch.topk(values.index_select(0, positions), k=k).indices
            positions = positions.index_select(0, local)
        kept_positions.append(positions)
    if not kept_positions:
        empty_indices = torch.empty((2, 0), device=row.device, dtype=torch.long)
        empty_values = torch.empty(0, device=values.device, dtype=values.dtype)
        return torch.sparse_coo_tensor(
            empty_indices, empty_values, adjacency.shape,
            device=values.device, dtype=values.dtype,
        ).coalesce()
    kept = torch.cat(kept_positions)
    return torch.sparse_coo_tensor(
        torch.stack([row.index_select(0, kept), col.index_select(0, kept)]),
        values.index_select(0, kept), adjacency.shape,
        device=values.device, dtype=values.dtype,
    ).coalesce()


def sparse_columns_to_dense(adjacency, columns):
    adjacency = to_sparse_coo(adjacency)
    columns = columns.to(device=adjacency.device, dtype=torch.long)
    n, _ = adjacency.shape
    output = torch.zeros(
        (n, columns.numel()), device=adjacency.device, dtype=adjacency.dtype
    )
    if columns.numel() == 0 or adjacency._nnz() == 0:
        return output
    inverse = torch.full((n,), -1, device=adjacency.device, dtype=torch.long)
    inverse[columns] = torch.arange(columns.numel(), device=adjacency.device)
    row, col = adjacency.indices()
    target_col = inverse.index_select(0, col)
    keep = target_col >= 0
    output.index_put_(
        (row[keep], target_col[keep]), adjacency.values()[keep], accumulate=True
    )
    return output


def sparse_row_block_to_dense(adjacency, start, end):
    adjacency = to_sparse_coo(adjacency)
    output = torch.zeros(
        (end - start, adjacency.shape[1]),
        device=adjacency.device, dtype=adjacency.dtype,
    )
    if adjacency._nnz() == 0:
        return output
    row, col = adjacency.indices()
    keep = (row >= start) & (row < end)
    output.index_put_(
        (row[keep] - start, col[keep]), adjacency.values()[keep], accumulate=True
    )
    return output


def sparse_values_at(adjacency, row, col):
    """Gather sparse values for an edge list; absent edges return zero."""
    adjacency = to_sparse_coo(adjacency).coalesce()
    n = adjacency.shape[1]
    keys = adjacency.indices()[0] * n + adjacency.indices()[1]
    order = torch.argsort(keys)
    keys = keys.index_select(0, order)
    values = adjacency.values().index_select(0, order)
    query = row * n + col
    position = torch.searchsorted(keys, query)
    valid = position < keys.numel()
    safe_position = position.clamp(max=max(keys.numel() - 1, 0))
    output = torch.zeros(query.numel(), device=row.device, dtype=adjacency.dtype)
    if keys.numel() > 0:
        matched = valid & (keys.index_select(0, safe_position) == query)
        output[matched] = values.index_select(0, safe_position[matched])
    return output


def select_cluster_balanced_anchors(q, confidence, anchors_per_cluster):
    """Select high-confidence anchors from each current pseudo-cluster.

    The per-cluster quota is controlled by ``anchors_per_cluster`` and any
    unfilled positions are assigned to the highest-confidence remaining cells.
    """
    q = q.detach()
    confidence = confidence.detach()
    n, cluster_count = q.shape
    per_cluster = max(int(anchors_per_cluster), 1)
    target = min(per_cluster * cluster_count, n)
    if target <= 0:
        raise ValueError("At least one anchor is required.")

    base, remainder = divmod(target, cluster_count)
    quotas = [base + (1 if cluster < remainder else 0) for cluster in range(cluster_count)]
    pseudo = torch.argmax(q, dim=1)
    selected = []
    selected_mask = torch.zeros(n, device=q.device, dtype=torch.bool)
    for cluster, quota in enumerate(quotas):
        if quota == 0:
            continue
        members = torch.nonzero(pseudo == cluster, as_tuple=False).reshape(-1)
        if members.numel() == 0:
            continue
        take = min(quota, members.numel())
        local = torch.topk(confidence.index_select(0, members), k=take).indices
        chosen = members.index_select(0, local)
        selected.append(chosen)
        selected_mask[chosen] = True

    current = sum(item.numel() for item in selected)
    if current < target:
        remaining = torch.nonzero(~selected_mask, as_tuple=False).reshape(-1)
        take = min(target - current, remaining.numel())
        if take > 0:
            local = torch.topk(confidence.index_select(0, remaining), k=take).indices
            selected.append(remaining.index_select(0, local))
    anchors = torch.cat(selected) if selected else torch.topk(confidence, k=target).indices
    return anchors[:target].detach()

def unified_node_anchor_similarity(z1, z2, h1, h2, anchors, mix):
    """Compute mixed cross-view similarities between all cells and selected anchors."""
    attribute = 0.5 * (
        z1 @ z2.index_select(0, anchors).t()
        + z2 @ z1.index_select(0, anchors).t()
    )
    structure = 0.5 * (
        h1 @ h2.index_select(0, anchors).t()
        + h2 @ h1.index_select(0, anchors).t()
    )
    return mix * attribute + (1.0 - mix) * structure


def unified_similarity_block(z1, z2, h1, h2, start, end, mix):
    attribute = 0.5 * (
        z1[start:end] @ z2.t() + z2[start:end] @ z1.t()
    )
    structure = 0.5 * (
        h1[start:end] @ h2.t() + h2[start:end] @ h1.t()
    )
    return mix * attribute + (1.0 - mix) * structure


def build_anchor_relation_target(
    z1,
    z2,
    h1,
    h2,
    q,
    confidence,
    anchors,
    A_original,
    A_knn,
    mix,
    maturity,
    eta_struct,
    use_pair_reliability=True,
    use_soft_cluster_relation=True,
):
    # Keep relation targets detached while allowing gradients through the structural mixture weight.
    p_observed = sparse_columns_to_dense(A_original, anchors)
    p_attribute = sparse_columns_to_dense(A_knn, anchors)
    p_struct = (
        eta_struct * p_observed + (1.0 - eta_struct) * p_attribute
    ).clamp(0.0, 1.0)
    semantic_similarity = unified_node_anchor_similarity(
        z1, z2, h1, h2, anchors, mix
    )
    p_semantic = similarity_to_unit_interval(semantic_similarity).detach()
    pair_confidence = pair_reliability(
        z1, z2,
        z1.index_select(0, anchors), z2.index_select(0, anchors),
        confidence, confidence.index_select(0, anchors),
        enabled=use_pair_reliability,
    )
    cluster_probability = cluster_relation(
        q, q.index_select(0, anchors), soft=use_soft_cluster_relation
    )
    alpha = torch.as_tensor(
        maturity, device=z1.device, dtype=z1.dtype
    ).detach().clamp(0.0, 1.0)
    target = (
        (1.0 - alpha) * p_struct
        + alpha * (
            pair_confidence * cluster_probability
            + (1.0 - pair_confidence) * p_semantic
        )
    ).clamp(0.0, 1.0)
    return target, pair_confidence.detach()


def soft_threshold(delta, threshold):
    delta = delta.detach()
    return torch.sign(delta) * torch.relu(torch.abs(delta) - threshold)


def lowrank_relation_loss(
    u,
    anchors,
    relation_target,
    pair_confidence,
    sparse_threshold,
    maturity,
    eps=1e-8,
):
    """Compute confidence-weighted low-rank relation reconstruction loss.

    Anchor rows form the low-rank reference, detached residuals are soft-thresholded
    into a sparse corruption term, and the reconstruction loss is scaled by maturity.
    """
    v = u.index_select(0, anchors)
    lowrank = u @ v.t()
    signed_target = 2.0 * relation_target - 1.0
    residual = signed_target - lowrank
    corruption = soft_threshold(residual, sparse_threshold)

    omega = 1.0 + pair_confidence.detach()
    reconstruction = signed_target - lowrank - corruption
    base_loss = (
        torch.sum(omega * reconstruction.square())
        / omega.sum().clamp_min(eps)
    )
    xi = torch.as_tensor(
        maturity, device=u.device, dtype=u.dtype
    ).detach().clamp(0.0, 1.0)
    loss = xi * base_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("L_pur contains NaN or Inf.")

    stats = {
        "mean_abs_e": float(torch.mean(torch.abs(corruption)).detach().cpu()),
        "e_sparsity": float(torch.mean((torch.abs(corruption) <= eps).float()).cpu()),
        "u_std": float(torch.mean(torch.std(u.detach(), dim=0, unbiased=False)).cpu()),
        "mean_anchor_relation": float(relation_target.mean().detach().cpu()),
        "maturity": float(xi.detach().cpu()),
        "base_purification_loss": float(base_loss.detach().cpu()),
    }
    return loss, corruption, lowrank, stats


def _sparse_from_topk(score, start, k, n):
    values, columns = torch.topk(score, k=k, dim=1)
    rows = torch.arange(start, start + score.shape[0], device=score.device)
    rows = rows[:, None].expand_as(columns)
    return rows.reshape(-1), columns.reshape(-1), values.reshape(-1)


def build_semantic_candidate_graph(z, k, chunk_size):
    n = z.shape[0]
    k = min(max(int(k), 1), max(n - 1, 1))
    rows_all, cols_all = [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        score = z[start:end] @ z.t()
        local_rows = torch.arange(start, end, device=z.device)
        score[torch.arange(end - start, device=z.device), local_rows] = float("-inf")
        rows, cols, _ = _sparse_from_topk(score, start, k, n)
        rows_all.append(rows)
        cols_all.append(cols)
    rows = torch.cat(rows_all)
    cols = torch.cat(cols_all)
    values = torch.ones(rows.numel(), device=z.device, dtype=z.dtype)
    return torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (n, n),
        device=z.device, dtype=z.dtype,
    ).coalesce()


def _candidate_union(A_original, A_knn, semantic_graph):
    graphs = [to_sparse_coo(A_original), to_sparse_coo(A_knn), to_sparse_coo(semantic_graph)]
    indices = torch.cat([graph.indices() for graph in graphs], dim=1)
    values = torch.ones(indices.shape[1], device=indices.device, dtype=graphs[0].dtype)
    union = torch.sparse_coo_tensor(
        indices, values, graphs[0].shape, device=values.device, dtype=values.dtype
    ).coalesce()
    return remove_self_loops(union)


def _direct_relation_edge_scores(
    row,
    col,
    z1,
    z2,
    h1,
    h2,
    q,
    confidence,
    A_original,
    A_knn,
    mix,
    maturity,
    eta_struct,
    use_pair_reliability,
    use_soft_cluster_relation,
):
    p_struct = (
        eta_struct * sparse_values_at(A_original, row, col)
        + (1.0 - eta_struct) * sparse_values_at(A_knn, row, col)
    ).clamp(0.0, 1.0)
    attribute = 0.5 * (
        torch.sum(z1.index_select(0, row) * z2.index_select(0, col), dim=1)
        + torch.sum(z2.index_select(0, row) * z1.index_select(0, col), dim=1)
    )
    structure = 0.5 * (
        torch.sum(h1.index_select(0, row) * h2.index_select(0, col), dim=1)
        + torch.sum(h2.index_select(0, row) * h1.index_select(0, col), dim=1)
    )
    p_semantic = similarity_to_unit_interval(mix * attribute + (1.0 - mix) * structure)
    pair_confidence = pair_reliability_edges(
        z1, z2, confidence, row, col, enabled=use_pair_reliability
    )
    cluster_probability = cluster_relation_edges(
        q, row, col, soft=use_soft_cluster_relation
    )
    alpha = torch.as_tensor(maturity, device=row.device, dtype=z1.dtype)
    return (
        (1.0 - alpha) * p_struct
        + alpha * (
            pair_confidence * cluster_probability
            + (1.0 - pair_confidence) * p_semantic
        )
    ).clamp(0.0, 1.0)


def _build_full_chunked_topology(
    u,
    z1,
    z2,
    h1,
    h2,
    q,
    confidence,
    A_original,
    A_knn,
    mix,
    maturity,
    eta_struct,
    relation_k,
    chunk_size,
    use_lowrank_relation,
    use_pair_reliability,
    use_soft_cluster_relation,
):
    n = z1.shape[0]
    k = min(max(int(relation_k), 1), max(n - 1, 1))
    rows_all, cols_all, values_all = [], [], []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        if use_lowrank_relation:
            similarity = u[start:end] @ u.t()
            node_reliability = torch.sqrt(
                confidence[start:end, None].clamp_min(0.0)
                * confidence[None, :].clamp_min(0.0)
            )
            score = similarity_to_unit_interval(similarity) * node_reliability
        else:
            p_struct = (
                eta_struct * sparse_row_block_to_dense(A_original, start, end)
                + (1.0 - eta_struct) * sparse_row_block_to_dense(A_knn, start, end)
            ).clamp(0.0, 1.0)
            p_semantic = similarity_to_unit_interval(
                unified_similarity_block(z1, z2, h1, h2, start, end, mix)
            )
            pair_confidence = pair_reliability(
                z1[start:end], z2[start:end], z1, z2,
                confidence[start:end], confidence,
                enabled=use_pair_reliability,
            )
            cluster_probability = cluster_relation(
                q[start:end], q, soft=use_soft_cluster_relation
            )
            alpha = torch.as_tensor(maturity, device=z1.device, dtype=z1.dtype)
            score = (
                (1.0 - alpha) * p_struct
                + alpha * (
                    pair_confidence * cluster_probability
                    + (1.0 - pair_confidence) * p_semantic
                )
            ).clamp(0.0, 1.0)

        local_rows = torch.arange(start, end, device=z1.device)
        score[torch.arange(end - start, device=z1.device), local_rows] = float("-inf")
        if not torch.isfinite(score[score != float("-inf")]).all():
            raise FloatingPointError("Topology score contains NaN or Inf.")
        rows, cols, values = _sparse_from_topk(score, start, k, n)
        rows_all.append(rows)
        cols_all.append(cols)
        values_all.append(values)

    rows = torch.cat(rows_all)
    cols = torch.cat(cols_all)
    values = torch.cat(values_all).clamp(0.0, 1.0)
    graph = torch.sparse_coo_tensor(
        torch.stack([rows, cols]), values, (n, n),
        device=z1.device, dtype=z1.dtype,
    ).coalesce()
    return symmetrize_sparse(graph), float(values.mean().detach().cpu())


def _build_candidate_topology(
    u,
    z,
    z1,
    z2,
    h1,
    h2,
    q,
    confidence,
    A_original,
    A_knn,
    mix,
    maturity,
    eta_struct,
    relation_k,
    semantic_candidate_k,
    chunk_size,
    use_lowrank_relation,
    use_pair_reliability,
    use_soft_cluster_relation,
):
    semantic_graph = build_semantic_candidate_graph(
        z, semantic_candidate_k, chunk_size
    )
    candidates = _candidate_union(A_original, A_knn, semantic_graph)
    row, col = candidates.indices()
    if use_lowrank_relation:
        similarity = torch.sum(
            u.index_select(0, row) * u.index_select(0, col), dim=1
        )
        node_reliability = torch.sqrt(
            confidence.index_select(0, row).clamp_min(0.0)
            * confidence.index_select(0, col).clamp_min(0.0)
        )
        values = similarity_to_unit_interval(similarity) * node_reliability
    else:
        values = _direct_relation_edge_scores(
            row, col, z1, z2, h1, h2, q, confidence,
            A_original, A_knn, mix, maturity, eta_struct,
            use_pair_reliability, use_soft_cluster_relation,
        )
    if not torch.isfinite(values).all():
        raise FloatingPointError("Candidate topology score contains NaN or Inf.")
    weighted = torch.sparse_coo_tensor(
        candidates.indices(), values.clamp(0.0, 1.0), candidates.shape,
        device=values.device, dtype=values.dtype,
    ).coalesce()
    pruned = prune_sparse_topk(weighted, relation_k)
    return symmetrize_sparse(pruned), float(values.mean().detach().cpu())


def build_purified_topology(
    u,
    z,
    z1,
    z2,
    h1,
    h2,
    q,
    confidence,
    A_original,
    A_knn,
    mix,
    maturity,
    eta_struct,
    relation_k,
    chunk_size,
    topology_mode,
    full_threshold,
    semantic_candidate_k,
    use_lowrank_relation=True,
    use_pair_reliability=True,
    use_soft_cluster_relation=True,
):
    n = z.shape[0]
    resolved_mode = topology_mode
    if resolved_mode == "auto":
        resolved_mode = "full_chunked" if n <= full_threshold else "candidate_sparse"
    if use_lowrank_relation and u is None:
        raise ValueError("U is required when low-rank relation purification is enabled.")
    if resolved_mode == "full_chunked":
        graph, mean_score = _build_full_chunked_topology(
            u, z1, z2, h1, h2, q, confidence, A_original, A_knn,
            mix, maturity, eta_struct, relation_k, chunk_size,
            use_lowrank_relation, use_pair_reliability, use_soft_cluster_relation,
        )
    elif resolved_mode == "candidate_sparse":
        graph, mean_score = _build_candidate_topology(
            u, z, z1, z2, h1, h2, q, confidence, A_original, A_knn,
            mix, maturity, eta_struct, relation_k,
            semantic_candidate_k, chunk_size, use_lowrank_relation,
            use_pair_reliability, use_soft_cluster_relation,
        )
    else:
        raise ValueError(f"Unknown topology mode: {topology_mode}")
    return graph.coalesce(), mean_score, resolved_mode


def topology_change_ratio(previous, current):
    previous = to_sparse_coo(previous).coalesce()
    current = to_sparse_coo(current).coalesce()
    n = previous.shape[1]
    old_key = (
        previous.indices()[0] * n + previous.indices()[1]
    ).detach().cpu().numpy()
    new_key = (
        current.indices()[0] * n + current.indices()[1]
    ).detach().cpu().numpy()
    old_set = set(old_key.tolist())
    new_set = set(new_key.tolist())
    union = old_set | new_set
    if not union:
        return 0.0
    return 1.0 - len(old_set & new_set) / len(union)


def adaptive_topology_update(previous, purified, maturity, k):
    """Blend the current and purified graphs using the supplied maturity weight.

    The merged graph is clipped, pruned to the strongest outgoing relations, and
    symmetrized before it is returned.
    """
    previous = to_sparse_coo(previous)
    purified = to_sparse_coo(purified)
    xi = torch.as_tensor(
        maturity, device=previous.device, dtype=previous.dtype
    ).detach().clamp(0.0, 1.0)
    indices = torch.cat([previous.indices(), purified.indices()], dim=1)
    values = torch.cat([
        (1.0 - xi) * previous.values(), xi * purified.values()
    ])
    merged = torch.sparse_coo_tensor(
        indices, values, previous.shape, device=values.device, dtype=values.dtype
    ).coalesce()
    merged = torch.sparse_coo_tensor(
        merged.indices(), merged.values().clamp(0.0, 1.0), merged.shape,
        device=values.device, dtype=values.dtype,
    ).coalesce()
    pruned = prune_sparse_topk(merged, k)
    return symmetrize_sparse(pruned)

