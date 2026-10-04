import csv
import os

import numpy as np
import torch
from torch import optim
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from model import ScRARENetwork
from opt import args
from relation import (
    build_anchor_relation_target,
    build_purified_topology,
    adaptive_topology_update,
    lowrank_relation_loss,
    normalize_adjacency,
    select_cluster_balanced_anchors,
    sparse_edge_count,
    to_sparse_coo,
    topology_change_ratio,
)
from reliability import (
    conformalized_node_confidence,
    legacy_top_h_mask,
    square_euclidean_distance,
    student_t_soft_assignment,
)
from setup import setup_args
from utils import (
    comprehensive_similarity,
    confidence_weighted_clustering_loss,
    hard_self_supervision_loss,
    high_confidence_cluster_loss,
    laplacian_filtering,
    legacy_high_confidence_indices,
    legacy_pseudo_matrix,
    load_graph_data,
    run_kmeans,
    setup_seed,
    reliability_aware_contrastive_loss,
)

import h5py

    
def refresh_clustering(z, cluster_num):
    """Refresh K-means pseudo-labels and cluster centers from the current embedding."""
    pseudo_cpu, centers, inertia = run_kmeans(z.detach(), cluster_num)
    centers = centers.detach().to(device=z.device, dtype=z.dtype)
    return {
        "centers": centers,
        "pseudo": pseudo_cpu.to(device=z.device, dtype=torch.long),
        "inertia": inertia,
    }


def current_assignment(z, clustering_state, calibration_seed):
    """Compute soft assignments, assignment entropy, and calibrated node confidence."""
    distance = square_euclidean_distance(z, clustering_state["centers"])
    q = student_t_soft_assignment(
        distance, gamma=args.cluster_gamma, eps=args.eps
    )
    compatibility, confidence, uncertainty, conformal_margin = (
        conformalized_node_confidence(
            q.detach(), eps=args.eps, seed=calibration_seed
        )
    )
    return q, confidence, uncertainty, compatibility, conformal_margin

def update_legacy_weights(model, pseudo, dense_similarity, binary_confidence):
    high_nodes, high_both_views = legacy_high_confidence_indices(binary_confidence)
    positive, matrix = legacy_pseudo_matrix(
        pseudo,
        dense_similarity,
        binary_confidence.numel(),
        beta=args.beta,
        eps=args.eps,
    )
    with torch.no_grad():
        model.pos_weight[high_both_views] = positive[high_both_views]
        if high_both_views.numel() > 0:
            row = high_both_views[:, None].expand(-1, high_both_views.numel())
            col = high_both_views[None, :].expand(high_both_views.numel(), -1)
            model.pos_neg_weight[row, col] = matrix[row, col]
    return high_nodes


def clustering_accuracy(y_true, y_pred):
    """Permutation-invariant clustering ACC via Hungarian matching."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"Label length mismatch: y_true={y_true.shape[0]}, "
            f"y_pred={y_pred.shape[0]}"
        )
    if y_true.size == 0:
        return 0.0

    _, true_ids = np.unique(y_true, return_inverse=True)
    _, pred_ids = np.unique(y_pred, return_inverse=True)

    n_true = int(true_ids.max()) + 1
    n_pred = int(pred_ids.max()) + 1
    size = max(n_true, n_pred)

    contingency = np.zeros((size, size), dtype=np.int64)
    np.add.at(contingency, (pred_ids, true_ids), 1)

    # Maximize matched samples by minimizing the negated contingency matrix.
    row_ind, col_ind = linear_sum_assignment(-contingency)
    matched = contingency[row_ind, col_ind].sum()
    return float(matched) / float(y_true.size)


def evaluate_q(q, labels):
    """Return ACC, NMI, ARI and AMI in percentage.

    This evaluator is intentionally self-contained and does not call utils.eva(),
    so evaluation remains independent of legacy utility code.
    """
    predicted = torch.argmax(q.detach(), dim=1).cpu().numpy()
    labels_np = np.asarray(labels).reshape(-1)

    accuracy = clustering_accuracy(labels_np, predicted)
    nmi = normalized_mutual_info_score(
        labels_np,
        predicted,
        average_method="arithmetic",
    )
    ari = adjusted_rand_score(labels_np, predicted)
    ami = adjusted_mutual_info_score(
        labels_np,
        predicted,
        average_method="arithmetic",
    )

    return tuple(
        100.0 * value
        for value in (accuracy, nmi, ari, ami)
    )


def assert_finite_gradients(model):
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise FloatingPointError(f"Gradient for '{name}' contains NaN or Inf.")


def train_one_run(dataset_name, run_seed):
    setup_seed(run_seed)
    feature, labels, A_original, A_knn, node_num, cluster_num = load_graph_data(
        dataset_name, show_details=False
    )
    feature_filtered = laplacian_filtering(
        A_original, feature, args.t, eps=args.eps
    )

    device = torch.device(args.device)
    feature_filtered = feature_filtered.to(device)
    A_original = to_sparse_coo(A_original, device=device)
    A_knn = to_sparse_coo(A_knn, device=device)
    A_dyn = A_original.clone().coalesce()  # A_dyn^(0)=A exactly.

    # Limit relation-embedding width by the number of clusters and latent dimension.
    relation_rank = min(cluster_num, args.dims)
    enable_projector = args.method == "proposed" and args.use_lowrank_relation
    model = ScRARENetwork(
        input_dim=feature_filtered.shape[1],
        hidden_dim=args.dims,
        act=args.activate,
        n_num=node_num,
        cluster_num=cluster_num,
        method=args.method,
        relation_rank=relation_rank if enable_projector else 0,
        allocate_legacy_weights=(args.method == "legacy" or args.legacy_weighting),
        eps=args.eps,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.method == "proposed":
        A_knn_encoder = normalize_adjacency(
            A_knn, mode=args.structural_normalization, eps=args.eps
        )
        A_dyn_encoder = normalize_adjacency(
            A_dyn, mode=args.structural_normalization, eps=args.eps
        )
    else:
        # Feed raw adjacency rows to the sparse structural encoders.
        A_knn_encoder = A_knn
        A_dyn_encoder = A_dyn

    clustering_state = None
    anchor_indices = None
    maturity = torch.tensor(0.0, device=device)
    high_conf_nodes = None
    # Metric order: ACC, NMI, ARI, AMI.
    best_metrics = (0.0, 0.0, 0.0, 0.0)
    final_metrics = best_metrics
    topology_stats = {
        "change_ratio": 0.0,
        "mean_relation_score": 0.0,
        "mode": "disabled" if not args.use_dynamic_topology else "warmup",
    }
    relation_stats = {
        "mean_abs_e": 0.0,
        "e_sparsity": 1.0,
        "u_std": 0.0,
        "mean_anchor_relation": 0.0,
    }

    progress = tqdm(range(args.epochs), desc=f"{dataset_name}/{args.method}/seed={run_seed}")
    for epoch in progress:
        model.train()
        z1, z2, h1, h2 = model(feature_filtered, A_dyn_encoder, A_knn_encoder)
        z = 0.5 * (z1 + z2)

        clustering_due = clustering_state is None or (
            epoch % args.clustering_update_interval == 0
        )
        if clustering_due:
            clustering_state = refresh_clustering(z, cluster_num)

        # Recompute assignment statistics from the current embedding; refresh
        # K-means centers only at the configured clustering interval.
        q, confidence, uncertainty, compatibility, conformal_margin = (
            current_assignment(z, clustering_state, calibration_seed=run_seed)
        )
        maturity = (1.0 - uncertainty.mean()).detach().clamp(0.0, 1.0)
        binary_confidence = legacy_top_h_mask(conformal_margin, args.tao)

        # Update the dynamic topology on the same interval as cluster-center refresh.
        topology_due = clustering_due

        # Refresh anchors from current pseudo-clusters and node confidence each epoch.
        if args.method == "proposed" and args.use_lowrank_relation:
            anchor_indices = select_cluster_balanced_anchors(
                q.detach(), confidence.detach(),
                anchors_per_cluster=args.anchors_per_cluster,
            )

        if args.method == "legacy":
            mix = model.similarity_mix(constrained=False)
            if clustering_due and epoch >= args.warmup_epochs:
                dense_similarity = comprehensive_similarity(z1, z2, h1, h2, mix)
                high_conf_nodes = update_legacy_weights(
                    model,
                    clustering_state["pseudo"],
                    dense_similarity,
                    binary_confidence,
                )
            loss_con, mean_hard_weight = reliability_aware_contrastive_loss(
                z1, z2, h1, h2, mix, node_num,
                temperature=args.temperature,
                chunk_size=args.topology_chunk_size,
                legacy_pos_neg_weight=model.pos_neg_weight,
                legacy_pos_weight=model.pos_weight,
            )
            if epoch >= args.warmup_epochs:
                loss_clu = high_confidence_cluster_loss(
                    z, clustering_state["centers"], high_conf_nodes,
                    gamma=args.cluster_gamma,
                )
                logits1, logits2 = model.semantic_logits(z1, z2)
                loss_s = hard_self_supervision_loss(
                    logits1,
                    logits2,
                    clustering_state["pseudo"],
                    high_conf_nodes,
                )
            else:
                loss_clu = z.new_zeros(())
                loss_s = z.new_zeros(())
            loss_pur = z.new_zeros(())
            total_loss = (
                loss_con
                + args.lambda_cluster * loss_clu
                + args.lambda_hard_self * loss_s
            )
        else:
            mix = model.similarity_mix(constrained=True)
            if args.legacy_weighting:
                if clustering_due:
                    dense_similarity = comprehensive_similarity(
                        z1, z2, h1, h2, mix
                    )
                    high_conf_nodes = update_legacy_weights(
                        model,
                        clustering_state["pseudo"],
                        dense_similarity,
                        binary_confidence,
                    )
                loss_con, mean_hard_weight = reliability_aware_contrastive_loss(
                    z1, z2, h1, h2, mix, node_num,
                    temperature=args.temperature,
                    chunk_size=args.topology_chunk_size,
                    legacy_pos_neg_weight=model.pos_neg_weight,
                    legacy_pos_weight=model.pos_weight,
                )
            else:
                # Compute contrastive loss with reliability-based pair reweighting.
                loss_con, mean_hard_weight = reliability_aware_contrastive_loss(
                    z1, z2, h1, h2, mix, node_num,
                    temperature=args.temperature,
                    chunk_size=args.topology_chunk_size,
                    q=q.detach(),
                    confidence=confidence.detach(),
                    beta=args.beta,
                    use_pair_reliability=args.use_pair_reliability,
                    use_soft_cluster_relation=args.use_soft_cluster_relation,
                )

            # Encourage confident cells to remain close to their soft-assignment centers.
            if args.use_continuous_confidence:
                loss_clu = confidence_weighted_clustering_loss(
                    z,
                    clustering_state["centers"],
                    q,
                    confidence,
                    eps=args.eps,
                )
            else:
                high_conf_nodes, _ = legacy_high_confidence_indices(binary_confidence)
                loss_clu = high_confidence_cluster_loss(
                    z,
                    clustering_state["centers"],
                    high_conf_nodes,
                    gamma=args.cluster_gamma,
                )

            # Build cell-anchor targets and optimize the low-rank relation factor.
            loss_pur = z.new_zeros(())
            if args.use_lowrank_relation and anchor_indices is not None:
                relation_target, anchor_pair_confidence = build_anchor_relation_target(
                    z1, z2, h1, h2, q.detach(), confidence.detach(),
                    anchor_indices, A_original, A_knn, mix,
                    maturity, model.structure_prior_mix(),
                    use_pair_reliability=args.use_pair_reliability,
                    use_soft_cluster_relation=args.use_soft_cluster_relation,
                )
                u = model.relation_embedding(z)
                if not torch.isfinite(u).all():
                    raise FloatingPointError("Relation factor U contains NaN or Inf.")
                loss_pur, _, _, relation_stats = lowrank_relation_loss(
                    u,
                    anchor_indices,
                    relation_target,
                    anchor_pair_confidence,
                    args.relation_sparse_threshold,
                    maturity=maturity,
                    eps=args.eps,
                )

            # Combine contrastive, clustering, and weighted relation-purification losses.
            total_loss = loss_con + loss_clu + args.lambda_r * loss_pur

        if not torch.isfinite(total_loss):
            raise FloatingPointError(f"Total loss is non-finite at epoch {epoch}.")
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        assert_finite_gradients(model)
        if args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        optimizer.step()
        if args.method == "proposed":
            with torch.no_grad():
                model.alpha.clamp_(0.0, 1.0)

        # Recompute relation embeddings after the optimizer step before updating A_dyn.
        if (
            args.method == "proposed"
            and args.use_dynamic_topology
            and topology_due
        ):
            model.eval()
            with torch.no_grad():
                tz1, tz2, th1, th2 = model(
                    feature_filtered, A_dyn_encoder, A_knn_encoder
                )
                tz = 0.5 * (tz1 + tz2)
                (
                    tq, tconfidence, tuncertainty, _, _
                ) = current_assignment(
                    tz, clustering_state, calibration_seed=run_seed
                )
                topology_maturity = (
                    1.0 - tuncertainty.mean()
                ).detach().clamp(0.0, 1.0)
                tu = model.relation_embedding(tz) if args.use_lowrank_relation else None
                A_pur, mean_score, resolved_mode = build_purified_topology(
                    tu, tz, tz1, tz2, th1, th2, tq.detach(), tconfidence,
                    A_original, A_knn, model.similarity_mix(constrained=True),
                    topology_maturity, model.structure_prior_mix(),
                    args.relation_k, args.topology_chunk_size,
                    args.topology_mode, args.topology_full_threshold,
                    args.semantic_candidate_k,
                    use_lowrank_relation=args.use_lowrank_relation,
                    use_pair_reliability=args.use_pair_reliability,
                    use_soft_cluster_relation=args.use_soft_cluster_relation,
                )
                updated = adaptive_topology_update(
                    A_dyn, A_pur, topology_maturity, args.relation_k
                )
                change = topology_change_ratio(A_dyn, updated)
                A_dyn = updated.detach().coalesce()
                A_dyn_encoder = normalize_adjacency(
                    A_dyn,
                    mode=args.structural_normalization,
                    eps=args.eps,
                )
                topology_stats = {
                    "change_ratio": change,
                    "mean_relation_score": mean_score,
                    "mode": resolved_mode,
                }
        elif args.method == "proposed" and not args.use_dynamic_topology:
            topology_stats["mode"] = "fixed"

        should_log = epoch % args.log_interval == 0 or epoch == args.epochs - 1
        if should_log:
            epoch_metrics = evaluate_q(q, labels)
            if epoch_metrics[0] >= best_metrics[0]:
                best_metrics = epoch_metrics
            anchor_number = 0 if anchor_indices is None else int(anchor_indices.numel())
            if args.method == "proposed":
                parameter_log = (
                    f"xi={maturity.item():.4f} "
                    f"alpha={model.similarity_mix(constrained=True).item():.4f} "
                    f"eta={model.structure_prior_mix().item():.4f} "
                    f"lambda_R={args.lambda_r:.4f} "
                )
            else:
                parameter_log = ""
            message = (
                f"epoch={epoch:04d} L_total={total_loss.item():.6f} "
                f"L_con={loss_con.item():.6f} L_clu={loss_clu.item():.6f} "
                f"L_pur={loss_pur.item():.6f} "
                f"ACC={epoch_metrics[0]:.2f} NMI={epoch_metrics[1]:.2f} "
                f"ARI={epoch_metrics[2]:.2f} AMI={epoch_metrics[3]:.2f} "
                f"mean_margin={conformal_margin.mean().item():.4f} "
                f"mean_e={uncertainty.mean().item():.4f} "
                f"mean_c={confidence.mean().item():.4f} "
                f"{parameter_log}anchors={anchor_number} "
                f"A_dyn_edges={sparse_edge_count(A_dyn)} "
                f"topo_change={topology_stats['change_ratio']:.4f} "
                f"mean_relation={topology_stats['mean_relation_score']:.4f} "
                f"mean|B|={relation_stats['mean_abs_e']:.4f} "
                f"B_sparsity={relation_stats['e_sparsity']:.4f} "
                f"mean_W={mean_hard_weight.item():.4f} "
                f"U_std={relation_stats['u_std']:.4f} "
                f"topo_mode={topology_stats['mode']}"
            )
            progress.write(message)

    # Evaluate the final model state; keep best-epoch metrics only for diagnostics.
    model.eval()
    with torch.no_grad():
        fz1, fz2, fh1, fh2 = model(
            feature_filtered, A_dyn_encoder, A_knn_encoder
        )
        fz = 0.5 * (fz1 + fz2)
        fq, _, _, _, _ = current_assignment(
            fz, clustering_state, calibration_seed=run_seed
        )
    final_metrics = evaluate_q(fq, labels)
    return best_metrics, final_metrics


def append_result(path, dataset_name, method, run_seed, best_metrics, final_metrics):
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if not exists:
            writer.writerow([
                "dataset", "method", "seed", "best_acc", "best_nmi",
                "best_ari", "best_ami", "final_acc", "final_nmi",
                "final_ari", "final_ami",
            ])
        writer.writerow([
            dataset_name, method, run_seed,
            *[f"{value:.4f}" for value in best_metrics],
            *[f"{value:.4f}" for value in final_metrics],
        ])


def main():
    setup_args(args.dataset)
    all_final = []
    for run_index in range(args.runs):
        run_seed = args.seed + run_index
        best_metrics, final_metrics = train_one_run(args.dataset, run_seed)
        append_result(
            args.result_file,
            args.dataset,
            args.method,
            run_seed,
            best_metrics,
            final_metrics,
        )
        all_final.append(final_metrics)
        print(
            "Run [{}/{}] seed={}: final ACC={:.2f}, NMI={:.2f}, ARI={:.2f}, AMI={:.2f} "
            "(best-by-ACC diagnostic: {:.2f}/{:.2f}/{:.2f}/{:.2f})".format(
                run_index + 1, args.runs, run_seed,
                *final_metrics, *best_metrics
            )
        )

    values = np.asarray(all_final, dtype=np.float64)
    names = ["ACC", "NMI", "ARI", "AMI"]
    print("=" * 72)
    print(f"Final-model summary over {args.runs} run(s):")
    for column, name in enumerate(names):
        print(f"{name}: {values[:, column].mean():.2f} +/- {values[:, column].std():.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
