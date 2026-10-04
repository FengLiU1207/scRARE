import numpy as np
import torch
from scipy import stats


def square_euclidean_distance(z, centers):
    """Squared Euclidean distances, shape [N,K], with round-off clamping."""
    z2 = torch.sum(z * z, dim=1, keepdim=True)
    c2 = torch.sum(centers * centers, dim=1).unsqueeze(0)
    distance = z2 + c2 - 2.0 * (z @ centers.t())
    return distance.clamp_min(0.0)


def student_t_soft_assignment(distance, gamma=1.0, eps=1e-8):
    if gamma <= 0:
        raise ValueError("cluster_gamma must be positive")
    numerator = torch.pow(1.0 + distance / gamma, -1.0)
    denominator = numerator.sum(dim=1, keepdim=True).clamp_min(eps)
    q = numerator / denominator
    if not torch.isfinite(q).all():
        raise FloatingPointError("Student-t assignment Q contains NaN or Inf.")
    return q


def normalized_assignment_entropy(q, eps=1e-8):
    """u_i=-sum_k q_ik log q_ik/log(K), safely handling K=1."""
    if q.shape[1] <= 1:
        return torch.zeros(q.shape[0], device=q.device, dtype=q.dtype)
    log_k = q.new_tensor(float(np.log(q.shape[1]))).clamp_min(eps)
    entropy = -torch.sum(q * torch.log(q.clamp_min(eps)), dim=1) / log_k
    return entropy.clamp(0.0, 1.0)


class BetaMixtureModel:
    """Numerically guarded two-component beta mixture for distance reliability."""

    def __init__(self, max_iter=50, tol=1e-4, eps=1e-8):
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.eps = float(eps)

    @staticmethod
    def _beta_parameters(mean, variance, eps):
        mean = float(np.clip(mean, 1e-4, 1.0 - 1e-4))
        max_variance = max(mean * (1.0 - mean) - 1e-6, eps)
        variance = float(np.clip(variance, eps, max_variance))
        concentration = max(mean * (1.0 - mean) / variance - 1.0, 2e-3)
        alpha = max(mean * concentration, 1e-3)
        beta = max((1.0 - mean) * concentration, 1e-3)
        return alpha, beta

    def fit_predict(self, values):
        x = np.asarray(values, dtype=np.float64).reshape(-1)
        if x.size == 0:
            return np.empty(0, dtype=np.float32)
        if not np.isfinite(x).all():
            raise FloatingPointError("BMM input contains NaN or Inf.")
        x = np.clip(x, 1e-4, 1.0 - 1e-4)

        # Return a neutral score when the input has too little variation to fit two components.
        if float(np.ptp(x)) < self.tol or float(np.var(x)) < self.eps:
            return np.full(x.shape, 0.5, dtype=np.float32)

        means = np.array([np.percentile(x, 25), np.percentile(x, 75)], dtype=np.float64)
        variances = np.full(2, max(float(np.var(x)), self.eps), dtype=np.float64)
        priors = np.array([0.5, 0.5], dtype=np.float64)
        responsibilities = np.full((x.size, 2), 0.5, dtype=np.float64)

        for _ in range(self.max_iter):
            previous = np.concatenate([means.copy(), variances.copy(), priors.copy()])
            log_joint = np.empty((x.size, 2), dtype=np.float64)
            for component in range(2):
                alpha, beta = self._beta_parameters(
                    means[component], variances[component], self.eps
                )
                log_joint[:, component] = (
                    np.log(max(priors[component], self.eps))
                    + stats.beta.logpdf(x, alpha, beta)
                )

            row_max = np.max(log_joint, axis=1, keepdims=True)
            stable = np.exp(log_joint - row_max)
            responsibilities = stable / np.maximum(
                stable.sum(axis=1, keepdims=True), self.eps
            )
            if not np.isfinite(responsibilities).all():
                return np.clip(1.0 - x, 0.0, 1.0).astype(np.float32)

            mass = responsibilities.sum(axis=0) + self.eps
            priors = mass / float(x.size)
            means = (responsibilities * x[:, None]).sum(axis=0) / mass
            centered = x[:, None] - means[None, :]
            variances = (responsibilities * centered * centered).sum(axis=0) / mass

            current = np.concatenate([means, variances, priors])
            if np.max(np.abs(current - previous)) < self.tol:
                break

        clean_component = int(np.argmin(means))
        posterior = responsibilities[:, clean_component]
        if not np.isfinite(posterior).all():
            posterior = 1.0 - x
        return np.clip(posterior, 0.0, 1.0).astype(np.float32)


def estimate_beta_mixture_reliability(
    distance,
    q,
    max_iter=50,
    tol=1e-4,
    eps=1e-8,
):
    """Return r_i and the assignment-weighted distance d_i."""
    weighted_distance = torch.sum(q * distance, dim=1)
    if not torch.isfinite(weighted_distance).all():
        raise FloatingPointError("Assignment-weighted distances contain NaN or Inf.")
    d_min = weighted_distance.min()
    d_max = weighted_distance.max()
    spread = d_max - d_min
    if spread <= eps:
        normalized = torch.full_like(weighted_distance, 0.5)
    else:
        normalized = (weighted_distance - d_min) / spread.clamp_min(eps)
    normalized = normalized.clamp(1e-4, 1.0 - 1e-4)

    bmm = BetaMixtureModel(max_iter=max_iter, tol=tol, eps=eps)
    posterior = bmm.fit_predict(normalized.detach().cpu().numpy())
    reliability = torch.as_tensor(
        posterior, device=distance.device, dtype=distance.dtype
    ).clamp(0.0, 1.0)
    return reliability, weighted_distance



def conformalized_node_confidence(q, eps=1e-8, seed=0):
    """Estimate node confidence from soft assignments with two-fold calibration.

    The function splits cells into two folds, samples one pseudo-cluster per
    calibration cell from its assignment distribution, builds opposite-fold
    nonconformity references, and returns cluster compatibility, node confidence,
    normalized assignment entropy, and the top-two compatibility margin.
    """
    if q.ndim != 2:
        raise ValueError(f"Q must be 2-D, got shape {tuple(q.shape)}")
    if not torch.isfinite(q).all():
        raise FloatingPointError("CNRC received NaN or Inf in Q.")

    n, cluster_count = q.shape
    if n == 0:
        empty = q.new_empty((0,))
        return q.new_empty((0, cluster_count)), empty, empty, empty

    uncertainty = normalized_assignment_entropy(q.detach(), eps=eps)
    if cluster_count <= 1:
        compatibility = q.new_ones((n, cluster_count))
        margin = q.new_ones((n,))
        confidence = (margin * (1.0 - uncertainty)).clamp(0.0, 1.0)
        return (
            compatibility.detach(), confidence.detach(),
            uncertainty.detach(), margin.detach(),
        )

    # Build empirical calibration references on CPU with an isolated random generator.
    q_cpu = q.detach().float().cpu().clamp_min(0.0)
    q_cpu = q_cpu / q_cpu.sum(dim=1, keepdim=True).clamp_min(float(eps))

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))

    permutation = torch.randperm(n, generator=generator)
    fold_id = torch.empty(n, dtype=torch.long)
    fold_id[permutation[::2]] = 0
    fold_id[permutation[1::2]] = 1

    # Sample one pseudo-cluster from each cell's current soft assignment.
    pseudo = torch.multinomial(
        q_cpu, num_samples=1, replacement=True, generator=generator
    ).squeeze(1)
    compatibility_cpu = torch.empty((n, cluster_count), dtype=q_cpu.dtype)

    all_indices = torch.arange(n, dtype=torch.long)
    for target_fold in (0, 1):
        target_idx = all_indices[fold_id == target_fold]
        calib_idx = all_indices[fold_id != target_fold]
        if target_idx.numel() == 0:
            continue
        if calib_idx.numel() == 0:
            # Fall back to the target assignments when no opposite-fold samples exist.
            compatibility_cpu[target_idx] = q_cpu.index_select(0, target_idx)
            continue

        calib_labels = pseudo.index_select(0, calib_idx)
        calib_q = q_cpu.index_select(0, calib_idx)
        local_row = torch.arange(calib_idx.numel(), dtype=torch.long)
        calib_scores = 1.0 - calib_q[local_row, calib_labels]
        pooled_scores = torch.sort(calib_scores).values
        target_q = q_cpu.index_select(0, target_idx)

        for cluster in range(cluster_count):
            cluster_scores = calib_scores[calib_labels == cluster]
            # Use pooled opposite-fold scores when a cluster has too few calibration samples.
            if cluster_scores.numel() < 2:
                reference = pooled_scores
            else:
                reference = torch.sort(cluster_scores).values

            test_scores = 1.0 - target_q[:, cluster]
            # Convert the target nonconformity into an empirical compatibility score.
            left = torch.searchsorted(reference, test_scores, right=False)
            count_ge = reference.numel() - left
            compatibility_cpu[target_idx, cluster] = (
                count_ge.to(target_q.dtype) + 1.0
            ) / float(reference.numel() + 1)

    if not torch.isfinite(compatibility_cpu).all():
        raise FloatingPointError("CNRC compatibility scores contain NaN or Inf.")

    top2 = torch.topk(compatibility_cpu, k=2, dim=1).values
    margin_cpu = (top2[:, 0] - top2[:, 1]).clamp(0.0, 1.0)

    compatibility = compatibility_cpu.to(device=q.device, dtype=q.dtype)
    margin = margin_cpu.to(device=q.device, dtype=q.dtype)
    confidence = (margin * (1.0 - uncertainty)).clamp(0.0, 1.0)

    return (
        compatibility.detach(), confidence.detach(),
        uncertainty.detach(), margin.detach(),
    )


def estimate_conformal_reliability(distance, q, eps=1e-8, seed=0):
    """Return the calibration margin together with assignment-weighted distance."""
    if q.shape[0] != distance.shape[0]:
        raise ValueError("Distance/Q node counts do not match.")
    _, _, _, margin = conformalized_node_confidence(q, eps=eps, seed=seed)
    weighted_distance = torch.sum(q.detach() * distance.detach(), dim=1)
    return margin, weighted_distance


def estimate_bmm_reliability(
    distance,
    q,
    max_iter=50,
    tol=1e-4,
    eps=1e-8,
):
    """Compatibility wrapper that reuses conformal reliability estimation.

    ``max_iter`` and ``tol`` are accepted for API compatibility and are ignored.
    """
    del max_iter, tol
    return estimate_conformal_reliability(distance, q, eps=eps, seed=0)

def legacy_top_h_mask(reliability, filtering_rate):
    """Return a binary mask that keeps the highest-scoring fraction of nodes."""
    n = reliability.numel()
    if n == 0:
        return torch.zeros_like(reliability)
    keep = int(n * (1.0 - float(filtering_rate)))
    keep = min(max(keep, 1), n)
    indices = torch.topk(reliability, k=keep, largest=True).indices
    mask = torch.zeros_like(reliability)
    mask[indices] = 1.0
    return mask


def node_reliability(reliability, q, continuous=True, filtering_rate=0.6, eps=1e-8):
    uncertainty = normalized_assignment_entropy(q, eps=eps)
    if continuous:
        confidence = reliability * (1.0 - uncertainty)
    else:
        confidence = legacy_top_h_mask(reliability, filtering_rate)
    return confidence.clamp(0.0, 1.0).detach(), uncertainty.detach()


def pair_reliability(
    z1_rows,
    z2_rows,
    z1_cols,
    z2_cols,
    c_rows,
    c_cols,
    enabled=True,
):
    """Parameter-free pair reliability for an arbitrary row/column block.

    Because all attribute embeddings are L2-normalized, each cosine similarity
    lies in [-1, 1], hence the cross-view disagreement lies in [0, 2].  We
    therefore map disagreement to an agreement score by the exact bounded
    normalization 1 - Delta/2 instead of introducing a temperature tau_u.
    """
    shape = (z1_rows.shape[0], z1_cols.shape[0])
    if not enabled:
        return torch.ones(shape, device=z1_rows.device, dtype=z1_rows.dtype)
    s1 = (z1_rows @ z1_cols.t()).clamp(-1.0, 1.0)
    s2 = (z2_rows @ z2_cols.t()).clamp(-1.0, 1.0)
    disagreement = torch.abs(s1 - s2).clamp(0.0, 2.0)
    agreement = (1.0 - 0.5 * disagreement).clamp(0.0, 1.0)
    node_pair = torch.sqrt(
        c_rows.clamp_min(0.0).unsqueeze(1)
        * c_cols.clamp_min(0.0).unsqueeze(0)
    )
    return (node_pair * agreement).clamp(0.0, 1.0).detach()


def pair_reliability_edges(z1, z2, confidence, row, col, enabled=True):
    """Parameter-free c_ij on a sparse candidate edge list."""
    if not enabled:
        return torch.ones(row.numel(), device=z1.device, dtype=z1.dtype)
    s1 = torch.sum(z1.index_select(0, row) * z1.index_select(0, col), dim=1).clamp(-1.0, 1.0)
    s2 = torch.sum(z2.index_select(0, row) * z2.index_select(0, col), dim=1).clamp(-1.0, 1.0)
    disagreement = torch.abs(s1 - s2).clamp(0.0, 2.0)
    agreement = (1.0 - 0.5 * disagreement).clamp(0.0, 1.0)
    node_pair = torch.sqrt(
        confidence.index_select(0, row).clamp_min(0.0)
        * confidence.index_select(0, col).clamp_min(0.0)
    )
    return (node_pair * agreement).clamp(0.0, 1.0).detach()

def cluster_relation(q_rows, q_cols, soft=True):
    if soft:
        return (q_rows @ q_cols.t()).clamp(0.0, 1.0).detach()
    labels_rows = torch.argmax(q_rows, dim=1)
    labels_cols = torch.argmax(q_cols, dim=1)
    return (labels_rows[:, None] == labels_cols[None, :]).to(q_rows.dtype).detach()


def cluster_relation_edges(q, row, col, soft=True):
    if soft:
        return torch.sum(
            q.index_select(0, row) * q.index_select(0, col), dim=1
        ).clamp(0.0, 1.0).detach()
    labels = torch.argmax(q, dim=1)
    return (labels.index_select(0, row) == labels.index_select(0, col)).to(q.dtype)


def similarity_to_unit_interval(similarity):
    """For normalized embeddings and a convex view mixture, S is in [-1,1]."""
    return ((similarity + 1.0) * 0.5).clamp(0.0, 1.0)
