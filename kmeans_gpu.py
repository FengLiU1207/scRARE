import torch


def initialize(x, num_clusters):
    if x.shape[0] < num_clusters:
        raise ValueError(
            f"KMeans requires N>=K, got N={x.shape[0]} and K={num_clusters}."
        )
    indices = torch.randperm(x.shape[0], device=x.device)[:num_clusters]
    return x.index_select(0, indices).clone()


def pairwise_distance(data1, data2, device=None):
    device = data1.device if device is None else torch.device(device)
    data1 = data1.to(device)
    data2 = data2.to(device)
    distance = (
        torch.sum(data1 * data1, dim=1, keepdim=True)
        + torch.sum(data2 * data2, dim=1).unsqueeze(0)
        - 2.0 * (data1 @ data2.t())
    )
    return distance.clamp_min(0.0)


def pairwise_cosine(data1, data2, device=None, eps=1e-8):
    device = data1.device if device is None else torch.device(device)
    data1 = data1.to(device)
    data2 = data2.to(device)
    data1 = data1 / data1.norm(dim=1, keepdim=True).clamp_min(eps)
    data2 = data2 / data2.norm(dim=1, keepdim=True).clamp_min(eps)
    return (1.0 - data1 @ data2.t()).clamp(0.0, 2.0)


def kmeans(
    X,
    num_clusters,
    distance="euclidean",
    tol=1e-4,
    device=torch.device("cpu"),
    max_iter=500,
    initialization_trials=20,
):
    """Device-safe KMeans with deterministic seeded empty-cluster recovery."""
    device = torch.device(device)
    x = X.detach().float().to(device)
    if x.ndim != 2:
        raise ValueError(f"KMeans expects a 2-D tensor, got shape {tuple(x.shape)}")
    if not torch.isfinite(x).all():
        raise FloatingPointError("KMeans input contains NaN or Inf.")
    if num_clusters < 1 or num_clusters > x.shape[0]:
        raise ValueError(f"Invalid number of clusters K={num_clusters} for N={x.shape[0]}.")

    if distance == "euclidean":
        distance_function = pairwise_distance
    elif distance == "cosine":
        distance_function = pairwise_cosine
    else:
        raise NotImplementedError(f"Unsupported KMeans distance: {distance}")

    best_score = None
    centers = None
    for _ in range(max(int(initialization_trials), 1)):
        candidate = initialize(x, num_clusters)
        score = distance_function(x, candidate, device=x.device).min(dim=1).values.sum()
        if best_score is None or score < best_score:
            best_score = score
            centers = candidate

    for _ in range(max(int(max_iter), 1)):
        distances = distance_function(x, centers, device=x.device)
        assignments = torch.argmin(distances, dim=1)
        previous = centers.clone()
        nearest_distance = distances.min(dim=1).values
        for cluster in range(num_clusters):
            members = torch.nonzero(assignments == cluster, as_tuple=False).reshape(-1)
            if members.numel() == 0:
                # Re-seed with the currently worst represented point. This avoids
                # an unseeded random branch and is stable under fixed seeds.
                replacement = torch.argmax(nearest_distance)
                centers[cluster] = x[replacement]
                nearest_distance[replacement] = -1.0
            else:
                centers[cluster] = x.index_select(0, members).mean(dim=0)
        shift = torch.sum(torch.norm(centers - previous, dim=1))
        if not torch.isfinite(shift):
            raise FloatingPointError("KMeans center update produced NaN or Inf.")
        if shift.square().item() < tol:
            break

    final_distance = distance_function(x, centers, device=x.device)
    assignments = torch.argmin(final_distance, dim=1)
    return assignments.cpu(), centers.detach()


def kmeans_predict(
    X,
    cluster_centers,
    distance="euclidean",
    device=torch.device("cpu"),
):
    device = torch.device(device)
    x = X.detach().float().to(device)
    centers = cluster_centers.detach().float().to(device)
    if distance == "euclidean":
        distances = pairwise_distance(x, centers, device=device)
    elif distance == "cosine":
        distances = pairwise_cosine(x, centers, device=device)
    else:
        raise NotImplementedError(f"Unsupported KMeans distance: {distance}")
    return torch.argmin(distances, dim=1).cpu()
