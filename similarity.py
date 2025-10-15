import numpy as np
from scipy.ndimage import gaussian_filter1d


def cosine_distance(v1, v2):
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 1.0
    similarity = np.dot(v1, v2) / (norm1 * norm2)
    return 1.0 - similarity


def cross_bin_dissimilarity(v1, v2, sigma=1.0):
    v1_smooth = gaussian_filter1d(v1, sigma)
    v2_smooth = gaussian_filter1d(v2, sigma)
    return np.linalg.norm(v1_smooth - v2_smooth)


def jensen_shannon_divergence(p, q, eps=1e-10):
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    p = p / (np.sum(p) + eps)
    q = q / (np.sum(q) + eps)
    m = 0.5 * (p + q)
    def kl_divergence(a, b):
        return np.sum(a * np.log((a + eps) / (b + eps)))
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)


def jaccard_similarity(p, q, eps=1e-10):
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    return np.sum(np.minimum(p, q)) / (np.sum(np.maximum(p, q)) + eps)
