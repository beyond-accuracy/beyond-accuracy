import numpy as np


def compute_cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Cosine similarity between two causal vectors."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a < 1e-8 or norm_b < 1e-8:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))


def compute_gt_alignment(causal_vector: np.ndarray, gt_binary: np.ndarray) -> float:
    """Cosine similarity between a model causal vector and a binary GT label."""
    return compute_cosine_similarity(causal_vector, gt_binary.astype(float))


def build_similarity_matrix(vectors: dict) -> tuple:
    """Build an N×N cosine similarity matrix from a model name → causal_vector dict."""
    model_names = list(vectors.keys())
    n = len(model_names)
    mat = np.zeros((n, n))
    for i, m_a in enumerate(model_names):
        for j, m_b in enumerate(model_names):
            mat[i, j] = compute_cosine_similarity(vectors[m_a], vectors[m_b])
    return mat, model_names
