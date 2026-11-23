"""Precision at k."""


def precision_at_k(recommended, hidden, k):
    """Tỉ lệ gợi ý đúng trong Top-K: |Top-K ∩ hidden| / k."""
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    return len(set(recommended[:k]) & set(hidden)) / k


def max_recall_at_k(hidden, k):
    """Recall@k cao nhất có thể đạt: khi cả k gợi ý đều đúng, là min(k, |hidden|) / |hidden|."""
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    return min(k, len(hidden)) / len(hidden)
