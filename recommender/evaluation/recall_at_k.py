"""Recall at k."""


def recall_at_k(recommended, hidden, k):
    """Tỉ lệ bài bị giấu xuất hiện trong Top-K gợi ý: |Top-K ∩ hidden| / |hidden|."""
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    return len(set(recommended[:k]) & set(hidden)) / len(hidden)
