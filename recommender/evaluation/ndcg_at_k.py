"""NDCG at k."""
import math


def ndcg_at_k(recommended, hidden, k):
    """DCG của Top-K chia cho DCG lý tưởng (mọi bài bị giấu xếp lên đầu), độ liên quan nhị phân."""
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    hidden = set(hidden)
    dcg = sum(1.0 / math.log2(rank + 2) for rank, item in enumerate(recommended[:k]) if item in hidden)
    idcg = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(hidden), k)))
    return dcg / idcg
