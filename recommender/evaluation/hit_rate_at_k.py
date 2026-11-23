"""Hit rate at k."""


def hit_rate_at_k(recommended, hidden, k):
    """1 nếu Top-K có ít nhất một bài bị giấu, ngược lại 0. Trung bình trên nhiều playlist
    là tỉ lệ playlist mà người dùng thấy ít nhất một gợi ý trúng."""
    if not hidden:
        raise ValueError("hidden rỗng: playlist này không đánh giá được")
    hidden = set(hidden)
    return float(any(item in hidden for item in recommended[:k]))
