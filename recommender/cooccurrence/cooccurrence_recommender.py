"""
Co-occurrence recommender (item-item).

Điểm của bài j với một seed S (mặc định, đúng công thức đề bài):
    score(j) = tổng theo i thuộc S của count(i, j) / sqrt(count(i) * count(j))
trong đó count(i, j) là số playlist train chứa cả i và j.

Bản tổng quát có 4 tham số, mặc định trùng công thức trên:
    score(j) = tổng theo playlist p chứa j của  (tổng theo i thuộc S ∩ p của count(i)^-seed_exponent)^playlist_exponent
               chia cho count(j)^item_exponent, và bỏ các bài j có count(j) < min_count.
Với playlist_exponent = 1, phần tử số đúng bằng tổng theo i thuộc S của count(i, j) / count(i)^seed_exponent.
seed_exponent quyết định bài seed hiếm hay phổ biến được tin hơn, item_exponent là
mức phạt bài phổ biến, min_count loại bài quá hiếm (dễ được điểm cao do trùng hợp),
playlist_exponent > 1 làm playlist chứa nhiều bài seed cùng lúc (giống playlist đang
gợi ý hơn) có tiếng nói lớn hơn hẳn playlist chỉ chứa một bài seed.

Ma trận đầy đủ X^T X có thể tới hàng trăm triệu ô khác 0, không vừa RAM máy cá
nhân, nên điểm được tính khi cần bằng hai phép nhân thưa, cho kết quả y hệt:
    score = (X^T (X w)^playlist_exponent) / count^item_exponent,  với w[i] = count(i)^-seed_exponent nếu i thuộc S

Mô hình chỉ cần 4 mảng chỉ số của X (dạng CSR theo playlist và CSC theo bài),
nên các mảng này có thể là np.memmap đọc thẳng từ đĩa thay vì nạp hết vào RAM.
"""
import numpy as np


def gather(indptr, indices, ids):
    """Nối các đoạn indices[indptr[i]:indptr[i+1]] của mọi i trong ids, trả kèm độ dài từng đoạn."""
    starts = np.asarray(indptr[ids], dtype=np.int64)
    lengths = np.asarray(indptr[ids + 1], dtype=np.int64) - starts
    offsets = np.repeat(starts - (np.cumsum(lengths) - lengths), lengths)
    return indices[offsets + np.arange(lengths.sum())], lengths


class CooccurrenceRecommender:
    """fallback dùng khi không đủ bài có điểm > 0."""

    def __init__(self, row_indptr, row_indices, col_indptr, col_indices, fallback,
                 seed_exponent=0.5, item_exponent=0.5, min_count=1, playlist_exponent=1.0):
        self.row_indptr, self.row_indices = row_indptr, row_indices
        self.col_indptr, self.col_indices = col_indptr, col_indices
        self.count = np.diff(np.asarray(col_indptr)).astype(np.float64)
        self.fallback = fallback
        self.seed_exponent = seed_exponent
        self.item_exponent = item_exponent
        self.min_count = min_count
        self.playlist_exponent = playlist_exponent

    @classmethod
    def from_matrix(cls, X, fallback, **params):
        """X: ma trận CSR nhị phân playlist x bài, mỗi ô chỉ xuất hiện một lần."""
        X = X.tocsr()
        X.sum_duplicates()
        X_csc = X.tocsc()
        return cls(X.indptr, X.indices, X_csc.indptr, X_csc.indices, fallback, **params)

    def seed_playlists(self, seed_items, seed_exponent=None):
        """Các playlist train chứa ít nhất một bài seed, kèm trọng số = tổng count(i)^-seed_exponent
        của các bài seed i nằm trong playlist đó."""
        if seed_exponent is None:
            seed_exponent = self.seed_exponent
        n_items = len(self.count)
        seed_items = np.array(
            [i for i in set(seed_items) if 0 <= i < n_items and self.count[i] > 0], dtype=np.int64
        )
        if len(seed_items) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)

        playlists, lengths = gather(self.col_indptr, self.col_indices, seed_items)
        weights = np.repeat(self.count[seed_items] ** -seed_exponent, lengths)
        unique_playlists, inverse = np.unique(playlists, return_inverse=True)
        return unique_playlists.astype(np.int64), np.bincount(inverse, weights=weights)

    def raw_scores(self, seed_items, seed_exponent=None, playlist_exponent=None):
        """Điểm trước khi phạt bài phổ biến (xem công thức ở đầu file)."""
        if playlist_exponent is None:
            playlist_exponent = self.playlist_exponent
        playlists, weights = self.seed_playlists(seed_items, seed_exponent)
        items, lengths = gather(self.row_indptr, self.row_indices, playlists)
        return np.bincount(items, weights=np.repeat(weights ** playlist_exponent, lengths), minlength=len(self.count))

    def score(self, seed_items):
        """Trả về mảng điểm cộng dồn của mọi bài (chưa loại seed)."""
        raw = self.raw_scores(seed_items)
        with np.errstate(divide="ignore", invalid="ignore"):
            scores = raw / self.count ** self.item_exponent
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0)
        scores[self.count < self.min_count] = 0.0
        return scores

    def rank(self, candidates, candidate_scores, seed, n):
        """Top-n trong candidates (đã loại seed, điểm > 0), bù bằng bài phổ biến nếu thiếu."""
        if len(candidates) > n:
            top = np.argpartition(-candidate_scores, n - 1)[:n]
            candidates, candidate_scores = candidates[top], candidate_scores[top]
        # Bài cùng điểm xếp theo id, để kết quả lặp lại được giữa các lần chạy
        order = np.lexsort((candidates, -candidate_scores))
        recommended = [int(i) for i in candidates[order]]

        # k = 0, hoặc seed có quá ít bài đồng xuất hiện: bù bằng bài phổ biến
        if len(recommended) < n:
            taken = set(seed) | set(recommended)
            recommended += self.fallback.recommend(taken, n - len(recommended))
        return recommended

    def recommend(self, seed_items, n, scores=None):
        """scores: kết quả score(seed_items) nếu đã tính sẵn, để khỏi tính lại."""
        seed = set(seed_items)
        scores = self.score(seed_items) if scores is None else scores.copy()
        known_seed = [i for i in seed if 0 <= i < len(scores)]
        scores[known_seed] = 0.0
        candidates = np.flatnonzero(scores)
        return self.rank(candidates, scores[candidates], seed, n)
