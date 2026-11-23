"""
Hybrid recommender: Co-occurrence kết hợp gợi ý theo tên playlist.

Tên playlist cho biết khá nhiều về gu ("workout", "christmas", "country"...), và là
thông tin duy nhất khi k = 0. TitleRecommender chấm mỗi bài bằng số playlist train
cùng tên (đã chuẩn hóa) có chứa bài đó.

HybridRecommender cộng hai điểm sau khi chia mỗi loại cho điểm cao nhất của nó
(để cùng thang 0-1):
    score = cooccurrence / max(cooccurrence) + title_weight * title / max(title)
Không có seed thì chỉ còn điểm theo tên; không có cả tên khớp thì dùng bài phổ biến.
"""
import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import gather
from recommender.hybrid.playlist_titles import normalize_title


class TitleRecommender:
    """row_titles[r] là id tên của playlist train ở hàng r (-1: không có tên),
    title_ids: {tên đã chuẩn hóa: id}."""

    def __init__(self, row_indptr, row_indices, row_titles, title_ids, n_items):
        self.row_indptr, self.row_indices = row_indptr, row_indices
        self.title_ids = title_ids
        self.n_items = n_items
        row_titles = np.asarray(row_titles)
        # Xếp các hàng theo id tên để lấy mọi playlist cùng tên bằng một lát cắt
        self.rows_by_title = np.argsort(row_titles, kind="stable").astype(np.int64)
        self.title_starts = np.searchsorted(row_titles[self.rows_by_title], np.arange(len(title_ids) + 1))

    def n_playlists(self, title):
        t = self.title_ids.get(normalize_title(title or ""))
        return 0 if t is None else int(self.title_starts[t + 1] - self.title_starts[t])

    def scores(self, title):
        """Số playlist train cùng tên chứa từng bài (mảng toàn 0 nếu tên chưa gặp trong train)."""
        t = self.title_ids.get(normalize_title(title or ""))
        if t is None:
            return np.zeros(self.n_items)
        rows = self.rows_by_title[self.title_starts[t]: self.title_starts[t + 1]]
        items, _ = gather(self.row_indptr, self.row_indices, rows)
        return np.bincount(items, minlength=self.n_items).astype(np.float64)


def combine(cooccurrence_scores, title_scores, title_weight):
    """Cộng hai loại điểm sau khi đưa mỗi loại về thang 0-1."""
    scores = np.zeros_like(cooccurrence_scores)
    for part, weight in ((cooccurrence_scores, 1.0), (title_scores, title_weight)):
        top = part.max()
        if top > 0 and weight > 0:
            scores += weight * part / top
    return scores


class HybridRecommender:
    uses_title = True  # evaluate_models truyền thêm tên playlist cho mô hình có cờ này

    def __init__(self, cooccurrence, titles, title_weight):
        self.cooccurrence = cooccurrence
        self.titles = titles
        self.title_weight = title_weight
        self.fallback = cooccurrence.fallback

    def score(self, seed_items, title=None):
        return combine(self.cooccurrence.score(seed_items), self.titles.scores(title), self.title_weight)

    def recommend(self, seed_items, n, title=None, scores=None):
        """scores: kết quả score(seed_items, title) nếu đã tính sẵn, để khỏi tính lại."""
        seed = set(seed_items)
        scores = self.score(seed_items, title) if scores is None else scores.copy()
        known_seed = [i for i in seed if 0 <= i < len(scores)]
        scores[known_seed] = 0.0
        candidates = np.flatnonzero(scores)
        return self.cooccurrence.rank(candidates, scores[candidates], seed, n)
