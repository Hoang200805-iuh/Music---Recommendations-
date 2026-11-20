"""
Đọc mô hình ALS đã xuất (không cần Spark) và gợi ý cho một playlist mới.

Định dạng xuất, trong một thư mục:
    item_factors.npy   float32, m x r: vector của m bài (itemFactors của ALSModel)
    track_uris.txt     m dòng, URI Spotify của từng hàng trong item_factors.npy
    als_meta.json      (không bắt buộc) {"reg_param": 0.1, "alpha": 1.0}: tham số lúc huấn luyện

Xuất từ PySpark (một lần, khi huấn luyện), với `model` là ALSModel và `track_index`
là DataFrame (track_id, track_uri) dùng khi đánh số bài cho ALS:
    rows = model.itemFactors.join(track_index, model.itemFactors.id == track_index.track_id) \\
                            .select("track_uri", "features").collect()
    np.save("item_factors.npy", np.array([r.features for r in rows], dtype=np.float32))
    Path("track_uris.txt").write_text("\\n".join(r.track_uri for r in rows) + "\\n")

Playlist mới chưa có vector nên phải "gấp vào" (fold-in): giữ nguyên vector các bài Y,
giải một bước ALS phản hồi ngầm (Hu, Koren, Volinsky 2008) cho vector u của playlist:
    u = (YᵀY + alpha · Y_Sᵀ Y_S + reg · I)⁻¹ · (1 + alpha) · Y_Sᵀ 1
với Y_S là vector các bài seed. Điểm của bài j là Y_j · u.
"""
import json
from pathlib import Path

import numpy as np


class ALSRecommender:
    """factors: vector của các bài đã xuất; rows[item] là hàng của bài item trong factors (-1: không có)."""

    def __init__(self, factors, rows, fallback, reg_param=0.1, alpha=1.0):
        self.factors = np.asarray(factors, dtype=np.float32)
        self.rows = np.asarray(rows, dtype=np.int64)
        self.fallback = fallback
        self.reg_param = reg_param
        self.alpha = alpha
        self.gram = self.factors.T.astype(np.float64) @ self.factors.astype(np.float64)  # YᵀY, r x r
        # Hàng -> id bài, để đổi điểm theo hàng thành điểm theo id
        self.items = np.full(len(self.factors), -1, dtype=np.int64)
        known = np.flatnonzero(self.rows >= 0)
        self.items[self.rows[known]] = known

    def user_vector(self, seed_items):
        """Vector của playlist từ các bài seed có trong mô hình (None nếu không bài nào có)."""
        seed_rows = [int(self.rows[i]) for i in set(seed_items) if 0 <= i < len(self.rows) and self.rows[i] >= 0]
        if not seed_rows:
            return None
        Y_s = self.factors[seed_rows].astype(np.float64)
        A = self.gram + self.alpha * (Y_s.T @ Y_s) + self.reg_param * np.eye(self.gram.shape[0])
        return np.linalg.solve(A, (1 + self.alpha) * Y_s.sum(axis=0))

    def score(self, seed_items):
        """Điểm của mọi bài theo id (0 cho bài không có trong mô hình hoặc khi không có seed nào trong mô hình)."""
        scores = np.zeros(len(self.rows))
        u = self.user_vector(seed_items)
        if u is not None:
            matched = self.items >= 0
            scores[self.items[matched]] = self.factors[matched] @ u
        return scores

    def recommend(self, seed_items, n, scores=None):
        """Top-n bài có điểm cao nhất, bỏ bài seed; không có seed nào trong mô hình thì dùng bài phổ biến."""
        seed = set(seed_items)
        if scores is None:
            scores = self.score(seed_items)
        candidates = self.items[self.items >= 0]
        if not scores[candidates].any():
            return self.fallback.recommend(seed, n)
        candidates = candidates[~np.isin(candidates, list(seed))]
        if len(candidates) > n:
            top = np.argpartition(-scores[candidates], n - 1)[:n]
            candidates = candidates[top]
        order = np.lexsort((candidates, -scores[candidates]))
        recommended = [int(i) for i in candidates[order]]
        if len(recommended) < n:
            recommended += self.fallback.recommend(seed | set(recommended), n - len(recommended))
        return recommended


def load_als(als_dir, uri_index, fallback):
    """Đọc thư mục ALS đã xuất; uri_index: app_index.UriIndex để ghép URI với id bài của chỉ mục.

    Trả về (ALSRecommender, số bài ghép được), hoặc (None, 0) nếu thư mục chưa có file.
    """
    als_dir = Path(als_dir)
    if not (als_dir / "item_factors.npy").exists():
        return None, 0
    factors = np.load(als_dir / "item_factors.npy")
    uris = (als_dir / "track_uris.txt").read_text(encoding="utf-8").split()
    if len(uris) != len(factors):
        raise ValueError(f"track_uris.txt có {len(uris)} dòng nhưng item_factors.npy có {len(factors)} hàng")
    params = {}
    if (als_dir / "als_meta.json").exists():
        params = json.loads((als_dir / "als_meta.json").read_text(encoding="utf-8"))
    rows = np.full(len(uri_index), -1, dtype=np.int64)
    keep = []
    for row, uri in enumerate(uris):
        item = uri_index.item(uri)
        if item is not None:
            rows[item] = len(keep)
            keep.append(row)
    model = ALSRecommender(factors[keep], rows, fallback, reg_param=params.get("reg_param", 0.1),
                           alpha=params.get("alpha", 1.0))
    return model, len(keep)
