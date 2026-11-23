"""Gợi ý cho playlist mới từ item factors học bằng Spark implicit ALS.

Train/val/test chia theo playlist nên không thể dùng userFactors của train cho
playlist đánh giá. Fold-in chỉ dùng seed, giữ nguyên item factors Y:

    (Y.T Y + alpha * Ys.T Ys + regParam * |S| * I) x
        = (1 + alpha) * sum(Ys)

Ys gồm các bài seed phân biệt đã gặp trong train. Đây là hệ phương trình
implicit ALS với rating=1, confidence=1+alpha và regularization ALS-WR của Spark.
Không có seed đã biết thì dùng Popularity, giống Co-occurrence.
"""
import numpy as np


class ALSRecommender:
    def __init__(self, item_factors, fallback, reg_param=0.1, alpha=10.0):
        self.item_factors = np.asarray(item_factors, dtype=np.float32)
        if self.item_factors.ndim != 2 or not all(self.item_factors.shape):
            raise ValueError("item_factors phải là ma trận không rỗng")
        if not np.isfinite(reg_param) or reg_param <= 0 or not np.isfinite(alpha) or alpha < 0:
            raise ValueError("Cần reg_param > 0 và alpha >= 0")
        self.fallback = fallback
        self.reg_param = float(reg_param)
        self.alpha = float(alpha)
        self.rank = self.item_factors.shape[1]
        self.gram = np.zeros((self.rank, self.rank), dtype=np.float64)
        # Chuyển từng khối sang float64, tránh nhân đôi toàn bộ ma trận trong RAM.
        for start in range(0, len(self.item_factors), 65536):
            block = self.item_factors[start:start + 65536].astype(np.float64)
            if not np.isfinite(block).all():
                raise ValueError("item_factors chứa NaN/Infinity")
            self.gram += block.T @ block

    def fold_in(self, seed_items):
        """Vector playlist từ seed; bài chưa gặp và bài lặp không thêm confidence."""
        known = sorted({int(i) for i in seed_items if 0 <= i < len(self.item_factors)})
        if not known:
            return None
        factors = self.item_factors[known].astype(np.float64)
        lhs = self.gram + self.alpha * (factors.T @ factors)
        lhs.flat[::self.rank + 1] += self.reg_param * len(known)
        rhs = (1.0 + self.alpha) * factors.sum(axis=0)
        return np.linalg.solve(lhs, rhs)

    def recommend(self, seed_items, n):
        if n <= 0:
            return []
        seed = set(seed_items)
        user = self.fold_in(seed)
        if user is None:
            return self.fallback.recommend(seed, n)
        # float32 để phép nhân không tạo bản sao float64 của toàn bộ item factors.
        scores = self.item_factors @ user.astype(np.float32)
        known_seed = [i for i in seed if 0 <= i < len(scores)]
        scores[known_seed] = -np.inf
        candidates = np.flatnonzero(np.isfinite(scores))
        if len(candidates) > n:
            # Giữ tie ở ranh giới Top-N theo ID, không phụ thuộc argpartition.
            candidate_scores = scores[candidates]
            cutoff = np.partition(candidate_scores, len(candidates) - n)[len(candidates) - n]
            higher = candidates[candidate_scores > cutoff]
            tied = candidates[candidate_scores == cutoff][:n - len(higher)]
            candidates = np.concatenate((higher, tied))
        order = np.lexsort((candidates, -scores[candidates]))
        recommended = candidates[order].tolist()
        if len(recommended) < n:
            recommended += self.fallback.recommend(seed | set(recommended), n - len(recommended))
        return recommended
