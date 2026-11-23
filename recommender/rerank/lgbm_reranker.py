"""
Mô hình hai bước: Hybrid chọn ứng viên, LightGBM (LambdaRank) xếp hạng lại.

Xem recommender/rerank/features.py cho các đặc trưng và
recommender/rerank/train_reranker.py cho cách huấn luyện (trên tập validation).
"""
import numpy as np

MIN_SEEDS = 1  # playlist có ít bài seed hơn thì giữ thứ tự của mô hình gốc (k = 0: xếp lại không tốt hơn)


class RerankRecommender:
    uses_title = True  # evaluate_models truyền thêm tên playlist cho mô hình có cờ này

    def __init__(self, builder, booster):
        """builder: FeatureBuilder; booster: lightgbm.Booster đã huấn luyện."""
        self.builder = builder
        self.booster = booster
        self.fallback = builder.cooccurrence.fallback

    @classmethod
    def from_file(cls, builder, model_path):
        import lightgbm  # chỉ cần khi dùng mô hình này

        return cls(builder, lightgbm.Booster(model_file=str(model_path)))

    def score(self, seed_items, title=None):
        """(ứng viên theo thứ tự của mô hình gốc, điểm LightGBM của chúng hoặc None nếu không xếp lại)."""
        candidates, features = self.builder.build(seed_items, title)
        if len(seed_items) < MIN_SEEDS:
            return candidates, None
        return candidates, self.booster.predict(features)

    def recommend(self, seed_items, n, title=None, scored=None):
        """scored: kết quả score(seed_items, title) nếu đã tính sẵn, để khỏi tính lại."""
        candidates, scores = self.score(seed_items, title) if scored is None else scored
        if scores is not None:
            # Hòa điểm thì giữ thứ tự của mô hình gốc
            candidates = candidates[np.argsort(-scores, kind="stable")]
        recommended = [int(i) for i in candidates[:n]]
        if len(recommended) < n:
            recommended += self.fallback.recommend(set(seed_items) | set(recommended), n - len(recommended))
        return recommended
