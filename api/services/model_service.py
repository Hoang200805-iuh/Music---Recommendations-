"""
Nạp mọi artifact mô hình một lần lúc API khởi động. Không dùng Spark.

    MUSIC_APP_INDEX  thư mục chỉ mục (recommender.app_index): bảng popularity (số đếm MapReduce),
                     ma trận train để tính Co-occurrence, tên playlist, URI và tên bài,
                     mô hình LightGBM (reranker_lgbm.txt, nếu có)
    MUSIC_ALS_DIR    thư mục ALS đã xuất (recommender.als.load_als), không bắt buộc

Các mảng lớn được đọc qua memory-map nên API chỉ chiếm vài trăm MB RAM.
"""
import os
import time
from pathlib import Path

from recommender import app_index
from recommender.als.load_als import load_als
from recommender.neighbors.user_knn_recommender import UserKNNRecommender
from recommender.rerank import build_features

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_DIR = ROOT / "hdfs_sim/music/analytics/app_index"
DEFAULT_ALS_DIR = ROOT / "hdfs_sim/music/analytics/als"

# Tên mô hình trong API -> mô tả ngắn (trả về ở /health)
DESCRIPTIONS = {
    "popularity": "Bài phổ biến nhất theo số đếm MapReduce",
    "cooccurrence": "Co-occurrence, điểm cosine",
    "cooccurrence_tuned": "Co-occurrence tinh chỉnh trên val",
    "hybrid": "Co-occurrence tinh chỉnh + tên playlist",
    "rerank": "Hybrid + User-User chọn 300 ứng viên, LightGBM xếp hạng lại",
    "als": "ALS (phân rã ma trận), gấp playlist mới vào bằng một bước ALS",
}
# Mô hình dùng khi request không chỉ định: tốt nhất trong các mô hình đã nạp
PREFERENCE = ("rerank", "hybrid", "cooccurrence_tuned", "cooccurrence", "als", "popularity")


class ModelService:
    def __init__(self, index_dir=None, als_dir=None):
        start = time.time()
        self.index_dir = Path(index_dir or os.environ.get("MUSIC_APP_INDEX", DEFAULT_INDEX_DIR))
        als_dir = Path(als_dir or os.environ.get("MUSIC_ALS_DIR", DEFAULT_ALS_DIR))
        if not (self.index_dir / "test_meta.json").exists():
            raise RuntimeError(f"Không thấy chỉ mục ở {self.index_dir}: dựng bằng python -m recommender.app_index")
        data = app_index.load(self.index_dir)
        if data["uris"] is None:
            raise RuntimeError(f"Chỉ mục {self.index_dir} chưa có uris.bin: chạy python -m recommender.app_index "
                               "--flat ... --split-file ... --out <chỉ mục> --uris-only")
        self.meta = data["meta"]
        self.counts = data["counts"]
        self.names = data["names"]
        self.uris = data["uris"]
        self.titles = data["hybrid"].titles
        # Dùng để giải thích gợi ý (api/services/explain_service.py), cùng cấu hình với mô hình rerank
        self.cooccurrence = data["hybrid"].cooccurrence
        self.neighbors = UserKNNRecommender(self.cooccurrence, **build_features.NEIGHBORS)
        models = {name: data[name] for name in ("popularity", "cooccurrence", "cooccurrence_tuned", "hybrid", "rerank")}
        models["als"], self.n_als_tracks = load_als(als_dir, self.uris, data["popularity"])
        self.models = {name: model for name, model in models.items() if model is not None}
        self.missing = sorted(set(models) - set(self.models))
        self.default_model = next(name for name in PREFERENCE if name in self.models)
        self.load_seconds = round(time.time() - start, 1)
        self.started_at = time.time()
