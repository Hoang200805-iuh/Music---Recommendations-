"""Nạp bundle ALS để chấm trên cùng vocabulary với các mô hình khác."""
import json
from pathlib import Path

import numpy as np

from recommender.als.recommend_als import ALSRecommender
from recommender.als.train_als import MAX_ITER, file_digest, vocab_digest


def validate_bundle(path, vocab=None, train_sha256=None):
    path = Path(path)
    with (path / "metadata.json").open(encoding="utf-8") as f:
        metadata = json.load(f)
    if (metadata.get("format_version") != 1 or metadata.get("implicitPrefs") is not True
            or metadata.get("rating") != 1 or metadata.get("maxIter") != MAX_ITER):
        raise ValueError("Bundle không phải implicit ALS rating=1, maxIter=10 được hỗ trợ")
    if vocab is not None and metadata["vocab_sha256"] != vocab_digest(vocab):
        raise ValueError("Vocabulary ALS không khớp train hiện tại; cần train lại trên đúng split/dữ liệu")
    if train_sha256 is not None and metadata.get("train_sha256") != train_sha256:
        raise ValueError("ALS đã học trên split/tương tác train khác; cần train lại trên đúng dữ liệu")
    factors = np.load(path / "item_factors.npy", mmap_mode="r", allow_pickle=False)
    if factors.shape != (metadata["n_items"], metadata["rank"]):
        raise ValueError("Kích thước item factors không khớp metadata")
    if metadata.get("item_factors_sha256") and file_digest(path / "item_factors.npy") != metadata["item_factors_sha256"]:
        raise ValueError("SHA-256 item factors không khớp; bundle có thể ghi dở hoặc hỏng")
    return metadata, factors


def load_als(path, fallback, vocab=None, train_sha256=None):
    metadata, factors = validate_bundle(path, vocab, train_sha256)
    return ALSRecommender(factors, fallback, metadata["regParam"], metadata["alpha"])
