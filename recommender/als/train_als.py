"""Huấn luyện Spark MLlib implicit ALS chỉ trên train, rating nhị phân = 1.

Chạy: python -m recommender.als.train_als --help
PySpark chỉ được import khi train; đánh giá bundle đã lưu không cần JVM.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile
import time

import numpy as np

from recommender.als.recommend_als import ALSRecommender

MAX_ITER = 10
DEFAULT_PARAMS = {"rank": 50, "regParam": 0.1, "alpha": 10.0}


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def vocab_digest(vocab):
    """Fingerprint bao gồm cả URI và thứ tự ID để phát hiện ánh xạ không khớp."""
    digest = hashlib.sha256()
    for uri, item in sorted(vocab.items(), key=lambda pair: pair[1]):
        encoded = uri if isinstance(uri, bytes) else uri.encode("utf-8")
        digest.update(int(item).to_bytes(8, "little"))
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def training_digest(train):
    """Phát hiện split/tương tác train khác dù tập URI và ID tình cờ vẫn giống."""
    digest = hashlib.sha256(vocab_digest(train.vocab).encode("ascii"))
    arrays = (np.asarray(train.X.shape), np.asarray(train.row_pids), train.X.indptr, train.X.indices)
    for values in arrays:
        digest.update(len(values).to_bytes(8, "little"))
        for start in range(0, len(values), 1_000_000):
            digest.update(np.asarray(values[start:start + 1_000_000], dtype="<i8").tobytes())
    return digest.hexdigest()


def add_spark_arguments(parser):
    parser.add_argument("--master", default="local[*]", help="Spark master; dữ liệu tạm cần thấy từ mọi executor")
    parser.add_argument("--driver-memory", default="4g")
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument("--temp-dir", help="thư mục chứa TSV tạm playlist/item/rating")


def create_spark(master="local[*]", driver_memory="4g", app_name="PlaylistImplicitALS"):
    try:
        from pyspark.sql import SparkSession
    except ImportError as exc:
        raise RuntimeError("Cần cài requirements-als.txt và Java tương thích để train ALS") from exc
    spark = (SparkSession.builder.master(master).appName(app_name)
             .config("spark.driver.memory", driver_memory).getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


@contextmanager
def training_ratings(spark, train, temp_dir=None):
    """Nạp CSR train thành Spark DataFrame (user=int, item=int, rating=float).

    Mỗi hàng CSR là một playlist (user được mã hóa liên tiếp). Ghi TSV theo
    luồng để tránh tạo hàng chục triệu Python tuples trên driver. File tạm được
    giữ đến khi tất cả các lần fit xong, rồi dọn cùng Spark cache.
    """
    from pyspark import StorageLevel
    X = train.X
    if not X.shape[0] or not X.shape[1] or not X.nnz:
        raise ValueError("Tập train không có tương tác")
    if max(X.shape) > np.iinfo(np.int32).max:
        raise ValueError("Spark ALS cần user/item trong miền int32")
    with tempfile.TemporaryDirectory(prefix="playlist-als-", dir=temp_dir) as directory:
        path = Path(directory) / "ratings.tsv"
        with path.open("w", encoding="ascii", newline="\n") as out:
            last_log = time.monotonic()
            print(f"[ratings] Ghi TSV tạm: {path}", flush=True)
            for user in range(X.shape[0]):
                items = X.indices[X.indptr[user]:X.indptr[user + 1]]
                out.writelines(f"{user}\t{int(item)}\t1\n" for item in items)
                if user % 10000 == 0 and time.monotonic() - last_log >= 10:
                    print(f"[ratings] {user:,}/{X.shape[0]:,} playlist", flush=True)
                    last_log = time.monotonic()
        print("[ratings] Đọc/cache TSV trong Spark...", flush=True)
        ratings = (spark.read.option("sep", "\t").option("mode", "FAILFAST")
                   .schema("user INT, item INT, rating FLOAT")
                   .csv(path.resolve().as_uri()).persist(StorageLevel.MEMORY_AND_DISK))
        try:
            count = ratings.count()
            print(f"ALS train: {X.shape[0]:,} playlist, {X.shape[1]:,} bài, {count:,} rating=1", flush=True)
            yield ratings
        finally:
            ratings.unpersist(blocking=True)


def fit_als(ratings, rank=50, regParam=0.1, alpha=10.0, seed=42, num_blocks=10):
    """maxIter luôn bằng 10; không train bằng các seed/hidden của val/test."""
    from pyspark.ml.recommendation import ALS
    before = set(_persistent_rdds(ratings.sparkSession))
    try:
        model = ALS(userCol="user", itemCol="item", ratingCol="rating", implicitPrefs=True,
                    rank=int(rank), regParam=float(regParam), alpha=float(alpha), maxIter=MAX_ITER,
                    seed=seed, numUserBlocks=num_blocks, numItemBlocks=num_blocks,
                    nonnegative=False, coldStartStrategy="nan").fit(ratings)
    except Exception:
        for rdd_id, rdd in _persistent_rdds(ratings.sparkSession).items():
            if rdd_id not in before:
                rdd.unpersist(False)
        raise
    model._als_cached_rdds = [rdd for rdd_id, rdd in _persistent_rdds(ratings.sparkSession).items()
                              if rdd_id not in before]
    return model


def _persistent_rdds(spark):
    # ALS persist Scala RDDs bên dưới Dataset. Dataset.unpersist() chỉ dọn SQL
    # cache, nên cần giữ các RDD mới của mỗi lần fit để không tích lũy 7 models.
    iterator = spark.sparkContext._jsc.sc().getPersistentRDDs().iterator()
    result = {}
    while iterator.hasNext():
        entry = iterator.next()
        result[int(entry._1())] = entry._2()
    return result


def extract_item_factors(model, n_items):
    """Thu item factors theo ID; không collect user factors hay ma trận dự đoán."""
    factors = np.zeros((n_items, model.rank), dtype=np.float32)
    seen = np.zeros(n_items, dtype=bool)
    for row in model.itemFactors.toLocalIterator():
        item = row["id"]
        if not 0 <= item < n_items or seen[item]:
            raise ValueError("ID item factors không khớp train")
        factors[item] = row["features"]
        seen[item] = True
    if not seen.all():
        raise ValueError("Spark không trả đủ item factors của train")
    return factors


def release_factors(model):
    model.userFactors.unpersist()
    model.itemFactors.unpersist()
    for rdd in getattr(model, "_als_cached_rdds", ()):
        rdd.unpersist(False)


def save_bundle(out_dir, recommender, vocab, params, selection=None, train_sha256=None):
    """Lưu dữ liệu inference và metadata; không ghi đè Spark native model."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    factor_tmp = out / "item_factors.npy.tmp"
    with factor_tmp.open("wb") as f:
        np.save(f, recommender.item_factors, allow_pickle=False)
    factor_sha256 = file_digest(factor_tmp)
    factor_tmp.replace(out / "item_factors.npy")
    items_tmp = out / "items.tsv.tmp"
    with items_tmp.open("w", encoding="utf-8", newline="\n") as f:
        for uri, item in sorted(vocab.items(), key=lambda pair: pair[1]):
            text = uri.decode("utf-8") if isinstance(uri, bytes) else uri
            f.write(f"{item}\t{text}\n")
    items_tmp.replace(out / "items.tsv")
    metadata = {"format_version": 1, "implicitPrefs": True, "rating": 1, "maxIter": MAX_ITER,
                "rank": int(params["rank"]), "regParam": float(params["regParam"]),
                "alpha": float(params["alpha"]), "n_items": len(vocab), "vocab_sha256": vocab_digest(vocab),
                "train_sha256": train_sha256, "item_factors_sha256": factor_sha256,
                "inference": "seed_fold_in_als_wr", "selection": selection}
    from recommender.evaluation.data_cache import write_json
    write_json(out / "metadata.json", metadata)


def main():
    from recommender.evaluation.evaluate_models import load_train
    from recommender.evaluation.split_playlists import read_assignment
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--out", required=True, help="thư mục bundle ALS")
    ap.add_argument("--rank", type=int, default=DEFAULT_PARAMS["rank"])
    ap.add_argument("--reg-param", type=float, default=DEFAULT_PARAMS["regParam"])
    ap.add_argument("--alpha", type=float, default=DEFAULT_PARAMS["alpha"])
    ap.add_argument("--save-spark-model", action="store_true", help="lưu thêm ALSModel gốc trong out/spark_model")
    ap.add_argument("--cache-dir", help="cache dữ liệu train đã nạp")
    add_spark_arguments(ap)
    args = ap.parse_args()
    from recommender.evaluation.data_cache import load_train_cached
    train = load_train_cached(args.flat, read_assignment(args.split_file), args.cache_dir)
    params = {"rank": args.rank, "regParam": args.reg_param, "alpha": args.alpha}
    spark = create_spark(args.master, args.driver_memory)
    try:
        with training_ratings(spark, train, args.temp_dir) as ratings:
            model = fit_als(ratings, **params, num_blocks=args.num_blocks)
            try:
                recommender = ALSRecommender(extract_item_factors(model, len(train.vocab)), None,
                                             args.reg_param, args.alpha)
                save_bundle(args.out, recommender, train.vocab, params, train_sha256=training_digest(train))
                if args.save_spark_model:
                    model.write().save((Path(args.out) / "spark_model").resolve().as_uri())
            finally:
                release_factors(model)
    finally:
        spark.stop()
    print(f"Đã lưu ALS: {args.out}", flush=True)


if __name__ == "__main__":
    main()
