"""
Bước 2 của xếp hạng lại: huấn luyện LightGBM (LambdaRank) trên đặc trưng của tập val,
rồi so sánh với mô hình gốc (Hybrid + User-User) trên cùng các playlist test.

80% playlist val để huấn luyện, 20% còn lại để dừng sớm (early stopping); chia theo pid
nên một playlist không nằm ở cả hai phần. Tập test chỉ dùng để báo cáo kết quả.
Chỉ xếp lại playlist có bài seed (k >= 1): với k = 0 chỉ có tên playlist, xếp lại
không tốt hơn mô hình gốc (thử trên val), nên giữ nguyên thứ tự gốc.

Cách dùng (sau recommender.rerank.build_features):
    python -m recommender.rerank.train_reranker --features-dir <thư mục đặc trưng> \
        --out <kết quả.csv> --importance-out <độ quan trọng.csv> --model-out <reranker.txt>
"""
import argparse
import csv
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from recommender.evaluation.evaluate_models import TOP_N, mean_ci
from recommender.evaluation.hit_rate_at_k import hit_rate_at_k
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import precision_at_k
from recommender.evaluation.r_precision import r_precision
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.rerank.build_features import SEED_SIZES
from recommender.rerank.features import FEATURE_NAMES
from recommender.rerank.lgbm_reranker import MIN_SEEDS

METRICS = ["recall@10", "recall@50", "ndcg@10", "precision@10", "hit@10", "r_precision"]
PARAMS = {
    "objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [10, 50],
    # Cây nhỏ, học chậm, phạt L2: cấu hình tốt nhất trên phần val giữ lại
    "learning_rate": 0.02, "num_leaves": 15, "min_data_in_leaf": 1000, "lambda_l2": 10.0,
    "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "seed": 42, "verbose": -1,
}


def load_split(features_dir, split):
    """Gộp các file đặc trưng của một tập; thêm 'k' của từng playlist và vị trí bắt đầu của nó."""
    parts = [dict(np.load(Path(features_dir) / f"features_{split}_k{k}.npz")) | {"k": k} for k in SEED_SIZES]
    data = {name: np.concatenate([p[name] for p in parts])
            for name in ("features", "labels", "candidates", "groups", "hidden", "hidden_lengths", "pids")}
    data["k"] = np.concatenate([np.full(len(p["groups"]), p["k"]) for p in parts])
    data["starts"] = np.concatenate([[0], np.cumsum(data["groups"])])
    data["hidden_starts"] = np.concatenate([[0], np.cumsum(data["hidden_lengths"])])
    return data


def select(data, case_mask):
    """Chỉ giữ các playlist có case_mask = True."""
    rows = np.repeat(case_mask, data["groups"])
    hidden_rows = np.repeat(case_mask, data["hidden_lengths"])
    out = {name: data[name][rows] for name in ("features", "labels", "candidates")}
    out["hidden"] = data["hidden"][hidden_rows]
    for name in ("groups", "hidden_lengths", "pids", "k"):
        out[name] = data[name][case_mask]
    out["starts"] = np.concatenate([[0], np.cumsum(out["groups"])])
    out["hidden_starts"] = np.concatenate([[0], np.cumsum(out["hidden_lengths"])])
    return out


def dataset(data, reference=None):
    """Bỏ các playlist không có bài đáp án nào trong ứng viên: LambdaRank không học được gì từ chúng."""
    has_positive = np.add.reduceat(data["labels"], data["starts"][:-1]) > 0
    data = select(data, has_positive)
    return lgb.Dataset(data["features"], data["labels"], group=data["groups"], feature_name=FEATURE_NAMES,
                       reference=reference, free_raw_data=True)


def evaluate(data, scores, artist_of):
    """{(k, mô hình): {chỉ số: giá trị từng playlist}} cho mô hình gốc ("base") và đã xếp lại ("rerank")."""
    values = {}
    for c in range(len(data["groups"])):
        start, end = data["starts"][c], data["starts"][c + 1]
        hidden = set(data["hidden"][data["hidden_starts"][c]: data["hidden_starts"][c + 1]].tolist())
        candidates = data["candidates"][start:end]
        n = max(TOP_N, len(hidden))
        reranked = candidates
        if data["k"][c] >= MIN_SEEDS:
            reranked = candidates[np.argsort(-scores[start:end], kind="stable")]
        rankings = {"base": candidates, "rerank": reranked}
        for name, ranking in rankings.items():
            recommended = ranking[:n].tolist()
            metrics = values.setdefault((int(data["k"][c]), name), {m: [] for m in METRICS})
            metrics["recall@10"].append(recall_at_k(recommended, hidden, 10))
            metrics["recall@50"].append(recall_at_k(recommended, hidden, 50))
            metrics["ndcg@10"].append(ndcg_at_k(recommended, hidden, 10))
            metrics["precision@10"].append(precision_at_k(recommended, hidden, 10))
            metrics["hit@10"].append(hit_rate_at_k(recommended, hidden, 10))
            metrics["r_precision"].append(r_precision(recommended, hidden, artist_of))
    return values


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-dir", required=True)
    ap.add_argument("--out", required=True, help="CSV kết quả trên test")
    ap.add_argument("--importance-out", required=True, help="CSV độ quan trọng của đặc trưng")
    ap.add_argument("--model-out", required=True, help="file mô hình LightGBM")
    ap.add_argument("--holdout", type=float, default=0.2, help="tỉ lệ playlist val dùng để dừng sớm")
    args = ap.parse_args()

    start = time.time()
    artists = np.load(Path(args.features_dir) / "artists.npz")
    item_artists = artists["item_artists"]
    unseen_artists = dict(zip(artists["unseen_ids"].tolist(), artists["unseen_artists"].tolist()))

    def artist_of(item):
        return item_artists[item] if item >= 0 else unseen_artists[item]

    val = load_split(args.features_dir, "val")
    pids = np.unique(val["pids"])
    holdout_pids = np.random.default_rng(42).choice(pids, int(len(pids) * args.holdout), replace=False)
    in_holdout = np.isin(val["pids"], holdout_pids)
    with_seed = val["k"] >= MIN_SEEDS
    train_data = dataset(select(val, ~in_holdout & with_seed))
    valid_data = dataset(select(val, in_holdout & with_seed), reference=train_data)
    del val
    print(f"Đã nạp đặc trưng val | {time.time() - start:.0f}s", flush=True)

    booster = lgb.train(PARAMS, train_data, num_boost_round=3000, valid_sets=[valid_data], valid_names=["holdout"],
                        callbacks=[lgb.early_stopping(150), lgb.log_evaluation(50)])
    booster.save_model(args.model_out, num_iteration=booster.best_iteration)
    print(f"Huấn luyện xong: {booster.best_iteration} cây | {time.time() - start:.0f}s", flush=True)

    importance = booster.feature_importance("gain", iteration=booster.best_iteration)
    with open(args.importance_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["feature", "gain_share"])
        for i in np.argsort(-importance):
            writer.writerow([FEATURE_NAMES[i], round(float(importance[i] / importance.sum()), 4)])

    del train_data, valid_data
    test = load_split(args.features_dir, "test")
    scores = booster.predict(test["features"], num_iteration=booster.best_iteration)
    values = evaluate(test, scores, artist_of)

    fields = ["k", "model", "n_playlists"] + [f"{m}{s}" for m in METRICS for s in ("", "_ci95")]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for k in SEED_SIZES:
            base, rerank = values[(k, "base")], values[(k, "rerank")]
            # "rerank - base": chênh lệch trên từng playlist, khoảng tin cậy hẹp hơn so sánh hai trung bình
            diff = {m: np.subtract(rerank[m], base[m]) for m in METRICS}
            for name, metrics in (("hybrid_knn", base), ("rerank", rerank), ("rerank - base", diff)):
                row = {"k": k, "model": name, "n_playlists": len(metrics["ndcg@10"])}
                for m in METRICS:
                    row[m], row[f"{m}_ci95"] = (round(v, 4) for v in mean_ci(metrics[m]))
                writer.writerow(row)
                print(row, flush=True)
    print(f"Xong | {time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
