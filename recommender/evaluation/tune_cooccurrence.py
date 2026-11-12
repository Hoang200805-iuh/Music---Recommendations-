"""
Chọn tham số cho Co-occurrence tổng quát trên tập validation.

Thử lưới seed_exponent x item_exponent x min_count. Bộ (0.5, 0.5, 1) chính là công
thức gốc của đề bài, nên bảng kết quả có sẵn dòng để so sánh. Chỉ chạy trên val:
tập test chỉ dùng một lần ở cuối (evaluate_models) để báo kết quả, tránh chọn tham
số theo đúng tập dùng để chấm.

Điểm thô (trước khi phạt bài phổ biến) chỉ phụ thuộc seed_exponent, nên mỗi playlist
chỉ tính điểm thô một lần cho mỗi seed_exponent rồi thử mọi item_exponent, min_count.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.evaluation.tune_cooccurrence --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --out <kết quả.csv>
"""
import argparse
import csv
import itertools
import time

import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.evaluation.evaluate_models import TOP_N, load_eval_playlists, load_train, mean_ci, sample_cases
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import precision_at_k
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import read_assignment
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

SEED_EXPONENTS = (0.0, 0.5, 1.0)
ITEM_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MIN_COUNTS = (1, 3, 10)
SEED_SIZES = (1, 5, 25)


def tune(model, cases_by_k):
    """Trả về {(seed_exp, item_exp, min_count, k): {chỉ số: danh sách giá trị theo playlist}}."""
    count = model.count
    results = {}
    for k, cases in cases_by_k.items():
        start = time.time()
        for _, seed_items, hidden in cases:
            seed = set(seed_items)
            for seed_exp in SEED_EXPONENTS:
                raw = model.raw_scores(seed_items, seed_exp)
                known_seed = [i for i in seed if 0 <= i < len(raw)]
                raw[known_seed] = 0.0
                candidates = np.flatnonzero(raw)
                candidate_raw, candidate_count = raw[candidates], count[candidates]
                for item_exp, min_count in itertools.product(ITEM_EXPONENTS, MIN_COUNTS):
                    keep = candidate_count >= min_count
                    scores = candidate_raw[keep] / candidate_count[keep] ** item_exp
                    recommended = model.rank(candidates[keep], scores, seed, TOP_N)
                    metrics = results.setdefault((seed_exp, item_exp, min_count, k), {
                        "recall@10": [], "recall@50": [], "ndcg@10": [], "precision@10": []})
                    metrics["recall@10"].append(recall_at_k(recommended, hidden, 10))
                    metrics["recall@50"].append(recall_at_k(recommended, hidden, 50))
                    metrics["ndcg@10"].append(ndcg_at_k(recommended, hidden, 10))
                    metrics["precision@10"].append(precision_at_k(recommended, hidden, 10))
        print(f"k={k}: {len(cases)} playlist | {time.time() - start:.0f}s", flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--popularity", required=True, help="đầu ra MapReduce đếm trên tập train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-playlists", type=int, default=2000)
    args = ap.parse_args()

    start = time.time()
    assignment = read_assignment(args.split_file)
    train = load_train(args.flat, assignment)
    vocab = train.vocab
    playlists = load_eval_playlists(args.flat, assignment, ["val"], vocab)[0]["val"]
    counts = {vocab[uri.encode("utf-8")]: c for uri, c in read_mapreduce_counts(args.popularity).items()}
    model = CooccurrenceRecommender.from_matrix(train.X, PopularityRecommender(counts))
    del train
    print(f"Đã nạp dữ liệu | {time.time() - start:.0f}s", flush=True)

    cases_by_k = {k: sample_cases(playlists, k, args.max_playlists) for k in SEED_SIZES}
    results = tune(model, cases_by_k)

    fields = ["seed_exponent", "item_exponent", "min_count", "k", "n_playlists",
              "recall@10", "recall@50", "ndcg@10", "precision@10"]
    rows = []
    for (seed_exp, item_exp, min_count, k), metrics in sorted(results.items()):
        row = {"seed_exponent": seed_exp, "item_exponent": item_exp, "min_count": min_count, "k": k,
               "n_playlists": len(metrics["ndcg@10"])}
        row.update({name: mean_ci(values)[0] for name, values in metrics.items()})
        rows.append(row)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Chọn bộ tham số có NDCG@10 trung bình cao nhất trên cả 3 giá trị k
    by_params = {}
    for row in rows:
        by_params.setdefault((row["seed_exponent"], row["item_exponent"], row["min_count"]), []).append(row["ndcg@10"])
    ranking = sorted(by_params.items(), key=lambda item: -np.mean(item[1]))
    print("\nTop 5 bộ tham số theo NDCG@10 trung bình (val):", flush=True)
    for params, values in ranking[:5]:
        print(f"  seed_exp={params[0]} item_exp={params[1]} min_count={params[2]}: {np.mean(values):.4f}")
    print(f"  Công thức gốc (0.5, 0.5, 1): {np.mean(by_params[(0.5, 0.5, 1)]):.4f}")


if __name__ == "__main__":
    main()
