"""
Chọn tham số chuẩn hóa theo độ dài playlist (RP3beta) cho Co-occurrence, trên tập validation.

Thử lưới seed_exponent x length_exponent x item_exponent, giữ min_count từ
tune_cooccurrence. Bộ (0.5, 0, 0.25) chính là cooccurrence_tuned; các bộ có
seed_exponent = length_exponent là RP3beta đúng nghĩa (alpha, beta = item_exponent).

Với mỗi playlist, các playlist train chứa bài seed và các bài trong đó chỉ cần
lấy một lần; mỗi bộ tham số chỉ tính lại phép cộng dồn cuối.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.evaluation.tune_rp3beta --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --out <kết quả.csv>
"""
import argparse
import csv
import time

import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender, gather
from recommender.evaluation.evaluate_models import TOP_N, load_eval_playlists, load_train, mean_ci, sample_cases
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import precision_at_k
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import read_assignment
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

SEED_EXPONENTS = (0.5, 0.75, 1.0)
LENGTH_EXPONENTS = (0.0, 0.25, 0.5, 0.75, 1.0)
ITEM_EXPONENTS = (0.25, 0.5)
SEED_SIZES = (1, 5, 25)
BASELINE = (0.5, 0.0, 0.25)  # cooccurrence_tuned


def tune(model, cases_by_k):
    """Trả về {(seed_exp, length_exp, item_exp, k): {chỉ số: danh sách giá trị theo playlist}}."""
    count = model.count
    dropped = count < model.min_count
    # Chia cho count^item_exp đúng như CooccurrenceRecommender.score để điểm (và thứ tự bài cùng điểm) khớp từng bit
    divisors = {item_exp: count ** item_exp for item_exp in ITEM_EXPONENTS}
    results = {}
    for k, cases in cases_by_k.items():
        start = time.time()
        for _, seed_items, hidden in cases:
            seed = set(seed_items)
            known_seed = [i for i in seed if 0 <= i < len(count)]
            playlists, _ = model.seed_playlists(seed_items)
            items, lengths = gather(model.row_indptr, model.row_indices, playlists)
            playlist_sizes = lengths.astype(np.float64)
            for seed_exp in SEED_EXPONENTS:
                # Tập playlist không đổi theo seed_exp, chỉ đổi trọng số
                _, weights = model.seed_playlists(seed_items, seed_exp)
                for length_exp in LENGTH_EXPONENTS:
                    raw = np.bincount(items, weights=np.repeat(weights / playlist_sizes ** length_exp, lengths),
                                      minlength=len(count))
                    raw[known_seed] = 0.0
                    raw[dropped] = 0.0
                    for item_exp, divisor in divisors.items():
                        with np.errstate(divide="ignore", invalid="ignore"):
                            scores = np.nan_to_num(raw / divisor, nan=0.0, posinf=0.0)
                        candidates = np.flatnonzero(scores)
                        recommended = model.rank(candidates, scores[candidates], seed, TOP_N)
                        metrics = results.setdefault((seed_exp, length_exp, item_exp, k), {
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
    ap.add_argument("--min-count", type=int, default=10, help="giữ như kết quả tune_cooccurrence")
    ap.add_argument("--max-playlists", type=int, default=1000)
    args = ap.parse_args()

    start = time.time()
    assignment = read_assignment(args.split_file)
    train = load_train(args.flat, assignment)
    vocab = train.vocab
    playlists = load_eval_playlists(args.flat, assignment, ["val"], vocab)[0]["val"]
    counts = {vocab[uri.encode("utf-8")]: c for uri, c in read_mapreduce_counts(args.popularity).items()}
    model = CooccurrenceRecommender.from_matrix(train.X, PopularityRecommender(counts), min_count=args.min_count)
    del train
    print(f"Đã nạp dữ liệu | {time.time() - start:.0f}s", flush=True)

    cases_by_k = {k: sample_cases(playlists, k, args.max_playlists) for k in SEED_SIZES}
    results = tune(model, cases_by_k)

    fields = ["seed_exponent", "length_exponent", "item_exponent", "k", "n_playlists",
              "recall@10", "recall@50", "ndcg@10", "precision@10"]
    rows = []
    for (seed_exp, length_exp, item_exp, k), metrics in sorted(results.items()):
        row = {"seed_exponent": seed_exp, "length_exponent": length_exp, "item_exponent": item_exp, "k": k,
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
        by_params.setdefault((row["seed_exponent"], row["length_exponent"], row["item_exponent"]), []).append(row)
    ranking = sorted(by_params.items(), key=lambda item: -np.mean([r["ndcg@10"] for r in item[1]]))
    print("\nTop 5 bộ tham số theo NDCG@10 trung bình (val):", flush=True)
    for params, group in ranking[:5] + [(BASELINE, by_params[BASELINE])]:
        label = "cooccurrence_tuned" if params == BASELINE else \
            f"seed_exp={params[0]} length_exp={params[1]} item_exp={params[2]}"
        print(f"  {label}: ndcg@10 {np.mean([r['ndcg@10'] for r in group]):.4f}, "
              f"recall@10 {np.mean([r['recall@10'] for r in group]):.4f}, "
              f"recall@50 {np.mean([r['recall@50'] for r in group]):.4f}", flush=True)


if __name__ == "__main__":
    main()
