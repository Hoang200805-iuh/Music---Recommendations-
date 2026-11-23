"""
Chọn tham số cho HybridRecommender trên tập validation.

Giữ nguyên tham số Co-occurrence đã chọn ở tune_cooccurrence, thử thêm:
    playlist_exponent: playlist chứa nhiều bài seed cùng lúc được tin hơn bao nhiêu
    title_weight: trọng số của điểm theo tên playlist (0 = không dùng tên)
Bộ (1, 0) chính là cooccurrence_tuned, nên bảng kết quả có sẵn dòng để so sánh.

Với mỗi playlist, các playlist train chứa bài seed chỉ cần tìm một lần; mỗi
playlist_exponent chỉ tính lại phép cộng dồn cuối.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.evaluation.tune_hybrid --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --titles <playlist_titles.tsv> --out <kết quả.csv>
"""
import argparse
import csv
import itertools
import time

import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender, gather
from recommender.evaluation.evaluate_models import TOP_N, load_eval_playlists, load_train, mean_ci, sample_cases
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import precision_at_k
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import read_assignment
from recommender.hybrid.hybrid_recommender import TitleRecommender, combine
from recommender.hybrid.playlist_titles import read_titles
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

PLAYLIST_EXPONENTS = (1.0, 1.5, 2.0, 3.0)
TITLE_WEIGHTS = (0.0, 0.1, 0.25, 0.5, 1.0)
SEED_SIZES = (0, 1, 5, 25)


def tune(model, titles, cases_by_k, playlist_titles):
    """Trả về {(playlist_exp, title_weight, k): {chỉ số: danh sách giá trị theo playlist}}."""
    count = model.count
    with np.errstate(divide="ignore"):
        penalty = np.where(count >= model.min_count, count ** -model.item_exponent, 0.0)
    results = {}
    for k, cases in cases_by_k.items():
        start = time.time()
        for pid, seed_items, hidden in cases:
            seed = set(seed_items)
            known_seed = [i for i in seed if 0 <= i < len(count)]
            title_scores = titles.scores(playlist_titles.get(pid))
            playlists, weights = model.seed_playlists(seed_items)
            items, lengths = gather(model.row_indptr, model.row_indices, playlists)
            for playlist_exp in PLAYLIST_EXPONENTS:
                raw = np.bincount(items, weights=np.repeat(weights ** playlist_exp, lengths), minlength=len(count))
                cooccurrence_scores = raw * penalty
                for title_weight in TITLE_WEIGHTS:
                    scores = combine(cooccurrence_scores, title_scores, title_weight)
                    scores[known_seed] = 0.0
                    candidates = np.flatnonzero(scores)
                    recommended = model.rank(candidates, scores[candidates], seed, TOP_N)
                    metrics = results.setdefault((playlist_exp, title_weight, k), {
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
    ap.add_argument("--titles", required=True, help="đầu ra recommender.hybrid.playlist_titles")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tuned", nargs=3, type=float, default=(0.5, 0.25, 10),
                    metavar=("SEED_EXP", "ITEM_EXP", "MIN_COUNT"), help="tham số Co-occurrence đã chọn")
    ap.add_argument("--max-playlists", type=int, default=1000)
    args = ap.parse_args()

    start = time.time()
    assignment = read_assignment(args.split_file)
    train = load_train(args.flat, assignment)
    vocab = train.vocab
    playlists = load_eval_playlists(args.flat, assignment, ["val"], vocab)[0]["val"]
    counts = {vocab[uri.encode("utf-8")]: c for uri, c in read_mapreduce_counts(args.popularity).items()}
    seed_exp, item_exp, min_count = args.tuned
    model = CooccurrenceRecommender.from_matrix(train.X, PopularityRecommender(counts), seed_exponent=seed_exp,
                                                item_exponent=item_exp, min_count=int(min_count))
    playlist_titles = read_titles(args.titles)
    title_ids = {}
    row_titles = np.array([title_ids.setdefault(playlist_titles[pid], len(title_ids)) if pid in playlist_titles
                           else -1 for pid in train.row_pids], dtype=np.int32)
    titles = TitleRecommender(model.row_indptr, model.row_indices, row_titles, title_ids, len(model.count))
    del train
    print(f"Đã nạp dữ liệu: {len(title_ids):,} tên playlist khác nhau trong train | {time.time() - start:.0f}s",
          flush=True)

    cases_by_k = {k: sample_cases(playlists, k, args.max_playlists) for k in SEED_SIZES}
    results = tune(model, titles, cases_by_k, playlist_titles)

    fields = ["playlist_exponent", "title_weight", "k", "n_playlists", "recall@10", "recall@50", "ndcg@10",
              "precision@10"]
    rows = []
    for (playlist_exp, title_weight, k), metrics in sorted(results.items()):
        row = {"playlist_exponent": playlist_exp, "title_weight": title_weight, "k": k,
               "n_playlists": len(metrics["ndcg@10"])}
        row.update({name: mean_ci(values)[0] for name, values in metrics.items()})
        rows.append(row)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Chọn bộ tham số có NDCG@10 trung bình cao nhất trên cả 4 giá trị k
    by_params = {}
    for row in rows:
        by_params.setdefault((row["playlist_exponent"], row["title_weight"]), []).append(row)
    ranking = sorted(by_params.items(), key=lambda item: -np.mean([r["ndcg@10"] for r in item[1]]))
    print("\nTop 5 bộ tham số theo NDCG@10 trung bình (val):", flush=True)
    for params, group in ranking[:5] + [((1.0, 0.0), by_params[(1.0, 0.0)])]:
        label = "cooccurrence_tuned" if params == (1.0, 0.0) else f"playlist_exp={params[0]} title_weight={params[1]}"
        print(f"  {label}: ndcg@10 {np.mean([r['ndcg@10'] for r in group]):.4f}, "
              f"recall@10 {np.mean([r['recall@10'] for r in group]):.4f}, "
              f"recall@50 {np.mean([r['recall@50'] for r in group]):.4f}", flush=True)


if __name__ == "__main__":
    main()
