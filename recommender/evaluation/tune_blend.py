"""
Chọn tham số khi trộn thêm User-User kNN vào mô hình Hybrid, trên tập validation.

Giữ nguyên Co-occurrence tinh chỉnh và trọng số tên playlist (title_weight) đã chọn,
thử thêm:
    n_neighbors, similarity_exponent: cấu hình User-User kNN
    neighbor_weight: trọng số của điểm User-User (0 = Hybrid hiện tại;
                     "only" = bỏ Co-occurrence, chỉ dùng User-User + tên)
Chỉ chấm k = 1, 5, 25 vì User-User cần bài seed.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.evaluation.tune_blend --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --titles <playlist_titles.tsv> --out <kết quả.csv>
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
from recommender.hybrid.hybrid_recommender import TitleRecommender, combine
from recommender.hybrid.playlist_titles import read_titles
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

N_NEIGHBORS = (100, 500, 2000)
SIMILARITY_EXPONENTS = (1.0, 2.0)
NEIGHBOR_WEIGHTS = (0.25, 0.5, 1.0, 2.0, "only")
LENGTH_EXPONENT = 0.5  # cosine
SEED_SIZES = (1, 5, 25)
BASELINE = (0, 0, 0.0)  # Hybrid hiện tại, không có User-User


def tune(model, titles, title_weight, cases_by_k, playlist_titles):
    """Trả về {(n_neighbors, similarity_exp, neighbor_weight, k): {chỉ số: danh sách giá trị}}.

    Phép tính giống hệt UserKNNRecommender.score và HybridRecommender.score, chỉ dùng lại
    phần chung (điểm Co-occurrence, điểm theo tên, độ giống của các playlist) giữa các bộ tham số.
    """
    n_items = len(model.count)
    results = {}

    def record(params, k, scores, seed, known_seed, hidden):
        scores[known_seed] = 0.0
        candidates = np.flatnonzero(scores)
        recommended = model.rank(candidates, scores[candidates], seed, TOP_N)
        metrics = results.setdefault((*params, k), {"recall@10": [], "recall@50": [], "ndcg@10": [],
                                                    "precision@10": []})
        metrics["recall@10"].append(recall_at_k(recommended, hidden, 10))
        metrics["recall@50"].append(recall_at_k(recommended, hidden, 50))
        metrics["ndcg@10"].append(ndcg_at_k(recommended, hidden, 10))
        metrics["precision@10"].append(precision_at_k(recommended, hidden, 10))

    for k, cases in cases_by_k.items():
        start = time.time()
        for pid, seed_items, hidden in cases:
            seed = set(seed_items)
            known_seed = [i for i in seed if 0 <= i < n_items]
            cooccurrence_scores = model.score(seed_items)
            title_scores = titles.scores(playlist_titles.get(pid))
            record(BASELINE, k, combine(cooccurrence_scores, title_scores, title_weight), seed, known_seed, hidden)

            playlists, overlap = model.seed_playlists(seed_items, seed_exponent=0.0)
            sizes = (np.asarray(model.row_indptr[playlists + 1], dtype=np.float64)
                     - np.asarray(model.row_indptr[playlists], dtype=np.float64))
            similarity = overlap / sizes ** LENGTH_EXPONENT
            for n_neighbors in N_NEIGHBORS:
                top_playlists, top_similarity = playlists, similarity
                if len(playlists) > n_neighbors:
                    top = np.argpartition(-similarity, n_neighbors - 1)[:n_neighbors]
                    top_playlists, top_similarity = playlists[top], similarity[top]
                items, lengths = gather(model.row_indptr, model.row_indices, top_playlists)
                for similarity_exp in SIMILARITY_EXPONENTS:
                    neighbor_scores = np.bincount(items, weights=np.repeat(top_similarity ** similarity_exp, lengths),
                                                  minlength=n_items)
                    for weight in NEIGHBOR_WEIGHTS:
                        if weight == "only":
                            scores = combine(neighbor_scores, title_scores, title_weight)
                        else:
                            scores = combine(cooccurrence_scores, title_scores, title_weight, (neighbor_scores, weight))
                        record((n_neighbors, similarity_exp, weight), k, scores, seed, known_seed, hidden)
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
    ap.add_argument("--title-weight", type=float, default=0.1, help="trọng số tên đã chọn ở tune_hybrid")
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
    print(f"Đã nạp dữ liệu | {time.time() - start:.0f}s", flush=True)

    cases_by_k = {k: sample_cases(playlists, k, args.max_playlists) for k in SEED_SIZES}
    results = tune(model, titles, args.title_weight, cases_by_k, playlist_titles)

    fields = ["n_neighbors", "similarity_exponent", "neighbor_weight", "k", "n_playlists",
              "recall@10", "recall@50", "ndcg@10", "precision@10"]
    rows = []
    for (n_neighbors, similarity_exp, weight, k), metrics in results.items():
        row = {"n_neighbors": n_neighbors, "similarity_exponent": similarity_exp, "neighbor_weight": weight, "k": k,
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
        by_params.setdefault((row["n_neighbors"], row["similarity_exponent"], row["neighbor_weight"]), []).append(row)
    ranking = sorted(by_params.items(), key=lambda item: -np.mean([r["ndcg@10"] for r in item[1]]))
    print("\nTop 5 bộ tham số theo NDCG@10 trung bình (val):", flush=True)
    for params, group in ranking[:5] + [(BASELINE, by_params[BASELINE])]:
        label = "hybrid hiện tại" if params == BASELINE else \
            f"n_neighbors={params[0]} similarity_exp={params[1]} neighbor_weight={params[2]}"
        print(f"  {label}: ndcg@10 {np.mean([r['ndcg@10'] for r in group]):.4f}, "
              f"recall@10 {np.mean([r['recall@10'] for r in group]):.4f}, "
              f"recall@50 {np.mean([r['recall@50'] for r in group]):.4f}", flush=True)


if __name__ == "__main__":
    main()
