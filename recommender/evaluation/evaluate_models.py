"""
Đánh giá các mô hình gợi ý trên tập val/test theo kiểu giữ seed - giấu đáp án.

Với mỗi playlist đánh giá và mỗi k: k bài đầu là seed, phần còn lại (trừ các bài
đã có trong seed) là đáp án. Playlist không còn bài nào để giấu thì bị bỏ qua với
k đó, nên k càng lớn thì càng ít playlist đủ điều kiện.

Độ đo: Recall@10, Recall@50, NDCG@10 (theo đề bài), thêm Precision@10, Hit Rate@10
và R-precision có tính điểm nghệ sĩ (độ đo chính của RecSys Challenge 2018).

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.evaluation.evaluate_models --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --out <results.csv>
"""
import argparse
import array
import csv
import math
import random
import time
from collections import namedtuple

import numpy as np
from scipy import sparse

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.evaluation.hit_rate_at_k import hit_rate_at_k
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import precision_at_k
from recommender.evaluation.r_precision import r_precision
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import read_assignment
from recommender.hybrid.hybrid_recommender import HybridRecommender, TitleRecommender
from recommender.hybrid.playlist_titles import read_titles
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

SEED_SIZES = (0, 1, 5, 25)
TOP_N = 50
METRICS = ("recall@10", "recall@50", "ndcg@10", "precision@10", "hit@10", "r_precision")

TrainData = namedtuple("TrainData", "X vocab row_pids item_artists artist_ids")


def load_train(flat_path, assignment):
    """Ma trận CSR nhị phân playlist x bài của tập train, từ điển uri -> id, pid của
    từng hàng, và id nghệ sĩ của từng bài (artist_ids: tên nghệ sĩ -> id)."""
    vocab = {}
    artist_ids = {}
    item_artists = array.array("i")
    pid_to_row = {}
    rows = array.array("i")
    cols = array.array("i")
    with open(flat_path, "rb") as f:
        for line in f:
            pid, uri, _, artist = line.rstrip(b"\r\n").split(b"\t", 3)
            pid = int(pid)
            if assignment[pid] == "train":
                rows.append(pid_to_row.setdefault(pid, len(pid_to_row)))
                item = vocab.get(uri)
                if item is None:
                    item = vocab[uri] = len(vocab)
                    item_artists.append(artist_ids.setdefault(artist, len(artist_ids)))
                cols.append(item)
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32),
         (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(len(pid_to_row), len(vocab)),
    )
    X.data[:] = 1  # playlist lặp lại một bài vẫn chỉ tính là 1
    return TrainData(X, vocab, list(pid_to_row), np.frombuffer(item_artists, dtype=np.int32), artist_ids)


def load_eval_playlists(flat_path, assignment, splits, vocab, artist_ids=None):
    """Trả về ({tập: {pid: các id bài theo đúng thứ tự trong playlist}}, {id âm: id nghệ sĩ}).

    Bài không có trong train nhận id âm: không gợi ý được nhưng vẫn nằm trong đáp
    án, đúng với tình huống thực tế. Nghệ sĩ chưa gặp trong train được thêm vào artist_ids.
    """
    unseen = {}
    unseen_artists = {}
    playlists = {s: {} for s in splits}
    with open(flat_path, "rb") as f:
        for line in f:
            pid, uri, _, artist = line.rstrip(b"\r\n").split(b"\t", 3)
            pid = int(pid)
            split = assignment[pid]
            if split in playlists:
                item = vocab.get(uri)
                if item is None:
                    item = unseen.get(uri)
                    if item is None:
                        item = unseen[uri] = -1 - len(unseen)
                        if artist_ids is not None:
                            unseen_artists[item] = artist_ids.setdefault(artist, len(artist_ids))
                playlists[split].setdefault(pid, array.array("i")).append(item)
    return playlists, unseen_artists


def mean_ci(values):
    """Trung bình và nửa khoảng tin cậy 95% (xấp xỉ chuẩn)."""
    values = np.asarray(values)
    return float(values.mean()), float(1.96 * values.std(ddof=1) / math.sqrt(len(values)))


def sample_cases(playlists, k, max_playlists, seed=42):
    """Tối đa max_playlists bộ (pid, seed, đáp án); cùng seed thì luôn ra cùng các playlist."""
    order = sorted(playlists)
    random.Random(seed).shuffle(order)
    cases = []
    for pid in order:
        tracks = playlists[pid]
        seed_items = list(tracks[:k])
        hidden = set(tracks[k:]) - set(seed_items)
        if hidden:
            cases.append((pid, seed_items, hidden))
            if len(cases) == max_playlists:
                break
    return cases


def evaluate(models, playlists, seed_sizes, max_playlists, seed=42, titles=None, artist_of=None):
    """Mọi mô hình chấm trên cùng các playlist để so sánh được theo từng k.

    titles: {pid: tên playlist} cho mô hình có cờ uses_title; artist_of(item) -> nghệ sĩ,
    dùng cho R-precision (None thì R-precision chỉ tính trúng bài).
    """
    titles = titles or {}
    results = []
    for k in seed_sizes:
        cases = sample_cases(playlists, k, max_playlists, seed)
        for name, model in models.items():
            start = time.time()
            values = {metric: [] for metric in METRICS}
            for pid, seed_items, hidden in cases:
                # R-precision chấm |hidden| gợi ý đầu nên cần danh sách đủ dài
                n = max(TOP_N, len(hidden))
                if getattr(model, "uses_title", False):
                    recommended = model.recommend(seed_items, n, title=titles.get(pid))
                else:
                    recommended = model.recommend(seed_items, n)
                values["recall@10"].append(recall_at_k(recommended, hidden, 10))
                values["recall@50"].append(recall_at_k(recommended, hidden, 50))
                values["ndcg@10"].append(ndcg_at_k(recommended, hidden, 10))
                values["precision@10"].append(precision_at_k(recommended, hidden, 10))
                values["hit@10"].append(hit_rate_at_k(recommended, hidden, 10))
                values["r_precision"].append(r_precision(recommended, hidden, artist_of))
            row = {"k": k, "model": name, "n_playlists": len(cases)}
            for metric in METRICS:
                row[metric], row[f"{metric}_ci95"] = mean_ci(values[metric])
            row["seconds"] = round(time.time() - start, 1)
            results.append(row)
            print(row, flush=True)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--popularity", required=True, help="đầu ra MapReduce đếm trên tập train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--splits", nargs="+", default=["test"], choices=["val", "test"])
    ap.add_argument("--max-playlists", type=int, default=10000)
    ap.add_argument("--tuned", nargs=3, type=float, metavar=("SEED_EXP", "ITEM_EXP", "MIN_COUNT"),
                    help="thêm mô hình cooccurrence_tuned với tham số chọn từ tune_cooccurrence (trên val)")
    ap.add_argument("--hybrid", nargs=2, type=float, metavar=("PLAYLIST_EXP", "TITLE_WEIGHT"),
                    help="thêm mô hình hybrid (cần --tuned và --titles) với tham số chọn từ tune_hybrid (trên val)")
    ap.add_argument("--titles", help="đầu ra recommender.hybrid.playlist_titles")
    args = ap.parse_args()
    if args.hybrid and not (args.tuned and args.titles):
        ap.error("--hybrid cần cả --tuned và --titles")

    start = time.time()
    assignment = read_assignment(args.split_file)
    train = load_train(args.flat, assignment)
    X, vocab = train.X, train.vocab
    playlists, unseen_artists = load_eval_playlists(args.flat, assignment, args.splits, vocab, train.artist_ids)
    print(f"Train: {X.shape[0]:,} playlist, {X.shape[1]:,} bài, {X.nnz:,} ô | {time.time() - start:.0f}s", flush=True)

    uri_counts = read_mapreduce_counts(args.popularity)
    n_mismatch = len(uri_counts.keys() ^ {uri.decode("utf-8") for uri in vocab})
    if n_mismatch:
        raise SystemExit(f"Đầu ra MapReduce lệch {n_mismatch} bài so với train: có thể đã đếm trên tập khác")
    counts = {vocab[uri.encode("utf-8")]: c for uri, c in uri_counts.items()}

    distinct = np.asarray(X.sum(axis=0)).ravel()
    n_diff = sum(1 for item, c in counts.items() if c != distinct[item])
    print(f"Bài có số đếm MapReduce (lượt) khác số playlist phân biệt: {n_diff:,}", flush=True)

    popularity = PopularityRecommender(counts)
    cooccurrence = CooccurrenceRecommender.from_matrix(X, popularity)
    models = {"popularity": popularity, "cooccurrence": cooccurrence}
    titles = None
    if args.tuned:
        seed_exp, item_exp, min_count = args.tuned
        # Dùng chung các mảng chỉ số với mô hình gốc, chỉ khác tham số
        arrays = (cooccurrence.row_indptr, cooccurrence.row_indices, cooccurrence.col_indptr, cooccurrence.col_indices)
        tuned_params = {"seed_exponent": seed_exp, "item_exponent": item_exp, "min_count": int(min_count)}
        models["cooccurrence_tuned"] = CooccurrenceRecommender(*arrays, popularity, **tuned_params)
        if args.hybrid:
            playlist_exp, title_weight = args.hybrid
            titles = read_titles(args.titles)
            title_ids = {}
            row_titles = np.array([title_ids.setdefault(titles[pid], len(title_ids)) if pid in titles else -1
                                   for pid in train.row_pids], dtype=np.int32)
            models["hybrid"] = HybridRecommender(
                CooccurrenceRecommender(*arrays, popularity, playlist_exponent=playlist_exp, **tuned_params),
                TitleRecommender(arrays[0], arrays[1], row_titles, title_ids, X.shape[1]),
                title_weight,
            )
    del X
    train = train._replace(X=None)  # giải phóng ma trận gốc, mô hình đã giữ bản CSR/CSC riêng

    item_artists = train.item_artists

    def artist_of(item):
        return item_artists[item] if item >= 0 else unseen_artists[item]

    fields = ["split", "k", "model", "n_playlists"]
    for metric in METRICS:
        fields += [metric, f"{metric}_ci95"]
    fields.append("seconds")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for split in args.splits:
            print(f"== {split}: {len(playlists[split]):,} playlist ==", flush=True)
            for row in evaluate(models, playlists[split], SEED_SIZES, args.max_playlists,
                                titles=titles, artist_of=artist_of):
                writer.writerow({"split": split, **row})
                f.flush()


if __name__ == "__main__":
    main()
