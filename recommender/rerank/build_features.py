"""
Bước 1 của xếp hạng lại: tạo đặc trưng cho các playlist val (để huấn luyện) và test (để đánh giá).

Mỗi (tập, k) được lưu thành một file features_<tập>_k<k>.npz trong --out-dir; file đã có
thì bỏ qua, nên chạy lại sau khi bị dừng sẽ làm tiếp phần còn thiếu.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.rerank.build_features --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --titles <playlist_titles.tsv> --out-dir <thư mục>
"""
import argparse
import time
from pathlib import Path

import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.evaluation.evaluate_models import load_eval_playlists, load_train, sample_cases
from recommender.evaluation.split_playlists import read_assignment
from recommender.hybrid.hybrid_recommender import TitleRecommender
from recommender.hybrid.playlist_titles import read_titles
from recommender.neighbors.user_knn_recommender import UserKNNRecommender
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts
from recommender.rerank.features import FeatureBuilder

SEED_SIZES = (0, 1, 5, 25)
# Tham số đã chọn trên val: tune_cooccurrence, tune_hybrid, tune_blend
TUNED = {"seed_exponent": 0.5, "item_exponent": 0.25, "min_count": 10}
TITLE_WEIGHT = 0.1
NEIGHBORS = {"n_neighbors": 100, "similarity_exponent": 1.0}
NEIGHBOR_WEIGHT = 0.5


def load_builder(flat, split_file, popularity_path, titles_path, splits):
    """Trả về (FeatureBuilder, {tập: {pid: bài}}, {pid: tên playlist})."""
    assignment = read_assignment(split_file)
    train = load_train(flat, assignment)
    vocab = train.vocab
    playlists, unseen_artists = load_eval_playlists(flat, assignment, splits, vocab, train.artist_ids)
    counts = {vocab[uri.encode("utf-8")]: c for uri, c in read_mapreduce_counts(popularity_path).items()}
    model = CooccurrenceRecommender.from_matrix(train.X, PopularityRecommender(counts), **TUNED)
    playlist_titles = read_titles(titles_path)
    title_ids = {}
    row_titles = np.array([title_ids.setdefault(playlist_titles[pid], len(title_ids)) if pid in playlist_titles
                           else -1 for pid in train.row_pids], dtype=np.int32)
    titles = TitleRecommender(model.row_indptr, model.row_indices, row_titles, title_ids, len(model.count))
    neighbors = UserKNNRecommender(model, **NEIGHBORS)
    builder = FeatureBuilder(model, titles, neighbors, train.item_artists, TITLE_WEIGHT, NEIGHBOR_WEIGHT,
                             unseen_artists=unseen_artists)
    return builder, playlists, playlist_titles


def build_cases(builder, cases, playlist_titles, label):
    features, labels, candidates, groups, hidden, hidden_lengths, pids = [], [], [], [], [], [], []
    start = time.time()
    for done, (pid, seed_items, hidden_items) in enumerate(cases, 1):
        cands, feats = builder.build(seed_items, playlist_titles.get(pid))
        hidden_array = np.fromiter(hidden_items, dtype=np.int64)
        features.append(feats)
        candidates.append(cands)
        labels.append(np.isin(cands, hidden_array).astype(np.int8))
        groups.append(len(cands))
        hidden.append(hidden_array)
        hidden_lengths.append(len(hidden_array))
        pids.append(pid)
        if done % 250 == 0:
            print(f"  {label}: {done}/{len(cases)} | {time.time() - start:.0f}s", flush=True)
    return {
        "features": np.concatenate(features), "labels": np.concatenate(labels),
        "candidates": np.concatenate(candidates), "groups": np.array(groups, dtype=np.int32),
        "hidden": np.concatenate(hidden), "hidden_lengths": np.array(hidden_lengths, dtype=np.int32),
        "pids": np.array(pids, dtype=np.int64),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--popularity", required=True, help="đầu ra MapReduce đếm trên tập train")
    ap.add_argument("--titles", required=True, help="đầu ra recommender.hybrid.playlist_titles")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--val-playlists", type=int, default=6000, help="số playlist val mỗi k (để huấn luyện)")
    ap.add_argument("--test-playlists", type=int, default=2000, help="số playlist test mỗi k (để đánh giá)")
    ap.add_argument("--seed-sizes", type=int, nargs="+", default=list(SEED_SIZES))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    builder, playlists, playlist_titles = load_builder(args.flat, args.split_file, args.popularity,
                                                       args.titles, ["val", "test"])
    # Nghệ sĩ của mọi bài, để tính R-precision khi đánh giá
    unseen = builder.unseen_artists
    np.savez(out_dir / "artists.npz", item_artists=builder.item_artists,
             unseen_ids=np.array(list(unseen), dtype=np.int64), unseen_artists=np.array(list(unseen.values())))
    print(f"Đã nạp dữ liệu | {time.time() - start:.0f}s", flush=True)

    for split, max_playlists in (("val", args.val_playlists), ("test", args.test_playlists)):
        for k in args.seed_sizes:
            path = out_dir / f"features_{split}_k{k}.npz"
            if path.exists():
                print(f"Bỏ qua {path.name} (đã có)", flush=True)
                continue
            cases = sample_cases(playlists[split], k, max_playlists)
            data = build_cases(builder, cases, playlist_titles, f"{split} k={k}")
            np.savez(path, **data)
            print(f"Đã lưu {path.name}: {len(cases)} playlist, {len(data['labels']):,} dòng | "
                  f"{time.time() - start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
