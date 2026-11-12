"""
Xem thử 2 mô hình gợi ý trên một phần dữ liệu train, tốn ít RAM.

Chỉ nạp N playlist train đầu tiên trong file (mặc định 30.000) rồi dừng đọc, nên
số liệu ở đây chỉ để xem mô hình hoạt động thế nào. Kết quả đánh giá chính thức
lấy từ recommender.evaluation.evaluate_models.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.demo --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --query "One Dance" "Wonderwall" --examples 2 --k 5
"""
import argparse
import array
import random

import numpy as np
from scipy import sparse

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import read_assignment
from recommender.popularity.popularity_recommender import PopularityRecommender

TOP_N = 10
MAX_TEST_PLAYLISTS = 200  # chỉ cần vài playlist để xem, giữ nhiều hơn chỉ tốn RAM


def load_subset(flat_path, assignment, max_train_playlists):
    """Trả về ma trận train, tên bài theo id và các playlist test gặp trên đường đọc."""
    vocab = {}
    names = []
    pid_to_row = {}
    rows = array.array("i")
    cols = array.array("i")
    test_playlists = {}
    with open(flat_path, "rb") as f:
        for line in f:
            pid, uri, track_name, artist_name = line.rstrip(b"\r\n").split(b"\t")[:4]
            pid = int(pid)
            split = assignment[pid]
            if split == "train":
                if pid not in pid_to_row and len(pid_to_row) == max_train_playlists:
                    break
                rows.append(pid_to_row.setdefault(pid, len(pid_to_row)))
                item = vocab.get(uri)
                if item is None:
                    item = vocab[uri] = len(vocab)
                    names.append(f"{artist_name.decode('utf-8')} – {track_name.decode('utf-8')}")
                cols.append(item)
            elif split == "test" and (pid in test_playlists or len(test_playlists) < MAX_TEST_PLAYLISTS):
                label = f"{artist_name.decode('utf-8')} – {track_name.decode('utf-8')}"
                test_playlists.setdefault(pid, []).append((uri, label))

    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int32),
         (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(len(pid_to_row), len(vocab)),
    )
    X.data[:] = 1  # playlist lặp lại một bài vẫn chỉ tính là 1
    return X, vocab, names, test_playlists


def split_test_playlist(tracks, k, vocab):
    """Tách playlist test thành seed (k bài đầu) và đáp án, cùng dạng id với vocab."""
    seed_tracks, rest = tracks[:k], tracks[k:]
    seed = [vocab[uri] for uri, _ in seed_tracks if uri in vocab]
    seed_uris = {uri for uri, _ in seed_tracks}
    # Bài chưa có trong phần train đã nạp nhận id âm: vẫn tính vào đáp án
    hidden = {vocab.get(uri, -1 - i) for i, (uri, _) in enumerate(rest) if uri not in seed_uris}
    return seed, hidden


def find_track(query, names, counts):
    """Bài phổ biến nhất có tên khớp query, hoặc None nếu không có."""
    query = query.lower()
    matches = [i for i, name in enumerate(names) if query in name.lower()]
    return max(matches, key=lambda i: counts[i]) if matches else None


def show_side_by_side(models, seed, names, hidden=None):
    for model_name, model in models.items():
        recommended = model.recommend(seed, TOP_N)
        print(f"  {model_name}:")
        for rank, item in enumerate(recommended, 1):
            hit = "  <-- đúng" if hidden and item in hidden else ""
            print(f"    {rank:2}. {names[item]}{hit}")
        if hidden:
            print(f"    Recall@{TOP_N} = {recall_at_k(recommended, hidden, TOP_N):.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--max-train-playlists", type=int, default=30000)
    ap.add_argument("--query", nargs="*", default=[], help="tên bài dùng làm seed, mỗi tên chạy riêng một lần")
    ap.add_argument("--examples", type=int, default=2, help="số playlist test đem ra xem")
    ap.add_argument("--k", type=int, default=5, help="số bài đầu dùng làm seed cho playlist test")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    X, vocab, names, test_playlists = load_subset(args.flat, read_assignment(args.split_file), args.max_train_playlists)
    print(f"Đã nạp {X.shape[0]:,} playlist train, {X.shape[1]:,} bài, {X.nnz:,} ô")

    count = np.asarray(X.sum(axis=0)).ravel()
    popularity = PopularityRecommender({i: int(c) for i, c in enumerate(count)})
    models = {"Popularity": popularity, "Co-occurrence": CooccurrenceRecommender.from_matrix(X, popularity)}

    for query in args.query:
        item = find_track(query, names, count)
        print(f'\n=== Seed: "{query}" ===')
        if item is None:
            print("  Không tìm thấy bài này trong phần dữ liệu đã nạp")
            continue
        print(f"  Khớp với: {names[item]} ({count[item]:,} playlist)")
        show_side_by_side(models, [item], names)

    eligible = [pid for pid in sorted(test_playlists) if len(test_playlists[pid]) >= args.k + TOP_N]
    picked = random.Random(args.seed).sample(eligible, min(args.examples, len(eligible)))
    for pid in picked:
        tracks = test_playlists[pid]
        print(f"\n=== Playlist test pid={pid}: {len(tracks)} bài, giữ {args.k} bài đầu làm seed ===")
        for _, label in tracks[: args.k]:
            print(f"  seed: {label}")
        seed, hidden = split_test_playlist(tracks, args.k, vocab)
        show_side_by_side(models, seed, names, hidden)


if __name__ == "__main__":
    main()
