"""
Chỉ mục cho app: dựng một lần từ dữ liệu đầy đủ, app mở lại bằng memory-map.

App không nạp file TSV 3,7GB vào RAM nữa mà đọc các file nhị phân này qua
memory-map: hệ điều hành chỉ đọc phần đang dùng và tự giải phóng khi thiếu RAM,
nên dùng được 100% tập train mà RAM riêng của app vẫn nhỏ.

Các file trong thư mục chỉ mục:
    row_indptr/row_indices.npy  ma trận train dạng CSR (playlist -> bài)
    col_indptr/col_indices.npy  cùng ma trận dạng CSC (bài -> playlist)
    counts.npy                  số đếm MapReduce trên train của từng bài
    ranked.npy                  id bài xếp theo độ phổ biến (giống PopularityRecommender)
    names.bin + name_offsets    tên "nghệ sĩ – bài" theo id
    search.bin + search_offsets tên viết thường xếp theo độ phổ biến, để tìm kiếm
    artists.npy                 id nghệ sĩ của từng bài (để chấm R-precision có điểm nghệ sĩ)
    row_titles.npy              id tên (đã chuẩn hóa) của từng playlist train, -1 = không có tên
    test_offsets/test_items.npy mẫu playlist test (id âm = bài không có trong train)
    test_meta.json              pid, tên của mẫu test, danh sách tên playlist, tên và nghệ sĩ
                                của các bài không có trong train

Cách dựng (chạy từ thư mục gốc repo):
    python -m recommender.app_index --flat <flat.tsv> --split-file <playlist_split.tsv> \
        --popularity <part-00000 đếm trên train> --titles <playlist_titles.tsv> --out <thư mục chỉ mục>
"""
import argparse
import array
import json
import mmap
import random
import time
from pathlib import Path

import numpy as np
from scipy import sparse

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.evaluation.split_playlists import read_assignment
from recommender.hybrid.hybrid_recommender import HybridRecommender, TitleRecommender
from recommender.hybrid.playlist_titles import read_titles
from recommender.popularity.popularity_recommender import PopularityRecommender, read_mapreduce_counts

N_TEST_PLAYLISTS = 500
# Chọn bằng tune_cooccurrence trên tập val, đã kiểm lại trên test (reports/week2_results_with_tuned.csv)
TUNED_PARAMS = {"seed_exponent": 0.5, "item_exponent": 0.25, "min_count": 10}
# Chọn bằng tune_hybrid trên tập val (reports/week2_tuning_hybrid_val.csv)
HYBRID_PARAMS = {"playlist_exponent": 1.0, "title_weight": 0.1}


def _write_strings(path_prefix, strings):
    """Ghi các chuỗi nối liền vào một file .bin, kèm mảng vị trí bắt đầu (n + 1 phần tử)."""
    offsets = np.zeros(len(strings) + 1, dtype=np.int64)
    with open(f"{path_prefix}.bin", "wb") as f:
        for i, s in enumerate(strings):
            data = s.encode("utf-8")
            f.write(data)
            offsets[i + 1] = offsets[i] + len(data)
    return offsets


def build(flat_path, split_path, popularity_path, titles_path, out_dir, n_test=N_TEST_PLAYLISTS, seed=42):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    assignment = read_assignment(split_path)
    test_pids = set(random.Random(seed).sample(sorted(p for p, s in assignment.items() if s == "test"), n_test))

    vocab = {}
    names = []
    artist_ids = {}
    item_artists = array.array("i")
    pid_to_row = {}
    rows = array.array("i")
    cols = array.array("i")
    test_tracks = {}
    start = time.time()
    with open(flat_path, "rb") as f:
        for line in f:
            pid, uri, track_name, artist_name = line.rstrip(b"\r\n").split(b"\t")[:4]
            pid = int(pid)
            split = assignment[pid]
            if split == "train":
                rows.append(pid_to_row.setdefault(pid, len(pid_to_row)))
                item = vocab.get(uri)
                if item is None:
                    item = vocab[uri] = len(vocab)
                    names.append(f"{artist_name.decode('utf-8')} – {track_name.decode('utf-8')}")
                    item_artists.append(artist_ids.setdefault(artist_name, len(artist_ids)))
                cols.append(item)
            elif pid in test_pids:
                label = f"{artist_name.decode('utf-8')} – {track_name.decode('utf-8')}"
                test_tracks.setdefault(pid, []).append((uri, label, artist_name))
    print(f"Đã đọc dữ liệu: {len(pid_to_row):,} playlist train, {len(vocab):,} bài | {time.time() - start:.0f}s", flush=True)

    # int8 đủ vì chỉ cần biết ô có hay không; ma trận vẫn nhị phân sau khi gộp ô trùng
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8),
         (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(len(pid_to_row), len(vocab)),
    )
    del rows, cols
    X.sum_duplicates()
    np.save(out_dir / "row_indptr.npy", X.indptr)
    np.save(out_dir / "row_indices.npy", X.indices)
    X_csc = X.tocsc()
    del X
    np.save(out_dir / "col_indptr.npy", X_csc.indptr)
    np.save(out_dir / "col_indices.npy", X_csc.indices)
    nnz = X_csc.nnz
    del X_csc

    counts = np.zeros(len(vocab), dtype=np.int64)
    n_unknown = 0
    for uri, c in read_mapreduce_counts(popularity_path).items():
        item = vocab.get(uri.encode("utf-8"))
        if item is None:
            n_unknown += 1
        else:
            counts[item] = c
    if n_unknown or not counts.all():
        raise SystemExit(f"Đầu ra MapReduce lệch so với train ({n_unknown} bài thừa, "
                         f"{int((counts == 0).sum())} bài thiếu): có thể đã đếm trên tập khác")
    # Cùng thứ tự với PopularityRecommender: số đếm giảm dần, cùng số đếm thì id nhỏ trước
    ranked = np.lexsort((np.arange(len(counts)), -counts)).astype(np.int32)
    np.save(out_dir / "counts.npy", counts)
    np.save(out_dir / "ranked.npy", ranked)

    np.save(out_dir / "artists.npy", np.frombuffer(item_artists, dtype=np.int32))
    playlist_titles = read_titles(titles_path)
    title_ids = {}
    row_titles = [title_ids.setdefault(playlist_titles[pid], len(title_ids)) if pid in playlist_titles else -1
                  for pid in pid_to_row]
    np.save(out_dir / "row_titles.npy", np.array(row_titles, dtype=np.int32))

    np.save(out_dir / "name_offsets.npy", _write_strings(out_dir / "names", names))
    np.save(out_dir / "search_offsets.npy",
            _write_strings(out_dir / "search", [names[i].lower() for i in ranked]))

    unseen = {}
    test_offsets = [0]
    test_items = []
    pids = sorted(test_tracks)
    for pid in pids:
        for uri, label, artist_name in test_tracks[pid]:
            item = vocab.get(uri)
            if item is None:
                artist = artist_ids.setdefault(artist_name, len(artist_ids))
                item = unseen.setdefault(uri, (-1 - len(unseen), label, artist))[0]
            test_items.append(item)
        test_offsets.append(len(test_items))
    np.save(out_dir / "test_offsets.npy", np.array(test_offsets, dtype=np.int64))
    np.save(out_dir / "test_items.npy", np.array(test_items, dtype=np.int32))
    with open(out_dir / "test_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "n_train_playlists": len(pid_to_row),
            "n_items": len(vocab),
            "nnz": int(nnz),
            "test_pids": pids,
            "test_titles": {str(pid): playlist_titles[pid] for pid in pids if pid in playlist_titles},
            "titles": list(title_ids),
            "unseen_names": {str(item): label for item, label, _ in unseen.values()},
            "unseen_artists": {str(item): artist for item, _, artist in unseen.values()},
        }, f, ensure_ascii=False)
    print(f"Đã ghi chỉ mục vào {out_dir} | tổng {time.time() - start:.0f}s", flush=True)


class NameStore:
    """Tra tên bài theo id và tìm kiếm, đọc thẳng từ đĩa qua memory-map."""

    def __init__(self, index_dir, ranked):
        index_dir = Path(index_dir)
        self._files = [open(index_dir / "names.bin", "rb"), open(index_dir / "search.bin", "rb")]
        self._names = mmap.mmap(self._files[0].fileno(), 0, access=mmap.ACCESS_READ)
        self._search = mmap.mmap(self._files[1].fileno(), 0, access=mmap.ACCESS_READ)
        self._name_offsets = np.load(index_dir / "name_offsets.npy")
        self._search_offsets = np.load(index_dir / "search_offsets.npy")
        self._ranked = ranked

    def __len__(self):
        return len(self._name_offsets) - 1

    def __getitem__(self, item):
        return self._names[self._name_offsets[item]: self._name_offsets[item + 1]].decode("utf-8")

    def search(self, query, limit):
        """Các bài khớp query, phổ biến nhất trước.

        Tên trong search.bin đã xếp theo độ phổ biến, nên chỉ cần quét từ đầu file
        và dừng khi đủ limit kết quả.
        """
        needle = query.strip().lower().encode("utf-8")
        if not needle:
            return []
        found = []
        pos = self._search.find(needle)
        while pos != -1 and len(found) < limit:
            rank = int(np.searchsorted(self._search_offsets, pos, side="right")) - 1
            # Bỏ kết quả nằm vắt qua ranh giới hai tên liền nhau
            if pos + len(needle) <= self._search_offsets[rank + 1]:
                found.append(int(self._ranked[rank]))
            pos = self._search.find(needle, int(self._search_offsets[rank + 1]))
        return found


def load(index_dir):
    """Mở chỉ mục: các mảng lớn dùng memory-map, chỉ mảng nhỏ được nạp vào RAM."""
    index_dir = Path(index_dir)

    def mapped(name):
        return np.load(index_dir / f"{name}.npy", mmap_mode="r")

    with open(index_dir / "test_meta.json", encoding="utf-8") as f:
        meta = json.load(f)
    ranked = np.load(index_dir / "ranked.npy")
    popularity = PopularityRecommender.from_ranked(ranked)
    test_offsets = np.load(index_dir / "test_offsets.npy")
    test_items = np.load(index_dir / "test_items.npy")
    # Các bản Co-occurrence dùng chung các mảng memory-map, chỉ khác tham số
    arrays = [mapped(name) for name in ("row_indptr", "row_indices", "col_indptr", "col_indices")]
    titles = TitleRecommender(arrays[0], arrays[1], np.load(index_dir / "row_titles.npy"),
                              {title: i for i, title in enumerate(meta["titles"])}, meta["n_items"])
    hybrid_cooccurrence = CooccurrenceRecommender(
        *arrays, popularity, playlist_exponent=HYBRID_PARAMS["playlist_exponent"], **TUNED_PARAMS)
    return {
        "meta": meta,
        "counts": np.load(index_dir / "counts.npy"),
        "names": NameStore(index_dir, ranked),
        "popularity": popularity,
        "cooccurrence": CooccurrenceRecommender(*arrays, popularity),
        "cooccurrence_tuned": CooccurrenceRecommender(*arrays, popularity, **TUNED_PARAMS),
        "hybrid": HybridRecommender(hybrid_cooccurrence, titles, HYBRID_PARAMS["title_weight"]),
        "artists": np.load(index_dir / "artists.npy"),
        "test_titles": {int(pid): title for pid, title in meta["test_titles"].items()},
        "test_playlists": {
            pid: test_items[test_offsets[i]: test_offsets[i + 1]] for i, pid in enumerate(meta["test_pids"])
        },
        "unseen_names": {int(k): v for k, v in meta["unseen_names"].items()},
        "unseen_artists": {int(k): v for k, v in meta["unseen_artists"].items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--flat", required=True)
    ap.add_argument("--split-file", required=True)
    ap.add_argument("--popularity", required=True, help="đầu ra MapReduce đếm trên tập train")
    ap.add_argument("--titles", required=True, help="đầu ra recommender.hybrid.playlist_titles")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    build(args.flat, args.split_file, args.popularity, args.titles, args.out)


if __name__ == "__main__":
    main()
