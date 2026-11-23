"""
Thí nghiệm hiệu năng: thời gian xây các bảng mô hình từ file slice MPD gốc.

Các bảng được xây (giống artifact API/app dùng):
    popularity   số lượt xuất hiện của từng bài (job MapReduce đếm độ phổ biến)
    co-occurrence ma trận playlist x bài dạng CSR và CSC (Co-occurrence tính điểm từ đây)
    titles       tên playlist đã chuẩn hóa (cho mô hình Hybrid)

Cách làm theo kiểu MapReduce:
    map     (song song) mỗi slice: đọc JSON, đếm bài, lấy danh sách bài và tên của từng playlist
    reduce  (tuần tự)   gộp số đếm, đánh số bài, nối các cặp (playlist, bài)
    build   (tuần tự)   dựng ma trận CSR/CSC, xếp hạng độ phổ biến
Số core bị giới hạn bằng CPU affinity của tiến trình (tiến trình con thừa hưởng), nên
"1 core" thật sự chỉ chạy trên 1 core; các core được chọn là core vật lý khác nhau.

Mỗi lần chạy ghi thêm một dòng vào --out. Cách dùng (từ thư mục gốc repo):
    python src/benchmark_build.py --data <thư mục slice> --fraction 0.1 --cores 4 --out reports/benchmark_build_runs.csv
"""
import argparse
import array
import csv
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

# Không dùng phép tính ma trận: tắt các luồng BLAS, nếu không mỗi tiến trình map giữ sẵn bộ đệm cho 16 luồng
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from recommender.hybrid.playlist_titles import normalize_title  # noqa: E402


def pin_cores(n_cores):
    """Giới hạn tiến trình vào n_cores core vật lý (logical 0, 2, 4...: mỗi core vật lý có 2 luồng)."""
    cpus = [2 * i for i in range(n_cores)]
    if os.cpu_count() < 2 * n_cores:
        cpus = list(range(n_cores))
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, cpus)
    else:
        import ctypes

        mask = sum(1 << c for c in cpus)
        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        if not kernel32.SetProcessAffinityMask(ctypes.c_void_p(kernel32.GetCurrentProcess()), ctypes.c_size_t(mask)):
            raise OSError("Không đặt được CPU affinity")
    return cpus


def select_slices(data_dir, fraction):
    """Lấy đều fraction số slice (sắp theo tên), để 10% trải đều cả bộ dữ liệu."""
    files = sorted(Path(data_dir).glob("mpd.slice.*.json"))
    step = round(1 / fraction)
    return files[::step]


def map_slice(path):
    """Map: một slice -> (số đếm bài, danh sách URI của từng playlist, tên đã chuẩn hóa). Slice hỏng: None."""
    try:
        with open(path, "rb") as f:
            data = json.load(f)
    except (ValueError, UnicodeDecodeError):
        return None
    counts = Counter()
    playlists = []
    titles = []
    for playlist in data["playlists"]:
        uris = [track["track_uri"] for track in playlist["tracks"]]
        counts.update(uris)
        playlists.append(uris)
        titles.append(normalize_title(playlist.get("name", "")))
    return counts, playlists, titles


def build(files, workers):
    timings = {"map_reduce_s": 0.0, "reduce_s": 0.0, "build_s": 0.0}
    popularity = Counter()
    vocab = {}
    rows = array.array("i")
    cols = array.array("i")
    title_ids = {}
    row_titles = array.array("i")
    n_bad = 0

    start = time.perf_counter()
    pool = Pool(workers) if workers > 1 else None
    results = pool.imap(map_slice, files, chunksize=1) if pool else map(map_slice, files)
    for result in results:
        reduce_start = time.perf_counter()
        if result is None:
            n_bad += 1
            continue
        counts, playlists, titles = result
        popularity.update(counts)
        for uris, title in zip(playlists, titles):
            row = len(row_titles)
            row_titles.append(title_ids.setdefault(title, len(title_ids)) if title else -1)
            for uri in uris:
                item = vocab.get(uri)
                if item is None:
                    item = vocab[uri] = len(vocab)
                rows.append(row)
                cols.append(item)
        timings["reduce_s"] += time.perf_counter() - reduce_start
    if pool:
        pool.close()
        pool.join()
    timings["map_reduce_s"] = time.perf_counter() - start

    build_start = time.perf_counter()
    X = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.int8), (np.frombuffer(rows, dtype=np.int32), np.frombuffer(cols, dtype=np.int32))),
        shape=(len(row_titles), len(vocab)),
    )
    X.sum_duplicates()
    X_csc = X.tocsc()
    counts = np.zeros(len(vocab), dtype=np.int64)
    for uri, c in popularity.items():
        counts[vocab[uri]] = c
    ranked = np.lexsort((np.arange(len(counts)), -counts))
    timings["build_s"] = time.perf_counter() - build_start
    timings["total_s"] = time.perf_counter() - start
    stats = {"n_slices": len(files) - n_bad, "n_bad_slices": n_bad, "n_playlists": X.shape[0],
             "n_tracks": X.shape[1], "nnz": X_csc.nnz, "n_titles": len(title_ids),
             "top_track_count": int(counts[ranked[0]])}
    return timings, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="thư mục các file mpd.slice.*.json")
    ap.add_argument("--fraction", type=float, required=True, help="tỉ lệ số slice, ví dụ 0.1 hoặc 1")
    ap.add_argument("--cores", type=int, required=True)
    ap.add_argument("--out", required=True, help="CSV, mỗi lần chạy thêm một dòng")
    ap.add_argument("--run", type=int, default=1, help="số thứ tự lần chạy lặp lại (chỉ để ghi vào CSV)")
    args = ap.parse_args()

    cpus = pin_cores(args.cores)
    files = select_slices(args.data, args.fraction)
    print(f"{len(files)} slice, {args.cores} core (CPU {cpus})", flush=True)
    timings, stats = build(files, workers=args.cores)
    row = {"fraction": args.fraction, "cores": args.cores, "run": args.run, **stats,
           **{k: round(v, 1) for k, v in timings.items()}}
    print(row, flush=True)
    out = Path(args.out)
    new = not out.exists()
    with open(out, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
