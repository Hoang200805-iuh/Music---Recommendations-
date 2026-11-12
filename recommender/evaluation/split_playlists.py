"""
Chia playlist theo pid thành train/val/test (mặc định 70/15/15).

Chia theo pid (không theo dòng) để mọi bài của một playlist nằm trọn trong một
tập. Chỉ lưu file ánh xạ pid -> tập thay vì tách 3 file TSV, để không tốn thêm
vài GB ổ đĩa; các bước sau lọc trực tiếp từ file TSV đã làm phẳng.

Cách dùng (chạy từ thư mục gốc repo):
    # Tạo file ánh xạ
    python -m recommender.evaluation.split_playlists --in <playlist_tracks_flat.tsv> --out <playlist_split.tsv>
    # Xuất các dòng của một tập ra stdout (làm input cho job MapReduce)
    python -m recommender.evaluation.split_playlists --in <flat.tsv> --out <playlist_split.tsv> --emit train
"""
import argparse
import random
import sys

SPLITS = ("train", "val", "test")


def assign_splits(pids, ratios=(0.70, 0.15, 0.15), seed=42):
    """Trả về dict pid -> tên tập; cùng pids và seed thì luôn ra cùng kết quả."""
    pids = sorted(pids)
    random.Random(seed).shuffle(pids)
    n_train = int(len(pids) * ratios[0])
    n_val = int(len(pids) * ratios[1])
    assignment = {}
    for i, pid in enumerate(pids):
        if i < n_train:
            assignment[pid] = "train"
        elif i < n_train + n_val:
            assignment[pid] = "val"
        else:
            assignment[pid] = "test"
    return assignment


def read_pids(flat_path):
    pids = set()
    with open(flat_path, "rb") as f:
        for line in f:
            pids.add(int(line[: line.index(b"\t")]))
    return pids


def write_assignment(assignment, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for pid in sorted(assignment):
            f.write(f"{pid}\t{assignment[pid]}\n")


def read_assignment(path):
    assignment = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            pid, split = line.rstrip("\n").split("\t")
            assignment[int(pid)] = split
    return assignment


def emit_split(flat_path, assignment, split, out):
    """Ghi nguyên các dòng thuộc một tập ra out (đọc/ghi dạng byte để giữ nguyên nội dung)."""
    with open(flat_path, "rb") as f:
        for line in f:
            if assignment[int(line[: line.index(b"\t")])] == split:
                out.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True, help="file ánh xạ pid <TAB> tập")
    ap.add_argument("--emit", choices=SPLITS, help="xuất các dòng của tập này ra stdout")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.emit:
        emit_split(args.in_path, read_assignment(args.out), args.emit, sys.stdout.buffer)
        return

    assignment = assign_splits(read_pids(args.in_path), seed=args.seed)
    write_assignment(assignment, args.out)
    counts = {s: sum(1 for v in assignment.values() if v == s) for s in SPLITS}
    print(f"Đã chia {len(assignment):,} playlist: {counts} -> {args.out}")


if __name__ == "__main__":
    main()
