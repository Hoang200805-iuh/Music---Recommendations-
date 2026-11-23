"""
Tên playlist từ file MPD gốc, đã chuẩn hóa, để gợi ý theo tên (k = 0: chỉ biết tên).

File làm phẳng không có tên playlist nên phải đọc lại các file mpd.slice.*.json.
Tên được chuẩn hóa để "Workout!!", "workout " và "WORKOUT 💪" coi là cùng một tên.

Đầu ra: mỗi dòng "pid <TAB> tên đã chuẩn hóa"; playlist có tên rỗng sau khi
chuẩn hóa (ví dụ chỉ có emoji) thì bỏ qua.

Cách dùng (chạy từ thư mục gốc repo):
    python -m recommender.hybrid.playlist_titles --in <thư mục mpd.slice.*.json> \
        --out hdfs_sim/music/processed/playlist_titles.tsv
"""
import argparse
import glob
import json
import os
import re
import sys
import unicodedata


def normalize_title(name):
    """Chữ thường, bỏ dấu câu/emoji, gộp khoảng trắng. Giữ chữ có dấu (tiếng Việt, ...)."""
    name = unicodedata.normalize("NFKC", name).lower()
    name = re.sub(r"[\W_]+", " ", name)
    return " ".join(name.split())


def read_titles(path):
    """{pid: tên đã chuẩn hóa}"""
    titles = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            pid, title = line.rstrip("\n").split("\t", 1)
            titles[int(pid)] = title
    return titles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "mpd.slice.*.json")))
    if not files:
        raise SystemExit(f"Không tìm thấy file mpd.slice.*.json trong {args.in_dir}")
    n_written = n_empty = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for i, fn in enumerate(files, 1):
            try:
                with open(fn, encoding="utf-8") as f:
                    playlists = json.load(f)["playlists"]
            except json.JSONDecodeError as e:
                # Giống flatten_for_mapreduce: file hỏng thì bỏ qua cả file
                print(f"LỖI JSON, bỏ qua {os.path.basename(fn)} ({e})", file=sys.stderr, flush=True)
                continue
            for pl in playlists:
                title = normalize_title(pl.get("name", ""))
                if title:
                    out.write(f"{pl['pid']}\t{title}\n")
                    n_written += 1
                else:
                    n_empty += 1
            if i % 100 == 0:
                print(f"[{i}/{len(files)}] {n_written:,} tên", flush=True)
    print(f"Đã ghi {n_written:,} tên, bỏ {n_empty:,} tên rỗng sau khi chuẩn hóa", flush=True)


if __name__ == "__main__":
    main()
