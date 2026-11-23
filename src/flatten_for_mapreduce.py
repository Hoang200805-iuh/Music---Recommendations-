"""
BƯỚC LÀM PHẲNG TẠM THỜI — chỉ để Người A có input đúng định dạng mà test job MapReduce,
KHÔNG thay thế cho bước ETL đầy đủ bằng PySpark của Người B (vốn còn phải xử lý
chuẩn hoá tiêu đề, khử playlist trùng, lọc playlist/bài hiếm...).

Định dạng đầu ra (hợp đồng giữa 2 người, để Người B ETL ra đúng format này cho tuần 2):
    Mỗi dòng = một lượt "bài xuất hiện trong playlist", các trường cách nhau bằng TAB:
        pid <TAB> track_uri <TAB> track_name <TAB> artist_name

Cách dùng:
    python3 flatten_for_mapreduce.py --in ../data/sample --out ../hdfs_sim/music/processed/playlist_tracks_flat.tsv
"""
import argparse
import glob
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.in_dir, "mpd.slice.*.json")))
    if not files:
        raise SystemExit(f"Không tìm thấy file mpd.slice.*.json trong {args.in_dir}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    n_lines = 0
    bad_files = []  # file JSON hỏng (ví dụ tải bị ngắt giữa chừng) -> bỏ qua, không dừng cả job

    with open(args.out, "w", encoding="utf-8") as out_f:
        for i, fn in enumerate(files, 1):
            if i == 1 or i % 20 == 0 or i == len(files):
                print(f"[{i}/{len(files)}] Đang đọc {os.path.basename(fn)} ... "
                      f"({n_lines:,} dòng đã ghi)", flush=True)
            try:
                with open(fn, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"  LOI JSON, bo qua file: {os.path.basename(fn)} ({e})",
                      file=sys.stderr, flush=True)
                bad_files.append(fn)
                continue

            for pl in data.get("playlists", []):
                pid = pl["pid"]
                for t in pl.get("tracks", []):
                    # loại bỏ tab/newline lẫn trong tên để không vỡ định dạng TSV
                    track_name = t["track_name"].replace("\t", " ").replace("\n", " ")
                    artist_name = t["artist_name"].replace("\t", " ").replace("\n", " ")
                    out_f.write(f"{pid}\t{t['track_uri']}\t{track_name}\t{artist_name}\n")
                    n_lines += 1

    print(f"\nĐã ghi {n_lines} dòng (bài-trong-playlist) vào {args.out}")
    if bad_files:
        print(f"\nCẢNH BÁO: {len(bad_files)} file JSON hỏng, đã BỎ QUA (không tính vào kết quả):")
        for bf in bad_files:
            print(f"  - {os.path.basename(bf)}")
        print("-> Các file này nhiều khả năng tải thiếu/hỏng. Ghi rõ số file này trong báo cáo;")
        print("   tải lại riêng các file đó từ Kaggle nếu muốn có đủ dữ liệu.")


if __name__ == "__main__":
    main()
