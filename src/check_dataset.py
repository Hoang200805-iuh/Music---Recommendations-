"""
Kiểm tra bộ dữ liệu MPD đã tải về: đếm số file slice, tổng số playlist/track,
mở thử vài file để chắc JSON không hỏng, so với con số chuẩn của bản gốc (1.000 file,
1.000.000 playlist).

Cách dùng:
    python3 check_dataset.py --data-dir ../data/raw
    python3 check_dataset.py --data-dir ../data/sample   # test với dữ liệu mẫu
"""
import argparse
import glob
import json
import os
import re
import sys

EXPECTED_SLICES = 1000
EXPECTED_PLAYLISTS = 1_000_000
SLICE_PATTERN = re.compile(r"mpd\.slice\.(\d+)-(\d+)\.json$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="Thư mục chứa các file mpd.slice.*.json")
    ap.add_argument("--sample-check", type=int, default=5,
                     help="Số file mở thử để kiểm tra JSON không hỏng")
    args = ap.parse_args()

    pattern = os.path.join(args.data_dir, "mpd.slice.*.json")
    files = sorted(glob.glob(pattern))

    print(f"Thư mục: {os.path.abspath(args.data_dir)}")
    print(f"Số file slice tìm thấy: {len(files)} (bản gốc có {EXPECTED_SLICES})")

    if len(files) == 0:
        print("\nKHÔNG tìm thấy file nào khớp mpd.slice.*.json — kiểm tra lại đường dẫn --data-dir.")
        sys.exit(1)

    if len(files) != EXPECTED_SLICES:
        print(f"CẢNH BÁO: số file khác {EXPECTED_SLICES}. Bản Kaggle có thể chỉ là một phần của MPD gốc.")
        print("-> Ghi rõ điều này trong báo cáo và điều chỉnh quy mô dữ liệu cho phù hợp.")

    # Kiểm tra tính liên tục của khoảng pid giữa các slice (phát hiện file thiếu)
    ranges = []
    for fn in files:
        m = SLICE_PATTERN.search(os.path.basename(fn))
        if m:
            ranges.append((int(m.group(1)), int(m.group(2))))
    ranges.sort()
    gaps = []
    for i in range(1, len(ranges)):
        prev_hi = ranges[i - 1][1]
        cur_lo = ranges[i][0]
        if cur_lo != prev_hi + 1:
            gaps.append((prev_hi, cur_lo))
    if gaps:
        print(f"\nCẢNH BÁO: phát hiện {len(gaps)} khoảng pid bị thiếu/không liên tục, ví dụ: {gaps[:5]}")
    else:
        print("\nCác khoảng pid liên tục, không thấy lỗ hổng rõ rệt giữa các slice.")

    # Mở thử vài file đầu/giữa/cuối để kiểm tra JSON hợp lệ + đếm playlist/track
    n = min(args.sample_check, len(files))
    check_indices = sorted(set([0, len(files) // 2, len(files) - 1] + list(range(n))))
    check_indices = [i for i in check_indices if 0 <= i < len(files)][: max(n, 3)]

    total_playlists_sampled = 0
    total_tracks_sampled = 0
    broken = []
    print(f"\nMở thử {len(check_indices)} file để kiểm tra JSON hợp lệ:")
    for idx in check_indices:
        fn = files[idx]
        try:
            with open(fn, "r", encoding="utf-8") as f:
                data = json.load(f)
            pls = data.get("playlists", [])
            n_tracks = sum(len(p.get("tracks", [])) for p in pls)
            total_playlists_sampled += len(pls)
            total_tracks_sampled += n_tracks
            print(f"  OK  {os.path.basename(fn):40s} playlists={len(pls):5d}  tracks={n_tracks:6d}")
        except Exception as e:
            broken.append((fn, str(e)))
            print(f"  LOI {os.path.basename(fn):40s} {e}")

    if broken:
        print(f"\nCẢNH BÁO: {len(broken)} file lỗi khi mở. Tải lại hoặc loại trừ các file này.")

    if total_playlists_sampled:
        avg_tracks = total_tracks_sampled / total_playlists_sampled
        est_total_playlists = len(files) * (total_playlists_sampled / len(check_indices))
        print(f"\nƯớc lượng từ mẫu: trung bình {avg_tracks:.1f} bài/playlist; "
              f"ước tính tổng playlist toàn bộ dữ liệu ~ {est_total_playlists:,.0f}")
        if abs(est_total_playlists - EXPECTED_PLAYLISTS) / EXPECTED_PLAYLISTS > 0.05 and len(files) == EXPECTED_SLICES:
            print("CẢNH BÁO: số playlist ước tính lệch >5% so với 1.000.000 của bản gốc.")

    print("\nXong. Lưu kết quả này lại (copy terminal output hoặc redirect > report.txt) để đưa vào báo cáo tuần 1.")


if __name__ == "__main__":
    main()
