"""
Vẽ nhanh 4-5 biểu đồ EDA (Exploratory Data Analysis).

PHIÊN BẢN 2 — xử lý kiểu "streaming" (đọc từng dòng, đếm bằng dict/Counter
thuần Python) thay vì nạp cả bảng vào pandas, vì file làm phẳng thật có ~52
triệu dòng và nạp hết vào DataFrame (4 cột text, lặp lại ở MỌI dòng) sẽ hết RAM
trên máy thường (đã gặp MemoryError). Cách này chỉ giữ trong RAM các dict đếm
có số lượng khóa nhỏ (số playlist, số nghệ sĩ, số bài) chứ không giữ 52 triệu
dòng thô.

Input 1 (--in, bắt buộc): file TSV đã làm phẳng, định dạng:
    pid <TAB> track_uri <TAB> track_name <TAB> artist_name
Input 2 (--popularity, khuyến khích dùng): kết quả popularity ĐÃ TÍNH SẴN từ
job MapReduce/Spark tuần 1, định dạng:
    track_uri <TAB> popularity_count <TAB> track_name <TAB> artist_name
  -> dùng lại kết quả này cho biểu đồ 2 & 4 thay vì tính lại từ đầu (đỡ tốn
     RAM hơn nhiều, và đảm bảo khớp với số đã đối chiếu MapReduce vs Spark).
  Nếu không truyền --popularity, script sẽ tự tính nhưng sẽ chậm/tốn RAM hơn
  trên dữ liệu thật lớn.

Output: 5 file PNG trong thư mục --out.

Cách dùng (khuyến khích — dùng lại kết quả popularity đã có):
    python eda_charts.py --in hdfs_sim\\music\\processed\\playlist_tracks_flat.tsv ^
        --popularity hdfs_sim\\music\\analytics\\track_popularity\\part-00000 ^
        --out reports\\figures
"""
import argparse
import os
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#4C78A8"
BLUE_DARK = "#2C4F6B"
TEXT_MUTED = "#6B7280"
GRID = "#E5E7EB"


def style_ax(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10, color=TEXT_MUTED)
    ax.set_ylabel(ylabel, fontsize=10, color=TEXT_MUTED)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def stream_flat_file(path):
    """Đọc file làm phẳng theo từng dòng, tính các số đếm cần thiết mà KHÔNG
    giữ 52 triệu dòng thô trong RAM — chỉ giữ các dict đếm (số khóa nhỏ hơn
    nhiều: số playlist, số nghệ sĩ)."""
    playlist_len = Counter()          # pid -> số bài trong playlist đó
    artist_occurrence = Counter()     # artist_name -> tổng số lượt xuất hiện
    artist_tracks = defaultdict(set)  # artist_name -> set(track_uri) duy nhất
    n_lines = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            pid, track_uri, _track_name, artist_name = parts[0], parts[1], parts[2], parts[3]

            playlist_len[pid] += 1
            artist_occurrence[artist_name] += 1
            artist_tracks[artist_name].add(track_uri)

            n_lines += 1
            if n_lines % 5_000_000 == 0:
                print(f"  ... đã đọc {n_lines:,} dòng", flush=True)

    n_tracks_per_artist = {a: len(s) for a, s in artist_tracks.items()}
    return n_lines, playlist_len, artist_occurrence, n_tracks_per_artist


def load_popularity_file(path):
    """Đọc kết quả popularity đã tính sẵn (từ MapReduce hoặc Spark), định dạng:
    track_uri <TAB> popularity_count <TAB> track_name <TAB> artist_name"""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            track_uri, count, track_name, artist_name = parts[0], int(parts[1]), parts[2], parts[3]
            rows.append((track_uri, count, track_name, artist_name))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def compute_popularity_from_scratch(path):
    """Phương án dự phòng nếu không có sẵn file --popularity: tính track
    popularity trực tiếp từ file làm phẳng. CẢNH BÁO: tốn RAM hơn nhiều so với
    dùng --popularity, vì phải giữ set(pid) cho từng track_uri."""
    track_pids = defaultdict(set)
    track_meta = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                continue
            pid, track_uri, track_name, artist_name = parts[0], parts[1], parts[2], parts[3]
            track_pids[track_uri].add(pid)
            if track_uri not in track_meta:
                track_meta[track_uri] = (track_name, artist_name)
    rows = [
        (uri, len(pids), track_meta[uri][0], track_meta[uri][1])
        for uri, pids in track_pids.items()
    ]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--popularity", dest="pop_path", default=None,
                     help="File popularity đã tính sẵn (MapReduce/Spark). Khuyến khích dùng trên dữ liệu thật.")
    ap.add_argument("--out", dest="out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Đang đọc file làm phẳng (streaming, có thể mất vài phút trên dữ liệu thật)...")
    n_lines, playlist_len, artist_occurrence, n_tracks_per_artist = stream_flat_file(args.in_path)
    n_playlists = len(playlist_len)
    n_artists = len(artist_occurrence)
    print(f"Đã đọc {n_lines:,} dòng | {n_playlists:,} playlist | {n_artists:,} nghệ sĩ")

    if args.pop_path:
        print(f"Dùng lại kết quả popularity có sẵn: {args.pop_path}")
        pop_rows = load_popularity_file(args.pop_path)
    else:
        print("Không có --popularity, tự tính từ file làm phẳng (chậm hơn)...")
        pop_rows = compute_popularity_from_scratch(args.in_path)
    n_tracks = len(pop_rows)
    print(f"Số track_uri duy nhất: {n_tracks:,}")

    # -------------------------------------------------------------------
    # 1) Phân phối độ dài playlist
    # -------------------------------------------------------------------
    lengths = list(playlist_len.values())
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(lengths, bins=30, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    avg_len = sum(lengths) / len(lengths)
    ax.axvline(avg_len, color=BLUE_DARK, linestyle="--", linewidth=1.5,
                label=f"Trung bình: {avg_len:.1f} bài")
    style_ax(ax, "1. Phân phối độ dài playlist", "Số bài trong playlist", "Số playlist")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "1_playlist_length_dist.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 2) Phân phối độ phổ biến bài hát (long-tail), rank vs. count, log-log
    # -------------------------------------------------------------------
    ranks = list(range(1, n_tracks + 1))
    counts = [r[1] for r in pop_rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ranks, counts, color=BLUE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    style_ax(ax, "2. Độ phổ biến bài hát (long-tail)", "Hạng bài hát (log)", "Số playlist chứa bài (log)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "2_track_popularity_longtail.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 3) Top 10 nghệ sĩ theo tổng số lượt xuất hiện
    # -------------------------------------------------------------------
    top_artists = artist_occurrence.most_common(10)[::-1]
    names = [a for a, _ in top_artists]
    vals = [v for _, v in top_artists]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(names, vals, color=BLUE, zorder=3)
    style_ax(ax, "3. Top 10 nghệ sĩ (số lượt bài xuất hiện trong playlist)", "Số lượt xuất hiện", "")
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "3_top_artists.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 4) Top 10 bài hát phổ biến nhất
    # -------------------------------------------------------------------
    top_tracks = pop_rows[:10][::-1]
    labels = [f"{artist} – {track}" for _uri, _cnt, track, artist in top_tracks]
    vals4 = [cnt for _uri, cnt, _track, _artist in top_tracks]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.barh(labels, vals4, color=BLUE, zorder=3)
    style_ax(ax, "4. Top 10 bài hát phổ biến nhất", "Số playlist chứa bài", "")
    for y, v in enumerate(vals4):
        ax.text(v, y, f" {v:,}", va="center", fontsize=8.5, color=TEXT_MUTED)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "4_top_tracks.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 5) Phân phối số bài (track_uri duy nhất) / nghệ sĩ
    # -------------------------------------------------------------------
    tpa_values = list(n_tracks_per_artist.values())
    max_v = max(tpa_values) if tpa_values else 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(tpa_values, bins=min(30, max_v), color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    style_ax(ax, "5. Phân phối số bài hát duy nhất / nghệ sĩ", "Số bài (track_uri duy nhất)", "Số nghệ sĩ")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "5_tracks_per_artist_dist.png"), dpi=150)
    plt.close(fig)

    print(f"\nĐã lưu 5 biểu đồ vào: {args.out_dir}")


if __name__ == "__main__":
    main()
