"""
Sinh dữ liệu mẫu đúng định dạng Million Playlist Dataset (MPD) để TEST pipeline
khi chưa có dữ liệu thật (ví dụ do mạng chặn Kaggle trong môi trường sandbox).

KHÔNG dùng file này thay cho dữ liệu thật trong báo cáo — chỉ dùng để kiểm tra
code chạy đúng trước khi chạy trên dữ liệu MPD thật mà bạn tải về máy mình.

Cách dùng:
    python3 generate_sample_mpd.py --out ../data/sample --num-slices 2 --playlists-per-slice 50
"""
import argparse
import json
import os
import random


ARTISTS = [
    ("Degiheugi", "spotify:artist:3V2paBX0y4OaBLYKlOP8qN"),
    ("Thievery Corporation", "spotify:artist:03r4iKL2g2442PT9n2UKsx"),
    ("The Mohawk Beat", "spotify:artist:5l7aLsAWAgApvPCqpbI2Ls"),
    ("Lofi Fruits Music", "spotify:artist:1pLZEt9eZTM9GcLJJFJGj3"),
    ("Chillhop Music", "spotify:artist:7CbxGLH3z0pQTLzHJUlvzH"),
    ("Taylor Swift", "spotify:artist:06HL4z0CvFAxyc27GXpf02"),
    ("Drake", "spotify:artist:3TVXtAsR1Inumwj472S9r4"),
    ("The Weeknd", "spotify:artist:1Xyo4u8uXC1ZmMpatF05PJ"),
    ("Billie Eilish", "spotify:artist:6qqNVTkY8uBg9cP3Jd7DAH"),
    ("Daft Punk", "spotify:artist:4tZwfgrHOc3mvqYlEYSvVi"),
]

MOODS = ["chill", "workout", "study", "party", "sad", "throwback", "roadtrip", "focus"]


def make_track(track_idx):
    artist_name, artist_uri = random.choice(ARTISTS)
    return {
        "pos": None,  # set by caller
        "artist_name": artist_name,
        "track_uri": f"spotify:track:{track_idx:022d}",
        "artist_uri": artist_uri,
        "track_name": f"Track {track_idx}",
        "album_uri": f"spotify:album:{(track_idx // 10):022d}",
        "duration_ms": random.randint(120000, 300000),
        "album_name": f"Album {track_idx // 10}",
    }


def make_playlist(pid, num_tracks_pool):
    mood = random.choice(MOODS)
    name = f"{mood} {random.choice(['mix', 'vibes', 'playlist', '2026', 'songs'])}"
    n_tracks = random.randint(5, 40)
    # Zipf-ish popularity: reuse a small pool of tracks heavily, long tail rarely.
    track_ids = random.choices(
        population=range(num_tracks_pool),
        weights=[1.0 / (i + 1) for i in range(num_tracks_pool)],
        k=n_tracks,
    )
    # de-dup while preserving order (a playlist shouldn't repeat a track)
    seen = set()
    uniq_ids = []
    for t in track_ids:
        if t not in seen:
            seen.add(t)
            uniq_ids.append(t)

    tracks = []
    for pos, tid in enumerate(uniq_ids):
        t = make_track(tid)
        t["pos"] = pos
        tracks.append(t)

    duration_ms = sum(t["duration_ms"] for t in tracks)
    return {
        "name": name,
        "collaborative": "false",
        "pid": pid,
        "modified_at": 1700000000 + pid * 1000,
        "num_tracks": len(tracks),
        "num_albums": len({t["album_uri"] for t in tracks}),
        "num_followers": random.randint(1, 50),
        "num_edits": random.randint(1, 20),
        "duration_ms": duration_ms,
        "num_artists": len({t["artist_uri"] for t in tracks}),
        "tracks": tracks,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/sample")
    ap.add_argument("--num-slices", type=int, default=2)
    ap.add_argument("--playlists-per-slice", type=int, default=50)
    ap.add_argument("--track-pool", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    pid = 0
    for s in range(args.num_slices):
        lo, hi = pid, pid + args.playlists_per_slice - 1
        playlists = []
        for _ in range(args.playlists_per_slice):
            playlists.append(make_playlist(pid, args.track_pool))
            pid += 1
        # Inject one exact-duplicate playlist on purpose (to test dedup logic later)
        if s == 0 and len(playlists) >= 2:
            dup = dict(playlists[0])
            dup["pid"] = pid
            dup["name"] = playlists[0]["name"] + " (copy)"
            playlists.append(dup)
            pid += 1

        payload = {
            "info": {
                "generated_on": "2026-10-26 00:00:00.000000",
                "slice": f"{lo}-{hi}",
                "version": "v1-sample",
            },
            "playlists": playlists,
        }
        fname = os.path.join(args.out, f"mpd.slice.{lo}-{hi}.json")
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        print(f"Wrote {fname} ({len(playlists)} playlists)")

    print(f"\nTổng cộng: {args.num_slices} file slice, {pid} playlist (mẫu, KHÔNG phải dữ liệu thật).")


if __name__ == "__main__":
    main()
