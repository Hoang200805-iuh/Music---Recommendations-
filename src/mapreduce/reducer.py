#!/usr/bin/env python3
"""
MapReduce Reducer — Track Popularity
Input (stdin, ĐÃ ĐƯỢC SẮP XẾP THEO KEY bởi Hadoop/sort): track_uri <TAB> 1 <TAB> track_name <TAB> artist_name
Output (stdout): track_uri <TAB> popularity_count <TAB> track_name <TAB> artist_name

Giả định quan trọng của Hadoop Streaming: các dòng cùng key (track_uri) luôn
đứng LIỀN NHAU sau bước shuffle&sort, nên reducer có thể cộng dồn theo kiểu
"streaming" (không cần load hết vào RAM) bằng cách theo dõi khi key đổi.

Chạy độc lập để test (bắt buộc có bước sort ở giữa, giống Hadoop sẽ tự làm):
    cat input.tsv | python3 mapper.py | sort -t$'\t' -k1,1 | python3 reducer.py
"""
import sys


def main():
    current_key = None
    current_count = 0
    current_track_name = None
    current_artist_name = None

    def flush():
        if current_key is not None:
            print(f"{current_key}\t{current_count}\t{current_track_name}\t{current_artist_name}")

    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            sys.stderr.write(f"SKIP malformed line: {line[:80]}\n")
            continue
        track_uri, cnt_str, track_name, artist_name = parts[0], parts[1], parts[2], parts[3]
        try:
            cnt = int(cnt_str)
        except ValueError:
            sys.stderr.write(f"SKIP bad count: {line[:80]}\n")
            continue

        if track_uri == current_key:
            current_count += cnt
        else:
            flush()
            current_key = track_uri
            current_count = cnt
            current_track_name = track_name
            current_artist_name = artist_name

    flush()  # đừng quên ghi nốt key cuối cùng


if __name__ == "__main__":
    main()
