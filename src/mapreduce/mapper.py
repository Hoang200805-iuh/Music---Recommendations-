#!/usr/bin/env python3
"""
MapReduce Mapper — Track Popularity
Input (stdin): mỗi dòng = pid <TAB> track_uri <TAB> track_name <TAB> artist_name
                (đầu ra của bước làm phẳng, xem flatten_for_mapreduce.py)
Output (stdout): track_uri <TAB> 1 <TAB> track_name <TAB> artist_name

Mỗi dòng input là một lượt bài xuất hiện trong MỘT playlist, nên mapper chỉ cần
phát ra (track_uri, 1) — việc cộng dồn số lần xuất hiện (= số playlist chứa bài,
giả sử mỗi playlist không lặp lại cùng 1 bài) do reducer làm sau khi Hadoop đã
gom theo key (shuffle & sort).

Chạy độc lập để test:
    cat input.tsv | python3 mapper.py | head
Chạy qua Hadoop Streaming: xem run_popularity_job.sh
"""
import sys


def main():
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) < 4:
            # dòng hỏng định dạng, bỏ qua (đếm số dòng lỗi ra stderr để debug)
            sys.stderr.write(f"SKIP malformed line: {line[:80]}\n")
            continue
        _pid, track_uri, track_name, artist_name = parts[0], parts[1], parts[2], parts[3]
        # key = track_uri -> Hadoop sẽ gom mọi dòng cùng track_uri lại với nhau trước khi reduce
        print(f"{track_uri}\t1\t{track_name}\t{artist_name}")


if __name__ == "__main__":
    main()
