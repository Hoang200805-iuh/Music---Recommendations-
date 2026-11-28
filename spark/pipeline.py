"""Làm sạch Spotify MPD bằng PySpark và xuất các bảng Parquet."""
import argparse
import json
import sys
import time
import uuid
from fractions import Fraction

from pyspark.sql import functions as F

from spark.cleaning import clean_until_stable
from spark.encode_ids import build_track_mapping
from spark.flatten_playlists import load_mpd
from spark.read_raw_data import DATASET_SOURCE, INPUT_FILE_PATTERN
from spark.save_parquet import save_outputs, write_json_file
from spark.spark_session import ParquetStages, create_spark_session, log, release, snapshot


def run_pipeline(spark, args):
    """Chạy ETL với thư mục staging riêng rồi dọn staging khi kết thúc."""
    if getattr(spark, "_mpd_stages", None) is not None:
        raise RuntimeError("Không chạy hai pipeline đồng thời trong cùng SparkSession.")
    output = args.output
    if output.rstrip("/\\") in ("", ".", ".."):
        raise ValueError("Cần thư mục output riêng.")
    hadoop_path = spark._jvm.org.apache.hadoop.fs.Path
    path = hadoop_path(output)
    fs = path.getFileSystem(spark._jsc.hadoopConfiguration())
    path = fs.makeQualified(path)
    if path.isRoot():
        raise ValueError("Cần thư mục output riêng.")
    if fs.exists(path) and not args.overwrite:
        raise FileExistsError(f"Output đã tồn tại: {output}. Chọn thư mục mới hoặc --overwrite.")

    fs.mkdirs(path)
    # Xóa dấu hoàn tất của lượt trước trước khi ghi lại các bảng.
    for name in ("_SUCCESS", "cleaning_report.json"):
        fs.delete(hadoop_path(path, name), False)
    root = path.toString()
    store = ParquetStages(spark, f"{root}/_staging/{uuid.uuid4().hex}")
    spark._mpd_stages = store
    try:
        return _run_pipeline(spark, args, root)
    finally:
        spark._mpd_stages = None
        try:
            store.close()
        except Exception as error:
            log(f"Không dọn hết thư mục tạm: {error}")


def _run_pipeline(spark, args, root):
    start = time.monotonic()
    log("Đọc MPD JSON và làm phẳng playlist")
    metadata, valid, events_uri, stats = load_mpd(spark, args.input)
    log(f"Đã đọc {stats['slice_files']} slice và {stats['input_playlists']:,} playlist")
    catalog = build_track_mapping(spark, valid)
    events = snapshot(events_uri.join(catalog.select("track_uri", "track_id"), "track_uri")
                      .select("pid", "pos", "track_id"))
    release(valid)

    playlist_tracks, audit, rounds = clean_until_stable(spark, events, args)
    used = playlist_tracks.select("track_id").distinct()
    mapping_full = snapshot(catalog.join(used, "track_id", "left_semi"))
    release(catalog)
    mapping = mapping_full.select("track_id", "track_uri")
    tracks = mapping_full.select("track_id", "track_name", "artist_name")
    counts = playlist_tracks.groupBy("pid").agg(F.count("*").cast("int").alias("num_tracks"))
    playlists = snapshot(metadata.join(counts, "pid").select(
        "pid", "name", "num_tracks", "num_followers"))
    release(metadata)

    save_outputs(root, playlists, tracks, playlist_tracks, mapping, audit, args)
    report = {
        **stats,
        "dataset_source": DATASET_SOURCE,
        "input_path": args.input,
        "input_file_pattern": INPUT_FILE_PATTERN,
        "output_playlists": playlists.count(),
        "output_tracks": tracks.count(),
        "output_playlist_track_rows": playlist_tracks.count(),
        "removed_duplicate_playlists": audit.count(),
        "threshold": args.near_threshold,
        "comparison": "Jaccard > threshold",
        "min_distinct_tracks_per_playlist": args.min_playlist_tracks,
        "min_distinct_playlists_per_track": args.min_track_playlists,
        "rounds": rounds,
        "elapsed_seconds": round(time.monotonic() - start, 2),
        "checks_passed": True,
        "split_created": False,
        "spark_version": spark.version,
        "temporary_storage": "rolling_zstd_parquet",
    }
    write_json_file(spark, f"{root}/cleaning_report.json", report)
    write_json_file(spark, f"{root}/_SUCCESS", {"ok": True})
    log(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True,
                   help="Thư mục dataset hoặc glob mpd.slice.*.json trên local/HDFS")
    p.add_argument("--output", required=True, help="Thư mục Parquet trên local/HDFS")
    p.add_argument("--master", help="Spark master, ví dụ local[2]")
    p.add_argument("--shuffle-partitions", type=int, default=64)
    p.add_argument("--min-playlist-tracks", type=int, default=5)
    p.add_argument("--min-track-playlists", type=int, default=5)
    p.add_argument("--near-threshold", default="0.9")
    p.add_argument("--max-core-rounds", type=int, default=100)
    p.add_argument("--max-clean-rounds", type=int, default=100)
    p.add_argument("--export-streaming", action="store_true", help="Xuất thêm JSONL GZIP cho Hadoop Streaming")
    p.add_argument("--overwrite", action="store_true", help="Ghi đè các bảng ở thư mục output đã có")
    return p


def main(argv=None):
    cli = parser()
    args = cli.parse_args(argv)
    if min(args.min_playlist_tracks, args.min_track_playlists, args.shuffle_partitions,
           args.max_core_rounds, args.max_clean_rounds) < 1:
        cli.error("Các tham số đếm phải từ 1 trở lên.")
    try:
        threshold = Fraction(args.near_threshold)
    except (ValueError, ZeroDivisionError):
        cli.error("--near-threshold phải là số nằm giữa 0 và 1.")
    if not 0 < threshold < 1:
        cli.error("--near-threshold phải nằm giữa 0 và 1.")

    spark = create_spark_session(args.master, args.shuffle_partitions)
    try:
        run_pipeline(spark, args)
    finally:
        spark.stop()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace")
    main()
