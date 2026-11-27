"""Kiểm tra và xuất bảng MPD đã làm sạch sang local hoặc HDFS."""
import json

from pyspark.sql import functions as F

from spark.spark_session import log, require_empty


def validate_outputs(playlists, tracks, playlist_tracks, mapping, args):
    """Kiểm tra khóa, số bài và các ngưỡng tần suất trước khi xuất."""
    require_empty(playlists.groupBy("pid").count().filter("count > 1"), "Trùng pid đầu ra.")
    require_empty(tracks.groupBy("track_id").count().filter("count > 1"), "Trùng track_id.")
    require_empty(mapping.groupBy("track_uri").count().filter("count > 1"), "Trùng track_uri.")
    require_empty(playlist_tracks.groupBy("pid", "pos").count().filter("count > 1"),
                  "Trùng cặp (pid, pos) đầu ra.")
    require_empty(playlist_tracks.join(playlists, "pid", "left_anti"), "Khóa ngoại pid bị lỗi.")
    require_empty(playlist_tracks.join(tracks, "track_id", "left_anti"), "Khóa ngoại track_id bị lỗi.")
    sizes = playlist_tracks.groupBy("pid").agg(
        F.count("*").alias("actual"), F.countDistinct("track_id").alias("unique_tracks"))
    require_empty(sizes.filter(F.col("unique_tracks") < args.min_playlist_tracks),
                  "Còn playlist dưới ngưỡng.")
    require_empty(playlists.join(sizes, "pid").filter("num_tracks != actual"), "Sai num_tracks.")
    require_empty(playlist_tracks.select("pid", "track_id").distinct().groupBy("track_id")
                  .count().filter(F.col("count") < args.min_track_playlists),
                  "Còn bài xuất hiện dưới ngưỡng playlist.")


def write_json_file(spark, path, value):
    """Ghi JSON UTF-8 qua filesystem Hadoop."""
    jpath = spark._jvm.org.apache.hadoop.fs.Path(path)
    fs = jpath.getFileSystem(spark._jsc.hadoopConfiguration())
    stream = fs.create(jpath, True)
    try:
        stream.write(bytearray(json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")))
    finally:
        stream.close()


def save_outputs(root, playlists, tracks, playlist_tracks, mapping, audit, args):
    """Xuất năm bảng Parquet ZSTD và tùy chọn tạo JSONL GZIP."""
    log("Kiểm tra khóa, tần suất và số bài")
    validate_outputs(playlists, tracks, playlist_tracks, mapping, args)
    mode = "overwrite" if args.overwrite else "errorifexists"
    tables = {"playlists": playlists, "tracks": tracks, "playlist_tracks": playlist_tracks,
              "track_mapping": mapping, "dedup_audit": audit}
    for name, df in tables.items():
        log(f"Ghi {name}")
        df.write.mode(mode).option("compression", "zstd").parquet(f"{root}/{name}")
    if args.export_streaming:
        lines = playlist_tracks.join(mapping, "track_id").select(F.to_json(F.struct(
            "pid", "pos", "track_id", "track_uri")).alias("value"))
        lines.write.mode(mode).option("compression", "gzip").text(f"{root}/streaming_tracks_jsonl")
