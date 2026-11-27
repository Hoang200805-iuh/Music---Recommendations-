"""Làm phẳng playlist và kiểm tra dữ liệu tương tác."""
from pyspark.sql import functions as F

from spark.read_raw_data import read_raw_mpd
from spark.spark_session import require_empty, snapshot


def clean_text(column):
    """Chuẩn hóa khoảng trắng và đổi null thành chuỗi rỗng."""
    return F.trim(F.regexp_replace(F.coalesce(column, F.lit("")), r"\s+", " "))


def load_mpd(spark, input_path):
    """Trả về metadata playlist, các dòng hợp lệ, tương tác và thống kê đầu vào."""
    source, file_count = read_raw_mpd(spark, input_path)
    # Chỉ lưu các bảng đã làm phẳng vào staging.
    raw = source.select(F.explode("playlists").alias("p")).select("p.*")
    metadata = snapshot(raw.select(
        "pid", clean_text(F.col("name")).alias("name"),
        F.greatest(F.coalesce("num_followers", F.lit(0)), F.lit(0)).alias("num_followers"),
    ))
    require_empty(metadata.filter(F.col("pid").isNull() | (F.col("pid") < 0)),
                  "pid phải là số nguyên không âm.")
    require_empty(metadata.groupBy("pid").count().filter("count > 1"),
                  "Trùng pid trong input. Kiểm tra các slice bị nạp lặp.")
    input_rows = raw.agg(F.coalesce(F.sum(F.when(F.col("tracks").isNotNull(),
        F.size("tracks")).otherwise(0)), F.lit(0)).alias("n")).first().n
    exploded = raw.select("pid", F.explode("tracks").alias("t")).select(
        "pid", F.col("t.pos").alias("pos"),
        F.trim(F.col("t.track_uri")).alias("track_uri"),
        clean_text(F.col("t.track_name")).alias("track_name"),
        clean_text(F.col("t.artist_name")).alias("artist_name"),
    )
    valid = snapshot(exploded.filter(
        F.col("pos").isNotNull() & (F.col("pos") >= 0)
        & F.col("track_uri").rlike(r"^spotify:track:[A-Za-z0-9]+$")
    ))
    require_empty(valid.groupBy("pid", "pos").agg(
        F.countDistinct("track_uri").alias("n")).filter("n > 1"),
        "Một cặp (pid, pos) có nhiều track_uri khác nhau.")
    # Giữ bài lặp ở các vị trí khác nhau trong cùng playlist.
    events = valid.select("pid", "pos", "track_uri").dropDuplicates()
    valid_count = valid.count()
    stats = {
        "slice_files": file_count,
        "input_playlists": metadata.count(),
        "input_track_rows": input_rows,
        "invalid_track_rows_removed": input_rows - valid_count,
        "duplicate_position_rows_removed": valid_count - events.count(),
    }
    return metadata, valid, events, stats
