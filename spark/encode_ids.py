"""Mã hóa Spotify track URI thành track_id có thể tái lập."""
from pyspark.sql import functions as F

from spark.spark_session import release, snapshot


def build_track_mapping(spark, valid):
    """Đánh số bài từ 1 theo thứ tự URI và chọn metadata nhất quán cho mỗi URI."""
    catalog = snapshot(valid.groupBy("track_uri").agg(F.min(F.struct(
        "track_name", "artist_name")).alias("m")).select("track_uri", "m.*"))
    if catalog.count() > 2_147_483_647:
        raise ValueError("Vượt giới hạn int32 cho track_id.")
    # Sắp xếp phân tán để giữ ID ổn định với cùng tập URI.
    indexed = catalog.orderBy("track_uri").rdd.zipWithIndex().map(
        lambda pair: (int(pair[1]) + 1, pair[0]["track_uri"],
                      pair[0]["track_name"], pair[0]["artist_name"]))
    schema = "track_id int, track_uri string, track_name string, artist_name string"
    result = snapshot(spark.createDataFrame(indexed, schema))
    release(catalog)
    return result
