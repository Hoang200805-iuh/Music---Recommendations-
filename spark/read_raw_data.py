"""Đọc các slice JSON của Spotify Million Playlist Dataset."""
from pyspark.sql import functions as F, types as T

from spark.spark_session import require_empty


DATASET_SOURCE = "https://www.kaggle.com/datasets/himanshuwagh/spotify-million"
INPUT_FILE_PATTERN = "mpd.slice.*.json"

TRACK_SCHEMA = T.StructType([
    T.StructField("track_uri", T.StringType()),
    T.StructField("track_name", T.StringType()),
    T.StructField("artist_name", T.StringType()),
    T.StructField("pos", T.IntegerType()),
])
PLAYLIST_SCHEMA = T.StructType([
    T.StructField("pid", T.LongType()),
    T.StructField("name", T.StringType()),
    T.StructField("num_tracks", T.IntegerType()),
    T.StructField("num_followers", T.LongType()),
    T.StructField("tracks", T.ArrayType(TRACK_SCHEMA)),
])
ROOT_SCHEMA = T.StructType([
    T.StructField("playlists", T.ArrayType(PLAYLIST_SCHEMA)),
])


def read_raw_mpd(spark, input_path):
    """Đọc thư mục hoặc glob trên local/HDFS và trả về DataFrame cùng số slice."""
    source = (spark.read.schema(ROOT_SCHEMA).option("multiLine", True)
              .option("recursiveFileLookup", True)
              .option("pathGlobFilter", INPUT_FILE_PATTERN)
              .option("mode", "FAILFAST").json(input_path))
    file_count = len(source.inputFiles())
    if file_count == 0:
        raise FileNotFoundError(f"Không tìm thấy {INPUT_FILE_PATTERN} ở {input_path}.")
    require_empty(source.filter(F.col("playlists").isNull()),
                  "JSON không có mảng playlists hoặc sai cấu trúc MPD.")
    return source, file_count
