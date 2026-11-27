"""Khởi tạo Spark và lưu các DataFrame trung gian vào Parquet nén ZSTD."""
import uuid

from pyspark.sql import SparkSession


def create_spark_session(master=None, shuffle_partitions=64):
    builder = SparkSession.builder.appName("MPD-Week1-PersonB")
    if master:
        builder = builder.master(master)
    spark = (builder.config("spark.sql.shuffle.partitions", shuffle_partitions)
             .config("spark.sql.adaptive.enabled", "true").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def log(message):
    print(f"[MPD] {message}", flush=True)


def has_rows(df):
    return bool(df.take(1))


def require_empty(df, message):
    if has_rows(df):
        raise ValueError(message)


class ParquetStages:
    """Quản lý các bản Parquet tạm của một lượt ETL."""

    def __init__(self, spark, root):
        self.spark, self.root = spark, root
        self.paths = {}

    def save(self, df):
        """Ghi và đọc lại một DataFrame để ngắt chuỗi tính toán cũ."""
        path = f"{self.root}/{uuid.uuid4().hex}"
        df.write.mode("errorifexists").option("compression", "zstd").parquet(path)
        result = self.spark.read.parquet(path)
        self.paths[id(result)] = (result, path)
        return result

    def delete_path(self, path):
        jpath = self.spark._jvm.org.apache.hadoop.fs.Path(path)
        fs = jpath.getFileSystem(self.spark._jsc.hadoopConfiguration())
        if fs.exists(jpath) and not fs.delete(jpath, True):
            raise OSError(f"Không xóa được dữ liệu tạm: {path}")

    def release(self, df):
        entry = self.paths.pop(id(df), None)
        if entry is not None:
            self.delete_path(entry[1])

    def close(self):
        self.paths.clear()
        self.delete_path(self.root)


def snapshot(df):
    """Lưu mốc tính toán bằng Parquet trong pipeline hoặc checkpoint khi chạy riêng."""
    store = getattr(df.sparkSession, "_mpd_stages", None)
    return store.save(df) if store else df.checkpoint(eager=True)


def release(df):
    store = getattr(df.sparkSession, "_mpd_stages", None)
    if store:
        store.release(df)
