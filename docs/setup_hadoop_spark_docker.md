# Cài Hadoop + Spark bằng Docker (chạy ở máy cá nhân)

Môi trường sandbox dùng để soạn code này **không chạy được Docker daemon** và
**không có mạng ra Kaggle/DockerHub**, nên phần cài đặt thật phải làm trên máy
của bạn. Dưới đây là cách nhanh nhất cho một môn học (chạy một máy, không cần
cluster nhiều node).

## Cách 1 — Image dựng sẵn (nhanh nhất)

```bash
docker run -d --name hadoop-spark -p 9870:9870 -p 8088:8088 -p 8888:8888 \
  -h hadoop-spark \
  bde2020/hadoop-base:2.0.0-hadoop3.2.1-java8
```

Hoặc dùng docker-compose với image `bde2020/hadoop-namenode`,
`bde2020/hadoop-datanode`, `bde2020/spark-master`, `bde2020/spark-worker`
(tìm "docker-hadoop-spark-workbench" trên GitHub — có sẵn file
`docker-compose.yml` cho đúng nhu cầu "1 node, dùng để học/demo").

## Cách 2 — Cài trực tiếp (không Docker)

1. Cài Java 8/11 (`sudo apt install openjdk-11-jdk`).
2. Tải Hadoop binary (hadoop.apache.org/releases.html), giải nén, set
   `HADOOP_HOME`, chạy ở chế độ **pseudo-distributed** (1 máy, nhưng có NameNode/
   DataNode/ResourceManager thật) theo hướng dẫn chính thức
   "Hadoop: Setting up a Single Node Cluster".
3. Tải Spark (spark.apache.org/downloads.html), giải nén, set `SPARK_HOME`.
   PySpark cũng cài được riêng qua `pip install pyspark` (không cần Spark riêng)
   nếu chỉ cần chạy local, không cần nói chuyện với YARN/HDFS thật.

## Kiểm tra đã chạy đúng

```bash
hdfs dfs -mkdir -p /music/raw/spotify
hdfs dfs -put data/raw/mpd.slice.*.json /music/raw/spotify/
hdfs dfs -ls /music/raw/spotify | head
```

## Chạy job MapReduce thật (sau khi có Hadoop)

```bash
cd src/mapreduce
USE_HADOOP=1 \
HADOOP_STREAMING_JAR=/path/to/hadoop-streaming-x.y.z.jar \
HDFS_INPUT=/music/processed/playlist_tracks_flat.tsv \
HDFS_OUTPUT=/music/analytics/track_popularity \
./run_popularity_job.sh
```

File jar `hadoop-streaming-*.jar` thường nằm ở
`$HADOOP_HOME/share/hadoop/tools/lib/`.

## Nếu không kịp cài Hadoop thật

Không sao — phần lớn báo cáo môn Big Data chấp nhận chạy "Hadoop Streaming
local" (chính là cách `run_popularity_job.sh` đang chạy mặc định: `cat | mapper
| sort | reducer`, đúng 3 bước map → shuffle/sort → reduce mà Hadoop làm, chỉ
khác là không phân tán). Ghi rõ trong báo cáo: "job được viết theo chuẩn Hadoop
Streaming, kiểm chứng logic bằng chạy local; đã có sẵn lệnh để chạy trên Hadoop
thật khi có cluster (xem run_popularity_job.sh, USE_HADOOP=1)".
