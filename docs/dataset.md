# Dataset

## Chạy Spark ETL từ JSON

Cài PySpark từ `requirements-als.txt` và dùng Java tương thích với phiên bản Spark.
Từ thư mục gốc dự án, chạy pipeline bằng lệnh sau.

```powershell
python -m spark.pipeline --input data/raw --output music_clean_compact --master 'local[2]'
```

Trong Bash có thể dùng `bash scripts/run_spark_etl.sh` với cùng các tham số.
`--input` nhận thư mục hoặc glob `mpd.slice.*.json`. Cả đầu vào và đầu ra có
thể là đường dẫn local hoặc URI `hdfs://...` khi Spark đã kết nối HDFS.
ETL xuất năm bảng Parquet, báo cáo `cleaning_report.json` và dấu `_SUCCESS`.
Thêm `--export-streaming` để xuất thêm JSONL GZIP cho Hadoop Streaming.

Notebook `notebooks/prepare-mdp.ipynb` chạy độc lập trên Kaggle. Có thể chạy test
ETL với dữ liệu mẫu trên máy có Spark và Java bằng lệnh sau.

```powershell
$env:RUN_SPARK_TESTS = '1'
python -m pytest tests/test_spark_pipeline.py -q
```

Trên Windows với Python 3.12 trở lên, dùng PySpark 4.1.2 trở lên để có
[bản vá lỗi Python worker](https://issues.apache.org/jira/browse/SPARK-53759).
Khi ghi Parquet trực tiếp trên Windows, Hadoop cần `winutils.exe` và
`hadoop.dll` tương thích. Có thể chạy ETL trong WSL/Linux.

## Dữ liệu Parquet đã làm sạch

MPD gốc là JSON. Bộ `music_clean_compact` là bản đã làm sạch, lưu bằng Parquet
(nén khoảng 286 MB, tính cả báo cáo/audit):

| Bảng | Số dòng | Cột dùng trong project |
|---|---:|---|
| `playlists` | 990.000 | `pid`, `name` |
| `playlist_tracks` | 63.403.784 | `pid`, `pos`, `track_id` |
| `tracks` | 591.653 | `track_id`, `track_name`, `artist_name` |
| `track_mapping` | 591.653 | `track_id`, `track_uri` |

`track_id` không liên tiếp và không phải ID của chỉ mục ứng dụng. Phải ghép với
`track_mapping` để giữ URI Spotify. Dữ liệu Parquet không đảm bảo thứ tự bài;
các bài của mỗi playlist phải sắp theo `pos` trước khi lấy k bài đầu làm seed.

Chuẩn bị dữ liệu cho luồng ALS/Hybrid hiện tại, từ gốc project:

```powershell
python -m src.prepare_compact_mpd --input music_clean_compact --out hdfs_sim/music
```

Script [prepare_compact_mpd.py](../src/prepare_compact_mpd.py) dùng SQLite để sắp xếp
trên ổ đĩa, xuất TSV cùng hợp đồng với các mô hình cũ. Chia theo playlist với seed
42: **693.000 train, 148.500 validation, 148.500 test**. Popularity chỉ đếm trên
train; bài trùng ở vị trí khác nhau vẫn được giữ để khớp cách đếm lượt hiện có.
Ma trận ALS sau đó chuyển tương tác sang nhị phân.

Đầu ra gồm dữ liệu phẳng, ánh xạ split, tiêu đề chuẩn hóa và popularity train.
Manifest lưu nguồn/cấu hình để lần chạy sau kiểm tra và dùng lại. Khi đổi dữ liệu
hoặc split, cần dựng lại chỉ mục và model tương ứng; không dùng artifact cũ với
vocabulary mới. Xem [hướng dẫn ALS](run_als_local.md).
