# Music Recommendation Big Data

Project structure for large-scale music ingestion, processing, recommendation, API, and dashboard workflows.
---

# Tuần 1 — Người A: Môi trường, dữ liệu, MapReduce

Trạng thái các việc trong plan tuần 1 của Người A:

| Việc | Trạng thái | Ghi chú |
|---|---|---|
| Cài Hadoop/Spark | ⚠️ Một phần | PySpark cài được qua pip; Hadoop/Docker thật cần chạy ở máy bạn — xem `docs/setup_hadoop_spark_docker.md` |
| Tải dữ liệu, đếm số file slice | ⚠️ Cần bạn chạy | Mạng ở đây bị chặn tới Kaggle — script đã viết và test xong bằng dữ liệu mẫu, xem `docs/download_dataset.md` |
| Đọc điều khoản dùng MPD | Xong | `docs/mpd_terms_of_use.md` |
| Viết job MapReduce (track popularity) | Xong, đã test | `src/mapreduce/` — khớp tuyệt đối với brute-force Python |

## Vì sao có phần chưa xong hẳn

Sandbox này không có mạng ra ngoài tới `kaggle.com` (bị proxy chặn) và không
chạy được Docker daemon. Vì vậy:
- Mọi script đã được **viết xong và kiểm thử bằng dữ liệu mẫu** tự sinh đúng
  định dạng MPD thật (`src/generate_sample_mpd.py`), để đảm bảo logic đúng.
- Bạn chỉ cần chạy lại 2-3 lệnh ở máy mình với dữ liệu thật — không phải viết
  gì thêm.

## Việc bạn cần làm trên máy mình

```bash
# 1. Tải dữ liệu thật (xem docs/download_dataset.md nếu cần chi tiết)
pip install kaggle
kaggle datasets download -d himanshuwagh/spotify-million -p data/raw --unzip

# 2. Đếm/kiểm tra dữ liệu — lưu output này cho báo cáo
cd src
python3 check_dataset.py --data-dir ../data/raw --sample-check 10 | tee ../reports/check_dataset_report.txt

# 3. Làm phẳng tạm (để có input cho job MapReduce)
python3 flatten_for_mapreduce.py --in ../data/raw --out ../hdfs_sim/music/processed/playlist_tracks_flat.tsv

# 4. Chạy job MapReduce (mặc định mô phỏng local, không cần Hadoop thật)
cd mapreduce
./run_popularity_job.sh
```

Nếu muốn chạy "Hadoop thật" (không bắt buộc, cộng điểm nếu giảng viên yêu cầu):
xem `docs/setup_hadoop_spark_docker.md` rồi chạy lại bước 4 với `USE_HADOOP=1`.

## Cấu trúc đã tạo

```
music-rec/
├── data/
│   ├── raw/              # (trống) — bạn tải dữ liệu thật vào đây
│   ├── sample/            # dữ liệu mẫu để test (2 slice, 101 playlist, KHÔNG phải dữ liệu thật)
│   └── parquet/           # (dành cho tuần 2, Người B)
├── docs/
│   ├── mpd_terms_of_use.md
│   ├── download_dataset.md
│   └── setup_hadoop_spark_docker.md
├── hdfs_sim/music/        # mô phỏng cấu trúc HDFS bằng thư mục local
│   ├── raw/spotify/
│   ├── processed/{playlists,tracks,playlist_tracks}/
│   ├── analytics/track_popularity/   <- kết quả job MapReduce
│   ├── models/
│   └── recommendations/
├── src/
│   ├── generate_sample_mpd.py        # sinh dữ liệu mẫu để test
│   ├── check_dataset.py              # đếm/kiểm tra dữ liệu (việc 2-3 tuần 1)
│   ├── flatten_for_mapreduce.py      # làm phẳng tạm, input cho MapReduce
│   └── mapreduce/
│       ├── mapper.py
│       ├── reducer.py
│       └── run_popularity_job.sh     # mô phỏng local HOẶC chạy Hadoop thật
└── reports/figures/        # (dành cho EDA ở tuần 1-2)
```

## Kết quả đã kiểm chứng (trên dữ liệu mẫu)

- Job chạy ra top bài phổ biến nhất đúng theo phân phối Zipf đã sinh trong dữ
  liệu mẫu (kiểm tra bằng mắt: track phổ biến nhất xuất hiện trong ~88/101
  playlist, đúng như kỳ vọng vì nó được set trọng số cao nhất lúc sinh mẫu).
- **Đối chiếu chéo:** đếm lại bằng `collections.Counter` thuần Python trên
  cùng dữ liệu, so với kết quả mapper/reducer — khớp tuyệt đối 262/262
  track_uri, không lệch dòng nào. Đây chính là bước "kiểm tra chéo MapReduce
  vs Spark" nêu trong plan (ở đây đối chiếu với Python trước; khi Người B có
  bản Spark ở tuần 3, đối chiếu thêm lần nữa).
- Script `check_dataset.py` đã tự phát hiện đúng một khoảng `pid` bị thiếu mà
  mình cố tình để trong dữ liệu mẫu (49 → 51), nghĩa là cảnh báo thiếu file
  hoạt động đúng.

## Việc cần phối hợp với Người B

Định dạng input cho job MapReduce (đã ghi trong docstring của
`flatten_for_mapreduce.py`):

```
pid <TAB> track_uri <TAB> track_name <TAB> artist_name
```

Ở tuần 2, khi Người B có bảng Parquet `playlist_tracks` + `tracks` thật, chỉ
cần export ra đúng 4 cột này (dạng TSV hoặc đổi mapper đọc trực tiếp từ
Parquet) là job MapReduce ở đây chạy được luôn trên dữ liệu thật, không cần
sửa `mapper.py`/`reducer.py`.
