# Tải dữ liệu thật (himanshuwagh/spotify-million)

Môi trường soạn code này bị chặn mạng tới kaggle.com, nên bước này **bạn cần tự
chạy trên máy mình** (hoặc trên Colab). Sau khi tải xong, các script trong
`src/` đã sẵn sàng dùng ngay, chỉ cần đổi `--data-dir` sang thư mục thật.

## Cách 1 — Kaggle CLI (khuyến nghị)

```bash
pip install kaggle
# Lấy API token: vào kaggle.com -> Account -> Create New API Token -> tải kaggle.json
mkdir -p ~/.kaggle && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d himanshuwagh/spotify-million -p data/raw --unzip
```

## Cách 2 — Tải thủ công

Vào https://www.kaggle.com/datasets/himanshuwagh/spotify-million, bấm Download,
giải nén vào `data/raw/`.

## Sau khi tải xong, chạy ngay:

```bash
cd src
python3 check_dataset.py --data-dir ../data/raw --sample-check 10
```

Script sẽ báo:
- Số file slice thực tế (so với 1.000 file của bản gốc — bản Kaggle có thể ít
  hơn, đã gặp trường hợp này trước đây nên đừng ngạc nhiên).
- Có khoảng `pid` nào bị thiếu giữa các slice không.
- Mở thử vài file để chắc JSON không hỏng.
- Ước lượng tổng số playlist từ mẫu.

**Lưu kết quả terminal này lại** (hoặc `> ../reports/check_dataset_report.txt`)
để đưa vào báo cáo tuần 1 — đây chính là "điều kiện xong" của việc kiểm tra dữ
liệu.

Sau đó chạy lại toàn bộ pipeline MapReduce trên dữ liệu thật:

```bash
python3 flatten_for_mapreduce.py --in ../data/raw --out ../hdfs_sim/music/processed/playlist_tracks_flat.tsv
cd mapreduce && ./run_popularity_job.sh
```

Lưu ý: với 1.000 file/1 triệu playlist, bước `flatten_for_mapreduce.py` (thuần
Python, đọc từng file) sẽ chậm — chỉ dùng để ra input test cho MapReduce job
của tuần 1. Việc làm phẳng "thật" bằng PySpark (có thể chạy song song, xử lý
được khối lượng này nhanh hơn nhiều) là phần việc của Người B, sẽ thay thế
script tạm này ở tuần 2.
