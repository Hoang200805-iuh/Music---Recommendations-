# Chạy bằng Docker

Cả hệ thống (API + giao diện) chạy bằng **một lệnh** từ thư mục gốc repo:

```bash
docker compose up --build
```

Rồi mở:

- Giao diện: http://localhost:8501 (trang "So sánh mô hình" và "Gợi ý qua API")
- API: http://localhost:8000/docs (thử request trên trình duyệt), http://localhost:8000/health

Dừng: `Ctrl+C`, hoặc `docker compose down` nếu chạy nền bằng `docker compose up --build -d`.

## Gồm những gì

| Service | Cổng | Lệnh | Ghi chú |
|---|---|---|---|
| `api` | 8000 | `uvicorn api.main:app` | Nạp chỉ mục và mô hình lúc khởi động (khoảng 5–10 giây), có healthcheck `/health` |
| `dashboard` | 8501 | `streamlit run dashboard/app.py` | Chỉ khởi động sau khi API báo sẵn sàng; gọi API qua `http://api:8000` |

Hai service dùng chung một image (`Dockerfile`, Python 3.12, thư viện trong `docker/requirements.txt`
cùng phiên bản với lúc huấn luyện). Không có Spark trong image.

## Cần có trước khi build

Image chứa sẵn **chỉ mục app** `hdfs_sim/music/analytics/app_index/` (khoảng 530MB, gồm cả mô hình LightGBM
`reranker_lgbm.txt`). Thư mục này không có trên git (dữ liệu MPD), nên máy mới cần dựng một lần theo
README ("Cách chạy lại", bước 1–3 và 7):

```bash
python -m recommender.app_index --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv \
    --split-file hdfs_sim/music/processed/playlist_split.tsv \
    --popularity hdfs_sim/music/analytics/track_popularity_train/part-00000 \
    --titles hdfs_sim/music/processed/playlist_titles.tsv --out hdfs_sim/music/analytics/app_index
cp hdfs_sim/music/analytics/rerank_features/reranker_lgbm.txt hdfs_sim/music/analytics/app_index/
```

Chỉ mục được chép vào image thay vì gắn từ máy ngoài vì API đọc nó bằng memory-map: đọc qua thư mục
chia sẻ từ Windows/macOS vào container chậm hơn nhiều. Dựng lại chỉ mục thì chạy lại `docker compose up --build`.

Mô hình **ALS** (không bắt buộc) được gắn từ `hdfs_sim/music/analytics/als/` lúc chạy, không cần build lại:
thả `item_factors.npy` và `track_uris.txt` vào đó (định dạng: `docs/api.md`) rồi `docker compose restart api`.

## Tài nguyên

Đo trên Docker Desktop (Windows 11, WSL2):

- Image 1,89GB (Python và thư viện khoảng 1,3GB, chỉ mục 530MB). Build lần đầu vài phút, các lần sau chỉ
  chép lại phần code đã đổi.
- `docker compose up` đến khi cả hai service chạy: khoảng 20 giây (API nạp chỉ mục, healthcheck báo sẵn sàng).
- RAM: API khoảng 400MB; dashboard khoảng 55MB khi chưa ai mở, tăng thêm vài trăm MB khi mở trang
  "So sánh mô hình" (trang này tự đọc chỉ mục, không qua API).
- Một request `/recommend` với mô hình LightGBM: khoảng 0,15–0,4 giây.
