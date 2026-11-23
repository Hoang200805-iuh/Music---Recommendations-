# Chạy bằng Docker

## Train/tuning ALS trước khi chạy Docker (PowerShell)

Image Docker hiện tại phục vụ API/dashboard, không chứa Spark/Java để train.
Dữ liệu Parquet đã được chuẩn bị thành bốn file trong `hdfs_sim/music`.
Trên máy hiện tại đã có `.venv` và dependency; không cần tạo lại môi trường.

**1. Vào project và kiểm tra môi trường/input:**

```powershell
cd 'D:\2026-2027\Bigdata\CK\Music---Recommendations-'
.\.venv\Scripts\python.exe scripts/run_als_local.py --check-only --master 'local[2]' --driver-memory 4g
```

Khi thấy `LOCAL_CHECK_OK` thì chạy bước tiếp theo. Dùng trực tiếp Python trong
`.venv` nên không cần kích hoạt PowerShell script. Đóng bớt ứng dụng chiếm RAM;
`4g` chỉ là heap Spark, còn Python/cache và hệ điều hành cũng cần bộ nhớ.

**2. Train, tuning trên validation rồi đánh giá trên test:**

```powershell
.\.venv\Scripts\python.exe scripts/run_als_local.py `
    --work-dir outputs/als_local `
    --master 'local[2]' --driver-memory 4g --num-blocks 10 `
    --val-playlists 1000 --test-playlists 10000
```

Lệnh này huấn luyện trên toàn bộ **693.000 playlist train**. Số 1.000 và 10.000
chỉ giới hạn playlist đánh giá ở mỗi mức seed, không thu nhỏ tập train.
Tuner thử `rank` (20, 50, 100), rồi `regParam` (0.01, 0.1, 1), rồi `alpha`
(1, 10, 40): tổng 7 cấu hình khác nhau, mỗi cấu hình 10 vòng lặp. Chọn theo
NDCG@10 trung bình trên validation; test dùng để báo cáo sau khi đã chọn.
Nếu chỉ muốn tuning, thêm `--tune-only` vào lệnh trên.

Nếu bị ngắt, chạy lại **đúng lệnh và work-dir**: trial hoàn tất được dùng lại,
trial đang dở phải train lại. Không dùng `--smoke-test` cho dữ liệu thật.

Kết quả trong `outputs/als_local`:

- `models/als_implicit/`: bundle tốt nhất gồm ba file cần để phục vụ gợi ý.
- `reports/als_tuning_val.csv`: kết quả tuning trên validation.
- `reports/results_with_als.csv`: so sánh 5 mô hình trên test (trừ khi dùng `--tune-only`).
- `pipeline.log`: log quá trình chạy; theo dõi bằng terminal PowerShell khác:

```powershell
Get-Content outputs/als_local/pipeline.log -Tail 30 -Wait
```

Kết thúc thành công sẽ có `LOCAL_DONE`. Nếu CSV đang mở trong Excel, báo cáo
có thể được lưu với tên `.latest.csv`; đường dẫn thực tế xuất hiện trong log.

### Nếu JVM báo thiếu bộ nhớ / paging file (errno=1455)

`The paging file is too small` và `Native memory allocation (mmap) failed`
nghĩa là Windows hết khả năng cấp phát bộ nhớ. Các lỗi Py4J/ConnectionRefused
phía sau là hệ quả JVM đã tắt. Cảnh báo `winutils.exe` không phải nguyên nhân
của lần lỗi này. JVM có thể chết ngay cả khi heap chưa đạt giới hạn `4g`.

1. Đóng các ứng dụng chiếm bộ nhớ, nhất là Docker Desktop nếu đang mở và chưa
   dùng để phục vụ API/dashboard; kiểm tra Task Manager → Memory → Committed.
2. Nếu vẫn gần giới hạn, mở `Win+R` → `sysdm.cpl` → Advanced → Performance /
   Settings → Advanced → Virtual memory / Change. Có thể bỏ chọn tự quản lý
   cho mọi ổ, chọn ổ D còn đủ dung lượng, chọn **System managed size**, bấm
   **Set** và xác nhận. Giữ pagefile đang có trên ổ C; khởi động lại nếu Windows
   yêu cầu. Đây là thay đổi cấu hình hệ thống do người dùng thực hiện.
3. Chạy lại với heap nhỏ hơn; vẫn giữ dữ liệu và cache đã tạo:

```powershell
.\.venv\Scripts\python.exe scripts/run_als_local.py --check-only --master 'local[2]' --driver-memory 2g
.\.venv\Scripts\python.exe scripts/run_als_local.py `
    --work-dir outputs/als_local `
    --master 'local[2]' --driver-memory 2g --num-blocks 10 `
    --val-playlists 1000 --test-playlists 10000
```

Heap 2g giảm nhu cầu bộ nhớ nhưng chưa đảm bảo đủ cho mọi cấu hình ALS trên toàn
bộ dữ liệu. Runner kiểm tra sớm commit còn trống tối thiểu bằng heap + 2 GiB dự
phòng Python/JVM; đây là ngưỡng kiểm tra, không phải dự đoán mức dùng tối đa.
Nếu chưa đủ, chương trình báo rõ và dừng trước khi đọc dữ liệu/train.
Không giảm `val-playlists` để xử lý lỗi đọc/cache ratings: tham số đó chỉ giảm
phần đánh giá, không giảm 43,8 triệu tương tác train.

Cache train/validation đã tạo được giữ lại. Lần lỗi ghi nhận chưa hoàn tất
trial đầu tiên nên cần train lại trial đó. Giữ `num-blocks` và các tham số thống
kê nếu muốn dùng cùng checkpoint; thay heap 4g → 2g không đổi fingerprint ALS.

Tham khảo [Microsoft: pagefile và giới hạn commit của Windows](https://learn.microsoft.com/en-us/troubleshoot/windows-client/performance/how-to-determine-the-appropriate-page-file-size-for-64-bit-versions-of-windows).

**3. Dựng chỉ mục từ cùng bộ dữ liệu và chép bundle ALS:**

```powershell
.\.venv\Scripts\python.exe -X utf8 -m recommender.app_index `
    --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv `
    --split-file hdfs_sim/music/processed/playlist_split.tsv `
    --popularity hdfs_sim/music/analytics/track_popularity_train/part-00000 `
    --titles hdfs_sim/music/processed/playlist_titles.tsv `
    --out hdfs_sim/music/analytics/app_index

$alsBundle = 'outputs/als_local/models/als_implicit'
$alsTarget = 'hdfs_sim/music/analytics/als'
New-Item -ItemType Directory -Force $alsTarget | Out-Null
Copy-Item "$alsBundle/item_factors.npy", "$alsBundle/items.tsv", "$alsBundle/metadata.json" -Destination $alsTarget -Force
```

Chỉ mục app chưa có trên máy hiện tại nên cần dựng trước khi build Docker.
LightGBM là tùy chọn: chỉ chép `reranker_lgbm.txt` nếu đã train lại trên đúng
bộ dữ liệu/split này; không chép model cũ sang chỉ mục mới.

**4. Build và mở ứng dụng:**

```powershell
docker compose up --build -d
```

Mở http://localhost:8000/health: trường `models` phải có `als`,
`n_als_tracks` lớn hơn 0. Mở http://localhost:8501 và chọn mô hình ALS khi
gọi API, hoặc xem ô ALS trong trang so sánh. API và dashboard đều được mount
bundle từ cùng thư mục. Khi chỉ đổi bundle, dùng:

```powershell
docker compose restart api dashboard
```

Chi tiết cache/checkpoint và các CLI riêng: [Chạy ALS local](run_als_local.md).

## Chạy dịch vụ

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
chép `item_factors.npy`, `items.tsv`, `metadata.json` vào đó (hoặc định dạng cũ
`item_factors.npy` + `track_uris.txt`, xem `docs/api.md`) rồi khởi động lại API/dashboard.

## Tài nguyên

Đo trên Docker Desktop (Windows 11, WSL2):

- Image 1,89GB (Python và thư viện khoảng 1,3GB, chỉ mục 530MB). Build lần đầu vài phút, các lần sau chỉ
  chép lại phần code đã đổi.
- `docker compose up` đến khi cả hai service chạy: khoảng 20 giây (API nạp chỉ mục, healthcheck báo sẵn sàng).
- RAM: API khoảng 400MB; dashboard khoảng 55MB khi chưa ai mở, tăng thêm vài trăm MB khi mở trang
  "So sánh mô hình" (trang này tự đọc chỉ mục, không qua API).
- Một request `/recommend` với mô hình LightGBM: khoảng 0,15–0,4 giây.
