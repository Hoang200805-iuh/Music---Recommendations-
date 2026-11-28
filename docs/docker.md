# Chạy bằng Docker

Cả hệ thống (API + giao diện) chạy bằng **một lệnh** từ thư mục gốc repo:

```bash
docker compose up --build
```

Rồi mở:

- Giao diện: http://localhost:8501 (trang "So sánh mô hình" và "Gợi ý qua API")
- API: http://localhost:8000/docs (thử request trên trình duyệt), http://localhost:8000/health

Dừng: `Ctrl+C`, hoặc `docker compose down` nếu chạy nền (`docker compose up --build -d`).

## Lần đầu trên máy mới: chuẩn bị dữ liệu

Dữ liệu MPD không được chia sẻ công khai (`docs/mpd_terms_of_use.md`) nên không có trên git. Lần chạy
đầu, service `setup` tự dựng chỉ mục của app từ một trong hai nguồn:

1. **Gói dữ liệu của nhóm** (cách nhanh, khoảng 10–15 phút): giải nén vào **thư mục gốc repo** để có đủ 4 file

   ```
   hdfs_sim/music/processed/playlist_tracks_flat.tsv
   hdfs_sim/music/processed/playlist_split.tsv
   hdfs_sim/music/processed/playlist_titles.tsv
   hdfs_sim/music/analytics/track_popularity_train/part-00000
   ```

   Cùng 4 file thì chỉ mục dựng ra giống hệt từng byte trên mọi máy (kể cả 500 playlist mẫu của tab
   "Thử trên playlist có thật"), nên giao diện và gợi ý giống hệt nhau.
2. **Dữ liệu MPD gốc** (khoảng 30 phút): đặt các file `mpd.slice.*.json` vào `data/raw/`, hoặc chỉ đường dẫn
   bằng `MPD_DATA_DIR` (`$env:MPD_DATA_DIR="D:\mpd\data"` trong PowerShell) rồi chạy lệnh trên.

Các lần sau `setup` bỏ qua ngay vì chỉ mục đã nằm trong volume `music_data`. Nếu đổi sang bộ dữ liệu hoặc
cách chia khác, `setup` tự nhận ra (số playlist train trong chỉ mục khác file chia tập) và dựng lại; muốn
xóa hẳn: `docker compose down -v`.

Mô hình **LightGBM** (`models/reranker_lgbm.txt`) có sẵn trên git, đã huấn luyện trên đúng cách chia của
4 file trên (550.900 / 118.050 / 118.050 playlist), nên không cần huấn luyện lại. File chỉ chứa các cây
quyết định, không chứa bài hát hay playlist nào.

## Thêm mô hình ALS (không bắt buộc)

Image Docker chỉ phục vụ API/dashboard, không chứa Spark/Java để train. ALS được train một lần bên ngoài
Docker trên **cùng 4 file đầu vào** ở trên (để so được với các mô hình khác), rồi chép bundle vào
`hdfs_sim/music/analytics/als/`. Chi tiết cài đặt, cache/checkpoint: [Chạy ALS local](run_als_local.md).
Không chạy `src/prepare_compact_mpd.py` khi đã dùng gói dữ liệu của nhóm (bước đó tạo bộ dữ liệu khác).

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

Lệnh này huấn luyện trên toàn bộ **550.900 playlist train**. Số 1.000 và 10.000
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


**3. Chép bundle ALS rồi chạy lại:**

```powershell
$alsBundle = 'outputs/als_local/models/als_implicit'
$alsTarget = 'hdfs_sim/music/analytics/als'
New-Item -ItemType Directory -Force $alsTarget | Out-Null
Copy-Item "$alsBundle/item_factors.npy", "$alsBundle/items.tsv", "$alsBundle/metadata.json" -Destination $alsTarget -Force
docker compose up --build -d      # lần đầu; nếu đang chạy thì: docker compose restart api dashboard
```

Mở http://localhost:8000/health: trường `models` phải có `als`, `n_als_tracks` lớn hơn 0. Trang
"So sánh mô hình" có thêm ô ALS; trang "Gợi ý qua API" có thêm mô hình `als`. Bundle được gắn vào cả API
và dashboard lúc chạy, không cần build lại image.

## Gồm những gì

| Service | Cổng | Việc |
|---|---|---|
| `setup` | | Chuẩn bị chỉ mục trong volume `music_data` (`docker/setup.sh`), rồi thoát |
| `api` | 8000 | `uvicorn api.main:app`, có healthcheck `/health`; chạy sau khi `setup` xong |
| `dashboard` | 8501 | `streamlit run dashboard/app.py`; chạy sau khi API sẵn sàng, gọi API qua `http://api:8000` |

`setup` (`docker/setup.sh`) làm theo thứ tự:

1. Volume đã có chỉ mục khớp dữ liệu: bỏ qua.
2. Máy đã dựng sẵn chỉ mục khớp dữ liệu ở `hdfs_sim/music/analytics/app_index/`: chép vào volume.
3. Máy có 4 file đầu vào trong `hdfs_sim/music/`: dựng chỉ mục từ đó.
4. Chỉ có dữ liệu MPD gốc: chạy pipeline như README (làm phẳng → chia tập → MapReduce mapper/sort/reducer
   trên train → tên playlist → dựng chỉ mục), xong thì xóa file làm phẳng tạm (3,7GB).

Cả ba service dùng chung một image (`Dockerfile`, Python 3.12, thư viện trong `docker/requirements.txt`
cùng phiên bản với lúc huấn luyện), không có Spark. Chỉ mục nằm trong volume chứ không gắn thẳng thư mục
trên máy, vì API đọc nó bằng memory-map: đọc qua thư mục chia sẻ từ Windows/macOS vào container chậm
hơn nhiều.

## Chạy không dùng Docker

```bash
pip install -r requirements.txt
python -m recommender.app_index --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv \
    --split-file hdfs_sim/music/processed/playlist_split.tsv \
    --popularity hdfs_sim/music/analytics/track_popularity_train/part-00000 \
    --titles hdfs_sim/music/processed/playlist_titles.tsv --out hdfs_sim/music/analytics/app_index
uvicorn api.main:app --port 8000                              # cửa sổ 1
streamlit run dashboard/app.py --server.address 127.0.0.1     # cửa sổ 2
```

## Tài nguyên

Đo trên Docker Desktop (Windows 11, WSL2, máy ảo Docker 16 CPU, 7,9GB RAM):

- Image khoảng 1GB (Python và thư viện, không có dữ liệu).
- Máy mới với gói dữ liệu của nhóm: `setup` dựng chỉ mục mất 11 phút (phần lớn là đọc file 3,7GB
  qua thư mục chia sẻ của Docker Desktop), RAM cao nhất 1,4GB; chỉ mục dựng ra giống hệt từng byte
  với chỉ mục trên máy đã dùng để đánh giá.
- Máy đã có chỉ mục: `docker compose up --build` đến khi cả hệ thống chạy khoảng 40 giây
  (chép 530MB chỉ mục vào volume 15 giây); các lần sau khoảng 20 giây.
- RAM khi chạy: API khoảng 400MB; dashboard khoảng 55MB khi chưa ai mở, tăng thêm vài trăm MB khi mở trang
  "So sánh mô hình" (trang này tự đọc chỉ mục, không qua API).
- Một request `/recommend` với mô hình LightGBM: khoảng 0,15–0,4 giây.
