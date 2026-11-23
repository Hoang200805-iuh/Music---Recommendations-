# Chạy ALS trên máy local

Tại thư mục dự án, cài dependency trong môi trường riêng (Python 3.12,
Java 17+ cho Spark 4):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-als.txt
```

Sau đó dùng script:

Nếu dữ liệu đầu vào là bộ Parquet đã làm sạch trong `music_clean_compact`,
chuẩn bị một lần trước khi train:

```powershell
python -m src.prepare_compact_mpd --input music_clean_compact --out hdfs_sim/music
```

Bước này ghép URI/tên bài từ `track_mapping` và `tracks`, sắp bài theo `pid, pos`,
chia playlist 70/15/15 với seed 42 và chỉ đếm popularity trên tập train.
SQLite sắp xếp trên ổ đĩa trong `outputs/prepare_compact` để hạn chế RAM.
Chạy lại với dữ liệu không đổi sẽ dùng lại các file đã xuất. Không trộn các
chỉ mục/model cũ với dữ liệu hoặc cách chia mới; dựng lại chỉ mục và train lại
trên cùng bộ đầu vào này. Lệnh sẽ từ chối ghi đè dữ liệu đã có không khớp.

```powershell
cd 'D:\2026-2027\Bigdata\CK\Music---Recommendations-'
python scripts/run_als_local.py
```

Mặc định thư mục làm việc là:

```text
D:\2026-2027\Bigdata\CK\Music---Recommendations-\outputs\als_local
```

Script sẽ kiểm tra input, ghi/đọc temp và khởi động Spark trước khi xử lý dữ liệu
lớn. Script dùng trực tiếp 4 file processed trong `hdfs_sim/music` của dự án,
không nhân bản dữ liệu. Source runtime nhỏ được lưu vào thư mục làm việc.
Spark shuffle, TSV tạm, cache, checkpoint và kết quả nằm trong
`outputs/als_local` của chính dự án. Tất cả đường dẫn dữ liệu đều trên D.

Nếu chủ động thêm `--copy-inputs`, việc copy có tiến độ và file `.part` để tiếp
tục nếu bị ngắt; input đúng kích thước/mtime sẽ được dùng lại. Cache train/val/test
giúp không phải đọc TSV nhiều lần. Mỗi trial ALS hoàn tất có checkpoint; trial
đang dở phải train lại. Chạy lại đúng lệnh và work-dir để tiếp tục.

Các tham số thống kê vẫn giữ nguyên: implicitPrefs=True, rating=1, maxIter=10;
rank → regParam → alpha, chỉ tune trên val; chấm cùng 5 mô hình trên cùng test.
Mặc định dùng local[4], Spark driver 4g, 1.000 val và 10.000 test mỗi mức seed.

Mỗi cấu hình ALS chạy trong một Python/JVM riêng. Khi trial xong, tiến trình
thoát và thư mục shuffle tạm riêng của trial được dọn trước khi mở trial tiếp.
Cách này tránh tích lũy shuffle của nhiều lần fit trong cùng một JVM. Cache,
model và checkpoint ở bên ngoài thư mục tạm nên vẫn được dùng khi chạy lại.
Các tham số/thuật toán/độ đo được giữ nguyên. Fingerprint phiên bản 2 dùng danh
sách mã nguồn cố định, giống nhau khi chạy tại dự án hoặc source runtime. Sửa bộ
quản lý tiến trình, ghi báo cáo hay tuner cooccurrence/hybrid không làm mất cache
ALS. Đổi thuật toán, độ đo, dữ liệu, số playlist lấy mẫu hoặc num-blocks vẫn làm
thay đổi fingerprint.

Checkpoint phiên bản cũ chỉ được chuyển nếu fingerprint cũ được tái lập từ bản
mã nguồn đã biết, dữ liệu/cấu hình khớp, bundle đúng SHA-256, và metrics/bộ tham
số của bundle khớp checkpoint. Metadata và các trial cũ được sao lưu trong
`work-dir/checkpoints/legacy_backups` trước khi chuyển. File item factors và các
checkpoint gốc được giữ nguyên; không cần train lại trial đã được kiểm chứng.

Nếu đang mở CSV bằng Excel khiến file bị khóa, chương trình ghi báo cáo sang
file `.latest.csv` trong cùng thư mục và tiếp tục. Nếu file đó cũng bị khóa,
chương trình dùng tên có timestamp. Đường dẫn thực tế được in trong log.
Đóng CSV đang mở nếu muốn báo cáo cập nhật lại đúng tên mặc định.

Lệnh tuning trực tiếp cũng dùng cùng bộ chạy tách tiến trình với script local:

```powershell
python -m recommender.als.tune_als `
    --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv `
    --split-file hdfs_sim/music/processed/playlist_split.tsv `
    --popularity hdfs_sim/music/analytics/track_popularity_train/part-00000 `
    --out outputs/als_local/reports/als_tuning_val.csv `
    --model-out outputs/als_local/models/als_implicit `
    --cache-dir outputs/als_local/cache `
    --checkpoint-dir outputs/als_local/checkpoints `
    --temp-dir outputs/als_local/temporary/ratings `
    --master 'local[4]' --driver-memory 4g --num-blocks 10
```

Giữ cùng đường dẫn model/cache/checkpoint và cấu hình để dùng lại kết quả giữa
hai cách chạy. Nếu bỏ ba đường dẫn tùy chọn, CLI tạo chúng trong thư mục
`<model-out>.work` cạnh model. CLI tuning và CLI `evaluate_models` đều tự xử lý
CSV bị khóa và in đường dẫn báo cáo thực tế. Chỉ chạy tuning sẽ không chạy bảng
so sánh 5 mô hình trên test.

Chỉ kiểm tra môi trường, chưa copy file dữ liệu lớn hoặc train:

```powershell
python scripts/run_als_local.py --check-only
```

Trên Windows, runner cũng báo commit còn trống (khả năng cấp phát từ RAM +
pagefile) và dừng sớm nếu thấp hơn heap Spark + 2 GiB dự phòng. Ngưỡng này
không đảm bảo mọi trial đủ bộ nhớ; trạng thái hệ thống có thể thay đổi trong
lúc chạy. Nếu gặp `paging file is too small` / `errno=1455`, xem phần xử lý
bộ nhớ trong [hướng dẫn Docker/train](docker.md).

Kiểm thử bằng dữ liệu mẫu riêng:

```powershell
python scripts/run_als_local.py --smoke-test --work-dir outputs/als_smoke --master 'local[2]' --driver-memory 1g --num-blocks 2
```

Đổi thư mục làm việc hoặc giới hạn playlist đánh giá:

```powershell
python scripts/run_als_local.py --work-dir outputs/als_local --val-playlists 1000 --test-playlists 10000
```

Nếu chỉ muốn tuning trước, thêm `--tune-only`. Giảm số playlist đánh giá không
giảm dữ liệu train. Các file kết quả nằm trong `work-dir/reports`, model trong
`work-dir/models/als_implicit`, log trong `work-dir/pipeline.log`.

Script tự đặt các biến môi trường của phiên chạy. Temp nhỏ mặc định của user dùng cho
gateway Java; dữ liệu ratings/shuffle lớn có thư mục riêng trên D. JAVA_HOME được
lấy từ JVM thật, không suy ra từ Oracle launcher trong javapath.

## Nạp ALS vào API và dashboard

Bundle gồm `item_factors.npy`, `items.tsv`, `metadata.json`. API và dashboard
ghép từng hàng bằng URI, không phụ thuộc thứ tự ID của chỉ mục ứng dụng.
Máy chỉ phục vụ gợi ý không cần PySpark/Java. Hybrid và LightGBM vẫn dùng chỉ mục
và cấu hình hiện có. Khi chưa có bundle, các mô hình còn lại vẫn chạy.

Sau khi train thành công, mở hai terminal tại gốc project, kích hoạt `.venv`,
và đặt biến sau trong **cả hai terminal** trước khi chạy dịch vụ:

```powershell
$env:MUSIC_ALS_DIR = (Resolve-Path outputs/als_local/models/als_implicit).Path
# Terminal 1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# Terminal 2
python -m streamlit run dashboard/app.py
```

Kiểm tra `/health`: `models` có `als`, `n_als_tracks` lớn hơn 0. Gửi
`POST /recommend` với `model="als"` hoặc xem ô ALS ở trang so sánh.
Mặc định bundle được tìm trong `hdfs_sim/music/analytics/als`.
Docker dùng bind mount thư mục này như cấu hình hiện có; chép **cả ba file**
bundle vào đó rồi khởi động lại API/dashboard.

Luồng ALS cần bốn file input liệt kê trong `scripts/run_als_local.py`.
Lệnh chuẩn bị Parquet ở trên tạo đủ bốn file này. Smoke test dùng dữ liệu mẫu riêng.
