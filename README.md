# Music Recommendation Big Data

Project structure for large-scale music ingestion, processing, recommendation, API, and dashboard workflows.

---

# Tuần 2 — Mô hình gợi ý và đánh giá

Toàn bộ chạy trên dữ liệu MPD thật (787.000 playlist đọc được, 788 file slice, bỏ 1 file hỏng).

| Việc | Trạng thái | Code |
|---|---|---|
| Chia train/val/test 70/15/15 theo `pid` | Xong | `recommender/evaluation/split_playlists.py` |
| Độ đo Recall@10, Recall@50, NDCG@10 | Xong | `recommender/evaluation/{recall_at_k,ndcg_at_k}.py` |
| Độ đo bổ sung: Precision@10, Hit Rate@10, R-precision | Xong | `recommender/evaluation/{precision_at_k,hit_rate_at_k,r_precision}.py` |
| Mô hình 1: Popularity (từ số đếm MapReduce) | Xong | `recommender/popularity/popularity_recommender.py` |
| Mô hình 2: Co-occurrence (cosine) | Xong | `recommender/cooccurrence/cooccurrence_recommender.py` |
| Tinh chỉnh Co-occurrence trên val | Xong | `recommender/evaluation/tune_cooccurrence.py` |
| Mô hình Hybrid: Co-occurrence + tên playlist | Xong | `recommender/hybrid/` |
| Đánh giá trên test | Xong | `recommender/evaluation/evaluate_models.py` |
| Giao diện thử mô hình (Streamlit, 100% dữ liệu) | Xong | `dashboard/`, `recommender/app_index.py` |

## Cách đánh giá

- **Chia tập:** mỗi `pid` được gán ngẫu nhiên (seed 42) vào train/val/test theo tỉ lệ 70/15/15,
  được 550.900 / 118.050 / 118.050 playlist. Chỉ lưu file `pid → tập` (`playlist_split.tsv`) để tiết kiệm ổ đĩa.
- **Không rò rỉ dữ liệu:** job MapReduce đếm độ phổ biến được chạy lại **chỉ trên train**
  (`hdfs_sim/music/analytics/track_popularity_train/`), mô hình chỉ học từ train.
- **Seed và đáp án:** với mỗi playlist test, giữ k bài đầu làm seed (k = 0 chỉ biết tên, 1, 5, 25),
  các bài còn lại là đáp án bị giấu. Bài không có trong train vẫn nằm trong đáp án (không gợi ý được,
  đúng với thực tế).
- **Chọn tham số trên val, báo cáo trên test:** tập test chỉ dùng một lần ở bước cuối.
- Mỗi k chấm trên 10.000 playlist test, mọi mô hình chấm trên **cùng** các playlist.
  Số sau dấu ± là nửa khoảng tin cậy 95%.

## Các mô hình

1. **Popularity:** gợi ý các bài có số đếm MapReduce cao nhất, bỏ bài đã có trong seed.
2. **Co-occurrence (đúng công thức đề bài):** `score(j) = Σ_{i ∈ seed} count(i,j) / sqrt(count(i)·count(j))`.
   Ma trận đầy đủ X^T X quá lớn cho RAM nên điểm được tính khi cần bằng hai phép nhân ma trận thưa
   `X^T (X w)`, cho kết quả y hệt công thức.
3. **Co-occurrence tinh chỉnh:** công thức tổng quát
   `count(i,j) / (count(i)^a · count(j)^b)` và bỏ bài có `count(j) < m`. Thử 45 bộ tham số trên val,
   chọn **a = 0,5; b = 0,25; m = 10**: phạt bài phổ biến nhẹ hơn và bỏ bài quá hiếm.
4. **Hybrid:** cộng điểm Co-occurrence tinh chỉnh với điểm theo **tên playlist**
   (bài hay có trong các playlist train cùng tên, tên đã chuẩn hóa), mỗi loại đưa về thang 0–1,
   trọng số tên **0,1** (chọn trên val). Đây là mô hình duy nhất làm tốt ở k = 0.

## Kết quả trên tập test

10.000 playlist mỗi k (`reports/week2_results_with_tuned.csv`). Ở k = 0 không có seed nên cả 3 mô hình
đều gợi ý bài phổ biến và cho cùng kết quả.

**NDCG@10** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0384 ± 0,0021 | 0,0375 ± 0,0021 | 0,0349 ± 0,0020 | 0,0318 ± 0,0019 |
| Co-occurrence | 0,0384 ± 0,0021 | 0,1924 ± 0,0050 | 0,2157 ± 0,0052 | 0,1834 ± 0,0047 |
| **Co-occurrence tinh chỉnh** | 0,0384 ± 0,0021 | **0,2312 ± 0,0052** | **0,2454 ± 0,0052** | **0,1967 ± 0,0046** |

**Recall@10**

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0062 | 0,0061 | 0,0058 | 0,0061 |
| Co-occurrence | 0,0062 | 0,0403 | 0,0494 | 0,0460 |
| **Co-occurrence tinh chỉnh** | 0,0062 | **0,0470** | **0,0537** | **0,0463** |

**Recall@50**

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0238 | 0,0235 | 0,0231 | 0,0228 |
| Co-occurrence | 0,0238 | 0,1126 | 0,1439 | 0,1427 |
| **Co-occurrence tinh chỉnh** | 0,0238 | **0,1316** | **0,1553** | **0,1435** |

**Precision@10**

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0366 | 0,0357 | 0,0330 | 0,0300 |
| Co-occurrence | 0,0366 | 0,1770 | 0,2025 | 0,1727 |
| **Co-occurrence tinh chỉnh** | 0,0366 | **0,2119** | **0,2288** | **0,1845** |

Kết quả của mô hình Hybrid (kèm Hit Rate@10 và R-precision) trên test nằm ở
`reports/week2_results_hybrid.csv`. Trên val (1.000 playlist mỗi k), so với Co-occurrence tinh chỉnh:

| k | NDCG@10 | Recall@50 |
|---|---|---|
| 0 | 0,042 → **0,168** | 0,025 → **0,102** |
| 1 | 0,248 → **0,266** | 0,144 → **0,155** |
| 5 | 0,262 → **0,268** | 0,165 → 0,167 |
| 25 | 0,218 → 0,221 | 0,151 → 0,151 |

### Nhận xét

- Co-occurrence vượt xa Popularity (gấp 5–7 lần ở mọi độ đo) ngay khi có 1 bài seed.
- Bản tinh chỉnh tốt hơn công thức gốc ở cả 3 mức k có seed (NDCG@10 +20% / +14% / +7%),
  khoảng tin cậy không chồng nhau. Kết quả trên test khớp với val, nên tham số không bị chọn "may".
- Tên playlist giúp nhiều nhất ở k = 0 (gấp 4 lần), giúp ít khi đã có seed.
- **Recall@10 thấp là bình thường:** playlist test giấu trung bình vài chục bài trong khi chỉ được gợi ý
  10 bài, nên Recall@10 tối đa thường chỉ khoảng 0,1–0,3. Precision@10 khoảng 0,2 nghĩa là trung bình
  2 trong 10 gợi ý trúng đúng bài người dùng đã chọn, giữa 1,7 triệu bài.
- Biết trước nhiều bài hơn không làm điểm tăng mãi: điểm cao nhất ở k = 5, giảm ở k = 25
  (đáp án ít đi và các bài cuối playlist khác gu hơn).
- Thử "tin playlist chứa nhiều bài seed hơn" (`playlist_exponent`) không cải thiện, nên giữ mức 1.

### Lưu ý về dữ liệu

- Số đếm MapReduce là **số lượt** bài xuất hiện, không phải số playlist chứa bài: 114.500 bài có hai
  con số này khác nhau (playlist lặp lại một bài). Popularity dùng đúng số đếm MapReduce theo đề;
  Co-occurrence dùng số playlist phân biệt.
- File `mpd.slice.807000-807999.json` bị hỏng nên bị bỏ qua (giống tuần 1).

## Giao diện thử mô hình

App Streamlit so sánh 4 mô hình trên **toàn bộ** tập train. Dữ liệu được dựng một lần thành các
file nhị phân rồi đọc bằng memory-map, nên app chỉ dùng khoảng 650MB RAM.

- Tab **Tự chọn bài:** tìm bài, thêm vào playlist, nhập tên playlist, xem 4 mô hình gợi ý.
- Tab **Thử trên playlist có thật:** giấu một playlist test, chọn k (0, 1, 5, 25; thêm 50, 100 để thử),
  xem từng mô hình đoán trúng bao nhiêu bài (đánh dấu ✓) cùng NDCG@10, Precision@10, Recall@50, R-precision.

## Cách chạy lại

```bash
pip install -r requirements.txt

# 1. Chia tập (sau khi đã có playlist_tracks_flat.tsv từ tuần 1)
python -m recommender.evaluation.split_playlists --in hdfs_sim/music/processed/playlist_tracks_flat.tsv \
    --out hdfs_sim/music/processed/playlist_split.tsv

# 2. Chạy MapReduce chỉ trên train
python -m recommender.evaluation.split_playlists --in hdfs_sim/music/processed/playlist_tracks_flat.tsv \
    --emit train --out hdfs_sim/music/processed/playlist_split.tsv > train_flat.tsv
cat train_flat.tsv | python src/mapreduce/mapper.py | sort | python src/mapreduce/reducer.py \
    > hdfs_sim/music/analytics/track_popularity_train/part-00000

# 3. Tên playlist (đọc lại file MPD gốc)
python -m recommender.hybrid.playlist_titles --in data/raw --out hdfs_sim/music/processed/playlist_titles.tsv

# 4. Tinh chỉnh trên val, rồi đánh giá trên test
COMMON="--flat hdfs_sim/music/processed/playlist_tracks_flat.tsv --split-file hdfs_sim/music/processed/playlist_split.tsv \
    --popularity hdfs_sim/music/analytics/track_popularity_train/part-00000"
python -m recommender.evaluation.tune_cooccurrence $COMMON --out reports/week2_tuning_val.csv --max-playlists 1000
python -m recommender.evaluation.tune_hybrid $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out reports/week2_tuning_hybrid_val.csv
python -m recommender.evaluation.evaluate_models $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out reports/week2_results_hybrid.csv --tuned 0.5 0.25 10 --hybrid 1.0 0.1

# 5. Dựng chỉ mục cho app (một lần, khoảng 2–3 phút) rồi mở app
python -m recommender.app_index $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out hdfs_sim/music/analytics/app_index
streamlit run dashboard/app.py

# Test
pytest tests/test_evaluation.py tests/test_recommender.py tests/test_app_index.py tests/test_dashboard.py
```

Thời gian và RAM trên laptop 16GB: đánh giá 10.000 playlist mỗi k mất khoảng 2,5 giờ, RAM cao nhất 1,9GB;
tinh chỉnh trên val khoảng 30–70 phút.

## File kết quả

| File | Nội dung |
|---|---|
| `reports/week2_baseline_results.csv` | Popularity và Co-occurrence gốc trên test |
| `reports/week2_results_with_tuned.csv` | Thêm Co-occurrence tinh chỉnh và Precision@10 |
| `reports/week2_results_hybrid.csv` | Cả 4 mô hình, thêm Hit Rate@10 và R-precision |
| `reports/week2_tuning_val.csv` | 45 bộ tham số Co-occurrence trên val |
| `reports/week2_tuning_hybrid_val.csv` | 20 bộ tham số Hybrid trên val |
| `reports/eda_charts/` | Biểu đồ EDA (`src/eda_charts.py`) |

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
