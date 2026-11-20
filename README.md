# Music Recommendation Big Data

Project structure for large-scale music ingestion, processing, recommendation, API, and dashboard workflows.

---

# API, giao diện và thí nghiệm hiệu năng

| Việc | Trạng thái | Code |
|---|---|---|
| FastAPI: `POST /recommend`, `GET /health` (thêm `GET /tracks/search` cho ô tìm bài) | Xong | `api/`, tài liệu `docs/api.md` |
| Giao diện gọi API: tên playlist + chọn bài tự gợi ý, Top-10 kèm lý do | Xong | `dashboard/pages/5_API_Recommend.py` |
| Thí nghiệm hiệu năng xây bảng mô hình: 10% / 100% slice, 1 / 4 core | Xong | `src/benchmark_build.py` |

## API

API nạp artifact đã xuất một lần lúc khởi động (chỉ mục memory-map, mô hình LightGBM, ALS nếu đã xuất),
không dùng Spark khi chạy: khởi động khoảng 5 giây, khoảng 420MB RAM. Chi tiết request/response,
định dạng xuất ALS: `docs/api.md`.

```bash
uvicorn api.main:app --port 8000          # tài liệu tương tác: http://localhost:8000/docs
```

Mỗi bài gợi ý có lý do tính từ dữ liệu train: **hay đi cùng bài X** (bài seed cùng nằm trong nhiều playlist
nhất), **cùng gu với playlist của bạn** (có trong 100 playlist train giống nhất), **cùng gu với các playlist
cùng tên**, **phổ biến**.

## Giao diện gọi API

Trang "Gợi ý qua API" của dashboard chỉ gọi API: nhập tên playlist, thêm bài bằng ô tìm kiếm tự gợi ý khi gõ,
chọn mô hình, xem Top-10 kèm lý do.

```bash
uvicorn api.main:app --port 8000                              # cửa sổ 1
streamlit run dashboard/app.py --server.address 127.0.0.1     # cửa sổ 2, chọn trang "Gợi ý qua API"
```

## Thí nghiệm hiệu năng: thời gian xây bảng mô hình

**Cách đo.** Xây các bảng mà API và app dùng, đọc thẳng từ file slice JSON gốc của MPD:
bảng popularity (đếm số lượt mỗi bài, như job MapReduce), ma trận playlist × bài dạng CSR và CSC
(Co-occurrence tính điểm từ đây) và bảng tên playlist đã chuẩn hóa. Pipeline theo kiểu MapReduce:

- **map** (song song): mỗi slice đọc JSON, đếm bài, lấy danh sách bài và tên của từng playlist;
- **reduce** (tuần tự): gộp số đếm, đánh số bài, nối các cặp (playlist, bài);
- **dựng ma trận** (tuần tự): CSR, CSC, xếp hạng độ phổ biến.

Số core bị giới hạn bằng CPU affinity (tiến trình con thừa hưởng), chọn các core vật lý khác nhau, nên
"1 core" thật sự chỉ dùng 1 core. 10% = 79 slice lấy đều (cứ 10 slice lấy 1). Mỗi cấu hình chạy trong
tiến trình riêng, đo RAM của cả cây tiến trình. 10% chạy 3 lần (file đã nằm trong cache), 100% chạy 2 lần
(25GB lớn hơn RAM nên lần nào cũng đọc lại từ đĩa). Máy: AMD Ryzen 7 6800H (8 core / 16 luồng), 16GB RAM,
SSD, Windows 11, Python 3.14. Mọi cấu hình cho ra đúng cùng số bài, số ô ma trận và số đếm.

| Dữ liệu | Core | Thời gian (giây) | Map + reduce | Reduce tuần tự | Dựng ma trận | Playlist/giây | Tăng tốc | Hiệu suất | RAM cao nhất |
|---|---|---|---|---|---|---|---|---|---|
| 10%: 79 slice, 79.000 playlist, 598.062 bài | 1 | 49,4 (47,1–51,5) | 48,5 | 9,8 | 0,9 | 1.598 | 1,00× | 100% | 302 MB |
| 10% | 4 | 15,8 (15,3–16,9) | 15,0 | 13,1 | 0,9 | 4.989 | **3,12×** | 78% | 669 MB |
| 100%: 787 slice, 787.000 playlist, 2.014.098 bài | 1 | 434,6 (419,9–449,3) | 428,6 | 85,5 | 6,1 | 1.811 | 1,00× | 100% | 1.347 MB |
| 100% | 4 | 132,8 (128,4–137,2) | 126,8 | 124,4 | 6,1 | 5.926 | **3,27×** | 82% | 1.928 MB |

Thời gian là trung bình các lần chạy (trong ngoặc: nhanh nhất – chậm nhất). Hiệu suất = tăng tốc / số core.
Số liệu từng lần chạy: `reports/benchmark_build_runs.csv`; bảng tổng hợp: `reports/benchmark_build_summary.csv`.

**Nhận xét**

- **Dữ liệu tăng 10 lần, thời gian tăng khoảng 8,5–8,8 lần**, gần tuyến tính; số bài chỉ tăng 3,4 lần vì
  phần lớn bài phổ biến đã xuất hiện ngay trong 10% dữ liệu.
- **4 core nhanh hơn 1 core khoảng 3,1–3,3 lần** (hiệu suất 78–82%), vì phần nặng nhất là đọc JSON
  nằm ở pha map, chạy song song được.
- **Giới hạn là pha reduce tuần tự:** với 4 core, gần như toàn bộ thời gian (124 trên 133 giây ở 100%) là
  tiến trình chính đánh số bài và nối dữ liệu; các tiến trình map đọc xong nhanh hơn tốc độ reduce xử lý.
  Thêm core cũng không nhanh hơn nữa (định luật Amdahl). Muốn tăng tiếp phải chia nhỏ cả reduce, như
  Hadoop/Spark làm (chia bài theo khóa cho nhiều reducer).
- Reduce ở 4 core (124 giây) chậm hơn ở 1 core (85 giây) vì tiến trình chính phải tranh 4 core với 4 tiến
  trình map và chờ nhận dữ liệu từ chúng.
- **RAM tăng khi thêm core** (1,35GB → 1,93GB ở 100%) vì mỗi tiến trình map giữ một slice đã đọc,
  và bảng đánh số bài lớn dần theo số bài.
- Thí nghiệm dùng pipeline Python mô phỏng MapReduce chứ không dùng Spark (máy chưa cài Spark).

```bash
python src/benchmark_build.py --data <thư mục mpd.slice.*.json> --fraction 0.1 --cores 4 \
    --out reports/benchmark_build_runs.csv
```

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
| Thử RP3beta (bước đi ngẫu nhiên) | Xong, không tốt hơn | `recommender/evaluation/tune_rp3beta.py` |
| User-User kNN, trộn vào Hybrid | Xong | `recommender/neighbors/`, `recommender/evaluation/tune_blend.py` |
| Mô hình hai giai đoạn: Hybrid chọn ứng viên, LightGBM xếp hạng lại | Xong | `recommender/rerank/` |
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
- Mỗi k chấm trên 2.000 playlist test, mọi mô hình chấm trên **cùng** các playlist.
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
5. **RP3beta** (thử, không giữ): Co-occurrence chia thêm cho độ dài playlist, tương đương bước đi ngẫu nhiên
   bài → playlist → bài. Thử 30 bộ tham số trên val, bộ tốt nhất chỉ hơn Co-occurrence tinh chỉnh
   0,0006 NDCG@10 (nằm trong sai số), nên không đánh giá trên test.
6. **User-User kNN:** tìm 100 playlist train giống seed nhất (độ giống `|seed ∩ p| / sqrt(|p|)`) rồi gợi ý
   bài trong đó. Chạy riêng thì ngang Co-occurrence; trộn vào Hybrid với trọng số 0,5 (chọn trên val
   trong 30 bộ) tăng nhẹ Recall@50, rõ nhất ở k = 25.
7. **Hybrid + LightGBM xếp hạng lại (mô hình hai giai đoạn):** Hybrid + User-User chọn 300 bài ứng viên,
   rồi LightGBM (LambdaRank) xếp lại theo 22 đặc trưng: điểm và thứ hạng của từng mô hình con, điểm
   Co-occurrence chỉ từ 5 bài seed cuối, số bài seed cùng nghệ sĩ, độ phổ biến của bài so với các bài seed,
   độ đa dạng nghệ sĩ của playlist… Huấn luyện trên val (6.000 playlist mỗi k, 20% giữ lại để dừng sớm,
   được 657 cây). Ở k = 0 giữ nguyên thứ tự gốc vì xếp lại không giúp gì. Cách làm theo đội vô địch
   RecSys Challenge 2018.

## Kết quả trên tập test

Cả 6 mô hình chấm trên **cùng 2.000 playlist test mỗi k** (`reports/week2_results_all_models.csv`).
Ở k = 0 không có bài seed nên 3 mô hình đầu đều gợi ý bài phổ biến, 3 mô hình sau đều dùng tên playlist,
nên kết quả trong mỗi nhóm giống nhau. Số in đậm là cao nhất của từng cột.

**NDCG@10** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0375 ± 0,0045 | 0,0364 ± 0,0044 | 0,0342 ± 0,0043 | 0,0334 ± 0,0042 |
| Co-occurrence | 0,0375 ± 0,0045 | 0,1908 ± 0,0110 | 0,2153 ± 0,0115 | 0,1814 ± 0,0104 |
| Co-occurrence tinh chỉnh | 0,0375 ± 0,0045 | 0,2306 ± 0,0114 | 0,2404 ± 0,0114 | 0,1929 ± 0,0103 |
| Hybrid | **0,1581 ± 0,0103** | 0,2440 ± 0,0117 | 0,2461 ± 0,0114 | 0,1952 ± 0,0103 |
| Hybrid + User-User | **0,1581 ± 0,0103** | 0,2451 ± 0,0115 | 0,2502 ± 0,0115 | 0,1950 ± 0,0104 |
| **Hybrid + LightGBM** | **0,1581 ± 0,0103** | **0,2577 ± 0,0116** | **0,2588 ± 0,0117** | **0,2187 ± 0,0109** |

**Recall@10** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0062 ± 0,0009 | 0,0058 ± 0,0008 | 0,0056 ± 0,0008 | 0,0064 ± 0,0014 |
| Co-occurrence | 0,0062 ± 0,0009 | 0,0401 ± 0,0033 | 0,0488 ± 0,0038 | 0,0455 ± 0,0041 |
| Co-occurrence tinh chỉnh | 0,0062 ± 0,0009 | 0,0465 ± 0,0032 | 0,0525 ± 0,0035 | 0,0450 ± 0,0037 |
| Hybrid | **0,0308 ± 0,0027** | 0,0492 ± 0,0033 | 0,0535 ± 0,0035 | 0,0451 ± 0,0036 |
| Hybrid + User-User | **0,0308 ± 0,0027** | 0,0499 ± 0,0034 | 0,0553 ± 0,0037 | 0,0460 ± 0,0038 |
| **Hybrid + LightGBM** | **0,0308 ± 0,0027** | **0,0523 ± 0,0034** | **0,0582 ± 0,0038** | **0,0542 ± 0,0044** |

**Recall@50** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0238 ± 0,0020 | 0,0231 ± 0,0019 | 0,0223 ± 0,0020 | 0,0229 ± 0,0025 |
| Co-occurrence | 0,0238 ± 0,0020 | 0,1138 ± 0,0067 | 0,1451 ± 0,0074 | 0,1441 ± 0,0076 |
| Co-occurrence tinh chỉnh | 0,0238 ± 0,0020 | 0,1321 ± 0,0066 | 0,1540 ± 0,0069 | 0,1418 ± 0,0071 |
| Hybrid | 0,0917 ± 0,0061 | 0,1405 ± 0,0067 | 0,1564 ± 0,0069 | 0,1427 ± 0,0071 |
| Hybrid + User-User | 0,0917 ± 0,0061 | 0,1425 ± 0,0067 | 0,1618 ± 0,0072 | 0,1490 ± 0,0074 |
| **Hybrid + LightGBM** | **0,0926 ± 0,0061** | **0,1492 ± 0,0069** | **0,1690 ± 0,0075** | **0,1620 ± 0,0077** |

**Precision@10** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0353 ± 0,0042 | 0,0343 ± 0,0041 | 0,0319 ± 0,0039 | 0,0307 ± 0,0038 |
| Co-occurrence | 0,0353 ± 0,0042 | 0,1731 ± 0,0102 | 0,2007 ± 0,0108 | 0,1720 ± 0,0099 |
| Co-occurrence tinh chỉnh | 0,0353 ± 0,0042 | 0,2105 ± 0,0107 | 0,2237 ± 0,0107 | 0,1819 ± 0,0097 |
| Hybrid | **0,1487 ± 0,0097** | 0,2230 ± 0,0109 | 0,2283 ± 0,0107 | 0,1839 ± 0,0097 |
| Hybrid + User-User | **0,1487 ± 0,0097** | 0,2241 ± 0,0108 | 0,2331 ± 0,0108 | 0,1835 ± 0,0098 |
| **Hybrid + LightGBM** | **0,1487 ± 0,0097** | **0,2359 ± 0,0108** | **0,2412 ± 0,0110** | **0,2060 ± 0,0103** |

**Hit Rate@10** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,1780 ± 0,0168 | 0,1725 ± 0,0166 | 0,1635 ± 0,0162 | 0,1620 ± 0,0162 |
| Co-occurrence | 0,1780 ± 0,0168 | 0,5540 ± 0,0218 | 0,6075 ± 0,0214 | 0,5745 ± 0,0217 |
| Co-occurrence tinh chỉnh | 0,1780 ± 0,0168 | 0,6460 ± 0,0210 | 0,6915 ± 0,0202 | 0,6230 ± 0,0212 |
| Hybrid | **0,4860 ± 0,0219** | 0,6710 ± 0,0206 | 0,7015 ± 0,0201 | 0,6250 ± 0,0212 |
| Hybrid + User-User | **0,4860 ± 0,0219** | 0,6780 ± 0,0205 | 0,7050 ± 0,0200 | 0,6180 ± 0,0213 |
| **Hybrid + LightGBM** | **0,4860 ± 0,0219** | **0,7105 ± 0,0199** | **0,7215 ± 0,0197** | **0,6610 ± 0,0208** |

**R-precision** (± nửa khoảng tin cậy 95%)

| Mô hình | k = 0 | k = 1 | k = 5 | k = 25 |
|---|---|---|---|---|
| Popularity | 0,0334 ± 0,0023 | 0,0326 ± 0,0022 | 0,0305 ± 0,0021 | 0,0283 ± 0,0020 |
| Co-occurrence | 0,0334 ± 0,0023 | 0,1178 ± 0,0061 | 0,1409 ± 0,0065 | 0,1318 ± 0,0060 |
| Co-occurrence tinh chỉnh | 0,0334 ± 0,0023 | 0,1379 ± 0,0060 | 0,1531 ± 0,0061 | 0,1352 ± 0,0057 |
| Hybrid | 0,1042 ± 0,0060 | 0,1477 ± 0,0061 | 0,1557 ± 0,0061 | 0,1351 ± 0,0056 |
| Hybrid + User-User | 0,1042 ± 0,0060 | 0,1492 ± 0,0061 | 0,1597 ± 0,0062 | 0,1382 ± 0,0058 |
| **Hybrid + LightGBM** | **0,1050 ± 0,0059** | **0,1571 ± 0,0063** | **0,1660 ± 0,0064** | **0,1488 ± 0,0061** |

Khoảng tin cậy ở trên là của từng mô hình riêng lẻ. Vì mọi mô hình chấm trên cùng playlist, so sánh
**từng cặp playlist** cho khoảng tin cậy hẹp hơn nhiều. Mức tăng của Hybrid + LightGBM so với Hybrid + User-User
(mô hình chọn ứng viên cho nó), `reports/week2_results_rerank.csv`:

| Độ đo | k = 1 | k = 5 | k = 25 |
|---|---|---|---|
| NDCG@10 | +0,0125 ± 0,0046 (+5%) | +0,0086 ± 0,0036 (+3%) | +0,0237 ± 0,0052 (+12%) |
| Recall@10 | +0,0024 ± 0,0015 (+5%) | +0,0029 ± 0,0013 (+5%) | +0,0082 ± 0,0029 (+18%) |
| Recall@50 | +0,0068 ± 0,0021 (+5%) | +0,0072 ± 0,0021 (+4%) | +0,0131 ± 0,0029 (+9%) |
| R-precision | +0,0079 ± 0,0021 (+5%) | +0,0062 ± 0,0016 (+4%) | +0,0106 ± 0,0022 (+8%) |
| Hit Rate@10 | +0,0325 ± 0,0120 (+5%) | +0,0165 ± 0,0104 (+2%) | +0,0430 ± 0,0145 (+7%) |

- Mọi mức tăng đều lớn hơn khoảng tin cậy, nên LightGBM tốt hơn thật chứ không phải do may.
- Ở k = 0 LightGBM giữ nguyên thứ tự gốc (thử xếp lại ở k = 0 làm kết quả kém đi). Recall@50 và R-precision
  ở k = 0 lệch Hybrid một chút do cách cắt các bài cùng điểm ở cuối danh sách 300 ứng viên.
- Đặc trưng quan trọng nhất (`reports/week2_rerank_feature_importance.csv`): thứ hạng và điểm của mô hình
  gốc (65%), tỉ lệ playlist cùng tên chứa bài (5%), thứ hạng User-User (4%), điểm Co-occurrence từ 5 bài seed
  cuối (4%), thứ hạng Co-occurrence (3,5%).
- Dữ liệu huấn luyện quan trọng: lần đầu chỉ dùng 1.500 playlist val mỗi k, mô hình dừng ở 23 cây và
  chỉ tăng NDCG@10 1–5%; tăng lên 6.000 playlist mỗi k (và cây nhỏ hơn, học chậm hơn) cho kết quả trên.
- Hạn chế: LightGBM chỉ xếp lại 300 ứng viên, bài đúng không lọt vào đó thì vẫn không gợi ý được.
- Các lần chấm trước (`reports/week2_results_with_tuned.csv`: 3 mô hình đầu trên 10.000 playlist mỗi k;
  `reports/week2_results_hybrid.csv`: 4 mô hình đầu trên 3.000 playlist) cho kết quả khớp với bảng trên,
  với khoảng tin cậy hẹp hơn.

### Nhận xét

- Co-occurrence vượt xa Popularity (gấp 5–7 lần ở mọi độ đo) ngay khi có 1 bài seed.
- Hybrid + LightGBM tốt nhất ở mọi độ đo và mọi k có bài seed; ở k = 0 ngang Hybrid.
- Bản tinh chỉnh tốt hơn công thức gốc ở cả 3 mức k có seed (NDCG@10 +20% / +14% / +7%),
  trên 10.000 playlist khoảng tin cậy không chồng nhau. Kết quả trên test khớp với val, nên tham số không bị chọn "may".
- Tên playlist giúp nhiều nhất ở k = 0: NDCG@10 và Recall@50 tăng khoảng 4 lần, tỉ lệ playlist có ít nhất
  1 gợi ý đúng tăng từ 17,8% lên 48,6%. Khi đã có seed thì tên chỉ giúp thêm một chút (rõ nhất ở k = 1),
  mức tăng nằm trong sai số của từng mô hình nên chưa khẳng định chắc được.
- Với 10 gợi ý, Co-occurrence tinh chỉnh và Hybrid trúng ít nhất 1 bài ở 62–70% playlist khi có seed,
  Hybrid + LightGBM ở 66–72%.
- **Recall@10 thấp là bình thường:** playlist test giấu trung bình vài chục bài trong khi chỉ được gợi ý
  10 bài, nên Recall@10 tối đa thường chỉ khoảng 0,1–0,3. Precision@10 khoảng 0,2 nghĩa là trung bình
  2 trong 10 gợi ý trúng đúng bài người dùng đã chọn, giữa 1,7 triệu bài.
- Biết trước nhiều bài hơn không làm điểm tăng mãi: điểm cao nhất ở k = 5, giảm ở k = 25
  (đáp án ít đi và các bài cuối playlist khác gu hơn).
- Thử "tin playlist chứa nhiều bài seed hơn" (`playlist_exponent`) không cải thiện, nên giữ mức 1.
- Các biến thể của cùng một ý tưởng "đếm đồng xuất hiện" (RP3beta, User-User kNN) đều cho kết quả ngang
  nhau: mô hình loại này đã gần chạm trần. Muốn tăng rõ phải thêm loại thông tin mới (nghệ sĩ, độ phổ biến,
  thứ tự bài) và học cách kết hợp, như mô hình hai giai đoạn: NDCG@10 ở k = 1 tăng từ 0,231
  (Co-occurrence tinh chỉnh) lên 0,258 (+12%), ở k = 25 từ 0,193 lên 0,219 (+13%).

### Lưu ý về dữ liệu

- Số đếm MapReduce là **số lượt** bài xuất hiện, không phải số playlist chứa bài: 114.500 bài có hai
  con số này khác nhau (playlist lặp lại một bài). Popularity dùng đúng số đếm MapReduce theo đề;
  Co-occurrence dùng số playlist phân biệt.
- File `mpd.slice.807000-807999.json` bị hỏng nên bị bỏ qua (giống tuần 1).

## Giao diện thử mô hình

App Streamlit so sánh 4 mô hình trên **toàn bộ** tập train: Co-occurrence, Co-occurrence tinh chỉnh,
Hybrid và Hybrid + LightGBM (Popularity không hiện thành ô riêng vì kết quả thấp, chỉ dùng để bù bài
khi thiếu ứng viên). Dữ liệu được dựng một lần thành các file nhị phân rồi đọc bằng memory-map, nên app
chỉ dùng khoảng 650MB RAM; mỗi lần gợi ý của LightGBM mất khoảng 0,3 giây.

- Tab **Tự chọn bài:** tìm bài, thêm vào playlist, nhập tên playlist, xem 4 mô hình gợi ý.
- Ô LightGBM ghi thứ hạng của mỗi bài ở mô hình gốc ("↑ từ #14": bài thứ 14 được kéo lên).
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

# 5. (Thử thêm, chọn trên val) RP3beta và trộn User-User kNN
python -m recommender.evaluation.tune_rp3beta $COMMON --out reports/week2_tuning_rp3beta_val.csv
python -m recommender.evaluation.tune_blend $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out reports/week2_tuning_blend_val.csv

# 6. Mô hình hai giai đoạn: tạo đặc trưng (khoảng 2 giờ), huấn luyện LightGBM và chấm trên test (vài phút)
python -m recommender.rerank.build_features $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out-dir hdfs_sim/music/analytics/rerank_features
python -m recommender.rerank.train_reranker --features-dir hdfs_sim/music/analytics/rerank_features \
    --out reports/week2_results_rerank.csv --importance-out reports/week2_rerank_feature_importance.csv \
    --model-out hdfs_sim/music/analytics/rerank_features/reranker_lgbm.txt
# Chấm cả 6 mô hình trên cùng 2.000 playlist test mỗi k (khoảng 2 giờ)
python -m recommender.evaluation.evaluate_models $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out reports/week2_results_all_models.csv --tuned 0.5 0.25 10 --hybrid 1.0 0.1 \
    --rerank hdfs_sim/music/analytics/rerank_features/reranker_lgbm.txt --max-playlists 2000

# 7. Dựng chỉ mục cho app (một lần, khoảng 2–3 phút), chép mô hình LightGBM vào rồi mở app
python -m recommender.app_index $COMMON --titles hdfs_sim/music/processed/playlist_titles.tsv \
    --out hdfs_sim/music/analytics/app_index
cp hdfs_sim/music/analytics/rerank_features/reranker_lgbm.txt hdfs_sim/music/analytics/app_index/
streamlit run dashboard/app.py

# Test
pytest tests/test_evaluation.py tests/test_recommender.py tests/test_app_index.py tests/test_dashboard.py
```

Thời gian và RAM trên laptop 16GB: đánh giá 10.000 playlist mỗi k (3 mô hình) mất khoảng 2,5 giờ,
3.000 playlist mỗi k (4 mô hình) khoảng 1,5 giờ, RAM cao nhất 2,1GB; tinh chỉnh trên val khoảng 30–70 phút
(trộn User-User khoảng 1,5 giờ). Tạo đặc trưng cho LightGBM dùng khoảng 1,6GB RAM, huấn luyện khoảng 1,1GB.

## File kết quả

| File | Nội dung |
|---|---|
| `reports/week2_results_all_models.csv` | **Cả 6 mô hình trên cùng 2.000 playlist test mỗi k** (bảng chính) |
| `reports/week2_baseline_results.csv` | Popularity và Co-occurrence gốc trên test |
| `reports/week2_results_with_tuned.csv` | Thêm Co-occurrence tinh chỉnh và Precision@10 |
| `reports/week2_results_hybrid.csv` | Cả 4 mô hình, thêm Hit Rate@10 và R-precision |
| `reports/week2_tuning_val.csv` | 45 bộ tham số Co-occurrence trên val |
| `reports/week2_tuning_hybrid_val.csv` | 20 bộ tham số Hybrid trên val |
| `reports/week2_tuning_rp3beta_val.csv` | 30 bộ tham số RP3beta trên val |
| `reports/week2_tuning_blend_val.csv` | 30 cách trộn User-User kNN vào Hybrid trên val |
| `reports/week2_results_rerank.csv` | Hybrid + User-User trước và sau khi LightGBM xếp lại, trên test, kèm chênh lệch từng cặp |
| `reports/week2_rerank_feature_importance.csv` | Độ quan trọng (gain) của 22 đặc trưng LightGBM |
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
