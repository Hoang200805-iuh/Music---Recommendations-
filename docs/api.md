# API

API gợi ý nhạc viết bằng FastAPI (`api/`). Mọi artifact được nạp **một lần lúc khởi động**,
không dùng Spark khi chạy.

## Chạy

```bash
pip install -r requirements.txt
uvicorn api.main:app --host 0.0.0.0 --port 8000      # hoặc: bash scripts/run_api.sh
```

Tài liệu tương tác (thử request ngay trên trình duyệt): http://localhost:8000/docs

## Artifact được nạp

| Biến môi trường | Mặc định | Nội dung |
|---|---|---|
| `MUSIC_APP_INDEX` | `hdfs_sim/music/analytics/app_index` | Chỉ mục từ `python -m recommender.app_index`: bảng popularity (số đếm MapReduce), ma trận train để tính Co-occurrence, tên playlist, URI và tên bài; mô hình LightGBM `reranker_lgbm.txt` nếu có |
| `MUSIC_ALS_DIR` | `hdfs_sim/music/analytics/als` | ALS đã xuất (không bắt buộc), xem bên dưới |

Ma trận lớn được đọc qua memory-map: API khởi động khoảng 5 giây, dùng khoảng 420MB RAM thực tế
trên toàn bộ tập train (1,7 triệu bài). Chỉ mục dựng trước khi có file URI cần ghi thêm một lần:

```bash
python -m recommender.app_index --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv \
    --split-file hdfs_sim/music/processed/playlist_split.tsv --out hdfs_sim/music/analytics/app_index --uris-only
```

Co-occurrence không lưu thành bảng cặp bài (quá lớn: hàng tỉ cặp) mà tính khi cần từ ma trận
playlist × bài đã xuất, cho kết quả y hệt công thức.

### Định dạng ALS đã xuất

Thư mục `MUSIC_ALS_DIR` gồm:

- `item_factors.npy`: float32, m × r, vector của m bài (`itemFactors` của `ALSModel`)
- `track_uris.txt`: m dòng, URI Spotify của từng hàng
- `als_meta.json` (không bắt buộc): `{"reg_param": 0.1, "alpha": 1.0}`, tham số lúc huấn luyện

Xuất một lần từ PySpark khi huấn luyện xong (`track_index` là bảng `track_id → track_uri` dùng khi đánh số bài):

```python
rows = model.itemFactors.join(track_index, model.itemFactors.id == track_index.track_id) \
                        .select("track_uri", "features").collect()
np.save("item_factors.npy", np.array([r.features for r in rows], dtype=np.float32))
Path("track_uris.txt").write_text("\n".join(r.track_uri for r in rows) + "\n")
```

Playlist mới chưa có vector nên API "gấp vào" bằng một bước ALS phản hồi ngầm (giữ nguyên vector các bài,
giải vector của playlist từ các bài seed). Chưa có thư mục này thì mô hình `als` không được nạp và
`/health` báo trong `missing_models`.

## GET /health

```json
{
  "status": "ok",
  "default_model": "rerank",
  "models": {"popularity": "...", "cooccurrence": "...", "cooccurrence_tuned": "...", "hybrid": "...", "rerank": "..."},
  "missing_models": ["als"],
  "n_tracks": 1692609,
  "n_train_playlists": 550900,
  "n_als_tracks": 0,
  "load_seconds": 5.0,
  "uptime_seconds": 12.3
}
```

## POST /recommend

Request (cần ít nhất một trong `seed_tracks` và `title`):

| Trường | Kiểu | Mặc định | Ý nghĩa |
|---|---|---|---|
| `seed_tracks` | list[str] | `[]` | URI Spotify các bài đã có trong playlist |
| `title` | str | `null` | Tên playlist |
| `n` | int (1–500) | 10 | Số bài gợi ý |
| `model` | str | mô hình mặc định | `popularity`, `cooccurrence`, `cooccurrence_tuned`, `hybrid`, `rerank`, `als` |

```bash
curl -X POST http://localhost:8000/recommend -H "Content-Type: application/json" \
  -d '{"seed_tracks": ["spotify:track:0UaMYEvWZi0ZqiDOoHU3YI"], "title": "workout", "n": 3}'
```

```json
{
  "model": "rerank",
  "n_seed_tracks_used": 1,
  "unknown_seed_tracks": [],
  "title_matched_playlists": 4505,
  "recommendations": [
    {"rank": 1, "track_uri": "spotify:track:...", "track_name": "Work It", "artist_name": "Missy Elliott",
     "score": 1.277, "fallback": false,
     "reasons": [
       {"type": "co_occurrence", "text": "Hay đi cùng “Missy Elliott – Lose Control (feat. Ciara & Fat Man Scoop)”: cùng có trong 1.182 playlist"},
       {"type": "similar_playlists", "text": "Cùng gu với playlist của bạn: có trong 16/100 playlist giống nhất"},
       {"type": "title", "text": "Cùng gu với các playlist tên “workout”: có trong 100/4.505 playlist"}
     ]},
    ...
  ]
}
```

- `unknown_seed_tracks`: bài seed không có trong tập train, bị bỏ qua.
- `score` tùy mô hình: số đếm MapReduce (`popularity`), điểm Co-occurrence, điểm Hybrid (0–1 mỗi phần),
  điểm LightGBM (chỉ để so thứ tự), tích vô hướng (`als`). `score = null, fallback = true`: bài được bù từ
  danh sách phổ biến vì mô hình không đủ bài có điểm.
- `reasons`: lý do gợi ý, tính từ dữ liệu train (`api/services/explain_service.py`), lý do mạnh trước:
  - `co_occurrence` "Hay đi cùng bài X": bài seed cùng nằm trong nhiều playlist nhất với bài được gợi ý;
  - `similar_playlists` "Cùng gu với playlist của bạn": có trong 100 playlist train giống playlist đang gợi ý nhất;
  - `title` "Cùng gu với các playlist tên …": có trong các playlist train cùng tên;
  - `popular` "Phổ biến": bài được bù từ danh sách phổ biến, mô hình `popularity`, hoặc không có lý do nào ở trên.
- Lỗi: thiếu cả seed lẫn tên, hoặc `n` ngoài 1–500 → 422; tên mô hình không có → 400.

Thời gian một request trên laptop: `popularity` khoảng 0,01 giây, Co-occurrence 0,2 giây, `hybrid` 0,3 giây,
`rerank` 0,8 giây.

## GET /tracks/search

Tìm bài theo tên bài hoặc tên nghệ sĩ (không phân biệt hoa thường), phổ biến nhất trước. Dùng cho ô tìm bài
tự gợi ý khi gõ của giao diện.

```bash
curl "http://localhost:8000/tracks/search?q=wonderwall&limit=3"
```

```json
[
  {"track_uri": "spotify:track:...", "track_name": "Wonderwall - Remastered", "artist_name": "Oasis", "n_playlists": 9575},
  {"track_uri": "spotify:track:...", "track_name": "Wonderwall", "artist_name": "Ryan Adams", "n_playlists": 1001},
  {"track_uri": "spotify:track:...", "track_name": "Wonderwall", "artist_name": "Boyce Avenue", "n_playlists": 124}
]
```

## Giao diện gọi API

Trang "Gợi ý qua API" của dashboard (`dashboard/pages/5_API_Recommend.py`) chỉ gọi API, không tự nạp mô hình:
nhập tên playlist, thêm bài bằng ô tìm kiếm tự gợi ý (gói `streamlit-searchbox`), chọn mô hình, xem Top-10 kèm lý do.

```bash
uvicorn api.main:app --port 8000                                  # cửa sổ 1
streamlit run dashboard/app.py --server.address 127.0.0.1         # cửa sổ 2; đổi địa chỉ API bằng MUSIC_API_URL
```
