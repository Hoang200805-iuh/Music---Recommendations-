# Dataset

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
