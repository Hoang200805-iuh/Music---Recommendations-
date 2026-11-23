# Tóm tắt kết quả Tuần 1 — Người A

*(Chạy thật trên máy, 2026-10-26, dataset: Kaggle himanshuwagh/spotify-million)*

## Dữ liệu

- Tải qua Kaggle, giải nén ra 788 file `mpd.slice.*.json`.
- **Phạm vi thực tế của bản Kaggle này: `pid` từ 0 đến 807999** (không phải đủ
  0–999999 / 1.000.000 playlist như bản MPD gốc).
- Trong phạm vi 808 slice khả dĩ, có 788 file, thiếu 20 file ở 2 khoảng:
  `9000-9999` và `81000-99999`.
- Lúc làm phẳng, phát hiện thêm 1 file hỏng (`mpd.slice.807000-807999.json`,
  lỗi JSON "Unterminated string" — khả năng do tải bị ngắt giữa chừng), đã loại
  trừ. Còn lại **787 file xử lý thành công**.
- Kết quả: **52.184.832 dòng "bài-trong-playlist"** sau khi làm phẳng.

## Job MapReduce — Track Popularity

- Viết theo chuẩn Hadoop Streaming (`mapper.py` / `reducer.py`), chạy test ở
  chế độ mô phỏng local (`cat | mapper | sort | reducer`) do môi trường chưa
  có Hadoop cluster thật.
- **Đối chiếu chéo với đếm bằng Python thuần (`collections.Counter`)** trên dữ
  liệu mẫu: khớp tuyệt đối 262/262 track_uri, 0 dòng lệch — xác nhận logic
  mapper/reducer đúng trước khi chạy trên dữ liệu thật.
- Chạy trên dữ liệu thật (52,184,832 dòng), kết quả top 10 bài phổ biến nhất
  theo số playlist chứa bài:

| Hạng | Số playlist | Nghệ sĩ | Bài hát |
|---|---|---|---|
| 1 | 36.543 | Kendrick Lamar | HUMBLE. |
| 2 | 34.128 | Drake | One Dance |
| 3 | 32.500 | DRAM | Broccoli (feat. Lil Yachty) |
| 4 | 32.419 | The Chainsmokers | Closer |
| 5 | 31.377 | Post Malone | Congratulations |
| 6 | 27.673 | Aminé | Caroline |
| 7 | 27.571 | KYLE | iSpy (feat. Lil Yachty) |
| 8 | 27.508 | Lil Uzi Vert | XO TOUR Llif3 |
| 9 | 27.429 | Migos | Bad and Boujee (feat. Lil Uzi Vert) |
| 10 | 27.128 | Khalid | Location |

Đây đều là các bản hit lớn giai đoạn 2016-2017, khớp với thời điểm MPD được
thu thập (2010-2017) — một dấu hiệu sanity-check tốt cho thấy pipeline đúng.

## Điều khoản sử dụng

- Chỉ dùng cho mục đích học tập/nghiên cứu, không thương mại, không phân phối
  lại dữ liệu gốc hay dữ liệu suy ra từ nó ra ngoài phạm vi môn học.
- Nguồn: [AIcrowd challenge rules](https://aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/challenge_rules),
  [Chen et al., 2018 (arXiv:1810.01520)](https://export.arxiv.org/pdf/1810.01520).
- Chi tiết đầy đủ: `docs/mpd_terms_of_use.md`.

## Hạn chế / ghi chú cho báo cáo

- Job MapReduce chạy ở chế độ mô phỏng local, chưa chạy trên Hadoop cluster
  thật (môi trường chưa cài Hadoop). Logic đã kiểm chứng đúng bằng đối chiếu
  chéo; có sẵn lệnh để chạy thật khi có Hadoop (`docs/setup_hadoop_spark_docker.md`).
- Bản dataset Kaggle chỉ có ~787.000/1.000.000 playlist so với MPD gốc — ảnh
  hưởng tới các con số thống kê tổng, cần ghi rõ khi so sánh với số liệu công
  bố của MPD gốc (nếu có).

## Bàn giao cho Tuần 2 (Người B)

Định dạng input đã thống nhất cho job MapReduce (và các bước sau):

```
pid <TAB> track_uri <TAB> track_name <TAB> artist_name
```

Khi có bảng Parquet `playlist_tracks` + `tracks` từ ETL PySpark, chỉ cần export
đúng 4 cột này (hoặc sửa mapper đọc thẳng từ Parquet) là tái sử dụng được toàn
bộ `mapper.py`/`reducer.py` hiện có.
