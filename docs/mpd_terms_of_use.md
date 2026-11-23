# Điều khoản sử dụng và trích dẫn — Million Playlist Dataset (MPD)

Tóm tắt từ điều khoản gốc của cuộc thi AIcrowd "Spotify Million Playlist Dataset Challenge"
(https://aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/challenge_rules),
nơi bộ dữ liệu được phát hành lần đầu. Bản Kaggle (himanshuwagh/spotify-million) là bản sao,
nên các điều khoản gốc vẫn áp dụng. Đây là tóm tắt, không phải tư vấn pháp lý — đọc lại trang
gốc nếu cần chắc chắn.

## Những điều cần tuân thủ trong project môn học

- **Chỉ dùng cho mục đích nghiên cứu/học tập.** Spotify chỉ cấp quyền sử dụng dữ liệu "trong phạm vi cần thiết" cho mục đích đã nêu (ở đây là đồ án môn Big Data), không cấp quyền nào khác.
- **Không dùng cho mục đích thương mại.** Không bán, không dùng dữ liệu (hay dữ liệu suy ra từ nó) để kiếm lợi.
- **Không phân phối lại dữ liệu gốc.** Không public lại file JSON gốc lên repo công khai, không chia sẻ cho người ngoài nhóm/môn học.
- **Không cố định danh người dùng.** Không được tìm cách suy luận ra danh tính người tạo playlist.
- **Mã nguồn làm ra kết quả nên công khai được.** Nếu nộp báo cáo/kết quả dựa trên dữ liệu này, phần code xử lý nên để public (ví dụ GitHub), nhưng **không kèm theo dữ liệu gốc**.

## Áp dụng cho repo của nhóm

- `.gitignore` phải loại trừ toàn bộ `data/raw/` và `data/sample/` (nếu sample lấy từ dữ liệu thật) — không commit file JSON gốc lên Git.
- Trong README của repo, ghi rõ: "Dataset không được đính kèm do điều khoản sử dụng của Spotify; tải về theo hướng dẫn tại [Kaggle link]."
- Báo cáo nộp môn học không public lên nơi ngoài phạm vi lớp học.

## Trích dẫn

Trang challenge không nêu yêu cầu trích dẫn bắt buộc cho riêng bộ dữ liệu, nhưng nên trích bài báo mô tả cuộc thi khi viết báo cáo:

> Chen, C.-W., Lamere, P., Schedl, M., & Zamani, H. (2018). Recsys Challenge 2018: Automatic Music Playlist Continuation. In *Proceedings of the 12th ACM Conference on Recommender Systems* (RecSys '18). ACM.
> [arXiv:1810.01520](https://arxiv.org/abs/1810.01520)

## Nguồn

- [Spotify Million Playlist Dataset Challenge — Challenge Rules (AIcrowd)](https://aicrowd.com/challenges/spotify-million-playlist-dataset-challenge/challenge_rules)
- [Chen et al., 2018 — Recsys Challenge 2018 report (arXiv)](https://export.arxiv.org/pdf/1810.01520)
