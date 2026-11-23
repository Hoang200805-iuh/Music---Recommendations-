#!/usr/bin/env bash
# Chuẩn bị chỉ mục cho API và dashboard, chạy tự động trước khi hai service đó khởi động.
#
# Chỉ mục nằm trong volume Docker (đọc memory-map nhanh), ở $INDEX. Ba trường hợp:
#   1. Volume đã có chỉ mục: bỏ qua.
#   2. Máy đã dựng sẵn chỉ mục (hdfs_sim/music/analytics/app_index, gắn vào $SEED): chép vào volume.
#   3. Chưa có: chạy cả pipeline từ dữ liệu MPD gốc (gắn vào $MPD), giống các bước trong README:
#      làm phẳng -> chia tập -> MapReduce đếm độ phổ biến trên train -> tên playlist -> dựng chỉ mục.
# Mô hình LightGBM lấy từ models/reranker_lgbm.txt trong repo (không cần huấn luyện lại).
set -euo pipefail

DATA=/app/hdfs_sim/music
INDEX=$DATA/analytics/app_index
SEED=/seed/app_index
MPD=/data/mpd
MODEL=/app/models/reranker_lgbm.txt
start=$(date +%s)
log() { echo "[setup $(( $(date +%s) - start ))s] $*"; }

if [ -f "$INDEX/test_meta.json" ] && [ -f "$INDEX/uris.bin" ]; then
  log "Chỉ mục đã có trong volume, bỏ qua."
elif [ -f "$SEED/test_meta.json" ] && [ -f "$SEED/uris.bin" ]; then
  log "Chép chỉ mục đã dựng sẵn trên máy vào volume..."
  rm -rf "$INDEX.tmp" && mkdir -p "$INDEX.tmp"
  cp -r "$SEED/." "$INDEX.tmp/"
  rm -rf "$INDEX" && mv "$INDEX.tmp" "$INDEX"
else
  if ! ls "$MPD"/mpd.slice.*.json >/dev/null 2>&1; then
    echo "LỖI: chưa có chỉ mục và không thấy dữ liệu MPD (mpd.slice.*.json) trong thư mục gắn vào $MPD." >&2
    echo "Tải MPD rồi đặt các file slice vào data/raw/ (hoặc đặt MPD_DATA_DIR=<thư mục slice>), chạy lại." >&2
    exit 1
  fi
  log "Dựng chỉ mục từ dữ liệu MPD gốc ($(ls "$MPD"/mpd.slice.*.json | wc -l) slice), mất khoảng 30 phút..."
  P=$DATA/processed
  A=$DATA/analytics
  mkdir -p "$P" "$A/track_popularity_train"

  log "1/5 Làm phẳng JSON thành pid, bài, tên bài, nghệ sĩ"
  python src/flatten_for_mapreduce.py --in "$MPD" --out "$P/playlist_tracks_flat.tsv"

  log "2/5 Chia train/val/test 70/15/15 theo pid"
  python -m recommender.evaluation.split_playlists --in "$P/playlist_tracks_flat.tsv" --out "$P/playlist_split.tsv"

  log "3/5 MapReduce đếm độ phổ biến trên train (mapper -> sort -> reducer)"
  python -m recommender.evaluation.split_playlists --in "$P/playlist_tracks_flat.tsv" \
      --out "$P/playlist_split.tsv" --emit train \
    | python src/mapreduce/mapper.py \
    | LC_ALL=C sort -t$'\t' -k1,1 -S 1G -T "$P" \
    | python src/mapreduce/reducer.py > "$A/track_popularity_train/part-00000"

  log "4/5 Tên playlist"
  python -m recommender.hybrid.playlist_titles --in "$MPD" --out "$P/playlist_titles.tsv"

  log "5/5 Dựng chỉ mục"
  rm -rf "$INDEX.tmp"
  python -m recommender.app_index --flat "$P/playlist_tracks_flat.tsv" --split-file "$P/playlist_split.tsv" \
      --popularity "$A/track_popularity_train/part-00000" --titles "$P/playlist_titles.tsv" --out "$INDEX.tmp"
  rm -rf "$INDEX" && mv "$INDEX.tmp" "$INDEX"
  # File làm phẳng (khoảng 3,7GB) chỉ cần để dựng chỉ mục
  rm -f "$P/playlist_tracks_flat.tsv"
fi

if [ ! -f "$INDEX/reranker_lgbm.txt" ] && [ -f "$MODEL" ]; then
  log "Thêm mô hình LightGBM"
  cp "$MODEL" "$INDEX/"
fi
log "Xong: $(du -sh "$INDEX" | cut -f1) trong $INDEX"
