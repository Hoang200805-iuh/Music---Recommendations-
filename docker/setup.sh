#!/usr/bin/env bash
# Chuẩn bị chỉ mục cho API và dashboard, chạy tự động trước khi hai service đó khởi động.
#
# Chỉ mục nằm trong volume Docker (đọc memory-map nhanh), ở $INDEX. Thư mục hdfs_sim/music của máy
# được gắn chỉ-đọc vào $SEED. Theo thứ tự:
#   1. Volume đã có chỉ mục khớp dữ liệu: bỏ qua.
#   2. Máy đã dựng sẵn chỉ mục khớp dữ liệu (hdfs_sim/music/analytics/app_index): chép vào volume.
#   3. Máy có 4 file đầu vào (hdfs_sim/music/processed/*.tsv và track_popularity_train/part-00000,
#      ví dụ giải nén từ gói dữ liệu của nhóm): dựng chỉ mục từ đó, khoảng 3 phút.
#   4. Chỉ có dữ liệu MPD gốc (gắn vào $MPD): chạy cả pipeline như README, khoảng 30 phút.
# "Khớp dữ liệu": số playlist train trong chỉ mục bằng số playlist train của file chia tập (nếu có file
# này), để không dùng nhầm chỉ mục dựng từ bộ dữ liệu hoặc cách chia khác.
# Mô hình LightGBM lấy từ models/reranker_lgbm.txt trong repo (không cần huấn luyện lại).
set -euo pipefail

DATA=/app/hdfs_sim/music
INDEX=$DATA/analytics/app_index
SEED=/seed/music
MPD=/data/mpd
start=$(date +%s)
log() { echo "[setup $(( $(date +%s) - start ))s] $*"; }

FLAT=processed/playlist_tracks_flat.tsv
SPLIT=processed/playlist_split.tsv
TITLES=processed/playlist_titles.tsv
POPULARITY=analytics/track_popularity_train/part-00000

# Số playlist train của file chia tập trên máy (rỗng nếu không có file)
EXPECTED_TRAIN=""
if [ -f "$SEED/$SPLIT" ]; then
  EXPECTED_TRAIN=$(awk -F'\t' '$2 == "train"' "$SEED/$SPLIT" | wc -l)
  log "File chia tập trên máy: $EXPECTED_TRAIN playlist train"
fi

index_ok() {  # $1: thư mục chỉ mục
  [ -f "$1/test_meta.json" ] && [ -f "$1/uris.bin" ] || return 1
  [ -z "$EXPECTED_TRAIN" ] && return 0
  local n
  n=$(python -c "import json, sys; print(json.load(open(sys.argv[1]))['n_train_playlists'])" "$1/test_meta.json")
  if [ "$n" != "$EXPECTED_TRAIN" ]; then
    log "Chỉ mục ở $1 có $n playlist train, khác file chia tập ($EXPECTED_TRAIN): không dùng"
    return 1
  fi
}

build_index() {  # $1: thư mục chứa processed/ và analytics/
  rm -rf "$INDEX.tmp"
  python -m recommender.app_index --flat "$1/$FLAT" --split-file "$1/$SPLIT" \
      --popularity "$1/$POPULARITY" --titles "$1/$TITLES" --out "$INDEX.tmp"
  rm -rf "$INDEX" && mv "$INDEX.tmp" "$INDEX"
}

if index_ok "$INDEX"; then
  log "Chỉ mục đã có trong volume, bỏ qua."
elif index_ok "$SEED/analytics/app_index"; then
  log "Chép chỉ mục đã dựng sẵn trên máy vào volume..."
  rm -rf "$INDEX.tmp" && mkdir -p "$INDEX.tmp"
  cp -r "$SEED/analytics/app_index/." "$INDEX.tmp/"
  rm -rf "$INDEX" && mv "$INDEX.tmp" "$INDEX"
elif [ -f "$SEED/$FLAT" ] && [ -f "$SEED/$SPLIT" ] && [ -f "$SEED/$TITLES" ] && [ -f "$SEED/$POPULARITY" ]; then
  log "Dựng chỉ mục từ 4 file đầu vào trên máy (hdfs_sim/music), khoảng 3 phút..."
  mkdir -p "$DATA/analytics"
  build_index "$SEED"
elif ls "$MPD"/mpd.slice.*.json >/dev/null 2>&1; then
  log "Dựng chỉ mục từ dữ liệu MPD gốc ($(ls "$MPD"/mpd.slice.*.json | wc -l) slice), khoảng 30 phút..."
  mkdir -p "$DATA/processed" "$DATA/analytics/track_popularity_train"

  log "1/5 Làm phẳng JSON thành pid, bài, tên bài, nghệ sĩ"
  python src/flatten_for_mapreduce.py --in "$MPD" --out "$DATA/$FLAT"

  log "2/5 Chia train/val/test 70/15/15 theo pid"
  python -m recommender.evaluation.split_playlists --in "$DATA/$FLAT" --out "$DATA/$SPLIT"

  log "3/5 MapReduce đếm độ phổ biến trên train (mapper -> sort -> reducer)"
  python -m recommender.evaluation.split_playlists --in "$DATA/$FLAT" --out "$DATA/$SPLIT" --emit train \
    | python src/mapreduce/mapper.py \
    | LC_ALL=C sort -t$'\t' -k1,1 -S 1G -T "$DATA/processed" \
    | python src/mapreduce/reducer.py > "$DATA/$POPULARITY"

  log "4/5 Tên playlist"
  python -m recommender.hybrid.playlist_titles --in "$MPD" --out "$DATA/$TITLES"

  log "5/5 Dựng chỉ mục"
  build_index "$DATA"
  # File làm phẳng (khoảng 3,7GB) chỉ cần để dựng chỉ mục
  rm -f "$DATA/$FLAT"
else
  echo "LỖI: chưa có chỉ mục và không thấy dữ liệu để dựng. Cần một trong hai:" >&2
  echo "  - 4 file đầu vào trong hdfs_sim/music (giải nén gói dữ liệu của nhóm vào thư mục gốc repo), hoặc" >&2
  echo "  - các file MPD gốc mpd.slice.*.json trong data/raw/ (hoặc đặt MPD_DATA_DIR=<thư mục slice>)." >&2
  exit 1
fi
log "Xong: $(du -sh "$INDEX" | cut -f1) trong $INDEX (mô hình LightGBM: models/reranker_lgbm.txt của repo)"
