#!/usr/bin/env bash
# Chạy job MapReduce "track popularity".
#
# MẶC ĐỊNH chạy ở chế độ MÔ PHỎNG LOCAL (cat | mapper | sort | reducer) — đây là
# cách chuẩn để TEST logic mapper/reducer trước khi chạy thật trên Hadoop, vì
# Hadoop Streaming về bản chất cũng làm đúng 3 bước này (map -> shuffle/sort -> reduce),
# chỉ khác là chạy phân tán trên nhiều máy.
#
# Khi đã có Hadoop thật (ví dụ cài qua Docker trên máy cá nhân), đổi USE_HADOOP=1
# và chỉnh HADOOP_STREAMING_JAR cho đúng đường dẫn.

set -euo pipefail

# Tự chọn python3 hoặc python, nhưng KIỂM TRA CHẠY THỬ THẬT (không chỉ xem có
# trong PATH hay không) — trên Windows, "python3" trong PATH có thể là shim giả
# của Microsoft Store (AppData\Local\Microsoft\WindowsApps\python3), chạy không
# ra gì trong pipe dù lệnh "command -v" vẫn thấy nó tồn tại.
PYTHON_BIN=""
for candidate in python3 python; do
  if command -v "$candidate" >/dev/null 2>&1; then
    out="$("$candidate" -c "print(1)" 2>/dev/null || true)"
    if [ "$out" = "1" ]; then
      PYTHON_BIN="$candidate"
      break
    fi
  fi
done
if [ -z "$PYTHON_BIN" ]; then
  echo "LOI: khong tim duoc python3/python nao chay duoc that su." >&2
  echo "Thu chay tay: python3 -c \"print(1)\"  va  python -c \"print(1)\"  de xem ban nao hoat dong." >&2
  exit 1
fi
echo "Dung Python: $PYTHON_BIN ($(command -v "$PYTHON_BIN"))"

USE_HADOOP="${USE_HADOOP:-0}"
INPUT="${1:-../../hdfs_sim/music/processed/playlist_tracks_flat.tsv}"
OUTPUT_DIR="${2:-../../hdfs_sim/music/analytics/track_popularity}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

if [ "$USE_HADOOP" = "1" ]; then
  echo "== Chạy job THẬT qua Hadoop Streaming =="
  HADOOP_STREAMING_JAR="${HADOOP_STREAMING_JAR:-/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-3.3.6.jar}"
  HDFS_INPUT="${HDFS_INPUT:-/music/processed/playlist_tracks_flat.tsv}"
  HDFS_OUTPUT="${HDFS_OUTPUT:-/music/analytics/track_popularity}"

  hdfs dfs -rm -r -f "$HDFS_OUTPUT" || true
  hadoop jar "$HADOOP_STREAMING_JAR" \
    -D mapreduce.job.reduces=4 \
    -files "$SCRIPT_DIR/mapper.py,$SCRIPT_DIR/reducer.py" \
    -mapper "$PYTHON_BIN mapper.py" \
    -reducer "$PYTHON_BIN reducer.py" \
    -input "$HDFS_INPUT" \
    -output "$HDFS_OUTPUT"

  echo "Kết quả nằm trên HDFS tại: $HDFS_OUTPUT"
  hdfs dfs -cat "$HDFS_OUTPUT/part-*" | sort -t$'\t' -k2,2 -nr | head -20
else
  echo "== Chạy MÔ PHỎNG LOCAL (test logic mapper/reducer) =="
  OUT_FILE="$OUTPUT_DIR/part-00000"
  cat "$INPUT" \
    | "$PYTHON_BIN" "$SCRIPT_DIR/mapper.py" 2> "$OUTPUT_DIR/mapper.stderr.log" \
    | sort -t$'\t' -k1,1 \
    | "$PYTHON_BIN" "$SCRIPT_DIR/reducer.py" 2> "$OUTPUT_DIR/reducer.stderr.log" \
    > "$OUT_FILE"

  echo "Đã ghi kết quả vào: $OUT_FILE"
  echo ""
  echo "Top 10 bài phổ biến nhất (theo số playlist chứa bài):"
  sort -t$'\t' -k2,2 -nr "$OUT_FILE" | head -10 | awk -F'\t' '{printf "  %-5s playlists  %s - %s\n", $2, $4, $3}'

  if [ -s "$OUTPUT_DIR/mapper.stderr.log" ] || [ -s "$OUTPUT_DIR/reducer.stderr.log" ]; then
    echo ""
    echo "CẢNH BÁO: có log lỗi/skip, xem $OUTPUT_DIR/*.stderr.log"
  fi
fi
