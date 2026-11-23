# Một image cho cả API (FastAPI) và dashboard (Streamlit); docker-compose.yml chọn lệnh chạy.
# Chỉ mục app (hdfs_sim/music/analytics/app_index, kể cả mô hình LightGBM) được chép vào image:
# đọc memory-map qua thư mục chia sẻ từ Windows/macOS vào container rất chậm.
FROM python:3.12-slim-bookworm

# LightGBM cần thư viện OpenMP
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    MUSIC_APP_INDEX=/app/hdfs_sim/music/analytics/app_index \
    MUSIC_ALS_DIR=/app/als
WORKDIR /app

COPY docker/requirements.txt docker/requirements.txt
RUN pip install --no-cache-dir -r docker/requirements.txt

# Dữ liệu lớn, ít đổi: chép trước code để đổi code không phải chép lại 530MB
COPY hdfs_sim/music/analytics/app_index hdfs_sim/music/analytics/app_index
COPY recommender recommender
COPY api api
COPY dashboard dashboard

RUN useradd --create-home app && mkdir -p /app/als && chown app /app/als
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
