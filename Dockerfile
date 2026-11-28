# Một image cho cả API (FastAPI), dashboard (Streamlit) và bước setup dựng chỉ mục;
# docker-compose.yml chọn lệnh chạy cho từng service.
# Chỉ mục không nằm trong image (dựng từ dữ liệu MPD, không được chia sẻ lại): service setup
# tạo nó trong volume music_data, xem docker/setup.sh.
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

COPY recommender recommender
COPY api api
COPY dashboard dashboard
COPY src src
COPY models/reranker_lgbm.txt models/reranker_lgbm.txt
COPY docker/setup.sh docker/setup.sh

# Thư mục dữ liệu thuộc user app: volume trống gắn vào đây sẽ nhận quyền này
RUN useradd --create-home app && mkdir -p /app/hdfs_sim/music /app/als \
    && chown -R app /app/hdfs_sim /app/als && sed -i 's/\r$//' docker/setup.sh
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
