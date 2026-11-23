"""
API entry point.

Chạy (từ thư mục gốc repo):
    uvicorn api.main:app --host 0.0.0.0 --port 8000
Tài liệu tương tác: http://localhost:8000/docs. Xem thêm docs/api.md.

Mọi artifact (chỉ mục, mô hình LightGBM, ALS đã xuất) được nạp một lần lúc khởi động,
không dùng Spark khi chạy.
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from api.routes import recommend, tracks
from api.services.model_service import ModelService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cho phép test gán sẵn app.state.models để khỏi nạp chỉ mục thật
    if getattr(app.state, "models", None) is None:
        app.state.models = ModelService()
    yield


app = FastAPI(title="Music Recommendation API", version="1.0", lifespan=lifespan)
app.include_router(recommend.router)
app.include_router(tracks.router)
