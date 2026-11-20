"""Recommendation routes: POST /recommend và GET /health."""
import time

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, model_validator

from api.services.model_service import DESCRIPTIONS
from api.services.recommendation_service import UnknownModelError, recommend

router = APIRouter()
MAX_N = 500


class RecommendRequest(BaseModel):
    seed_tracks: list[str] = Field(default_factory=list, max_length=1000,
                                   description="URI Spotify của các bài đã có, ví dụ spotify:track:...")
    title: str | None = Field(default=None, max_length=200, description="Tên playlist")
    n: int = Field(default=10, ge=1, le=MAX_N, description="Số bài gợi ý")
    model: str | None = Field(default=None, description="Tên mô hình; bỏ trống thì dùng mô hình mặc định")

    @model_validator(mode="after")
    def needs_seed_or_title(self):
        if not self.seed_tracks and not (self.title and self.title.strip()):
            raise ValueError("Cần ít nhất một trong hai: seed_tracks hoặc title")
        return self


class Reason(BaseModel):
    type: str = Field(description="co_occurrence, similar_playlists, title hoặc popular")
    text: str


class Recommendation(BaseModel):
    rank: int
    track_uri: str
    track_name: str
    artist_name: str
    score: float | None = Field(description="Điểm của mô hình; null nếu bài được bù từ danh sách phổ biến")
    fallback: bool = Field(description="Bài được bù từ danh sách phổ biến vì mô hình không đủ bài có điểm")
    reasons: list[Reason] = Field(description="Lý do gợi ý, lý do mạnh trước")


class RecommendResponse(BaseModel):
    model: str
    n_seed_tracks_used: int = Field(description="Số bài seed có trong tập train (mô hình dùng được)")
    unknown_seed_tracks: list[str] = Field(description="Bài seed không có trong tập train, bị bỏ qua")
    title_matched_playlists: int = Field(description="Số playlist train cùng tên (đã chuẩn hóa)")
    recommendations: list[Recommendation]


@router.post("/recommend", response_model=RecommendResponse)
def post_recommend(body: RecommendRequest, request: Request):
    service = request.app.state.models
    try:
        return recommend(service, body.seed_tracks, body.title, body.n, body.model)
    except UnknownModelError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/health")
def health(request: Request):
    service = request.app.state.models
    return {
        "status": "ok",
        "default_model": service.default_model,
        "models": {name: DESCRIPTIONS[name] for name in service.models},
        "missing_models": service.missing,
        "n_tracks": service.meta["n_items"],
        "n_train_playlists": service.meta["n_train_playlists"],
        "n_als_tracks": service.n_als_tracks,
        "load_seconds": service.load_seconds,
        "uptime_seconds": round(time.time() - service.started_at, 1),
    }
