"""Track routes: GET /tracks/search, dùng cho ô tìm bài tự gợi ý (autocomplete) của giao diện."""
import numpy as np
from fastapi import APIRouter, Query, Request
from pydantic import BaseModel

from api.services.recommendation_service import split_name

router = APIRouter()


class TrackMatch(BaseModel):
    track_uri: str
    track_name: str
    artist_name: str
    n_playlists: int


@router.get("/tracks/search", response_model=list[TrackMatch])
def search_tracks(request: Request, q: str = Query(min_length=1, max_length=100),
                  limit: int = Query(default=10, ge=1, le=50)):
    """Bài có tên bài hoặc tên nghệ sĩ chứa q (không phân biệt hoa thường), phổ biến nhất trước."""
    service = request.app.state.models
    indptr = service.cooccurrence.col_indptr
    results = []
    for item in service.names.search(q, limit):
        artist, track = split_name(service.names[item])
        results.append({"track_uri": service.uris[item], "track_name": track, "artist_name": artist,
                        "n_playlists": int(np.int64(indptr[item + 1]) - np.int64(indptr[item]))})
    return results
