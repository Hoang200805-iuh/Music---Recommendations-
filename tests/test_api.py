"""API tests: chỉ mục nhỏ của test_app_index, thêm ALS và LightGBM giả."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.services.model_service import ModelService
from recommender.als.load_als import ALSRecommender
from recommender.popularity.popularity_recommender import PopularityRecommender
from recommender.rerank.features import FEATURE_NAMES
from test_app_index import _build

# Bài a, b, c của chỉ mục nhỏ có id 0, 1, 2
FACTORS = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]], dtype=np.float32)


def _write_als(tmp_path, uris=("u:c", "u:a", "u:b", "u:unknown")):
    als_dir = tmp_path / "als"
    als_dir.mkdir()
    rows = {"u:a": FACTORS[0], "u:b": FACTORS[1], "u:c": FACTORS[2], "u:unknown": [5.0, 5.0]}
    np.save(als_dir / "item_factors.npy", np.array([rows[u] for u in uris], dtype=np.float32))
    (als_dir / "track_uris.txt").write_text("\n".join(uris) + "\n", encoding="utf-8")
    return als_dir


def _write_reranker(index_dir):
    lgb = pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    X = rng.random((200, len(FEATURE_NAMES)))
    data = lgb.Dataset(X, (X[:, 0] > 0.5).astype(int), group=[20] * 10)
    booster = lgb.train({"objective": "lambdarank", "verbose": -1, "min_data_in_leaf": 5}, data, num_boost_round=5)
    booster.save_model(str(index_dir / "reranker_lgbm.txt"))


@pytest.fixture
def client(tmp_path):
    _build(tmp_path)
    _write_reranker(tmp_path / "index")
    app.state.models = ModelService(tmp_path / "index", _write_als(tmp_path))
    with TestClient(app) as c:
        yield c
    app.state.models = None


def test_health_lists_loaded_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert set(body["models"]) == {"popularity", "cooccurrence", "cooccurrence_tuned", "hybrid", "rerank", "als"}
    assert body["default_model"] == "rerank" and body["missing_models"] == []
    assert body["n_tracks"] == 3 and body["n_als_tracks"] == 3


def test_recommend_returns_top_n_with_scores_and_names(client):
    body = client.post("/recommend", json={"seed_tracks": ["u:c"], "n": 2, "model": "cooccurrence"}).json()
    # Giống test_app_index: c chung playlist với b (điểm 1/√4) cao hơn a (1/√6)
    assert [r["track_uri"] for r in body["recommendations"]] == ["u:b", "u:a"]
    first = body["recommendations"][0]
    first.pop("reasons")  # kiểm riêng ở test_reasons_explain_each_recommendation
    assert first == {"rank": 1, "track_uri": "u:b", "track_name": "Bravo", "artist_name": "Bob",
                     "score": pytest.approx(0.5), "fallback": False}
    assert body["n_seed_tracks_used"] == 1 and body["unknown_seed_tracks"] == []


def test_unknown_seeds_are_reported_and_popularity_scores_are_counts(client):
    body = client.post("/recommend", json={"seed_tracks": ["u:zzz", "u:a"], "model": "popularity", "n": 2}).json()
    assert body["unknown_seed_tracks"] == ["u:zzz"] and body["n_seed_tracks_used"] == 1
    # Số đếm MapReduce: b = 2, c = 2 (a là seed nên bị bỏ)
    assert [(r["track_uri"], r["score"]) for r in body["recommendations"]] == [("u:b", 2.0), ("u:c", 2.0)]


def test_title_only_uses_title_and_default_model(client):
    body = client.post("/recommend", json={"title": "Rock!", "n": 3}).json()
    assert body["model"] == "rerank" and body["title_matched_playlists"] == 2
    # Playlist train tên "rock" chứa b hai lần, a và c mỗi bài một lần
    assert body["recommendations"][0]["track_uri"] == "u:b"
    assert all(r["score"] is not None for r in body["recommendations"])


def test_every_model_answers(client):
    for model in ("popularity", "cooccurrence", "cooccurrence_tuned", "hybrid", "rerank", "als"):
        body = client.post("/recommend", json={"seed_tracks": ["u:a"], "title": "rock", "n": 2, "model": model})
        assert body.status_code == 200, model
        uris = [r["track_uri"] for r in body.json()["recommendations"]]
        assert len(uris) == 2 and "u:a" not in uris, model


def test_bad_requests(client):
    assert client.post("/recommend", json={}).status_code == 422
    assert client.post("/recommend", json={"title": "  "}).status_code == 422
    assert client.post("/recommend", json={"title": "rock", "n": 0}).status_code == 422
    response = client.post("/recommend", json={"title": "rock", "model": "nope"})
    assert response.status_code == 400 and "nope" in response.json()["detail"]


def test_als_fold_in_prefers_tracks_close_to_seed():
    popularity = PopularityRecommender({0: 3, 1: 2, 2: 1})
    model = ALSRecommender(FACTORS, [0, 1, 2], popularity, reg_param=0.01)
    # Seed a = (1, 0): b = (0.9, 0.1) gần a hơn c = (0, 1)
    assert model.recommend([0], 2) == [1, 2]
    assert model.recommend([], 2) == [0, 1]  # không có seed: bài phổ biến


def test_als_rows_follow_uris_not_file_order(tmp_path):
    _build(tmp_path)
    service = ModelService(tmp_path / "index", _write_als(tmp_path))
    als = service.models["als"]
    # File xuất xếp c, a, b; sau khi ghép URI, hàng của bài a (id 0) phải là vector của a
    assert np.allclose(als.factors[als.rows[0]], FACTORS[0])
    assert np.allclose(als.factors[als.rows[2]], FACTORS[2])


def test_service_without_als_or_reranker(tmp_path):
    _build(tmp_path)
    service = ModelService(tmp_path / "index", tmp_path / "no_als")
    assert service.missing == ["als", "rerank"] and service.default_model == "hybrid"


def test_search_tracks_for_autocomplete(client):
    assert client.get("/tracks/search", params={"q": "BRA"}).json() == [
        {"track_uri": "u:b", "track_name": "Bravo", "artist_name": "Bob", "n_playlists": 2}]
    # a có trong 3 playlist nên đứng trước (cùng chứa chữ "a": Ann, Bravo, Cat...)
    assert [t["track_uri"] for t in client.get("/tracks/search", params={"q": "a", "limit": 2}).json()] == ["u:a", "u:b"]
    assert client.get("/tracks/search", params={"q": ""}).status_code == 422


def test_reasons_explain_each_recommendation(client):
    body = client.post("/recommend", json={"seed_tracks": ["u:c"], "title": "rock", "n": 2,
                                           "model": "cooccurrence"}).json()
    b = body["recommendations"][0]
    assert b["track_uri"] == "u:b"
    reasons = {r["type"]: r["text"] for r in b["reasons"]}
    # c và b cùng nằm trong playlist 2; playlist chứa c là 1 và 2 (láng giềng), b có trong playlist 2
    assert reasons["co_occurrence"] == "Hay đi cùng “Cat – Charlie”: cùng có trong 1 playlist"
    assert reasons["similar_playlists"] == "Cùng gu với playlist của bạn: có trong 1/2 playlist giống nhất"
    # Playlist tên "rock": 0 và 2, cả hai chứa b
    assert reasons["title"] == "Cùng gu với các playlist tên “rock”: có trong 2/2 playlist"
    assert [r["type"] for r in b["reasons"]] == ["co_occurrence", "similar_playlists", "title"]


def test_popular_reason_for_popularity_model(client):
    body = client.post("/recommend", json={"title": "jazz", "n": 1, "model": "popularity"}).json()
    assert body["recommendations"][0]["reasons"] == [{"type": "popular", "text": "Phổ biến: có trong 3 playlist"}]
