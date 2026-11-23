"""Training bundles must retain track identity across API/app vocabularies."""
import numpy as np
import pytest

from api.services.model_service import ModelService
from api.services.recommendation_service import recommend
from recommender.als.recommend_als import ALSRecommender
from recommender.als.train_als import save_bundle
from test_app_index import _build


def bundle(tmp_path):
    _build(tmp_path)
    factors = np.array([[0, 1], [1, 0], [0.9, 0.1]], dtype=np.float32)
    model = ALSRecommender(factors, None, reg_param=0.2, alpha=40)
    target = tmp_path / "als"
    # ALS rows intentionally differ from the application order a, b, c.
    save_bundle(target, model, {b"u:c": 0, b"u:a": 1, b"u:b": 2},
                {"rank": 2, "regParam": 0.2, "alpha": 40})
    return target


def test_training_bundle_serves_correct_uri_scores_and_parameters(tmp_path):
    path = bundle(tmp_path)
    service = ModelService(tmp_path / "index", path)
    als = service.models["als"]
    assert service.n_als_tracks == 3
    assert als.core.reg_param == 0.2 and als.core.alpha == 40
    assert np.allclose(als.factors[als.rows[0]], [1, 0])
    response = recommend(service, ["u:a"], None, 2, "als")
    assert [r["track_uri"] for r in response["recommendations"]] == ["u:b", "u:c"]
    assert all(not r["fallback"] for r in response["recommendations"])
    assert len(als.recommend([0], 1)) == 1


def test_no_known_seed_reports_popularity_fallback(tmp_path):
    service = ModelService(tmp_path / "index", bundle(tmp_path))
    response = recommend(service, ["u:missing"], None, 2, "als")
    assert all(r["fallback"] and r["score"] is None for r in response["recommendations"])


def test_bundle_missing_uri_rows_is_rejected(tmp_path):
    path = bundle(tmp_path)
    (path / "items.tsv").write_text("0\tu:c\n1\tu:a\n", encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        ModelService(tmp_path / "index", path)


def test_zero_als_scores_are_not_fallback(tmp_path):
    path = bundle(tmp_path)
    # Orthogonal factors produce a legitimate zero prediction for c.
    model = ALSRecommender(np.eye(3, dtype=np.float32), None)
    save_bundle(path, model, {b"u:a": 0, b"u:b": 1, b"u:c": 2},
                {"rank": 3, "regParam": 0.1, "alpha": 10})
    service = ModelService(tmp_path / "index", path)
    response = recommend(service, ["u:a"], None, 2, "als")
    assert all(r["score"] == 0 and not r["fallback"] for r in response["recommendations"])
