"""ALS fold-in, chống rò rỉ dữ liệu, tuning từng tham số và Spark smoke test."""
import importlib.util
import os

import numpy as np
import pytest

from recommender.als.load_als import load_als
from recommender.als.recommend_als import ALSRecommender
from recommender.als.train_als import (
    DEFAULT_PARAMS, MAX_ITER, create_spark, extract_item_factors, fit_als,
    release_factors, save_bundle, training_digest, training_ratings,
)
from recommender.als.tune_als import tune_one_at_a_time
from recommender.evaluation.evaluate_models import evaluate, load_train
from recommender.popularity.popularity_recommender import PopularityRecommender


FACTORS = np.array([[1, 0], [0.8, 0.2], [0, 1], [-0.5, -0.5]], dtype=np.float32)


def _model():
    return ALSRecommender(FACTORS, PopularityRecommender({0: 3, 1: 2, 2: 1, 3: 1}), reg_param=0.1, alpha=10)


def test_fold_in_matches_full_implicit_weighted_least_squares():
    model = _model()
    seed = [0, 2]
    Y = FACTORS.astype(np.float64)
    # Độc lập với dạng rút gọn: dựng confidence/preferences cho toàn catalog.
    preference = np.array([1, 0, 1, 0])
    confidence = np.diag(1 + model.alpha * preference)
    expected = np.linalg.solve(Y.T @ confidence @ Y + model.reg_param * len(seed) * np.eye(2),
                               Y.T @ confidence @ preference)
    assert model.fold_in(seed) == pytest.approx(expected)
    assert model.fold_in([0, 0, -1, 2, 100]) == pytest.approx(expected)


def test_als_ranks_full_catalog_excludes_seed_and_keeps_negative_scores():
    model = _model()
    result = model.recommend([0], 10)
    assert result == [1, 2, 3]
    assert all(type(item) is int for item in result)
    scores = FACTORS[result] @ model.fold_in([0])
    assert list(scores) == sorted(scores, reverse=True)
    assert scores[-1] < 0


def test_empty_or_unseen_seed_uses_same_popularity_fallback():
    model = _model()
    assert model.fold_in([-1, 100]) is None
    assert model.recommend([], 3) == model.fallback.recommend([], 3)
    assert model.recommend([-1], 3) == model.recommend([], 3)
    assert model.recommend([0, -1], 3) == model.recommend([0], 3)
    assert model.recommend([0], 0) == []


def test_top_n_ties_choose_smallest_item_ids_at_cutoff():
    model = ALSRecommender(np.ones((20, 2)), _model().fallback)
    assert model.recommend([0], 3) == [1, 2, 3]


def test_bundle_round_trip_and_rejects_different_item_id_mapping(tmp_path):
    vocab = {b"uri:a": 0, b"uri:b": 1, b"uri:c": 2, b"uri:d": 3}
    params = {**DEFAULT_PARAMS, "rank": 2}
    save_bundle(tmp_path, _model(), vocab, params, train_sha256="train-a")
    loaded = load_als(tmp_path, _model().fallback, vocab, "train-a")
    assert loaded.recommend([0], 3) == _model().recommend([0], 3)
    wrong_vocab = {**vocab, b"uri:a": 1, b"uri:b": 0}
    with pytest.raises(ValueError, match="Vocabulary"):
        load_als(tmp_path, _model().fallback, wrong_vocab)
    with pytest.raises(ValueError, match="split"):
        load_als(tmp_path, _model().fallback, vocab, "train-b")


def test_sequential_tuning_carries_winners_and_reuses_identical_trials():
    calls = []

    def trial(params):
        calls.append(params)
        # Optimum rank=100, regParam=0.01, alpha=40.
        score = params["rank"] / 1000 + (0.1 if params["regParam"] == 0.01 else 0)
        score += 0.2 if params["alpha"] == 40 else 0
        return [{"k": k, "ndcg@10": score} for k in (1, 5, 25)]

    best, rows = tune_one_at_a_time(trial)
    assert best == {"rank": 100, "regParam": 0.01, "alpha": 40.0}
    assert len(calls) == 7
    for stage, values in (("rank", (20, 50, 100)), ("regParam", (0.01, 0.1, 1)), ("alpha", (1, 10, 40))):
        group = [row for row in rows if row["stage"] == stage]
        assert {row[stage] for row in group} == set(values)
        assert len(group) == 9
        assert sum(row["selected"] for row in group) == 3
        assert all(row["maxIter"] == 10 for row in group)
        if stage == "regParam":
            assert all(row["rank"] == 100 and row["alpha"] == 10 for row in group)
        if stage == "alpha":
            assert all(row["rank"] == 100 and row["regParam"] == 0.01 for row in group)


def test_tuning_ties_keep_starting_parameters():
    best, _ = tune_one_at_a_time(lambda params: [{"ndcg@10": 0.1}])
    assert best == DEFAULT_PARAMS


def test_als_uses_existing_evaluation_metrics_and_seed_protocol():
    model = _model()
    results = evaluate({"als": model}, {100: [0, 1, 2, -1]}, [0, 1], 10)
    assert [row["k"] for row in results] == [0, 1]
    assert all(row["model"] == "als" and row["n_playlists"] == 1 for row in results)
    assert results[0]["recall@50"] == 0.75  # Bài chưa gặp vẫn ở mẫu số.
    assert results[1]["recall@50"] == pytest.approx(2 / 3)


def _small_train(tmp_path):
    flat = tmp_path / "flat.tsv"
    flat.write_text("0\ta\tA\tArtist A\n0\ta\tA\tArtist A\n0\tb\tB\tArtist B\n"
                    "1\tb\tB\tArtist B\n1\tc\tC\tArtist C\n2\tval-only\tV\tArtist V\n"
                    "3\ttest-only\tT\tArtist T\n", encoding="utf-8")
    return load_train(flat, {0: "train", 1: "train", 2: "val", 3: "test"})


def test_training_digest_detects_changed_split_even_with_identical_vocabulary(tmp_path):
    train = _small_train(tmp_path)
    changed = train._replace(row_pids=[10, 11])
    assert training_digest(train) != training_digest(changed)
    changed_matrix = train.X.copy()
    changed_matrix.indices[0] = 2
    assert training_digest(train) != training_digest(train._replace(X=changed_matrix))


@pytest.mark.skipif(os.environ.get("RUN_SPARK_TESTS") != "1" or importlib.util.find_spec("pyspark") is None,
                    reason="Đặt RUN_SPARK_TESTS=1 với PySpark và Java để chạy integration")
def test_real_spark_trains_binary_train_only_and_extracts_factors(tmp_path):
    train = _small_train(tmp_path)
    spark = create_spark("local[2]", "1g", "TestPlaylistALS")
    spark.conf.set("spark.sql.shuffle.partitions", "2")
    try:
        with training_ratings(spark, train, tmp_path) as ratings:
            data = {(row.user, row.item, row.rating) for row in ratings.collect()}
            assert data == {(0, 0, 1.0), (0, 1, 1.0), (1, 1, 1.0), (1, 2, 1.0)}
            model = fit_als(ratings, rank=2, regParam=0.1, alpha=10, num_blocks=2)
            try:
                assert model._java_obj.parent().getMaxIter() == MAX_ITER
                assert model._java_obj.parent().getImplicitPrefs()
                factors = extract_item_factors(model, len(train.vocab))
                assert factors.shape == (3, 2) and np.isfinite(factors).all()
                recommender = ALSRecommender(factors, PopularityRecommender({0: 1, 1: 2, 2: 1}))
                assert set(recommender.recommend([0], 2)) == {1, 2}
            finally:
                release_factors(model)
            assert all(not rdd.getStorageLevel().useMemory() and not rdd.getStorageLevel().useDisk()
                       for rdd in model._als_cached_rdds)
    finally:
        spark.stop()
