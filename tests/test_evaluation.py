"""Evaluation tests."""
import math

import pytest

from recommender.evaluation.evaluate_models import evaluate
from recommender.evaluation.hit_rate_at_k import hit_rate_at_k
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import max_recall_at_k, precision_at_k
from recommender.evaluation.r_precision import r_precision
from recommender.evaluation.recall_at_k import recall_at_k
from recommender.evaluation.split_playlists import assign_splits


def test_recall_counts_hits_over_all_hidden():
    assert recall_at_k(["a", "b", "c"], {"a", "c", "x", "y"}, 2) == 0.25
    assert recall_at_k(["a", "b", "c"], {"a", "c", "x", "y"}, 3) == 0.5


def test_ndcg_hand_computed():
    # Trúng ở vị trí 2 trong 3, có 2 bài bị giấu
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["x", "a", "y"], {"a", "b"}, 3) == pytest.approx(expected)


def test_ndcg_is_one_for_perfect_ranking():
    assert ndcg_at_k(["a", "b", "x"], {"a", "b"}, 3) == pytest.approx(1.0)


def test_precision_counts_hits_over_k():
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c", "x"}, 4) == 0.5
    assert precision_at_k(["a", "b", "c", "d"], {"a", "c", "x"}, 2) == 0.5


def test_max_recall_is_capped_by_number_of_hidden_tracks():
    # Playlist ở ảnh chụp: giấu 103 bài, chỉ có 10 gợi ý
    assert max_recall_at_k(set(range(103)), 10) == pytest.approx(10 / 103)
    assert max_recall_at_k({"a", "b"}, 10) == 1.0


def test_recall_reaches_its_max_when_every_recommendation_is_hidden():
    hidden = set(range(103))
    assert recall_at_k(list(range(10)), hidden, 10) == max_recall_at_k(hidden, 10)


def test_metrics_reject_empty_hidden():
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError):
        ndcg_at_k(["a"], set(), 1)


def test_split_ratios_are_exact_and_repeatable():
    first = assign_splits(range(1000))
    assert first == assign_splits(range(1000))
    values = list(first.values())
    assert (values.count("train"), values.count("val"), values.count("test")) == (700, 150, 150)


class _RecordingModel:
    def __init__(self):
        self.seeds = []

    def recommend(self, seed_items, n):
        self.seeds.append(list(seed_items))
        return [1, 2]


def test_evaluate_seeds_with_first_k_and_skips_playlists_without_hidden():
    model = _RecordingModel()
    # Playlist 2 chỉ có 2 bài nên không còn gì để giấu khi k = 2
    rows = evaluate({"m": model}, {1: [1, 2, 3, 6], 2: [4, 5]}, seed_sizes=[2], max_playlists=10)
    assert rows[0]["n_playlists"] == 1
    assert model.seeds == [[1, 2]]


def test_hit_rate_is_one_when_any_top_k_item_is_hidden():
    assert hit_rate_at_k(["x", "a", "y"], {"a", "b"}, 2) == 1.0
    assert hit_rate_at_k(["x", "a", "y"], {"a", "b"}, 1) == 0.0


def test_r_precision_scores_top_r_with_partial_artist_credit():
    artist = {"a": "A", "b": "B", "c": "C", "a2": "A", "b2": "B", "b3": "B", "x": "X"}.get
    hidden = {"a", "b", "c"}
    # R = 3: "a" trúng bài; "a2" cùng nghệ sĩ A nhưng A đã trúng; "b2" được 0.25 cho nghệ sĩ B
    assert r_precision(["a", "a2", "b2", "c"], hidden, artist) == pytest.approx((1 + 0.25) / 3)
    # Mỗi nghệ sĩ chỉ được cộng một lần
    assert r_precision(["b2", "b3", "x"], hidden, artist) == pytest.approx(0.25 / 3)
    # Không có thông tin nghệ sĩ thì chỉ tính trúng bài
    assert r_precision(["a", "a2", "b2"], hidden) == pytest.approx(1 / 3)


def test_r_precision_never_exceeds_one():
    artist = {"a": "A", "b": "B", "a2": "A", "b2": "B"}.get
    assert r_precision(["a", "b"], {"a", "b"}, artist) == 1.0
    assert r_precision(["a2", "b2"], {"a", "b"}, artist) == 0.25


class _TitleModel:
    uses_title = True

    def __init__(self):
        self.calls = []

    def recommend(self, seed_items, n, title=None):
        self.calls.append((title, n))
        return list(range(n))


def test_evaluate_passes_titles_and_asks_for_enough_items_for_r_precision():
    model = _TitleModel()
    hidden_count = 60
    rows = evaluate({"m": model}, {7: list(range(100, 100 + hidden_count))}, seed_sizes=[0], max_playlists=1,
                    titles={7: "chill"})
    assert model.calls == [("chill", hidden_count)]
    assert rows[0]["r_precision"] == 0.0 and "hit@10" in rows[0]
