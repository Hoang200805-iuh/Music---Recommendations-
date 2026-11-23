"""Recommender tests."""
import math

import numpy as np
import pytest
from scipy import sparse

from recommender.cooccurrence.cooccurrence_recommender import CooccurrenceRecommender
from recommender.hybrid.hybrid_recommender import HybridRecommender, TitleRecommender, combine
from recommender.neighbors.user_knn_recommender import UserKNNRecommender
from recommender.hybrid.playlist_titles import normalize_title
from recommender.popularity.popularity_recommender import PopularityRecommender

PLAYLISTS = [[0, 1, 2], [0, 1], [1, 3], [2, 3, 4], [0, 2, 4], [5]]


def _matrix():
    rows = [r for r, items in enumerate(PLAYLISTS) for _ in items]
    cols = [i for items in PLAYLISTS for i in items]
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.int32), (rows, cols)), shape=(6, 7))


def _model():
    X = _matrix()
    counts = {i: int(c) for i, c in enumerate(np.asarray(X.sum(axis=0)).ravel()) if c > 0}
    return CooccurrenceRecommender.from_matrix(X, PopularityRecommender(counts))


def _brute_force_score(seed, j):
    total = 0.0
    for i in seed:
        both = sum(1 for p in PLAYLISTS if i in p and j in p)
        count_i = sum(1 for p in PLAYLISTS if i in p)
        count_j = sum(1 for p in PLAYLISTS if j in p)
        if count_j:
            total += both / math.sqrt(count_i * count_j)
    return total


def test_popularity_from_ranked_array_returns_plain_ints():
    model = PopularityRecommender.from_ranked(np.array([3, 1, 2], dtype=np.int32))
    assert model.recommend([1], 2) == [3, 2]
    assert all(type(i) is int for i in model.recommend([], 3))


def test_cooccurrence_works_on_memmapped_arrays(tmp_path):
    X = _matrix()
    X_csc = X.tocsc()
    arrays = []
    for name, values in (("ri", X.indptr), ("rx", X.indices), ("ci", X_csc.indptr), ("cx", X_csc.indices)):
        np.save(tmp_path / f"{name}.npy", values)
        arrays.append(np.load(tmp_path / f"{name}.npy", mmap_mode="r"))
    fallback = _model().fallback
    from_disk = CooccurrenceRecommender(*arrays, fallback)
    assert from_disk.recommend([0, 3], 4) == _model().recommend([0, 3], 4)


def test_popularity_skips_seed_and_breaks_ties_by_id():
    model = PopularityRecommender({"a": 5, "c": 3, "b": 3, "d": 1})
    assert model.recommend(["a"], 2) == ["b", "c"]


@pytest.mark.parametrize("seed", [[0], [2, 3], [0, 1, 4]])
def test_cooccurrence_score_matches_formula(seed):
    scores = _model().score(seed)
    for j in range(7):
        assert scores[j] == pytest.approx(_brute_force_score(seed, j))


def _brute_force_general(seed, j, seed_exp, item_exp):
    total = 0.0
    count_j = sum(1 for p in PLAYLISTS if j in p)
    for i in seed:
        both = sum(1 for p in PLAYLISTS if i in p and j in p)
        count_i = sum(1 for p in PLAYLISTS if i in p)
        if count_j:
            total += both / (count_i ** seed_exp * count_j ** item_exp)
    return total


@pytest.mark.parametrize("seed_exp,item_exp", [(0.0, 1.0), (1.0, 0.0), (0.5, 0.25)])
def test_general_score_matches_formula(seed_exp, item_exp):
    base = _model()
    model = CooccurrenceRecommender(base.row_indptr, base.row_indices, base.col_indptr, base.col_indices,
                                    base.fallback, seed_exponent=seed_exp, item_exponent=item_exp)
    scores = model.score([0, 3])
    for j in range(7):
        assert scores[j] == pytest.approx(_brute_force_general([0, 3], j, seed_exp, item_exp))


def test_min_count_drops_rare_tracks():
    base = _model()
    # Bài 3 và 4 nằm trong 2 playlist; với min_count = 3 chỉ bài có từ 3 playlist trở lên được điểm
    model = CooccurrenceRecommender(base.row_indptr, base.row_indices, base.col_indptr, base.col_indices,
                                    base.fallback, min_count=3)
    scores = model.score([3])
    assert scores[4] == 0 and scores[1] > 0 and scores[2] > 0


def test_cooccurrence_excludes_seed_and_orders_by_score():
    recommended = _model().recommend([0], 3)
    assert 0 not in recommended
    scores = [_brute_force_score([0], j) for j in recommended]
    assert scores == sorted(scores, reverse=True)


def test_recommend_with_precomputed_scores_matches_and_keeps_them():
    model = _model()
    scores = model.score([0, 3])
    before = scores.copy()
    assert model.recommend([0, 3], 4, scores) == model.recommend([0, 3], 4)
    assert np.array_equal(scores, before)


def test_cooccurrence_falls_back_to_popularity_without_seed():
    model = _model()
    assert model.recommend([], 3) == model.fallback.recommend([], 3)


def test_cooccurrence_ignores_tracks_unseen_in_train():
    model = _model()
    assert model.recommend([-1], 3) == model.fallback.recommend([], 3)
    assert model.recommend([0, -1], 3) == model.recommend([0], 3)


def test_cooccurrence_fills_with_popular_tracks_when_few_candidates():
    # Bài 5 chỉ nằm một mình nên không có bài đồng xuất hiện nào
    recommended = _model().recommend([5], 3)
    assert len(recommended) == 3 and 5 not in recommended


def _brute_force_playlist_exponent(seed, j, playlist_exp):
    total = 0.0
    for p in PLAYLISTS:
        if j in p:
            total += sum(1 / math.sqrt(sum(1 for q in PLAYLISTS if i in q)) for i in seed if i in p) ** playlist_exp
    count_j = sum(1 for p in PLAYLISTS if j in p)
    return total / math.sqrt(count_j) if count_j else 0.0


@pytest.mark.parametrize("playlist_exp", [1.0, 2.0])
def test_playlist_exponent_matches_formula(playlist_exp):
    base = _model()
    model = CooccurrenceRecommender(base.row_indptr, base.row_indices, base.col_indptr, base.col_indices,
                                    base.fallback, playlist_exponent=playlist_exp)
    scores = model.score([0, 2])
    for j in range(7):
        assert scores[j] == pytest.approx(_brute_force_playlist_exponent([0, 2], j, playlist_exp))


def test_normalize_title_merges_case_punctuation_and_emoji():
    assert normalize_title("  WORKOUT!! 💪 ") == normalize_title("workout") == "workout"
    assert normalize_title("Nhạc   Chill_2024") == "nhạc chill 2024"
    assert normalize_title("🔥🔥") == ""


def _title_models(title_weight):
    base = _model()
    # Tên của 6 playlist trong PLAYLISTS; playlist 5 không có tên
    title_ids = {"rock": 0, "chill": 1}
    titles = TitleRecommender(base.row_indptr, base.row_indices, [0, 0, 1, 1, 0, -1], title_ids, 7)
    return titles, HybridRecommender(base, titles, title_weight)


def test_title_scores_count_playlists_with_same_title():
    titles, _ = _title_models(1.0)
    # "rock": playlist 0, 1, 4 = [0,1,2], [0,1], [0,2,4]
    assert list(titles.scores("Rock!")) == [3, 2, 2, 0, 1, 0, 0]
    assert titles.n_playlists("rock") == 3
    assert not titles.scores("jazz").any() and not titles.scores(None).any()


def test_hybrid_uses_title_without_seed_and_falls_back_to_popularity():
    titles, hybrid = _title_models(1.0)
    assert hybrid.recommend([], 3, title="chill") == [3, 1, 2]
    assert hybrid.recommend([], 3, title="jazz") == hybrid.fallback.recommend([], 3)


def test_hybrid_without_title_weight_equals_cooccurrence():
    _, hybrid = _title_models(0.0)
    assert hybrid.recommend([0], 4, title="chill") == _model().recommend([0], 4)


def test_hybrid_with_precomputed_scores_matches():
    _, hybrid = _title_models(0.5)
    scores = hybrid.score([3], "rock")
    assert hybrid.recommend([3], 4, title="rock", scores=scores) == hybrid.recommend([3], 4, title="rock")


def _brute_force_rp3beta(seed, j, alpha, beta):
    """Bước đi ngẫu nhiên bài -> playlist -> bài: P(i -> p) = 1/count(i), P(p -> j) = 1/|p|, mỗi bước mũ alpha."""
    count = lambda i: sum(1 for p in PLAYLISTS if i in p)
    total = sum((1 / count(i)) ** alpha * (1 / len(p)) ** alpha
                for i in seed for p in PLAYLISTS if i in p and j in p)
    return total / count(j) ** beta if count(j) else 0.0


@pytest.mark.parametrize("alpha,beta", [(1.0, 0.5), (0.7, 0.25)])
def test_length_exponent_gives_rp3beta(alpha, beta):
    base = _model()
    model = CooccurrenceRecommender(base.row_indptr, base.row_indices, base.col_indptr, base.col_indices,
                                    base.fallback, seed_exponent=alpha, item_exponent=beta, length_exponent=alpha)
    scores = model.score([0, 3])
    for j in range(7):
        assert scores[j] == pytest.approx(_brute_force_rp3beta([0, 3], j, alpha, beta))


def _brute_force_user_knn(seed, j, n_neighbors, length_exp, similarity_exp):
    sims = [(len(set(seed) & set(p)) / len(p) ** length_exp, p) for p in PLAYLISTS if set(seed) & set(p)]
    sims.sort(key=lambda x: -x[0])
    return sum(sim ** similarity_exp for sim, p in sims[:n_neighbors] if j in p)


@pytest.mark.parametrize("n_neighbors,similarity_exp", [(10, 1.0), (2, 2.0)])
def test_user_knn_matches_formula(n_neighbors, similarity_exp):
    model = UserKNNRecommender(_model(), n_neighbors=n_neighbors, similarity_exponent=similarity_exp)
    # Seed [0, 2]: độ giống của các playlist khác nhau nên top-2 không bị hòa
    scores = model.score([0, 2])
    for j in range(7):
        assert scores[j] == pytest.approx(_brute_force_user_knn([0, 2], j, n_neighbors, 0.5, similarity_exp))


def test_user_knn_without_seed_falls_back_to_popularity():
    model = UserKNNRecommender(_model())
    assert model.recommend([], 3) == model.fallback.recommend([], 3)
    assert model.recommend([0], 3, model.score([0])) == model.recommend([0], 3)


def test_hybrid_adds_neighbor_scores_on_the_same_scale():
    base = _model()
    titles, _ = _title_models(0.0)
    neighbors = UserKNNRecommender(base)
    hybrid = HybridRecommender(base, titles, 0.0, neighbors, neighbor_weight=1.0)
    c, u = base.score([0]), neighbors.score([0])
    assert np.allclose(hybrid.score([0]), c / c.max() + u / u.max())


def test_combine_handles_integer_scores_from_unseen_seeds():
    titles, _ = _title_models(0.1)
    neighbors = UserKNNRecommender(_model())
    # Bài -1 không có trong train: User-User không có láng giềng nào
    scores = combine(neighbors.score([-1]), titles.scores("rock"), 0.1)
    assert scores.dtype == np.float64 and scores.max() == pytest.approx(0.1)


def _feature_builder(n_candidates=6):
    from recommender.rerank.features import FeatureBuilder

    base = _model()
    titles, _ = _title_models(0.1)
    neighbors = UserKNNRecommender(base, n_neighbors=3)
    # Nghệ sĩ của 7 bài; bài seed -1 không có trong train, thuộc nghệ sĩ 1
    builder = FeatureBuilder(base, titles, neighbors, [0, 0, 1, 1, 2, 3, 3], 0.1, 0.5,
                             unseen_artists={-1: 1}, n_candidates=n_candidates)
    return builder, HybridRecommender(base, titles, 0.1, neighbors, neighbor_weight=0.5)


def test_rerank_candidates_follow_hybrid_order():
    from recommender.rerank.features import FEATURE_NAMES

    builder, hybrid = _feature_builder()
    for seed, title in (([0], "rock"), ([2, 3, -1], None), ([], "chill"), ([], None)):
        candidates, features = builder.build(seed, title)
        assert list(candidates) == hybrid.recommend(seed, 6, title=title)
        assert features.shape == (len(candidates), len(FEATURE_NAMES)) and np.isfinite(features).all()


def test_rerank_features_match_brute_force():
    from recommender.rerank.features import FEATURE_NAMES

    builder, _ = _feature_builder()
    seed = [2, 3, -1]
    candidates, features = builder.build(seed, None)
    column = {name: features[:, i] for i, name in enumerate(FEATURE_NAMES)}
    neighbor_playlists, _ = builder.neighbors.neighbors(seed)
    artists = [0, 0, 1, 1, 2, 3, 3]
    for row, j in enumerate(candidates):
        hits = sum(1 for p in neighbor_playlists if j in PLAYLISTS[p])
        assert column["knn_neighbor_share"][row] == pytest.approx(hits / len(neighbor_playlists))
        # Seed 2, 3 (nghệ sĩ 1) và -1 (nghệ sĩ 1)
        assert column["artist_seed_count"][row] == (3 if artists[j] == 1 else 0)
    assert column["n_seed"][0] == 3 and column["n_known_seed"][0] == 2
    assert column["seed_artist_diversity"][0] == pytest.approx(1 / 3)


def test_rerank_recommender_orders_by_booster_and_fills():
    from recommender.rerank.lgbm_reranker import RerankRecommender

    class ReverseBooster:  # điểm tăng theo thứ hạng gốc: đảo ngược danh sách
        def predict(self, features):
            return np.arange(len(features), dtype=float)

    builder, hybrid = _feature_builder(n_candidates=4)
    model = RerankRecommender(builder, ReverseBooster())
    assert model.recommend([0], 4, title="rock") == hybrid.recommend([0], 4, title="rock")[::-1]
    # Bù bằng bài phổ biến khi thiếu ứng viên: cả 5 bài khác seed trong dữ liệu nhỏ
    assert sorted(model.recommend([0], 5, title="rock")) == [1, 2, 3, 4, 5]
    # Không có bài seed: giữ thứ tự của mô hình gốc
    assert model.recommend([], 4, title="chill") == hybrid.recommend([], 4, title="chill")
    assert model.score([], "chill")[1] is None
    assert model.recommend([0], 4, "rock", scored=model.score([0], "rock")) == model.recommend([0], 4, "rock")
