"""
Đặc trưng cho bước xếp hạng lại (second stage), theo cách của đội vô địch RecSys Challenge 2018.

Bước 1 (mô hình gốc): Hybrid = Co-occurrence tinh chỉnh + tên playlist + User-User kNN
chọn ra N_CANDIDATES bài ứng viên cho mỗi playlist (bù bằng bài phổ biến nếu thiếu).
Bước 2: LightGBM xếp hạng lại các ứng viên này dựa trên các đặc trưng dưới đây.
Mô hình gốc chỉ cộng các điểm theo trọng số cố định; LightGBM học được cách kết hợp
phụ thuộc vào ngữ cảnh: ví dụ khi playlist chỉ có một nghệ sĩ thì bài cùng nghệ sĩ
đáng tin hơn, khi k lớn thì User-User đáng tin hơn Co-occurrence.

Mọi đặc trưng chỉ dùng tập train và phần seed của playlist, không dùng đáp án.
"""
from collections import Counter

import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import gather
from recommender.hybrid.hybrid_recommender import combine

N_CANDIDATES = 300
RECENT_SEEDS = 5  # số bài seed cuối dùng cho đặc trưng "gần đây"

FEATURE_NAMES = [
    # Điểm của bài ứng viên theo từng mô hình con (chia cho điểm cao nhất) và thứ hạng
    "base_score", "base_rank",
    "cooc_score", "cooc_rank", "cooc_recent_score",
    "knn_score", "knn_rank", "knn_neighbor_share",
    "title_score", "title_playlist_share",
    "is_popularity_fill",
    # Bài ứng viên
    "log_popularity", "log_popularity_vs_seed",
    "artist_seed_count", "artist_seed_share", "artist_candidate_count",
    # Playlist (giống nhau cho mọi ứng viên của một playlist)
    "n_seed", "n_known_seed", "seed_artist_diversity", "seed_log_popularity",
    "log_title_playlists", "max_neighbor_similarity",
]


def _rank_of(values):
    """Thứ hạng (0 = cao nhất) của từng phần tử theo giá trị giảm dần."""
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[np.argsort(-values, kind="stable")] = np.arange(len(values))
    return ranks


def _share_of_top(scores, candidates):
    """Điểm của các ứng viên chia cho điểm cao nhất trên mọi bài (0 nếu không bài nào có điểm)."""
    top = scores.max()
    return scores[candidates] / top if top > 0 else np.zeros(len(candidates))


class FeatureBuilder:
    """cooccurrence: CooccurrenceRecommender tinh chỉnh; titles: TitleRecommender;
    neighbors: UserKNNRecommender; item_artists[i]: id nghệ sĩ của bài train i;
    unseen_artists: {id âm: id nghệ sĩ} cho bài seed không có trong train."""

    def __init__(self, cooccurrence, titles, neighbors, item_artists, title_weight, neighbor_weight,
                 unseen_artists=None, n_candidates=N_CANDIDATES):
        self.cooccurrence = cooccurrence
        self.titles = titles
        self.neighbors = neighbors
        self.item_artists = np.asarray(item_artists)
        self.title_weight = title_weight
        self.neighbor_weight = neighbor_weight
        self.unseen_artists = unseen_artists or {}
        self.n_candidates = n_candidates
        self.log_count = np.log1p(np.asarray(cooccurrence.count))

    def artist_of(self, item):
        return int(self.item_artists[item]) if item >= 0 else self.unseen_artists.get(item, -1)

    def build(self, seed_items, title=None):
        """Trả về (ứng viên theo thứ tự của mô hình gốc, ma trận đặc trưng float32)."""
        model = self.cooccurrence
        n_items = len(model.count)
        seed = set(seed_items)
        known_seed = [i for i in seed if 0 <= i < n_items]

        cooc = model.score(seed_items)
        recent = seed_items[-RECENT_SEEDS:]
        cooc_recent = model.score(recent) if len(seed_items) > RECENT_SEEDS else cooc
        title_scores = self.titles.scores(title)

        playlists, similarity = self.neighbors.neighbors(seed_items)
        items, lengths = gather(model.row_indptr, model.row_indices, playlists)
        knn = np.bincount(items, weights=np.repeat(similarity ** self.neighbors.similarity_exponent, lengths),
                          minlength=n_items)

        base = combine(cooc, title_scores, self.title_weight, (knn, self.neighbor_weight))
        base[known_seed] = 0.0
        scored = np.flatnonzero(base)
        candidates = np.array(model.rank(scored, base[scored], seed, self.n_candidates), dtype=np.int64)

        # Số láng giềng chứa từng ứng viên (không trọng số)
        order = np.argsort(candidates)
        in_candidates = items[np.isin(items, candidates)]
        neighbor_hits = np.empty(len(candidates))
        neighbor_hits[order] = np.bincount(np.searchsorted(candidates[order], in_candidates),
                                           minlength=len(candidates))

        n_title = self.titles.n_playlists(title)
        seed_artists = Counter(self.artist_of(i) for i in seed_items)
        candidate_artists = self.item_artists[candidates]
        _, artist_inverse, artist_counts = np.unique(candidate_artists, return_inverse=True, return_counts=True)
        artist_seed_count = np.array([seed_artists.get(a, 0) for a in candidate_artists.tolist()],
                                     dtype=np.float32)
        seed_pop = float(self.log_count[known_seed].mean()) if known_seed else 0.0

        n_seed = len(seed_items)
        c_base = base[candidates]
        features = np.column_stack([
            _share_of_top(base, candidates),
            np.arange(len(candidates)),
            _share_of_top(cooc, candidates),
            _rank_of(cooc[candidates]),
            _share_of_top(cooc_recent, candidates),
            _share_of_top(knn, candidates),
            _rank_of(knn[candidates]),
            neighbor_hits / max(len(playlists), 1),
            _share_of_top(title_scores, candidates),
            title_scores[candidates] / n_title if n_title else np.zeros(len(candidates)),
            c_base == 0,
            self.log_count[candidates],
            self.log_count[candidates] - seed_pop,
            artist_seed_count,
            artist_seed_count / max(n_seed, 1),
            artist_counts[artist_inverse],
            np.full(len(candidates), n_seed),
            np.full(len(candidates), len(known_seed)),
            np.full(len(candidates), len(seed_artists) / n_seed if n_seed else 0.0),
            np.full(len(candidates), seed_pop),
            np.full(len(candidates), np.log1p(n_title)),
            np.full(len(candidates), similarity.max() if len(similarity) else 0.0),
        ]).astype(np.float32)
        return candidates, features
