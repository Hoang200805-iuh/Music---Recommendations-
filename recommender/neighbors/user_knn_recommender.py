"""
User-User kNN (láng giềng theo playlist).

Tìm n_neighbors playlist train giống playlist đang gợi ý nhất rồi gợi ý các bài
trong đó. Độ giống của playlist train p với seed S:
    sim(p) = |S ∩ p| / |p|^length_exponent
(length_exponent = 0.5 tương đương cosine, vì |S| giống nhau với mọi p nên bỏ qua).
Điểm của bài j:
    score(j) = tổng theo p trong n_neighbors láng giềng có chứa j của sim(p)^similarity_exponent

Khác Co-occurrence ở chỗ chỉ giữ một số ít playlist giống nhất, nên các playlist chỉ
tình cờ chứa một bài seed phổ biến không làm nhiễu điểm. Đội vô địch RecSys Challenge
2018 dùng song song cả hai loại láng giềng và trộn lại.

Dùng chung các mảng chỉ số (có thể là memmap) với một CooccurrenceRecommender.
"""
import numpy as np

from recommender.cooccurrence.cooccurrence_recommender import gather


class UserKNNRecommender:
    def __init__(self, cooccurrence, n_neighbors=500, length_exponent=0.5, similarity_exponent=1.0):
        self.cooccurrence = cooccurrence
        self.fallback = cooccurrence.fallback
        self.n_neighbors = n_neighbors
        self.length_exponent = length_exponent
        self.similarity_exponent = similarity_exponent

    def neighbors(self, seed_items):
        """Các playlist láng giềng và độ giống của chúng (chưa sắp xếp)."""
        model = self.cooccurrence
        # seed_exponent = 0 nên trọng số của mỗi playlist đúng bằng số bài seed nó chứa
        playlists, overlap = model.seed_playlists(seed_items, seed_exponent=0.0)
        sizes = (np.asarray(model.row_indptr[playlists + 1], dtype=np.float64)
                 - np.asarray(model.row_indptr[playlists], dtype=np.float64))
        similarity = overlap / sizes ** self.length_exponent
        if len(playlists) > self.n_neighbors:
            top = np.argpartition(-similarity, self.n_neighbors - 1)[:self.n_neighbors]
            playlists, similarity = playlists[top], similarity[top]
        return playlists, similarity

    def score(self, seed_items):
        """Trả về mảng điểm của mọi bài (chưa loại seed)."""
        playlists, similarity = self.neighbors(seed_items)
        model = self.cooccurrence
        items, lengths = gather(model.row_indptr, model.row_indices, playlists)
        return np.bincount(items, weights=np.repeat(similarity ** self.similarity_exponent, lengths),
                           minlength=len(model.count))

    def recommend(self, seed_items, n, scores=None):
        """scores: kết quả score(seed_items) nếu đã tính sẵn, để khỏi tính lại."""
        seed = set(seed_items)
        scores = self.score(seed_items) if scores is None else scores.copy()
        known_seed = [i for i in seed if 0 <= i < len(scores)]
        scores[known_seed] = 0.0
        candidates = np.flatnonzero(scores)
        return self.cooccurrence.rank(candidates, scores[candidates], seed, n)
