"""
Lý do cho từng bài gợi ý, dựa trên dữ liệu train (không phụ thuộc mô hình nào đã chọn bài):

    co_occurrence      "Hay đi cùng bài X": bài seed X cùng nằm trong nhiều playlist nhất với bài này
    similar_playlists  "Cùng gu với playlist của bạn": có trong các playlist train giống playlist đang
                       gợi ý nhất (100 láng giềng của User-User kNN)
    title              "Cùng gu với các playlist tên ...": có trong các playlist train cùng tên
    popular            "Phổ biến": bài được bù từ danh sách phổ biến, hoặc không có lý do nào ở trên

Số playlist chứa một bài lấy từ ma trận CSC (bài -> các playlist, đã sắp xếp), nên đếm số playlist
chung của hai bài chỉ là phép giao hai mảng đã sắp xếp.
"""
import numpy as np

MAX_SEEDS = 50  # playlist dài: chỉ so với 50 bài seed cuối (gần với phần đang gợi ý nhất)


def _fmt(n):
    return f"{int(n):,}".replace(",", ".")


class Explainer:
    def __init__(self, service):
        self.service = service
        self.cooccurrence = service.cooccurrence
        self.neighbors = service.neighbors

    def playlists_of(self, item):
        model = self.cooccurrence
        return np.asarray(model.col_indices[model.col_indptr[item]: model.col_indptr[item + 1]])

    def explain(self, items, seed, title, fallback, model_name):
        """Danh sách lý do (mỗi lý do {type, text}) cho từng bài trong items, xếp lý do mạnh trước."""
        service = self.service
        seeds = list(dict.fromkeys(seed))[-MAX_SEEDS:]
        seed_playlists = {i: self.playlists_of(i) for i in seeds}
        neighbor_playlists = np.sort(self.neighbors.neighbors(seed)[0]) if seed else np.zeros(0, dtype=np.int64)
        n_title = service.titles.n_playlists(title) if title else 0
        title_scores = service.titles.scores(title) if n_title else None

        all_reasons = []
        for item, is_fallback in zip(items, fallback):
            playlists = self.playlists_of(item)
            reasons = []
            if seeds:
                shared = {i: len(np.intersect1d(p, playlists, assume_unique=True)) for i, p in seed_playlists.items()}
                best = max(shared, key=shared.get)
                if shared[best]:
                    reasons.append({"type": "co_occurrence", "text": (
                        f"Hay đi cùng “{service.names[best]}”: cùng có trong {_fmt(shared[best])} playlist")})
            if len(neighbor_playlists):
                n_neighbors = len(np.intersect1d(neighbor_playlists, playlists, assume_unique=True))
                if n_neighbors:
                    reasons.append({"type": "similar_playlists", "text": (
                        f"Cùng gu với playlist của bạn: có trong {n_neighbors}/{len(neighbor_playlists)} "
                        "playlist giống nhất")})
            if title_scores is not None and title_scores[item] > 0:
                reasons.append({"type": "title", "text": (
                    f"Cùng gu với các playlist tên “{title}”: có trong {_fmt(title_scores[item])}/"
                    f"{_fmt(n_title)} playlist")})
            if is_fallback or model_name == "popularity" or not reasons:
                reasons.append({"type": "popular", "text": f"Phổ biến: có trong {_fmt(len(playlists))} playlist"})
            all_reasons.append(reasons)
        return all_reasons
