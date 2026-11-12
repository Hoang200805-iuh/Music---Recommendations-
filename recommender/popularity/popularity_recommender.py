"""Popularity-based recommender."""


def read_mapreduce_counts(path):
    """Đọc đầu ra job MapReduce: track_uri <TAB> count <TAB> track_name <TAB> artist_name."""
    counts = {}
    with open(path, "rb") as f:
        for line in f:
            parts = line.rstrip(b"\r\n").split(b"\t")
            counts[parts[0].decode("utf-8")] = int(parts[1])
    return counts


class PopularityRecommender:
    """Gợi ý các bài phổ biến nhất chưa có trong seed."""

    def __init__(self, counts):
        # Bài cùng số đếm xếp theo id, để kết quả không phụ thuộc thứ tự đọc file
        self.ranked_items = sorted(counts, key=lambda item: (-counts[item], item))

    @classmethod
    def from_ranked(cls, ranked_items):
        """Dùng thứ tự đã xếp sẵn (ví dụ mảng đọc từ đĩa), tránh tạo dict hàng triệu phần tử."""
        model = cls({})
        model.ranked_items = ranked_items
        return model

    def recommend(self, seed_items, n):
        seed = set(seed_items)
        recommended = []
        for item in self.ranked_items:
            if item not in seed:
                recommended.append(item.item() if hasattr(item, "item") else item)
                if len(recommended) == n:
                    break
        return recommended
