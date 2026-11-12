"""App index tests."""
from recommender import app_index

# 4 playlist train (pid 0-3) và 1 playlist test (pid 4), dạng file đã làm phẳng
FLAT = [
    (0, "u:a", "Alpha", "Ann"), (0, "u:b", "Bravo", "Bob"), (0, "u:a", "Alpha", "Ann"),
    (1, "u:a", "Alpha", "Ann"), (1, "u:c", "Charlie", "Cat"),
    (2, "u:b", "Bravo", "Bob"), (2, "u:c", "Charlie", "Cat"),
    (3, "u:a", "Alpha", "Ann"),
    (4, "u:a", "Alpha", "Ann"), (4, "u:z", "Zulu", "Zed"),
]
SPLIT = {0: "train", 1: "train", 2: "train", 3: "train", 4: "test"}
# Đếm theo lượt như MapReduce: playlist 0 lặp bài a nên a được 4
MAPREDUCE = {"u:a": 4, "u:b": 2, "u:c": 2}


def _build(tmp_path):
    flat = tmp_path / "flat.tsv"
    flat.write_text("".join(f"{p}\t{u}\t{t}\t{a}\n" for p, u, t, a in FLAT), encoding="utf-8")
    split = tmp_path / "split.tsv"
    split.write_text("".join(f"{p}\t{s}\n" for p, s in SPLIT.items()), encoding="utf-8")
    popularity = tmp_path / "part-00000"
    popularity.write_text("".join(f"{u}\t{c}\tx\ty\n" for u, c in MAPREDUCE.items()), encoding="utf-8")
    titles = tmp_path / "titles.tsv"
    titles.write_text("0\trock\n1\tchill\n2\trock\n4\trock\n", encoding="utf-8")
    app_index.build(flat, split, popularity, titles, tmp_path / "index", n_test=1)
    return app_index.load(tmp_path / "index")


def test_index_round_trip(tmp_path):
    data = _build(tmp_path)
    names = data["names"]
    assert [names[i] for i in range(len(names))] == ["Ann – Alpha", "Bob – Bravo", "Cat – Charlie"]
    assert data["popularity"].recommend([], 3) == [0, 1, 2]
    # c chung 1 playlist với a (a có 3 playlist) và 1 với b (b có 2): b được 1/√4, a chỉ 1/√6
    assert data["cooccurrence"].recommend([2], 2) == [1, 0]


def test_tuned_model_shares_arrays_with_base(tmp_path):
    data = _build(tmp_path)
    base, tuned = data["cooccurrence"], data["cooccurrence_tuned"]
    assert tuned.row_indices is base.row_indices and tuned.col_indices is base.col_indices
    assert tuned.item_exponent == app_index.TUNED_PARAMS["item_exponent"]
    # min_count = 10 loại hết các bài của bộ dữ liệu nhỏ này, nên chỉ còn bài phổ biến bù vào
    assert tuned.recommend([2], 2) == data["popularity"].recommend([2], 2)


def test_search_returns_most_popular_first_and_ignores_case(tmp_path):
    names = _build(tmp_path)["names"]
    assert names.search("A", 10) == [0, 1, 2]
    assert names.search("bravo", 10) == [1]
    assert names.search("charlie", 10) == [2]
    assert names.search("nope", 10) == []


def test_search_does_not_match_across_two_names(tmp_path):
    # Tên viết thường nằm liền nhau trong file: "ann – alpha" rồi "bob – bravo"
    assert _build(tmp_path)["names"].search("alphabob", 10) == []


def test_test_playlist_keeps_unseen_tracks_as_negative_ids(tmp_path):
    data = _build(tmp_path)
    assert list(data["test_playlists"][4]) == [0, -1]
    assert data["unseen_names"] == {-1: "Zed – Zulu"}


def test_index_keeps_titles_and_artists(tmp_path):
    data = _build(tmp_path)
    assert data["test_titles"] == {4: "rock"}
    # Playlist train tên "rock" là 0 và 2: chứa a, b (playlist 0) và b, c (playlist 2)
    assert list(data["hybrid"].titles.scores("ROCK")) == [1, 2, 1]
    assert list(data["artists"]) == [0, 1, 2]
    assert data["unseen_artists"] == {-1: 3}
