"""Dashboard tests."""
from dashboard.components.track_cards import cover_html, panel_html, track_row_html


def test_track_row_escapes_names_from_data():
    row = track_row_html("<b>Artist</b> – <script>alert(1)</script>")
    assert "<script>" not in row and "<b>Artist" not in row
    assert "&lt;script&gt;" in row


def test_track_row_splits_artist_and_title():
    row = track_row_html("Oasis – Wonderwall - Remastered")
    assert '<div class="title">Wonderwall - Remastered</div>' in row
    assert '<div class="artist">Oasis</div>' in row


def test_track_row_marks_hits_and_draws_score_bar():
    row = track_row_html("A – B", rank=1, value="0.300", fill=0.5, is_hit=True)
    assert "track hit" in row and "width:50%" in row


def test_cover_color_is_stable_for_same_name():
    assert cover_html("Oasis") == cover_html("Oasis")


def test_panel_shows_stats_only_when_given():
    assert "stat-value" not in panel_html("T", "S", [])
    panel = panel_html("T", "S", [], [("0.039", "Recall@10"), ("0.40", "Precision@10")])
    assert panel.count('class="stat"') == 2 and "0.039" in panel and "0.40" in panel


class _FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


def _fake_api(monkeypatch, up=True):
    import requests

    health = {"status": "ok", "default_model": "hybrid", "models": {"hybrid": "Hybrid", "popularity": "Phổ biến"},
              "missing_models": ["als"], "n_tracks": 3, "n_train_playlists": 4}
    result = {"model": "hybrid", "n_seed_tracks_used": 1, "unknown_seed_tracks": [], "title_matched_playlists": 2,
              "recommendations": [{"rank": 1, "track_uri": "u:b", "track_name": "Bravo", "artist_name": "Bob",
                                   "score": 0.5, "fallback": False,
                                   "reasons": [{"type": "co_occurrence", "text": "Hay đi cùng “Ann – Alpha”"}]}]}
    calls = []

    def get(self, url, **kwargs):
        if not up:
            raise requests.ConnectionError("down")
        return _FakeResponse(health)

    def post(self, url, json=None, **kwargs):
        calls.append(json)
        return _FakeResponse(result)

    monkeypatch.setattr(requests.Session, "get", get)
    monkeypatch.setattr(requests.Session, "post", post)
    return calls


def _api_page():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file("../dashboard/pages/5_API_Recommend.py", default_timeout=30)


def test_api_page_shows_error_when_api_is_down(monkeypatch):
    _fake_api(monkeypatch, up=False)
    at = _api_page()
    at.run()
    assert "Không kết nối được API" in at.error[0].value


def test_api_page_sends_title_and_seeds_and_shows_reasons(monkeypatch):
    calls = _fake_api(monkeypatch)
    at = _api_page()
    at.run()
    at.text_input[0].input("rock")
    at.session_state["api_seeds"] = ["u:a"]
    at.run()
    next(b for b in at.button if "Gợi ý" in b.label).click().run()
    assert calls == [{"seed_tracks": ["u:a"], "title": "rock", "n": 10, "model": "hybrid"}]
    page = "".join(str(h.proto) for h in at.get("html"))
    assert "Bravo" in page and "Hay đi cùng" in page
