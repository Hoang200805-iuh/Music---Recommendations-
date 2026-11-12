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
