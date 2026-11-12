"""Recommendation page."""
import html
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st

from dashboard.components.track_cards import apply_theme, hero_html, panel_html, track_row_html
from recommender import app_index
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import max_recall_at_k, precision_at_k
from recommender.evaluation.r_precision import r_precision
from recommender.evaluation.recall_at_k import recall_at_k

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "hdfs_sim/music/analytics/app_index"

TOP_N = 10  # số bài hiện ra
EVAL_N = 50  # số bài mô hình gợi ý để chấm Recall@50, giống evaluate_models
MAX_SEARCH_RESULTS = 30
QUICK_PICKS = ("Wonderwall", "Fly Me To The Moon", "One Dance", "Mr. Brightside")


@st.cache_resource(show_spinner="Đang mở chỉ mục dữ liệu...")
def load_app_data():
    # Dữ liệu lớn được memory-map từ đĩa, không nạp hết vào RAM (xem recommender/app_index.py)
    return SimpleNamespace(**app_index.load(INDEX_DIR))


def track_name(item, data):
    return data.names[item] if item >= 0 else data.unseen_names[item]


def artist_of(item, data):
    return data.artists[item] if item >= 0 else data.unseen_artists[item]


def n_to_recommend(hidden):
    # R-precision chấm |hidden| gợi ý đầu nên cần danh sách đủ dài, giống evaluate_models
    return max(EVAL_N, len(hidden or ()))


def recommendation_panel(title, subtitle, items, values, data, hidden):
    """items: n_to_recommend(hidden) bài gợi ý, chỉ hiện TOP_N bài đầu.

    values[i] là số hiển thị của bài i; số 0 nghĩa là bài được bù từ danh sách phổ biến.
    """
    top = max((values[i] for i in items[:TOP_N]), default=0)
    rows = []
    for rank, item in enumerate(items[:TOP_N], 1):
        value = values[item]
        if value > 0:
            label = f"{value:,}" if isinstance(value, (int, np.integer)) else f"{value:.3f}"
            fill = value / top
        else:
            label, fill = "bù phổ biến", None
        rows.append(track_row_html(data.names[item], rank, label, fill, bool(hidden) and item in hidden))

    stats = ()
    if hidden:
        # NDCG và Precision luôn đạt được 1; mức tối đa của Recall hiện một lần ở tiêu đề playlist
        n_hits = len(set(items[:TOP_N]) & hidden)
        n_hits_eval = len(set(items[:EVAL_N]) & hidden)
        stats = (
            (f"{ndcg_at_k(items, hidden, TOP_N):.3f}", f"NDCG@{TOP_N} · bài đúng càng ở trên càng cao"),
            (f"{precision_at_k(items, hidden, TOP_N):.2f}", f"Precision@{TOP_N} · đúng {n_hits}/{TOP_N} gợi ý"),
            (f"{recall_at_k(items, hidden, EVAL_N):.3f}",
             f"Recall@{EVAL_N} · top {EVAL_N} đúng {n_hits_eval}/{len(hidden)} bài"),
            (f"{r_precision(items, hidden, lambda i: artist_of(i, data)):.3f}",
             "R-precision · đúng bài 1 điểm, chỉ đúng nghệ sĩ 0.25"),
        )
    return panel_html(title, subtitle, rows, stats)


def cooccurrence_panel(title, subtitle, model, seed, data, hidden):
    if not seed:
        subtitle = "Chưa có bài seed nên tạm dùng bài phổ biến"
    scores = model.score(seed)
    items = model.recommend(seed, n_to_recommend(hidden), scores)
    return recommendation_panel(title, subtitle, items, scores, data, hidden)


def hybrid_panel(seed, title, data, hidden):
    model = data.hybrid
    n_same_title = model.titles.n_playlists(title)
    if n_same_title:
        source = f"{n_same_title:,} playlist train cùng tên “{title}”"
    else:
        source = "tên playlist chưa gặp trong train" if title else "chưa có tên playlist"
    if not seed and not n_same_title:
        subtitle = f"Chưa có bài seed, {source} nên tạm dùng bài phổ biến"
    else:
        subtitle = f"Co-occurrence tinh chỉnh + tên playlist · {source}"
    scores = model.score(seed, title)
    items = model.recommend(seed, n_to_recommend(hidden), title=title, scores=scores)
    return recommendation_panel("Hybrid (bài + tên playlist)", subtitle, items, scores, data, hidden)


def show_recommendations(seed, data, hidden=None, title=None):
    top_left, top_right = st.columns(2)
    with top_left:
        st.html(recommendation_panel(
            "Popularity", "Bài phổ biến nhất · số lượt đếm bằng MapReduce",
            data.popularity.recommend(seed, n_to_recommend(hidden)), data.counts, data, hidden,
        ))
    with top_right:
        st.html(cooccurrence_panel(
            "Co-occurrence", "Bài hay nằm chung playlist · điểm cosine", data.cooccurrence, seed, data, hidden,
        ))
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        params = app_index.TUNED_PARAMS
        st.html(cooccurrence_panel(
            "Co-occurrence tinh chỉnh",
            f"Phạt bài phổ biến nhẹ hơn (mũ {params['item_exponent']}) · bỏ bài có dưới {params['min_count']} playlist",
            data.cooccurrence_tuned, seed, data, hidden,
        ))
    with bottom_right:
        st.html(hybrid_panel(seed, title, data, hidden))


def add_seed(item):
    if item is not None and item not in st.session_state.seeds:
        st.session_state.seeds.append(item)


def manual_tab(data):
    st.session_state.setdefault("seeds", [])

    title = st.text_input("Tên playlist (không bắt buộc, chỉ mô hình Hybrid dùng)",
                          placeholder="Ví dụ: workout, chill, christmas, country")
    query = st.text_input("Tìm bài hát hoặc nghệ sĩ", placeholder="Ví dụ: Wonderwall, Frank Sinatra, One Dance")
    matches = data.names.search(query, MAX_SEARCH_RESULTS)
    if query and not matches:
        st.warning("Không tìm thấy bài nào trong tập train. Thử tên khác nhé.")
    if matches:
        picked = st.selectbox(
            f"{len(matches)} kết quả phổ biến nhất",
            matches,
            format_func=lambda i: f"{data.names[i]} ({data.counts[i]:,} lượt)",
        )
        if st.button("＋ Thêm vào playlist", type="primary"):
            add_seed(picked)
            st.rerun()

    st.html('<div class="section-label">Thử nhanh</div>')
    for column, name in zip(st.columns(len(QUICK_PICKS)), QUICK_PICKS):
        if column.button(name, key=f"quick_{name}", width="stretch"):
            add_seed(next(iter(data.names.search(name, 1)), None))
            st.rerun()

    st.html(f'<div class="section-label">Playlist của bạn ({len(st.session_state.seeds)} bài)</div>')
    if not st.session_state.seeds:
        st.info("Playlist đang trống nên ba mô hình đầu chỉ gợi ý bài phổ biến; Hybrid vẫn gợi ý được nếu có "
                "tên playlist. Hãy thêm vài bài bạn thích.")
    for item in list(st.session_state.seeds):
        row_col, remove_col = st.columns([8, 1], vertical_alignment="center")
        row_col.html(track_row_html(data.names[item]))
        if remove_col.button("✕", key=f"remove_{item}", help="Xóa khỏi playlist"):
            st.session_state.seeds.remove(item)
            st.rerun()

    st.divider()
    show_recommendations(st.session_state.seeds, data, title=title)


def test_tab(data):
    st.write("Giấu phần lớn một playlist có thật, rồi xem mô hình đoán lại được bao nhiêu bài bị giấu.")
    # 0, 1, 5, 25 là các mức của đề bài (dùng trong báo cáo); 50, 100 chỉ để thử thêm trong app
    k = st.radio("Số bài đầu dùng làm seed", [0, 1, 5, 25, 50, 100], index=2, horizontal=True,
                 format_func=lambda k: "0 (chỉ biết tên)" if k == 0 else str(k))

    eligible = [pid for pid, tracks in sorted(data.test_playlists.items()) if len(tracks) >= k + TOP_N]
    if st.button("🔀 Đổi playlist khác") or st.session_state.get("test_pid") not in eligible:
        st.session_state.test_pid = random.choice(eligible)
    pid = st.session_state.test_pid
    tracks = data.test_playlists[pid]
    title = data.test_titles.get(pid)

    # Giống evaluate_models: k bài đầu là seed, đáp án là phần còn lại trừ bài đã có trong seed
    seed = [int(i) for i in tracks[:k]]
    hidden = {int(i) for i in tracks[k:]} - set(seed)
    st.html(
        f'<div class="section-label">Playlist #{pid} “{html.escape(title or "không tên")}”: '
        f"{len(tracks)} bài, biết trước {k}, giấu {len(hidden)} · "
        f"Recall@{EVAL_N} tối đa {max_recall_at_k(hidden, EVAL_N):.3f} (mỗi mô hình chỉ gợi ý {EVAL_N} bài)</div>"
    )
    if seed:
        with st.expander("Các bài seed", expanded=k <= 5):
            st.html("".join(track_row_html(track_name(i, data)) for i in seed))
    show_recommendations(seed, data, hidden, title)
    with st.expander("Xem đáp án (các bài bị giấu)"):
        st.html("".join(track_row_html(track_name(int(i), data)) for i in tracks[k:]))


apply_theme()

# row_titles.npy có từ khi thêm mô hình Hybrid; chỉ mục cũ thiếu file này phải dựng lại
if not (INDEX_DIR / "row_titles.npy").exists():
    st.error(
        "Chưa có chỉ mục dữ liệu (hoặc chỉ mục cũ, chưa có tên playlist). Chạy một lần (từ thư mục gốc repo): "
        "python -m recommender.app_index --flat hdfs_sim/music/processed/playlist_tracks_flat.tsv "
        "--split-file hdfs_sim/music/processed/playlist_split.tsv "
        "--popularity hdfs_sim/music/analytics/track_popularity_train/part-00000 "
        "--titles hdfs_sim/music/processed/playlist_titles.tsv "
        "--out hdfs_sim/music/analytics/app_index"
    )
    st.stop()

data = load_app_data()
st.html(hero_html(
    "Hôm nay nghe gì",
    "Gợi ý nhạc theo gu của bạn",
    f"So sánh 4 mô hình trên toàn bộ tập train: {data.meta['n_train_playlists']:,} playlist "
    f"và {data.meta['n_items']:,} bài. Tab playlist có thật dùng {len(data.test_playlists)} playlist "
    "ngẫu nhiên trong tập test.",
))

manual, test = st.tabs(["Tự chọn bài", "Thử trên playlist có thật"])
with manual:
    manual_tab(data)
with test:
    test_tab(data)
