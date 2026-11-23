"""Recommendation page."""
import html
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import streamlit as st

from dashboard.components.track_cards import apply_theme, hero_html, panel_html, track_row_html
from recommender import app_index
from recommender.als.serving_als import load_als
from recommender.evaluation.ndcg_at_k import ndcg_at_k
from recommender.evaluation.precision_at_k import max_recall_at_k, precision_at_k
from recommender.evaluation.r_precision import r_precision
from recommender.evaluation.recall_at_k import recall_at_k

ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = Path(os.environ.get("MUSIC_APP_INDEX", ROOT / "hdfs_sim/music/analytics/app_index"))
ALS_DIR = Path(os.environ.get(
    "MUSIC_ALS_DIR",
    ROOT / "outputs/als_local/models/als_implicit",
))

TOP_N = 10  # số bài hiện ra
EVAL_N = 50  # số bài mô hình gợi ý để chấm Recall@50, giống evaluate_models
MAX_SEARCH_RESULTS = 30
QUICK_PICKS = ("Wonderwall", "Fly Me To The Moon", "One Dance", "Mr. Brightside")


@st.cache_resource(show_spinner="Đang mở chỉ mục dữ liệu...")
def load_app_data():
    # Dữ liệu lớn được memory-map từ đĩa, không nạp hết vào RAM (xem recommender/app_index.py)
    data = app_index.load(INDEX_DIR)
    data["als"], data["n_als_tracks"] = load_als(ALS_DIR, data["uris"], data["popularity"]) if data["uris"] is not None else (None, 0)
    return SimpleNamespace(**data)


def track_name(item, data):
    return data.names[item] if item >= 0 else data.unseen_names[item]


def artist_of(item, data):
    return data.artists[item] if item >= 0 else data.unseen_artists[item]


def n_to_recommend(hidden):
    # R-precision chấm |hidden| gợi ý đầu nên cần danh sách đủ dài, giống evaluate_models
    return max(EVAL_N, len(hidden or ()))


def score_labels(items, values):
    """Nhãn và độ dài thanh điểm của từng bài: values[i] là số hiển thị của bài i;
    số 0 nghĩa là bài được bù từ danh sách phổ biến."""
    top = max((values[i] for i in items), default=0)
    labels = {}
    for item in items:
        value = values[item]
        if value > 0:
            labels[item] = (f"{value:,}" if isinstance(value, (int, np.integer)) else f"{value:.3f}", value / top)
        else:
            labels[item] = ("bù phổ biến", None)
    return labels


def recommendation_panel(title, subtitle, items, values, data, hidden, labels=None):
    """items: n_to_recommend(hidden) bài gợi ý, chỉ hiện TOP_N bài đầu.

    values: số hiển thị của từng bài (xem score_labels); labels: {bài: (nhãn, thanh điểm)} để thay values.
    """
    labels = labels or score_labels(items[:TOP_N], values)
    rows = []
    for rank, item in enumerate(items[:TOP_N], 1):
        label, fill = labels[item]
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


def rerank_panel(seed, title, data, hidden):
    model = data.rerank
    candidates, scores = model.score(seed, title)
    items = model.recommend(seed, n_to_recommend(hidden), title=title, scored=(candidates, scores))
    if scores is None:
        subtitle = "Chưa có bài seed nên giữ nguyên thứ tự của Hybrid + User-User (xếp lại không giúp ích ở k = 0)"
    else:
        subtitle = (f"Hybrid + User-User chọn {len(candidates)} ứng viên, LightGBM xếp lại theo 22 đặc trưng"
                    " · nhãn: thứ hạng ở mô hình gốc")
    base_rank = {int(item): rank for rank, item in enumerate(candidates, 1)}
    labels = {}
    for rank, item in enumerate(items[:TOP_N], 1):
        before = base_rank.get(item)
        if before is None:
            labels[item] = ("bù phổ biến", None)
        elif before > rank:
            labels[item] = (f"↑ từ #{before}", None)
        elif before < rank:
            labels[item] = (f"↓ từ #{before}", None)
        else:
            labels[item] = (f"= #{before}", None)
    return recommendation_panel("Hybrid + LightGBM xếp hạng lại", subtitle, items, None, data, hidden, labels)


RERANK_NOTE = """
1. **Chọn ứng viên:** Hybrid (Co-occurrence tinh chỉnh + tên playlist) cộng thêm User-User kNN
   (100 playlist giống nhất) chọn ra 300 bài.
2. **Xếp hạng lại:** LightGBM (LambdaRank, 657 cây) chấm lại 300 bài dựa trên điểm của từng mô hình con,
   số bài seed cùng nghệ sĩ, độ phổ biến của bài so với các bài seed, độ đa dạng nghệ sĩ của playlist…

Huấn luyện trên tập validation; trên 2000 playlist test, NDCG@10 tăng 3–12% và Recall@50 tăng 4–9%
so với mô hình gốc ở k ≥ 1 (k = 0 giữ nguyên).

Nhãn **↑ từ #14** nghĩa là bài đứng thứ 14 ở mô hình gốc đã được LightGBM kéo lên.
"""


def show_recommendations(seed, data, hidden=None, title=None):
    # Popularity không hiện thành ô riêng (kết quả thấp), chỉ còn dùng để bù bài khi các mô hình thiếu ứng viên
    top_left, top_right = st.columns(2)
    with top_left:
        st.html(cooccurrence_panel(
            "Co-occurrence", "Bài hay nằm chung playlist · điểm cosine", data.cooccurrence, seed, data, hidden,
        ))
    with top_right:
        params = app_index.TUNED_PARAMS
        st.html(cooccurrence_panel(
            "Co-occurrence tinh chỉnh",
            f"Phạt bài phổ biến nhẹ hơn (mũ {params['item_exponent']}) · bỏ bài có dưới {params['min_count']} playlist",
            data.cooccurrence_tuned, seed, data, hidden,
        ))
    bottom_left, bottom_right = st.columns(2)
    with bottom_left:
        st.html(hybrid_panel(seed, title, data, hidden))
    with bottom_right:
        if data.rerank is None:
            st.info("Chưa có mô hình LightGBM: cài lightgbm và chép reranker_lgbm.txt (đầu ra của "
                    "recommender.rerank.train_reranker) vào thư mục chỉ mục, rồi khởi động lại app.")
        else:
            st.html(rerank_panel(seed, title, data, hidden))
    if data.als is not None:
        scores = data.als.score(seed)
        items = data.als.recommend(seed, n_to_recommend(hidden), scores)
        labels = {i: (f"{scores[i]:.3f}", None) if np.isfinite(scores[i]) else ("bù phổ biến", None)
                  for i in items}
        st.html(recommendation_panel("ALS", "Spark implicit ALS · suy luận từ các bài seed",
                                     items, scores, data, hidden, labels=labels))
    else:
        st.info("Chưa có bundle ALS. Train theo docs/run_als_local.md và đặt MUSIC_ALS_DIR tới thư mục bundle rồi khởi động lại app.")
    with st.expander("Mô hình Hybrid + LightGBM hoạt động thế nào?"):
        st.markdown(RERANK_NOTE)


def add_seed(item):
    if item is not None and item not in st.session_state.seeds:
        st.session_state.seeds.append(item)


def manual_tab(data):
    st.session_state.setdefault("seeds", [])

    title = st.text_input("Tên playlist (không bắt buộc, chỉ Hybrid và LightGBM dùng)",
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
        st.info("Playlist đang trống nên hai mô hình Co-occurrence chỉ gợi ý bài phổ biến; Hybrid và LightGBM vẫn gợi ý được nếu có "
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
    f"So sánh {3 + int(data.rerank is not None) + int(data.als is not None)} mô hình trên toàn bộ tập train: {data.meta['n_train_playlists']:,} playlist "
    f"và {data.meta['n_items']:,} bài. Tab playlist có thật dùng {len(data.test_playlists)} playlist "
    "ngẫu nhiên trong tập test.",
))

manual, test = st.tabs(["Tự chọn bài", "Thử trên playlist có thật"])
with manual:
    manual_tab(data)
with test:
    test_tab(data)
