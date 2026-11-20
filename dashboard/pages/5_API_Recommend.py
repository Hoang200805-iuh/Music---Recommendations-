"""
Trang gợi ý qua API: nhập tên playlist, chọn bài (tìm tự gợi ý khi gõ), gọi FastAPI, hiện Top-10 kèm lý do.

Trang này không nạp mô hình nào, mọi thứ đi qua API (api/main.py). Chạy API trước:
    uvicorn api.main:app --port 8000
Đổi địa chỉ API bằng biến môi trường MUSIC_API_URL.
"""
import html
import os
import time

import requests
import streamlit as st
from streamlit_searchbox import st_searchbox

from dashboard.components.track_cards import apply_theme, cover_html, hero_html

API_URL = os.environ.get("MUSIC_API_URL", "http://127.0.0.1:8000").rstrip("/")
TOP_N = 10
REASON_STYLE = {  # biểu tượng, màu chữ, màu nền
    "co_occurrence": ("🔗", "#F9A8D4", "rgba(236, 72, 153, .16)"),
    "similar_playlists": ("👥", "#C4B5FD", "rgba(168, 85, 247, .18)"),
    "title": ("🏷️", "#93C5FD", "rgba(59, 130, 246, .16)"),
    "popular": ("🔥", "#FCD34D", "rgba(245, 158, 11, .15)"),
}
CSS = """
.result { display: flex; gap: 12px; padding: 10px 8px; border-radius: 12px; }
.result:hover { background: rgba(255, 255, 255, .06); }
.result .body { min-width: 0; flex: 1; }
.reasons { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 6px; }
.reason { font-size: .74rem; padding: 3px 9px; border-radius: 999px; line-height: 1.4; }
.seed { display: flex; align-items: center; gap: 10px; }
"""

session = requests.Session()


@st.cache_data(ttl=30, show_spinner=False)
def api_health():
    response = session.get(f"{API_URL}/health", timeout=5)
    response.raise_for_status()
    return response.json()


def search_tracks(query):
    """Gọi khi người dùng gõ vào ô tìm bài; trả về các cặp (nhãn hiển thị, URI)."""
    if not query or not query.strip():
        return []
    try:
        response = session.get(f"{API_URL}/tracks/search", params={"q": query.strip(), "limit": 10}, timeout=5)
        response.raise_for_status()
    except requests.RequestException:
        return []
    options = []
    for track in response.json():
        label = f"{track['artist_name']} – {track['track_name']}"
        st.session_state.track_labels[track["track_uri"]] = label
        options.append((f"{label}  ·  {track['n_playlists']:,} playlist", track["track_uri"]))
    return options


def add_seed(uri):
    if uri and uri not in st.session_state.api_seeds:
        st.session_state.api_seeds.append(uri)
        st.session_state.pop("api_result", None)


def reason_chip(reason):
    icon, color, background = REASON_STYLE.get(reason["type"], ("•", "#D8D1E6", "rgba(255, 255, 255, .08)"))
    return (f'<span class="reason" style="color:{color};background:{background}">'
            f'{icon} {html.escape(reason["text"])}</span>')


def result_html(item):
    name = f"{item['artist_name']} – {item['track_name']}"
    chips = "".join(reason_chip(r) for r in item["reasons"])
    return (
        f'<div class="result"><div class="rank" style="padding-top:12px">{item["rank"]}</div>{cover_html(name)}'
        f'<div class="body"><div class="title">{html.escape(item["track_name"])}</div>'
        f'<div class="artist">{html.escape(item["artist_name"])}</div>'
        f'<div class="reasons">{chips}</div></div></div>'
    )


apply_theme()
st.html(f"<style>{CSS}</style>")
st.html(hero_html("Gợi ý qua API", "Tạo playlist của bạn",
                  f"Nhập tên playlist, chọn vài bài, rồi xem Top-{TOP_N} bài gợi ý kèm lý do. "
                  f"Mọi gợi ý lấy từ FastAPI tại {API_URL}."))

try:
    health = api_health()
except requests.RequestException:
    st.error(f"Không kết nối được API ở {API_URL}. Chạy API trước (từ thư mục gốc repo): "
             "`uvicorn api.main:app --port 8000`, rồi tải lại trang.")
    st.stop()

st.session_state.setdefault("api_seeds", [])
st.session_state.setdefault("track_labels", {})

left, right = st.columns([3, 2], gap="large")
with left:
    title = st.text_input("Tên playlist", placeholder="Ví dụ: workout, chill, christmas, country")
    st.html('<div class="section-label">Thêm bài hát</div>')
    picked = st_searchbox(search_tracks, placeholder="Gõ tên bài hoặc nghệ sĩ, ví dụ: Wonderwall, Adele…",
                          key="api_track_search", clear_on_submit=True, debounce=250)
    if picked:
        add_seed(picked)

    st.html(f'<div class="section-label">Playlist của bạn ({len(st.session_state.api_seeds)} bài)</div>')
    if not st.session_state.api_seeds:
        st.caption("Chưa có bài nào. Chỉ cần tên playlist cũng gợi ý được, thêm bài thì gợi ý sát gu hơn.")
    for uri in list(st.session_state.api_seeds):
        label = st.session_state.track_labels.get(uri, uri)
        row, remove = st.columns([9, 1], vertical_alignment="center")
        row.html(f'<div class="seed">{cover_html(label)}<div class="title">{html.escape(label)}</div></div>')
        if remove.button("✕", key=f"api_remove_{uri}", help="Xóa khỏi playlist"):
            st.session_state.api_seeds.remove(uri)
            st.session_state.pop("api_result", None)
            st.rerun()

with right:
    models = health["models"]
    model = st.selectbox("Mô hình", list(models), index=list(models).index(health["default_model"]),
                         format_func=lambda name: f"{name}: {models[name]}")
    if health["missing_models"]:
        st.caption(f"Chưa nạp: {', '.join(health['missing_models'])} (thiếu artifact khi API khởi động)")
    st.caption(f"API: {health['n_tracks']:,} bài, {health['n_train_playlists']:,} playlist train.")
    can_run = bool(st.session_state.api_seeds or title.strip())
    if st.button(f"Gợi ý Top-{TOP_N}", type="primary", disabled=not can_run, width="stretch"):
        start = time.time()
        try:
            response = session.post(f"{API_URL}/recommend", timeout=30, json={
                "seed_tracks": st.session_state.api_seeds, "title": title.strip() or None, "n": TOP_N, "model": model})
            response.raise_for_status()
            st.session_state.api_result = (response.json(), time.time() - start)
        except requests.RequestException as e:
            st.session_state.pop("api_result", None)
            st.error(f"API báo lỗi: {e}")
    if not can_run:
        st.caption("Nhập tên playlist hoặc thêm ít nhất một bài để gợi ý.")
    st.html("".join(
        f'<div class="reasons" style="margin-top:4px">{reason_chip({"type": t, "text": text})}</div>'
        for t, text in (("co_occurrence", "Hay đi cùng một bài trong playlist của bạn"),
                        ("similar_playlists", "Cùng gu với playlist của bạn"),
                        ("title", "Cùng gu với các playlist cùng tên"),
                        ("popular", "Phổ biến"))))

if "api_result" in st.session_state:
    result, seconds = st.session_state.api_result
    st.divider()
    st.html(f'<div class="section-label">Top-{len(result["recommendations"])} gợi ý · mô hình {result["model"]} · '
            f'{seconds:.2f} giây</div>')
    if result["unknown_seed_tracks"]:
        st.warning(f"{len(result['unknown_seed_tracks'])} bài không có trong dữ liệu train nên bị bỏ qua.")
    st.html('<div class="panel">' + "".join(result_html(item) for item in result["recommendations"]) + "</div>")
