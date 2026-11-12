"""Giao diện kiểu app nghe nhạc: CSS chung và các hàng bài hát dạng HTML."""
import html
import zlib

import streamlit as st

CSS = """
.stApp {
  background:
    radial-gradient(1000px 480px at 12% -8%, rgba(168, 85, 247, .38), transparent 70%),
    radial-gradient(800px 420px at 95% 0%, rgba(236, 72, 153, .22), transparent 70%),
    #120C1F;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 1120px; padding-top: 2.2rem; }

.hero { padding: 6px 0 18px; }
.hero-eyebrow { color: #C4B5FD; font-size: .78rem; font-weight: 700; letter-spacing: .14em; text-transform: uppercase; }
.hero-title { color: #fff; font-size: clamp(2rem, 5vw, 3.1rem); font-weight: 800; line-height: 1.1; margin: 6px 0 10px; }
.hero-note { color: #A79DBD; font-size: .9rem; }

.stButton button, .stFormSubmitButton button { border-radius: 999px; font-weight: 600; }
.stTextInput [data-baseweb="input"], .stSelectbox [data-baseweb="select"] > div { border-radius: 12px; }

.section-label { color: #fff; font-size: 1.05rem; font-weight: 700; margin: 4px 0 2px; }

.panel { background: linear-gradient(180deg, rgba(255, 255, 255, .08), rgba(255, 255, 255, .03));
  border: 1px solid rgba(255, 255, 255, .08); border-radius: 18px; padding: 18px 14px 12px; }
.panel-head { display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; padding: 0 8px 12px; }
.panel-title { color: #fff; font-size: 1.3rem; font-weight: 800; }
.panel-sub { color: #A79DBD; font-size: .82rem; margin-top: 2px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(90px, 1fr)); gap: 8px; margin: 0 8px 12px; }
.stat { background: rgba(255, 255, 255, .06); border-radius: 12px; padding: 8px 10px; }
.stat-value { color: #fff; font-size: 1.2rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.stat-label { color: #A79DBD; font-size: .72rem; }

.track { display: flex; align-items: center; gap: 12px; padding: 8px; border-radius: 10px; transition: background .15s; }
.track:hover { background: rgba(255, 255, 255, .08); }
.track.hit { background: rgba(52, 211, 153, .12); }
.rank { width: 20px; flex: none; color: #A79DBD; font-size: .85rem; text-align: right; }
.cover { width: 42px; height: 42px; flex: none; border-radius: 8px; display: flex; align-items: center;
  justify-content: center; color: #fff; font-weight: 800; }
.info { min-width: 0; flex: 1; }
.title, .artist { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.title { color: #fff; font-size: .92rem; font-weight: 600; }
.artist { color: #A79DBD; font-size: .8rem; }
.value { width: 82px; flex: none; text-align: right; color: #D8D1E6; font-size: .8rem; font-variant-numeric: tabular-nums; }
.bar { height: 4px; margin-top: 5px; border-radius: 999px; background: rgba(255, 255, 255, .12); overflow: hidden; }
.bar > div { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #A855F7, #EC4899); }
.tag { color: #C4B5FD; font-size: .72rem; }
.hit-mark { color: #34D399; font-weight: 800; flex: none; }
"""


def apply_theme():
    st.html(f"<style>{CSS}</style>")


def hero_html(eyebrow, title, note):
    return (
        f'<div class="hero"><div class="hero-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="hero-title">{html.escape(title)}</div>'
        f'<div class="hero-note">{html.escape(note)}</div></div>'
    )


def cover_html(name):
    """Dữ liệu không có ảnh bìa, nên sinh ô màu cố định theo tên bài."""
    hue = zlib.crc32(name.encode("utf-8")) % 360
    style = f"background:linear-gradient(135deg,hsl({hue},72%,58%),hsl({(hue + 50) % 360},75%,32%))"
    return f'<div class="cover" style="{style}">{html.escape(name[:1].upper())}</div>'


def track_row_html(name, rank=None, value=None, fill=None, is_hit=False):
    """name có dạng "nghệ sĩ – tên bài"; fill là tỉ lệ 0-1 của thanh điểm, None thì hiện dạng nhãn."""
    artist, _, title = name.partition(" – ")
    rank_part = f'<div class="rank">{rank}</div>' if rank is not None else ""
    if value is None:
        value_part = ""
    elif fill is None:
        value_part = f'<div class="value"><span class="tag">{html.escape(value)}</span></div>'
    else:
        value_part = (
            f'<div class="value">{html.escape(value)}'
            f'<div class="bar"><div style="width:{fill * 100:.0f}%"></div></div></div>'
        )
    hit_part = '<div class="hit-mark">✓</div>' if is_hit else ""
    return (
        f'<div class="track{" hit" if is_hit else ""}">{rank_part}{cover_html(artist)}'
        f'<div class="info"><div class="title">{html.escape(title)}</div>'
        f'<div class="artist">{html.escape(artist)}</div></div>{value_part}{hit_part}</div>'
    )


def panel_html(title, subtitle, rows, stats=()):
    """stats: các cặp (giá trị, nhãn) hiện thành hàng ô số liệu dưới tiêu đề."""
    stats_part = ""
    if stats:
        cells = "".join(
            f'<div class="stat"><div class="stat-value">{html.escape(value)}</div>'
            f'<div class="stat-label">{html.escape(label)}</div></div>'
            for value, label in stats
        )
        stats_part = f'<div class="stats">{cells}</div>'
    return (
        f'<div class="panel"><div class="panel-head"><div><div class="panel-title">{html.escape(title)}</div>'
        f'<div class="panel-sub">{html.escape(subtitle)}</div></div></div>{stats_part}{"".join(rows)}</div>'
    )
