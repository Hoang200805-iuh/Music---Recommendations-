"""
Dashboard entry point.

Cách chạy (từ thư mục gốc repo):
    streamlit run dashboard/app.py --server.address 127.0.0.1
"""
import sys
from pathlib import Path

import streamlit as st

# Streamlit chỉ thêm thư mục dashboard/ vào sys.path, cần thêm gốc repo để import recommender
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

st.set_page_config(page_title="Gợi ý nhạc", page_icon="🎧", layout="wide")

# Khai báo tường minh để sidebar không hiện các trang còn trống của khung repo
page = st.navigation([st.Page("pages/3_Recommendation.py", title="Gợi ý nhạc", icon="🎧")])
page.run()
