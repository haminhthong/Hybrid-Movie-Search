"""Giao diện web trực quan (Streamlit Application) của MovieScout AI.

Cung cấp trải nghiệm tìm kiếm phim hiện đại với phong cách Dark Glassmorphism,
hiển thị poster, thông tin đạo diễn, diễn viên, cốt truyện và phân tích ranking.
"""

import html
import logging
import os
from typing import Any

import requests
import streamlit as st

logger = logging.getLogger(__name__)
API_URL = os.getenv("MOVIESCOUT_API_URL", "http://localhost:8000").rstrip("/")
API_TIMEOUT_SECONDS = float(os.getenv("MOVIESCOUT_API_TIMEOUT", "60"))
MAX_TOP_N = max(1, min(20, int(os.getenv("RERANK_K", "20"))))

# Cấu hình trang Streamlit
st.set_page_config(
    page_title="MovieScout AI — Hybrid Semantic Movie Search",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách Modern Dark Theme & Glassmorphism
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .main-title {
        font-size: 2.75rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .movie-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.25);
        border-color: rgba(129, 140, 248, 0.3);
    }
    .movie-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f1f5f9;
        margin-bottom: 8px;
    }
    .badge-tag {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .meta-text {
        color: #94a3b8;
        font-size: 0.92rem;
        line-height: 1.6;
    }
    .meta-highlight {
        color: #e2e8f0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_escape(value: Any) -> str:
    """Mã hóa chuỗi an toàn trước khi chèn vào HTML."""
    return html.escape(str(value), quote=True)


def search_movies(query: str, top_n: int, genre: str, year: str, debug: bool = False) -> dict[str, Any]:
    """Gọi FastAPI endpoint /search."""
    response = requests.post(
        f"{API_URL}/search",
        json={"query": query, "top_n": top_n, "genre": genre, "year": year, "debug": debug},
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code == 422:
        detail = response.json().get("detail", "Tham số tìm kiếm không hợp lệ.")
        raise ValueError(detail if isinstance(detail, str) else str(detail))
    response.raise_for_status()
    return response.json()


def render_movie(rank: int, movie: dict[str, Any], show_debug: bool = False) -> None:
    """Hiển thị thẻ thông tin kết quả phim."""
    director = movie.get("director") or "Chưa rõ"
    cast = ", ".join(movie.get("cast", [])) or "Chưa rõ"
    plot = movie.get("overview") or "Chưa có tóm tắt"
    year = movie.get("release_year") or str(movie.get("release_date", ""))[:4] or "N/A"
    title = movie.get("title", "Không rõ tên")
    vote_avg = movie.get("vote_average", 0)
    poster_path = movie.get("poster_path", "")
    genres = ", ".join(movie.get("genres", []))

    col_poster, col_info = st.columns([1, 4]) if poster_path else (None, None)

    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">
            #{rank}. {safe_escape(title)}
            <span style="color: #94a3b8; font-size: 1rem; font-weight: normal;">
                ({safe_escape(year)})
            </span>
        </div>
        <div style="margin-bottom: 12px;">
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-right: 14px;">
                ⭐ TMDB: <b style="color: #facc15;">{safe_escape(vote_avg)}/10</b>
            </span>
            <span style="color: #94a3b8; font-size: 0.88rem;">
                🎭 {safe_escape(genres)}
            </span>
        </div>
        <div class="meta-text">
            <p style="margin-bottom: 6px;">
                <span class="meta-highlight">🎬 Đạo diễn:</span> {safe_escape(director)}
            </p>
            <p style="margin-bottom: 6px;">
                <span class="meta-highlight">👥 Diễn viên:</span> {safe_escape(cast)}
            </p>
            <p style="margin-bottom: 0;">
                <span class="meta-highlight">📖 Cốt truyện:</span> {safe_escape(plot)}
            </p>
        </div>
    </div>
    """

    if poster_path and col_poster and col_info:
        with col_poster:
            poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}"
            st.image(poster_url, use_container_width=True)
        with col_info:
            st.markdown(card_html, unsafe_allow_html=True)
            if show_debug and "evidence" in movie and movie["evidence"]:
                ev = movie["evidence"]
                with st.expander("🔍 Chi tiết xếp hạng (Retrieval Evidence)"):
                    st.caption(
                        f"• **Dense Rank:** {ev.get('dense_rank', 'N/A')} | "
                        f"• **BM25 Rank:** {ev.get('sparse_rank', 'N/A')} | "
                        f"• **RRF Rank:** {ev.get('rrf_rank', 'N/A')} (Score: {ev.get('rrf_score', 0):.4f}) | "
                        f"• **Cross-Encoder Score:** {movie.get('rerank_score', 'N/A')}"
                    )
    else:
        st.markdown(card_html, unsafe_allow_html=True)
        if show_debug and "evidence" in movie and movie["evidence"]:
            ev = movie["evidence"]
            with st.expander("🔍 Chi tiết xếp hạng (Retrieval Evidence)"):
                st.caption(
                    f"• **Dense Rank:** {ev.get('dense_rank', 'N/A')} | "
                    f"• **BM25 Rank:** {ev.get('sparse_rank', 'N/A')} | "
                    f"• **RRF Rank:** {ev.get('rrf_rank', 'N/A')} (Score: {ev.get('rrf_score', 0):.4f}) | "
                    f"• **Cross-Encoder Score:** {movie.get('rerank_score', 'N/A')}"
                )


# Header ứng dụng
st.markdown("<div class='main-title'>🎬 MovieScout AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>Hybrid Movie Search: Dense (MiniLM) + BM25 → RRF → Cross-Encoder Reranking</div>",
    unsafe_allow_html=True,
)

genres_list = [
    "All",
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War",
    "Western",
]

# Sidebar
with st.sidebar:
    st.header("⚙️ Cấu Hình")
    st.info(
        "**MovieScout AI** kết hợp ngữ nghĩa (Dense MiniLM) và từ khóa chính xác (BM25), "
        "hợp nhất bằng Reciprocal Rank Fusion và rerank bằng Cross-Encoder."
    )
    top_n_slider = st.slider(
        "Số kết quả hiển thị (top_n)",
        min_value=1,
        max_value=MAX_TOP_N,
        value=min(10, MAX_TOP_N),
    )
    debug_mode = st.checkbox("Hiển thị chi tiết ranking (Debug mode)", value=False)
    st.markdown("---")
    st.markdown("<b>Pipeline Kiến Trúc:</b>", unsafe_allow_html=True)
    st.caption("1. Dense Retrieval: all-MiniLM-L6-v2")
    st.caption("2. Lexical Retrieval: FastEmbed BM25")
    st.caption("3. Fusion: Reciprocal Rank Fusion (RRF)")
    st.caption("4. Two-Stage Reranker: ms-marco-MiniLM-L-6-v2")
    st.caption("5. Vector Store: Qdrant")

# Form tìm kiếm
with st.form("search_form"):
    col_genre, col_year = st.columns(2)
    genre_selected = col_genre.selectbox("Thể loại phim", genres_list)
    year_input = col_year.text_input(
        "Năm sản xuất",
        placeholder="Ví dụ: 2014 hoặc 2010-2020",
    )

    query_input = st.text_input(
        "Mô tả nội dung, ý tưởng hoặc cốt truyện phim",
        placeholder="Ví dụ: 'A team of explorers travels through a wormhole in space to save humanity...'",
    )
    submit_button = st.form_submit_button("🔍 Tìm Kiếm Phim", use_container_width=True)

if submit_button:
    if not query_input.strip():
        st.warning("⚠️ Vui lòng nhập mô tả cốt truyện phim trước khi tìm kiếm.")
        st.stop()

    try:
        with st.spinner("🚀 Đang truy hồi hybrid (Dense + BM25) và Cross-Encoder reranking..."):
            response = search_movies(
                query=query_input,
                top_n=top_n_slider,
                genre=genre_selected,
                year=year_input,
                debug=debug_mode,
            )

        movies = response.get("results", [])
        latency = response.get("latency_ms", 0.0)

        if not movies:
            st.info("💡 Không tìm thấy bộ phim nào phù hợp với điều kiện lọc và mô tả của bạn.")
        else:
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("Số kết quả tìm thấy", f"{len(movies)} phim")
            m_col2.metric("Thời gian xử lý", f"{latency:.2f} ms")

            st.write("### 🍿 Kết Quả Tìm Kiếm:")
            for rank, movie in enumerate(movies, start=1):
                render_movie(movie.get("rank", rank), movie, show_debug=debug_mode)

    except ValueError as exc:
        st.warning(f"⚠️ Tham số không hợp lệ: {exc}")
    except requests.Timeout:
        st.error("❌ Search API phản hồi quá thời gian. Vui lòng thử lại.")
    except requests.RequestException:
        logger.exception("Không thể gọi Search API tại %s", API_URL)
        st.error(f"❌ Không kết nối được Search API tại {API_URL}. Vui lòng khởi động backend API.")
