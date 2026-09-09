"""Giao diện web tương tác (Streamlit Application) của MovieScout AI.

Cung cấp trải nghiệm tìm kiếm phim hiện đại với phong cách thiết kế Dark Glassmorphism,
hiển thị trực quan điểm tương quan tương đối, độ trễ phản hồi (Latency),
thanh điểm tương quan (Relevance Progress Bar) và chi tiết metadata của bộ phim.
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
    page_title="MovieScout AI — Hybrid Semantic Search Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS cho phong cách Modern Dark Theme & Glassmorphism
st.markdown(
    """
    <style>
    /* Nền chính ứng dụng */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    /* Tiêu đề ứng dụng */
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

    /* Thẻ hiển thị bộ phim (Glassmorphism Card) */
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

    .badge-score {
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
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

    /* Thanh điểm tương quan Progress Bar custom */
    .score-bar-bg {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        height: 8px;
        width: 100%;
        margin-top: 8px;
        margin-bottom: 12px;
        overflow: hidden;
    }

    .score-bar-fill {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        height: 100%;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_escape(value: Any) -> str:
    """Mã hóa chuỗi an toàn trước khi chèn vào HTML để tránh lỗi XSS injection."""
    return html.escape(str(value), quote=True)


def search_movies(query: str, top_n: int, genre: str, year: str) -> dict[str, Any]:
    """Gọi FastAPI; UI không tự nạp model hoặc truy cập Qdrant trực tiếp."""

    response = requests.post(
        f"{API_URL}/search",
        json={"query": query, "top_n": top_n, "genre": genre, "year": year},
        timeout=API_TIMEOUT_SECONDS,
    )
    if response.status_code == 422:
        detail = response.json().get("detail", "Tham số tìm kiếm không hợp lệ.")
        raise ValueError(detail if isinstance(detail, str) else str(detail))
    response.raise_for_status()
    return response.json()


def render_movie(rank: int, movie: dict[str, Any]) -> None:
    """Hiển thị một thẻ thông tin kết quả phim sang trọng."""
    director = movie.get("director") or "Chưa rõ"
    cast = ", ".join(movie.get("cast", [])) or "Chưa rõ"
    plot = movie.get("overview") or "Chưa có tóm tắt"
    year = movie.get("release_year") or str(movie.get("release_date", ""))[:4] or "N/A"
    display_score = float(movie.get("display_score", 0.0))
    rerank_score = float(movie.get("rerank_score", 0.0))
    score_pct = int(max(0.0, min(display_score, 1.0)) * 100)
    title = movie.get("title", "Không rõ tên")
    vote_avg = movie.get("vote_average", 0)
    poster_path = movie.get("poster_path", "")
    genres = ", ".join(movie.get("genres", []))

    # Tạo cột hiển thị ảnh poster (nếu có) và thông tin chi tiết
    col_poster, col_info = st.columns([1, 4]) if poster_path else (None, None)

    card_html = f"""
    <div class="movie-card">
        <div class="movie-title">
            #{rank}. {safe_escape(title)}
            <span style="color: #94a3b8; font-size: 1rem; font-weight: normal;">
                ({safe_escape(year)})
            </span>
        </div>
        <div style="margin-bottom: 10px;">
            <span class="badge-score">Display relevance: {display_score:.4f}</span>
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 12px;">
                CE: {rerank_score:.4f}
            </span>
            <span style="color: #cbd5e1; font-size: 0.88rem; margin-left: 12px;">
                ⭐ TMDB Rating: <b style="color: #facc15;">{safe_escape(vote_avg)}/10</b>
            </span>
            <span style="color: #94a3b8; font-size: 0.88rem; margin-left: 12px;">
                🎭 {safe_escape(genres)}
            </span>
        </div>
        <div class="score-bar-bg">
            <div class="score-bar-fill" style="width: {score_pct}%;"></div>
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
    else:
        st.markdown(card_html, unsafe_allow_html=True)


# Header ứng dụng
st.markdown("<div class='main-title'>🎬 MovieScout AI</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-title'>BM25 + Dense Retrieval → RRF → Cross-Encoder Reranking</div>",
    unsafe_allow_html=True,
)

# Thư viện thể loại phim
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

# Sidebar thông tin dự án
with st.sidebar:
    st.header("⚙️ Cấu Hình & Thông Tin")
    st.info(
        "**MovieScout AI** truy hồi hai nhánh song song, hợp nhất bằng RRF "
        "và rerank candidate bằng Cross-Encoder. Query v1 hỗ trợ tiếng Anh."
    )
    top_n_slider = st.slider(
        "Số kết quả hiển thị (top_n)",
        min_value=1,
        max_value=MAX_TOP_N,
        value=min(10, MAX_TOP_N),
    )
    st.markdown("---")
    st.markdown("<b>Công nghệ sử dụng:</b>", unsafe_allow_html=True)
    st.caption(f"• Search API: {API_URL}")
    st.caption("• Vector DB: Qdrant Cloud / Local")
    st.caption("• Dense Model: sentence-transformers/all-MiniLM-L6-v2")
    st.caption("• Sparse Model: FastEmbed Qdrant/bm25")
    st.caption("• Reranker: ms-marco-MiniLM-L-6-v2")

# Form tìm kiếm
with st.form("search_form"):
    col_genre, col_year = st.columns(2)
    genre_selected = col_genre.selectbox("Thể loại phim", genres_list)
    year_input = col_year.text_input(
        "Năm sản xuất",
        placeholder="Ví dụ: 2014 hoặc 2000-2020",
    )

    query_input = st.text_input(
        "Mô tả nội dung cốt truyện bộ phim cần tìm",
        placeholder=(
            "Mô tả ý tưởng, ví dụ: 'A team of explorers travels through a wormhole "
            "in space to save humanity...'"
        ),
    )
    submit_button = st.form_submit_button("🔍 Tìm Kiếm Phim", use_container_width=True)

# Xử lý sự kiện Submit
if submit_button:
    if not query_input.strip():
        st.warning("⚠️ Vui lòng nhập mô tả cốt truyện phim trước khi bấm Tìm Kiếm.")
        st.stop()

    try:
        with st.spinner("🚀 Đang truy hồi vector và tính toán độ tương quan ngữ nghĩa..."):
            response = search_movies(
                query=query_input,
                top_n=top_n_slider,
                genre=genre_selected,
                year=year_input,
            )

        movies = response["results"]
        latency = response["latency_ms"]

        if not movies:
            st.info("💡 Không tìm thấy bộ phim nào phù hợp với điều kiện lọc và mô tả của bạn.")
        else:
            # Hiển thị các Metrics
            m_col1, m_col2, m_col3 = st.columns(3)
            m_col1.metric("Số kết quả tìm thấy", f"{len(movies)} phim")
            m_col2.metric("Thời gian phản hồi", f"{latency:.2f} ms")
            m_col3.metric("Index version", response["index_version"])

            st.write("")

            st.write("### 🍿 Danh Sách Phim Tương Quan Nhất:")
            for rank, movie in enumerate(movies, start=1):
                render_movie(movie.get("rank", rank), movie)

    except ValueError as exc:
        st.warning(f"⚠️ Tham số không hợp lệ: {exc}")
    except requests.Timeout:
        st.error("❌ Search API phản hồi quá thời gian. Vui lòng thử lại.")
    except requests.HTTPError:
        logger.exception("Search API trả về lỗi HTTP tại %s", API_URL)
        st.error("❌ Search API chưa sẵn sàng. Hãy kiểm tra /ready và index Qdrant.")
    except requests.RequestException:
        logger.exception("Không thể gọi Search API tại %s", API_URL)
        st.error(
            "❌ Không kết nối được Search API. "
            "Hãy khởi động FastAPI và kiểm tra MOVIESCOUT_API_URL."
        )
    except Exception:
        logger.exception("Tìm kiếm gặp sự cố hệ thống")
        st.error(
            "❌ Dịch vụ tìm kiếm tạm thời không sẵn sàng. "
            "Vui lòng kiểm tra lại kết nối Qdrant/API Key và thử lại."
        )
