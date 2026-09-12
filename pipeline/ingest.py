"""Module thu thập dữ liệu tự động từ The Movie Database (TMDB) REST API.

Tải thông tin metadata phim bao gồm: Tiêu đề, tóm tắt nội dung (overview),
đạo diễn (director), top 5 diễn viên chính (cast), từ khóa (keywords), điểm đánh giá (vote_average),
độ phổ biến (popularity) và hình ảnh poster.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

# Lấy khóa API TMDB từ biến môi trường
TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")
OUTPUT_FILE: Path = Path(__file__).resolve().parent / "data" / "movies_raw.csv"
BASE_URL: str = "https://api.themoviedb.org/3"

# Khoảng dừng giữa các request để hạn chế rate limit (tính bằng giây)
REQUEST_DELAY: float = 0.05
REQUEST_TIMEOUT: int = 15


def _request_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Thực hiện HTTP GET request tới TMDB API và trả về nội dung định dạng JSON.

    Args:
        path: Đường dẫn endpoint tương đối (ví dụ: "movie/550").
        params: Các tham số truy vấn URL (query parameters).

    Returns:
        Dict chứa dữ liệu JSON kết quả từ TMDB API.

    Raises:
        RuntimeError: Nếu biến môi trường TMDB_API_KEY chưa được cấu hình.
        requests.RequestException: Nếu có lỗi kết nối mạng hoặc lỗi HTTP status code.
    """
    api_key = os.getenv("TMDB_API_KEY", TMDB_API_KEY).strip()
    if not api_key:
        raise RuntimeError("Thiếu biến môi trường TMDB_API_KEY. Vui lòng thêm khóa API vào tệp .env.")

    url = f"{BASE_URL}/{path.lstrip('/')}"
    query_params = {"api_key": api_key, **params}

    response = requests.get(url, params=query_params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def fetch_genre_mapping() -> dict[int, str]:
    """Lấy danh sách mã định danh thể loại (genre ID) và tên tiếng Anh tương ứng từ TMDB.

    Returns:
        Dict[int, str]: Bảng ánh xạ {genre_id: genre_name}.
    """
    data = _request_json("genre/movie/list", {"language": "en-US"})
    return {
        int(item["id"]): str(item["name"])
        for item in data.get("genres", [])
        if item.get("id") is not None and item.get("name")
    }


def fetch_movie_details(movie_id: int) -> dict[str, Any]:
    """Truy vấn thông tin chi tiết của một bộ phim (Credits: Đạo diễn, Diễn viên; Keywords: Từ khóa).

    Args:
        movie_id: Mã số ID của phim trên TMDB.

    Returns:
        Dict chứa các thông tin ("director", "cast", "keywords").
    """
    try:
        data = _request_json(
            f"movie/{movie_id}",
            {"append_to_response": "credits,keywords"},
        )
    except requests.RequestException:
        logger.exception("Không thể lấy thông tin chi tiết của phim ID: %s", movie_id)
        return {"director": "", "cast": [], "keywords": []}
    finally:
        time.sleep(REQUEST_DELAY)

    crew = data.get("credits", {}).get("crew", [])
    director = next(
        (member.get("name", "") for member in crew if member.get("job") == "Director"),
        "",
    )
    cast = [member.get("name", "") for member in data.get("credits", {}).get("cast", [])[:5]]
    keywords = [item.get("name", "") for item in data.get("keywords", {}).get("keywords", [])]

    return {
        "director": director,
        "cast": [name for name in cast if name],
        "keywords": [keyword for keyword in keywords if keyword],
    }


def _release_year(release_date: str, fallback: int) -> int:
    """Lấy năm 4 chữ số từ chuỗi `release_date`. Trả về fallback nếu dữ liệu ngày lỗi."""
    try:
        return int(str(release_date)[:4])
    except (TypeError, ValueError):
        return fallback


def fetch_tmdb_movies_by_year(
    start_year: int = 2023,
    end_year: int = 2024,
    max_pages_per_year: int = 10,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Thu thập phim có overview trong phạm vi năm, không lọc theo rating.

    Args:
        start_year: Năm phát hành bắt đầu.
        end_year: Năm phát hành kết thúc.
        max_pages_per_year: Số trang tối đa cần tải cho mỗi năm (mỗi trang chứa 20 phim).
        output_file: Đường dẫn tệp CSV ghi dữ liệu thô.

    Returns:
        pd.DataFrame: DataFrame chứa toàn bộ dữ liệu phim đã thu thập.
    """
    current_year = pd.Timestamp.utcnow().year + 1
    if not 1888 <= start_year <= end_year <= current_year:
        raise ValueError(f"Khoảng năm không hợp lệ. Phải nằm trong khoảng 1888–{current_year}.")
    if max_pages_per_year <= 0:
        raise ValueError("max_pages_per_year phải lớn hơn 0.")

    genre_mapping = fetch_genre_mapping()
    records: list[dict[str, Any]] = []

    for year in range(start_year, end_year + 1):
        for page in range(1, max_pages_per_year + 1):
            try:
                data = _request_json(
                    "discover/movie",
                    {
                        "language": "en-US",
                        "page": page,
                        "sort_by": "popularity.desc",
                        "primary_release_year": year,
                    },
                )
            except requests.RequestException:
                logger.exception("Không thể tải trang %d cho năm %d", page, year)
                break

            movies = data.get("results", [])
            if not movies:
                break

            for movie in movies:
                movie_id = movie.get("id")
                if movie_id is None:
                    continue

                details = fetch_movie_details(int(movie_id))
                genres = [genre_mapping[item] for item in movie.get("genre_ids", []) if item in genre_mapping]
                release_date = str(movie.get("release_date", ""))

                records.append(
                    {
                        "movie_id": int(movie_id),
                        "title": movie.get("title", ""),
                        "overview": movie.get("overview", ""),
                        "release_date": release_date,
                        "release_year": _release_year(release_date, year),
                        "vote_average": movie.get("vote_average", 0.0),
                        "popularity": movie.get("popularity", 0.0),
                        "genres": json.dumps(genres, ensure_ascii=False),
                        "director": details["director"],
                        "cast": json.dumps(details["cast"], ensure_ascii=False),
                        "keywords": json.dumps(details["keywords"], ensure_ascii=False),
                        "original_language": movie.get("original_language", ""),
                        "poster_path": movie.get("poster_path") or "",
                    }
                )

            logger.info("Đã tải năm %d, trang %d | Tổng tích lũy: %d phim", year, page, len(records))
            time.sleep(REQUEST_DELAY)

    dataframe = pd.DataFrame(records)
    if dataframe.empty:
        logger.warning("TMDB API không trả về phim nào trong khoảng thời gian cấu hình.")
        return dataframe

    dataframe["overview"] = dataframe["overview"].fillna("").astype(str)
    dataframe = dataframe[dataframe["overview"].str.strip().str.len() > 0]
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first")

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_file, index=False, encoding="utf-8")

    logger.info("Ingestion hoàn tất: %d phim được lưu vào %s", len(dataframe), output_file)
    return dataframe


def main() -> None:
    """Chạy quy trình Ingestion với cấu hình mặc định."""
    fetch_tmdb_movies_by_year(start_year=1990, end_year=2026, max_pages_per_year=25)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    main()
