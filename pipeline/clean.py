"""Chuẩn hóa snapshot TMDB thành movie documents versioned."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd

from retrieval.config import DOCUMENT_SCHEMA_VERSION

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "data" / "movies_raw.csv"
OUTPUT_FILE = BASE_DIR / "data" / "movies_clean.csv"

TEXT_FIELDS: tuple[str, ...] = ("title", "overview", "director", "cast", "keywords", "genres")
LIST_FIELDS: tuple[str, ...] = ("cast", "keywords", "genres")
REQUIRED_COLUMNS: set[str] = {"movie_id", "title", "overview", "release_date", *TEXT_FIELDS}
DOCUMENT_FIELDS: tuple[tuple[str, str], ...] = (
    ("title", "Title"),
    ("director", "Director"),
    ("cast", "Cast"),
    ("genres", "Genres"),
    ("keywords", "Keywords"),
    ("overview", "Overview"),
)


def clean_text(value: Any) -> str:
    """Xóa HTML/URL, chuẩn hóa khoảng trắng và chuyển text sang lowercase.

    Hàm này giữ hành vi cũ để dùng cho tokenization/test. Metadata hiển thị được
    xử lý bằng ``_clean_metadata`` để không làm mất chữ hoa trong tên riêng.
    """

    if not isinstance(value, str) or pd.isna(value):
        return ""
    text = re.sub(r"<[^>]+>", " ", value)
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip().lower()


def _clean_metadata(value: Any) -> str:
    """Làm sạch metadata nhưng giữ nguyên cách viết tên phim/người."""

    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return ""
    text = re.sub(r"<[^>]+>", " ", str(value))
    text = re.sub(r"(?:https?://|www\.)\S+", " ", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def parse_list_value(value: Any) -> list[str]:
    """Đọc metadata dạng list JSON/list Python hoặc chuỗi comma-separated cũ."""

    if value is None or (not isinstance(value, (list, tuple)) and pd.isna(value)):
        return []
    if isinstance(value, (list, tuple)):
        return [_clean_metadata(item) for item in value if _clean_metadata(item)]

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, list):
        return [_clean_metadata(item) for item in parsed if _clean_metadata(item)]
    return [_clean_metadata(item) for item in text.split(",") if _clean_metadata(item)]


def _document_value(field: str, value: Any) -> str:
    """Chuyển field scalar/list thành text nhất quán cho embedding."""

    if field in LIST_FIELDS:
        return ", ".join(parse_list_value(value))
    return _clean_metadata(value)


def create_combined_text(row: pd.Series) -> str:
    """Tạo một search document duy nhất cho mỗi movie theo schema cố định."""

    parts = []
    for field, label in DOCUMENT_FIELDS:
        value = _document_value(field, row.get(field, ""))
        if value:
            parts.append(f"{label}: {value}")
    return ". ".join(parts)


def _release_years(dataframe: pd.DataFrame) -> pd.Series:
    """Lấy release year từ ngày, có fallback sang cột release_year."""

    years = pd.to_numeric(dataframe["release_date"].astype(str).str[:4], errors="coerce")
    if "release_year" in dataframe.columns:
        years = years.fillna(pd.to_numeric(dataframe["release_year"], errors="coerce"))
    return years.fillna(0).astype(int)


def process_documents(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> pd.DataFrame:
    """Validate, clean và ghi canonical movie table."""

    input_path = Path(input_file)
    output_path = Path(output_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dữ liệu đầu vào: {input_path}")

    dataframe = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing:
        raise ValueError(f"Dữ liệu thiếu các cột bắt buộc: {', '.join(sorted(missing))}")

    before_count = len(dataframe)
    dataframe["movie_id"] = pd.to_numeric(dataframe["movie_id"], errors="coerce")
    dataframe = dataframe.dropna(subset=["movie_id"]).copy()
    dataframe["movie_id"] = dataframe["movie_id"].astype(int)
    dataframe = dataframe[dataframe["movie_id"] > 0]

    # Scalar giữ cách viết để response đẹp; search document vẫn có cấu trúc label.
    for field in ("title", "overview", "director"):
        dataframe[field] = dataframe[field].map(_clean_metadata)
    for field in LIST_FIELDS:
        dataframe[field] = dataframe[field].map(
            lambda value: json.dumps(parse_list_value(value), ensure_ascii=False)
        )

    dataframe["release_year"] = _release_years(dataframe)
    dataframe["combined_text"] = dataframe.apply(create_combined_text, axis=1)
    dataframe["document_schema_version"] = DOCUMENT_SCHEMA_VERSION
    dataframe = dataframe[
        (dataframe["title"].str.len() > 0)
        & (dataframe["overview"].str.len() > 0)
        & (dataframe["combined_text"].str.len() > 0)
        & dataframe["release_year"].between(1888, 2100)
    ]
    dataframe = dataframe.drop_duplicates(subset=["movie_id"], keep="first").copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(output_path, index=False, encoding="utf-8")
    logger.info(
        "Làm sạch hoàn tất: %d/%d bản ghi hợp lệ, output=%s",
        len(dataframe),
        before_count,
        output_path,
    )
    return dataframe


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    process_documents()
