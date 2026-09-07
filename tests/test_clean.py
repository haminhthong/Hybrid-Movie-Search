"""Unit tests cho module làm sạch dữ liệu và phân tích năm."""

import pandas as pd
import pytest

from pipeline.clean import clean_text, create_combined_text, parse_list_value
from retrieval.search import parse_year


def test_clean_text_basic():
    """Kiểm tra làm sạch thẻ HTML, URL và khoảng trắng."""
    raw = "<b>Title</b>  http://example.com  overview text  "
    cleaned = clean_text(raw)
    assert cleaned == "title overview text"


def test_clean_text_empty_and_nan():
    """Kiểm tra xử lý đầu vào rỗng hoặc NaN."""
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_create_combined_text():
    """Kiểm tra ghép nối metadata thành combined_text."""
    row = pd.Series(
        {
            "title": "Interstellar",
            "director": "Christopher Nolan",
            "cast": "Matthew McConaughey",
            "genres": "Adventure, Drama",
            "keywords": "space, wormhole",
            "overview": "A team of explorers travels through a wormhole.",
        }
    )
    combined = create_combined_text(row)
    assert "Title: Interstellar" in combined
    assert "Director: Christopher Nolan" in combined
    assert "Overview: A team of explorers travels through a wormhole." in combined


def test_parse_list_value_supports_json_and_legacy_csv():
    assert parse_list_value('["Action", "Drama"]') == ["Action", "Drama"]
    assert parse_list_value("Action, Drama") == ["Action", "Drama"]


def test_parse_year_single():
    """Kiểm tra phân tích năm đơn dạng YYYY."""
    start, end = parse_year("2014")
    assert start == 2014
    assert end == 2014


def test_parse_year_range_dash():
    """Kiểm tra phân tích khoảng năm dạng YYYY-YYYY."""
    start, end = parse_year("2000-2020")
    assert start == 2000
    assert end == 2020


def test_parse_year_range_to():
    """Kiểm tra phân tích khoảng năm dạng YYYY to YYYY."""
    start, end = parse_year("2010 to 2015")
    assert start == 2010
    assert end == 2015


def test_parse_year_invalid_format():
    """Kiểm tra định dạng năm không hợp lệ ném ra ValueError."""
    with pytest.raises(ValueError, match="Định dạng năm không hợp lệ"):
        parse_year("invalid_year")


def test_parse_year_out_of_range():
    """Kiểm tra năm ngoài phạm vi ném ra ValueError."""
    with pytest.raises(ValueError):
        parse_year("1800")
