"""Service layer với cache LRU có version index trong cache key."""

import copy
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from .config import settings
from .query import QueryEncoder
from .search import MovieSearch

logger = logging.getLogger(__name__)


class SearchService:
    """Điều phối request, cache kết quả và đo latency."""

    def __init__(self, engine: MovieSearch | None = None) -> None:
        self.engine = engine or MovieSearch()
        self.cache: OrderedDict[tuple[str, str, int, str, str, bool], dict[str, Any]] = OrderedDict()
        self._cache_lock = threading.RLock()

    def _cached_search(
        self,
        normalized_query: str,
        top_n: int,
        genre: str,
        year: str,
        debug: bool,
    ) -> dict[str, Any]:
        """Đọc/ghi cache bằng query đã chuẩn hóa và version index hiện tại."""

        key = (
            settings.index_version,
            normalized_query,
            top_n,
            genre.casefold(),
            year.casefold(),
            debug,
        )
        with self._cache_lock:
            cached = self.cache.get(key)
            if cached is not None:
                self.cache.move_to_end(key)
                return cached

        result = self.engine.search(normalized_query, top_n, genre, year, debug)
        with self._cache_lock:
            self.cache[key] = result
            self.cache.move_to_end(key)
            while len(self.cache) > settings.cache_size:
                self.cache.popitem(last=False)
        return result

    def search(
        self,
        query: str,
        top_n: int = 10,
        genre: str = "",
        year: str = "",
        debug: bool = False,
    ) -> dict[str, Any]:
        """Thực hiện search, giữ bản sao độc lập của kết quả cache."""

        started = time.perf_counter()
        normalized_query = QueryEncoder.clean_query(query)
        if not normalized_query:
            raise ValueError("Truy vấn tìm kiếm không được để trống hoặc chỉ chứa ký tự đặc biệt.")
        if not 0 < top_n <= settings.rerank_k:
            raise ValueError(f"top_n phải nằm trong khoảng 1-{settings.rerank_k}.")

        result = copy.deepcopy(
            self._cached_search(
                normalized_query,
                top_n,
                genre.strip(),
                year.strip(),
                debug,
            )
        )
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        logger.info(
            "Search thành công | index=%s | results=%d | latency=%.2f ms",
            result["index_version"],
            len(result["results"]),
            result["latency_ms"],
        )
        return result
