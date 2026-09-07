"""Schema và loader cho relevance judgments dạng JSONL."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QueryJudgment:
    """Một query có thể có một hoặc nhiều document relevant theo grade."""

    query_id: str
    query: str
    query_type: str
    judgments: dict[str, int]

    @property
    def relevant_ids(self) -> set[str]:
        """Các movie có relevance > 0."""

        return {movie_id for movie_id, grade in self.judgments.items() if grade > 0}


def _parse_record(record: dict[str, Any], line_number: int) -> QueryJudgment:
    """Validate một record JSONL và chuẩn hóa movie id thành string."""

    required = {"query_id", "query", "query_type", "judgments"}
    missing = required.difference(record)
    if missing:
        raise ValueError(f"Dòng {line_number} thiếu trường: {', '.join(sorted(missing))}")
    judgments = record["judgments"]
    if not isinstance(judgments, dict) or not judgments:
        raise ValueError(f"Dòng {line_number} phải có judgments không rỗng.")
    normalized = {}
    for movie_id, grade in judgments.items():
        try:
            normalized[str(movie_id)] = int(grade)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Dòng {line_number} có judgment không hợp lệ.") from exc
        if not 0 <= normalized[str(movie_id)] <= 3:
            raise ValueError(f"Dòng {line_number}: relevance phải nằm trong khoảng 0-3.")
    if not str(record["query"]).strip():
        raise ValueError(f"Dòng {line_number} có query rỗng.")
    return QueryJudgment(
        query_id=str(record["query_id"]),
        query=str(record["query"]),
        query_type=str(record["query_type"]),
        judgments=normalized,
    )


def load_judgments(path: Path) -> list[QueryJudgment]:
    """Đọc JSONL và từ chối query_id trùng nhau."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy judgment file: {path}")
    items: list[QueryJudgment] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Dòng {line_number} không phải JSON hợp lệ.") from exc
        item = _parse_record(record, line_number)
        if item.query_id in seen:
            raise ValueError(f"query_id bị trùng: {item.query_id}")
        seen.add(item.query_id)
        items.append(item)
    if not items:
        raise ValueError("Judgment file không có query.")
    return items
