"""Adaptive router cũ chỉ để chạy experiment, không dùng cho v1 production."""

from typing import Literal


def classify_rrf_route(
    top_score: float,
    runner_up: float,
    *,
    minimum_score: float,
    gap: float,
) -> Literal["EASY", "HARD"]:
    """Giữ logic cũ để tái hiện benchmark, không xem RRF score là confidence."""

    return "EASY" if top_score >= minimum_score and top_score - runner_up >= gap else "HARD"
