"""CLI chạy Locked Test sau khi đã freeze cấu hình."""

from evaluation.benchmark import DEFAULT_LOCKED, evaluate_split
from evaluation.judgments import load_judgments


def main() -> None:
    """Đánh giá Locked Test sau khi cấu hình đã freeze."""

    evaluate_split(
        load_judgments(DEFAULT_LOCKED),
        output_dir=DEFAULT_LOCKED.parent / "reports" / "locked_test",
    )


if __name__ == "__main__":
    main()
