"""CLI đánh giá mô hình trên Test set."""

from evaluation.benchmark import DEFAULT_TEST, evaluate_split
from evaluation.judgments import load_judgments


def main() -> None:
    """Đánh giá ablation trên Test set."""
    evaluate_split(
        load_judgments(DEFAULT_TEST),
        output_dir=DEFAULT_TEST.parent / "reports" / "test",
    )


if __name__ == "__main__":
    main()
