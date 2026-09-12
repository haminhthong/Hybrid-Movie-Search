"""CLI đánh giá mô hình trên Dev set."""

from evaluation.benchmark import DEFAULT_DEV, evaluate_split
from evaluation.judgments import load_judgments


def main() -> None:
    """Đánh giá ablation trên Dev set."""
    evaluate_split(load_judgments(DEFAULT_DEV), output_dir=DEFAULT_DEV.parent / "reports" / "dev")


if __name__ == "__main__":
    main()
