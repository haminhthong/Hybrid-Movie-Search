"""CLI duy nhất dùng để đánh giá/tune trên Dev."""

from evaluation.benchmark import DEFAULT_DEV, evaluate_split
from evaluation.judgments import load_judgments


def main() -> None:
    """Đánh giá trên Dev; đây là split duy nhất dùng để chọn cấu hình."""

    evaluate_split(load_judgments(DEFAULT_DEV), output_dir=DEFAULT_DEV.parent / "reports" / "dev")


if __name__ == "__main__":
    main()
