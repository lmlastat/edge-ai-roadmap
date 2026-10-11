import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a folder of images.")
    parser.add_argument("folder", type=Path, help="folder to scan")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Scanning {args.folder}")


if __name__ == "__main__":
    main()
