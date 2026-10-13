import argparse
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from collections.abc import Iterator
from dataclasses import dataclass

from PIL import Image

import logging
log = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    path: Path
    width: int
    height: int
    digest: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a folder of images.")
    parser.add_argument("folder", type=Path, help="folder to scan")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()

def iter_images(folder: Path) -> Iterator[Path]:
    img_ext_list = {".jpg", ".jpeg", ".png", ".webp"}
    for p in folder.rglob("*"): #every file and folder
        if p.is_file():
            file_ext = p.suffix.lower()
            if file_ext in img_ext_list:
                yield p

def get_size(path: Path) -> tuple[int, int]:

    with Image.open(path) as img:
        img.verify()

    with Image.open(path) as img:
        width, height = img.size
        return (width, height)


def get_hex_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_histogram(infos: list[ImageInfo]) -> Counter:
    return Counter((i.width, i.height) for i in infos)

def find_duplicate_by_hash(infos: list[ImageInfo]) -> None:
    groups = defaultdict(list)

    for info in infos:
        groups[info.digest].append(info.path)

    for digest, paths in groups.items():
        if len(paths) > 1:
            print(f"Duplicate group ({digest[:8]}):")
            for path in paths:
                print(f"   {path.name}")

def populate_infos(images: list[Path]) -> tuple[list[ImageInfo], list[tuple[Path, Exception]]]:
    infos = []
    corrupt_list = []
    for image in images:
        log.debug(f"Processing {image}")
        try:
            width, height = get_size(image)
            digest = get_hex_digest(image)
            info = ImageInfo(path=image, width=width, height=height, digest=digest)
            infos.append(info)
        except Exception as exc:
            corrupt_list.append((image, exc))
    
    return infos, corrupt_list

def main() -> None:
    

    args = parse_args()

    logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(levelname)s %(message)s",
       )
    if not args.folder.is_dir():
        log.error(f"{args.folder} is not a folder")
        return

    log.info(f"Scanning {args.folder}")


    images = list(iter_images(args.folder))
    log.info(f"Found {len(images)} images")

    infos, corrupt_list = populate_infos(images)

    if corrupt_list:
        print("Corrupted files are:")
        for corrupt_file, exc in corrupt_list:
            print(f"- {corrupt_file}: {exc}")

    histogram = get_histogram(infos)
    for (w,h), count in histogram.most_common():
        label = f"{w}x{h}"
        print(f"{label:<10}", "#" * count)

    find_duplicate_by_hash(infos)

    


if __name__ == "__main__":
    main()
