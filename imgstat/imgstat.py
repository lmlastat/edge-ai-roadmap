import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize a folder of images.")
    parser.add_argument("folder", type=Path, help="folder to scan")
    return parser.parse_args()

def iter_images(folder):
    img_ext_list = {".jpg", ".jpeg", ".png", ".webp"}
    for p in folder.rglob("*"): #every file and folder
        if p.is_file():
            file_ext = p.suffix.lower()
            if file_ext in img_ext_list:
                yield p

from PIL import Image


def getSize(imagePath):

    def verifyImage(path):
        with Image.open(path) as img:
            img.verify()

    def getDimensions(path):
        with Image.open(path) as img:
            width, height = img.size
            return (width, height)

    verifyImage(imagePath)
    return getDimensions(imagePath)

from collections import Counter


import hashlib
from collections import defaultdict



def getHexDigest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    digest = h.hexdigest();
    return digest




def main() -> None:
    args = parse_args()
    def showScanningPrompt():
        print(f"Scanning {args.folder}")

    images = list(iter_images(args.folder))
    print(f"Found {len(images)} images")

    def getSizeOrCorruptList():
        corruptList = []
        for image in images:
            try:
                print(getSize(image))
            except Exception as exc:
                print(type(exc), exc)
                corruptList.append(image)
        if corruptList:
            print("Corrupted files are:")
            for corruptFile in corruptList:
                print(f"- {corruptFile}")
        else:
            print("No corrupted files")

    def getHistogram(images):
        sizes = Counter()
        for image in images:
            try:
                sizes[getSize(image)] += 1
            except:
                continue
        return sizes.most_common()


    showScanningPrompt()
   # getSizeOrCorruptList()

    #histogram = getHistogram(images)

    def findDuplicateByHash(images):
        groups = defaultdict(list)

        for image in images:
            try:
                digest = getHexDigest(image)
                groups[digest].append(image)
            except:
                continue

        for digest, paths in groups.items():
            if len(paths) > 1:
                print(f"Duplicate group ({digest[:8]}):")
                for path in paths:
                    print(f"   {path.name}")

    findDuplicateByHash(images)


   # for (w,h), count in histogram:
    #    label = f"{w}x{h}"
     #   print(f"{label:<10}", "#" * count)


if __name__ == "__main__":
    main()
