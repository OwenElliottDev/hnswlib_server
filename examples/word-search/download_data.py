"""Download GloVe word vectors and extract a single-dimension file.

GloVe vectors are pre-trained, meaningful word embeddings, so this example needs
no embedding model: each word already has a vector. We download the standard
"6B" pack (trained on Wikipedia + Gigaword) and pull out one dimensionality.

Note: the GloVe 6B archive is ~822 MB; the download is one-time and cached under
./data. Pass --dim to choose 50, 100, 200 or 300 (default 100).
"""

import argparse
import os
import sys
import urllib.request
import zipfile

GLOVE_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        mb = downloaded / 1e6
        total_mb = total_size / 1e6
        sys.stdout.write(f"\r  {pct:3d}%  {mb:7.1f} / {total_mb:.1f} MB")
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=100, choices=[50, 100, 200, 300])
    args = parser.parse_args()

    os.makedirs(DATA_DIR, exist_ok=True)
    member = f"glove.6B.{args.dim}d.txt"
    target = os.path.join(DATA_DIR, member)
    if os.path.exists(target):
        print(f"Already have {target}")
        return

    zip_path = os.path.join(DATA_DIR, "glove.6B.zip")
    if not os.path.exists(zip_path):
        print(f"Downloading {GLOVE_URL}")
        urllib.request.urlretrieve(GLOVE_URL, zip_path, _progress)
        print()
    else:
        print(f"Using cached {zip_path}")

    print(f"Extracting {member} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extract(member, DATA_DIR)
    print(f"Ready: {target}")
    print("(You can delete data/glove.6B.zip to reclaim ~822 MB once extracted.)")


if __name__ == "__main__":
    main()
