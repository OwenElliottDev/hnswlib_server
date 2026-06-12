"""Download the OurAirports dataset (~80k airports worldwide).

The CSV has latitude/longitude plus rich fields (type, country, elevation,
municipality, ...) that make for good geodegrees + filtering demos. It is a few
MB and is cached under ./data.
"""

import os
import sys
import urllib.request

URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TARGET = os.path.join(DATA_DIR, "airports.csv")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    mb = downloaded / 1e6
    if total_size > 0:
        pct = min(100, downloaded * 100 // total_size)
        sys.stdout.write(f"\r  {pct:3d}%  {mb:.1f} MB")
    else:
        sys.stdout.write(f"\r  {mb:.1f} MB")
    sys.stdout.flush()


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(TARGET):
        print(f"Already have {TARGET}")
        return
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, TARGET, _progress)
    print(f"\nReady: {TARGET}")


if __name__ == "__main__":
    main()
