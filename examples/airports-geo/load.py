# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
# ]
# ///
"""Ingest the OurAirports dataset into a GEODEGREES index.

The GEODEGREES space takes 2-D vectors of [latitude, longitude] in degrees and
returns great-circle distances in kilometers. Rich fields (type, country,
elevation, ...) are stored as metadata so they can be used as search filters.
"""

import csv
import os

import requests

SERVER = os.getenv("HNSW_SERVER", "http://localhost:8685")
DATA = os.path.join(os.path.dirname(__file__), "data", "airports.csv")
INDEX_NAME = "airports"
BATCH = 5000


def main():
    if not os.path.exists(DATA):
        raise FileNotFoundError(f"{DATA} not found. Run: python download_data.py")

    ids, vectors, metas = [], [], []
    with open(DATA, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                lat = float(row["latitude_deg"])
                lon = float(row["longitude_deg"])
            except (ValueError, KeyError):
                continue
            meta = {
                "name": row.get("name", ""),
                "type": row.get("type", ""),
                "iso_country": row.get("iso_country", ""),
                "municipality": row.get("municipality", ""),
                "ident": row.get("ident", ""),
                "iata": row.get("iata_code", ""),
                "lat": lat,
                "lon": lon,
            }
            elev = row.get("elevation_ft", "")
            if elev not in ("", None):
                try:
                    meta["elevation_ft"] = int(float(elev))
                except ValueError:
                    pass
            ids.append(len(ids))
            vectors.append([lat, lon])
            metas.append(meta)

    print(f"Parsed {len(ids)} airports with coordinates")

    requests.delete(f"{SERVER}/delete_index", json={"indexName": INDEX_NAME})
    requests.delete(f"{SERVER}/delete_index_from_disk", json={"indexName": INDEX_NAME})
    res = requests.post(
        f"{SERVER}/create_index",
        json={
            "indexName": INDEX_NAME,
            "dimension": 2,
            "spaceType": "GEODEGREES",
            "vectorType": "FLOAT32",
            "efConstruction": 256,
            "M": 16,
        },
    )
    res.raise_for_status()
    print(f"Created GEODEGREES index '{INDEX_NAME}'")

    total = len(ids)
    for start in range(0, total, BATCH):
        end = min(start + BATCH, total)
        res = requests.post(
            f"{SERVER}/add_documents",
            json={
                "indexName": INDEX_NAME,
                "ids": ids[start:end],
                "vectors": vectors[start:end],
                "metadatas": metas[start:end],
            },
        )
        res.raise_for_status()
        print(f"  added {end}/{total}", end="\r")

    print(f"\nDone. Loaded {total} airports into '{INDEX_NAME}'.")
    print("Start the UI with:  python app.py")


if __name__ == "__main__":
    main()
