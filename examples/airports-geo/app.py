# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "flask",
#     "requests",
# ]
# ///
"""Web UI for nearest-airport search over a GEODEGREES index.

Click anywhere on the map to find the closest airports (great-circle distance in
km), optionally filtered by airport type, country, name substring or elevation.
The browser talks to this Flask app, which builds the filter expression and
proxies the search to the hnswlib server.
"""

import argparse
import os
import time

import requests
from flask import Flask, jsonify, request, send_from_directory

SERVER = os.getenv("HNSW_SERVER", "http://localhost:8685")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
INDEX_NAME = "airports"

app = Flask(__name__)


def _clean(s):
    # filter literals are double-quoted; drop any quotes from user input
    return str(s).replace('"', "").strip()


def build_filter(body):
    clauses = []
    types = body.get("types") or []
    types = [_clean(t) for t in types if _clean(t)]
    if types:
        quoted = ", ".join(f'"{t}"' for t in types)
        clauses.append(f"type IN [{quoted}]")
    country = _clean(body.get("country", ""))
    if country:
        clauses.append(f'iso_country = "{country.upper()}"')
    name_contains = _clean(body.get("name_contains", ""))
    if name_contains:
        clauses.append(f'name CONTAINS "{name_contains}"')
    min_elev = body.get("min_elev")
    if min_elev not in (None, "", 0, "0"):
        try:
            clauses.append(f"elevation_ft >= {int(min_elev)}")
        except (TypeError, ValueError):
            pass
    return " AND ".join(clauses)


@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/search", methods=["POST"])
def search():
    body = request.get_json(force=True)
    try:
        lat = float(body["lat"])
        lon = float(body["lon"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "lat/lon required"}), 400
    k = int(body.get("k", 10))
    filter_str = build_filter(body)

    t0 = time.perf_counter()
    res = requests.post(
        f"{SERVER}/search",
        json={
            "indexName": INDEX_NAME,
            "queryVector": [lat, lon],
            "k": k,
            "efSearch": 256,
            "filter": filter_str,
            "returnMetadata": True,
        },
    )
    server_ms = round((time.perf_counter() - t0) * 1000, 2)
    if not res.ok:
        return jsonify({"error": f"server: {res.status_code} {res.text}"}), 502
    data = res.json()
    results = []
    for dist, meta in zip(data["distances"], data.get("metadatas", [])):
        results.append(
            {
                "name": meta.get("name", ""),
                "type": meta.get("type", ""),
                "country": meta.get("iso_country", ""),
                "municipality": meta.get("municipality", ""),
                "ident": meta.get("ident", ""),
                "iata": meta.get("iata", ""),
                "elevation_ft": meta.get("elevation_ft"),
                "lat": meta.get("lat"),
                "lon": meta.get("lon"),
                "distance_km": round(dist, 1),
            }
        )
    return jsonify({"filter": filter_str, "server_ms": server_ms, "results": results})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=5002)
    args = parser.parse_args()
    print(f"Open http://localhost:{args.port}  (hnswlib server: {SERVER})")
    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
