# Airports (geodegrees + filtering + map UI)

Nearest-airport search over the [OurAirports](https://ourairports.com/data/)
dataset (~80k airports worldwide) using the `GEODEGREES` space, which returns
**great-circle distances in kilometers** between `[latitude, longitude]` points.

Click anywhere on a map to find the closest airports, and narrow results with
metadata **filters** — airport type, country, name substring and minimum
elevation — all evaluated server-side by the hnswlib filter DSL.

```
browser (Leaflet map) ──▶ Flask app ──▶ hnswlib server (:8685)
   map click → lat/lon      builds filter   GEODEGREES index + metadata
```

## Run it

1. Start the hnswlib server in Docker (build the image once from the repo root —
   see [`../README.md`](../README.md)):
   ```bash
   docker run --rm -p 8685:8685 -v "$PWD/indices:/indices" hnswlib_server:local
   ```

2. Download the dataset (a few MB):
   ```bash
   cd examples/airports-geo
   uv run python download_data.py
   ```

3. Load the airports into a GEODEGREES index:
   ```bash
   uv run --with requests python load.py
   ```

4. Start the map UI and open <http://localhost:5002>:
   ```bash
   uv run --with requests --with flask python app.py
   ```

## Things to try

- Click near a city to see the nearest airports ranked by distance in km.
- Tick **medium**/**small** types to include smaller fields.
- Set country to `JP` and click near Tokyo — only Japanese airports come back.
- Name contains `international`, min elevation `5000` ft — high-altitude intl airports.

## How it works

- `load.py` stores each airport as a 2-D `[lat, lon]` vector in a `GEODEGREES`
  index, with `type`, `iso_country`, `name`, `municipality`, `elevation_ft` (and
  `lat`/`lon` for plotting) saved as metadata.
- `app.py` turns the UI controls into a filter expression such as
  `type IN ["large_airport"] AND iso_country = "JP" AND elevation_ft >= 100`, then
  calls `/search` with that filter and `returnMetadata: true`.
- Distances come straight back in kilometers from the geodegrees (haversine) space.
