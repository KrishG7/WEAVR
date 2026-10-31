# OpenStreetMap basemap — scope, decisions and results

An optional **Show geography** layer for the blended-map and
extreme-probability views: the 0.25° forecast grid drawn over a basemap built
from OpenStreetMap data, served from a local file so the dashboard works
offline, and drawing **no national boundaries** (an official outline you
supply is drawn on top). The plain canvas maps remain the default and the
fallback. This document is the scoping record (sections for steps 01–06) plus
the final verification (step 07, the last section).

**Quick start**

```bash
brew install pmtiles                     # once
python scripts/build_basemap.py          # ~41 s, 148 MB, gitignored
uvicorn dashboard.api:app --port 8000    # then tick "Show geography"
python scripts/check_basemap_boundaries.py   # optional: confirm no borders
```

Optional: put your official boundary at `data/basemap/india-boundary.geojson`.
To turn the feature off, simply do not build the tile file: the toggle then
falls back to the plain grid with a caption.

## What exists today

| Item | Fact |
|---|---|
| Frontend | Vanilla HTML/CSS/JS in `dashboard-web/`, plain `<script>` tags, no build step, no third-party JS |
| Server | FastAPI (`dashboard/api.py`), API under `/api/…`, static frontend mounted last at `/` |
| Grid | 0.25° regular lat/lon, lat 6.5–38.5 and lon 66.5–100.0 (129 × 135 cells); values sit at cell centres so a cell spans ±0.125° around its coordinate ([grid convention](grid-and-time-convention.md)) |
| Blended map | `<canvas>` heat-map, one `fillRect` per cell, discrete rain-bin colours from `/api/colors`, rows flipped (latitude ascending, canvas y = 0 at top), **no geography** |
| Extreme-probability map | `<canvas>` two-pass heat-map: a continuous 5-stop probability scale plus a grey overlay for cells where EMOS-CSG had no fittable bin |

## Decisions (made with the user)

| Question | Decision |
|---|---|
| India boundary | The user will supply an official-depiction GeoJSON. Until it exists, the map draws **no outline** and shows a visible "no official boundary configured" note. The basemap never draws boundaries itself. |
| Where | Both map views (blended, extreme-probability), national and zoomed, **off by default** behind a "Show geography" toggle. The existing canvas stays the default and the fallback. |
| Offline | Fully offline. Tiles come from a local file; no runtime request to any remote host, and never to the public OSM tile servers. |
| Dependency | One vendored library pair, MapLibre GL JS plus the PMTiles plug-in (about 1 MB, committed with licences). No CDN, no npm, no build step. |

### Rejected options

| Option | Why rejected |
|---|---|
| Leaflet + `tile.openstreetmap.org` | Bakes OSM's own boundaries into the picture, needs the network, and breaches the public tile servers' usage policy for a demo. |
| Third-party hosted tiles | Adds an account/key and an uptime risk, and still draws boundaries unless a style can switch them off. |
| Stretching the existing canvas over the map | Web Mercator is not linear in latitude, so cells would be misplaced by kilometres, more so toward the north. Cells are drawn as polygons instead. |

## Tile source and measured size

- **Source:** the public Protomaps daily build, `https://build.protomaps.com/YYYYMMDD.pmtiles`
  (checked: the 2026-09-28 build exists, answers range requests, and is
  138.4 GB for the planet, which we do not download).
- **Method:** `pmtiles extract … --bbox=66.5,6.5,100.0,38.5 --maxzoom=Z --dry-run`
  (pmtiles CLI installed with Homebrew). A dry run reads directory headers and
  writes no file.
- **Measured archive size for the India bounding box:**

| Max zoom | Tiles kept | Archive size | Data transferred | Requests |
|---|---|---|---|---|
| 8 | 798 | **22 MB** | 23 MB | 24 |
| 10 | 10,830 | **148 MB** | 155 MB | 51 |
| 11 | 40,404 | **305 MB** | 321 MB | 61 |

Recommendation for step 02: **max zoom 10** (148 MB). It is enough to
identify a district's roads and towns at 0.25° cell scale; zoom 11 doubles the
size for detail that a 25 km grid cannot use.

- **Layers in the build** (from its metadata): `boundaries`, `buildings`,
  `earth`, `landcover`, `landuse`, `places`, `pois`, `roads`, `water`.
  **`boundaries` must be left out of the map style.** Buildings and POIs are
  also omitted, since they add nothing at this scale. Text labels stay off in
  the first version (the `places` layer names disputed areas, and the style
  would need local glyphs anyway).
- **Licence and attribution:** the build's own attribution is "© OpenStreetMap";
  its description says it is derived from OpenStreetMap and Natural Earth.
  OpenStreetMap data is ODbL, which allows building and hosting our own tiles
  with attribution. The map must show **© OpenStreetMap contributors**.
  Re-check the build's licence page when step 02 records the manifest.

## The one non-negotiable rule

The basemap draws **no national or disputed boundaries**. OSM depicts
boundaries as mapped by contributors, which differs from the Survey of India's
depiction in Kashmir and Arunachal Pradesh, and showing a non-official map
can be a legal problem in India. WEAVR overlays the user-supplied official
boundary on top. This is tested in steps 03, 06 and 07, not just intended.

## Files each later step will touch

| Step | Files |
|---|---|
| 02 | `scripts/build_basemap.py`, `tests/test_build_basemap.py`, `data/basemap/` (gitignored), this doc |
| 03 | `dashboard-web/vendor/`, `dashboard-web/basemap/style.json`, `dashboard/api.py`, `tests/test_dashboard_basemap.py` |
| 04 | `dashboard-web/js/charts/basemapMap.js`, `blendedMap.js`, `index.html` |
| 05 | `basemapMap.js`, `extremeProbability.js` |
| 06 | `dashboard/api.py`, `basemapMap.js`, a border-check script/test |
| 07 | docs and verification only |

## Building the basemap (step 02)

```bash
python scripts/build_basemap.py --dry-run      # estimate only, downloads nothing
python scripts/build_basemap.py --source-url https://build.protomaps.com/20260928.pmtiles
```

Requires the `pmtiles` CLI (`brew install pmtiles`). The script extracts only
the India bounding box (66.5, 6.5, 100.0, 38.5, the common grid's extent) up
to max zoom 10, writes to `india.pmtiles.partial` first and renames on
success (so an interrupted run never leaves a truncated file that looks
valid), then writes `data/basemap/manifest.json`. A second run is a no-op
while the file still matches the manifest; `--force` rebuilds. With no
`--source-url` it uses the newest daily build in the last ten days.
Pass an explicit URL when you want a reproducible build.

**Measured (2026-09-29):** 148.0 MB archive (147,968,833 bytes), 155 MB
transferred, 51 requests, about 41 s wall time. Header check: bounds
66.5–100.0 E, 6.5–38.5 N, zoom 0–10, 13,091 addressed tiles, 10,830 tile
entries. `data/basemap/` is gitignored (`git check-ignore` confirms it).

The manifest records source URL, bounding box, max zoom, tool version, size,
SHA-256, build time and the attribution string.

### Layers in the file, and which carry boundaries

| Layer | Use in the map style |
|---|---|
| `boundaries` | **Carries national, disputed and administrative boundaries. Never included.** |
| `earth`, `water`, `landcover`, `landuse` | Polygons: land, water bodies, cover. Included. |
| `roads` | Lines: roads, rail, ferries. Included, muted. |
| `places` | Point labels, including names of disputed areas. Excluded until local fonts exist and the labels are reviewed. |
| `buildings`, `pois` | Excluded (nothing useful at 0.25° scale). |

Only `boundaries` is documented as holding boundary lines. That comes from the
build's layer list, not from decoding the tiles, so step 06 still has to
verify that no other layer carries a boundary feature.

## Serving the basemap (step 03)

**Vendored libraries** (`dashboard-web/vendor/`, versions and licences in
`VERSIONS.md`): MapLibre GL JS 5.24.0 (1.06 MB) and PMTiles 4.5.0 (20 KB),
both BSD-3-Clause. Version 5 was chosen over the newest 6.x because 6.x is
ES-module only, which the no-build-step frontend cannot load with plain
script tags.

**Style** (`dashboard-web/basemap/style.json`): background, `earth`,
`landcover`, `landuse`, `water` (polygons and lines) and two `roads` layers
(major from zoom 5, minor from zoom 8). No `boundaries`, `places`,
`buildings` or `pois`, no symbol layers, no glyphs or sprite, no remote URLs.
The vector source URL is a placeholder (`pmtiles://__BASEMAP_TILES_URL__`);
`basemapMap.js` (step 04) replaces it with an absolute
`pmtiles://<origin>/basemap/india.pmtiles` at runtime.

**Routes** (`dashboard/api.py`):

| Route | Behaviour |
|---|---|
| `GET /basemap/india.pmtiles` | Streams `data/basemap/india.pmtiles`; answers `Range` requests with `206 Partial Content`. Missing file: `404` plain text telling you to run `scripts/build_basemap.py`. |
| `GET /api/basemap/status` | `{"available": bool, "bytes": int \| null}` so the frontend can fall back before creating a map. |

**Tests** (`tests/test_dashboard_basemap.py`, 12): status with and without the
file; full, range (`bytes 10-19/1024`) and missing-file responses; the real
file's first bytes read as `PMTiles`; the style contains no forbidden source
layers, no symbol layers, no boundary-like names, no remote URLs, uses only
layers that exist in the tile file, and names OpenStreetMap in its
attribution.

**Browser check (2026-09-29):** on a fresh port, the vendored libraries and
the real tile file rendered the Kerala coast at zoom 7 (land, sea, roads,
"© OpenStreetMap contributors" attribution) with no map errors, and every
network request went to `127.0.0.1` only. `curl -r 0-99` on the tile route
returns `206` with `Content-Range: bytes 0-99/147968833`.

## The map component and the blended map (step 04)

`dashboard-web/js/charts/basemapMap.js` is one reusable module
(`BasemapMap.isAvailable()`, `BasemapMap.create(container)`,
`handle.setCells({latitude, longitude, colorAt})`, `handle.destroy()`).
The blended-map view (`blendedMap.js`) gets a **Show geography** checkbox,
off by default, that swaps its canvas for this map.

**How cells are placed.** Each cell is a GeoJSON polygon in a MapLibre fill
layer, not a picture stretched over a bounding box (Web Mercator is not
linear in latitude, so an image would misplace cells by kilometres). Grid
coordinates are cell centres, so each polygon spans half a grid step either
side; the step is read from the data and checked to be regular. Dry cells
(bin 0) are left out so the basemap shows through; the rest use the same
discrete IMD colours as the canvas.

**Fallback.** The toggle only takes effect if the libraries loaded, WebGL is
available and `/api/basemap/status` says the tile file exists; any failure
while creating the map (including a 10 s load timeout) unchecks the box,
shows the plain canvas and the caption "Basemap unavailable -- showing the
plain grid." The map creation *promise* is cached, so changing the lead while
the first map is still loading joins that creation instead of starting a
second one; a single map is reused across lead changes and view switches.

**Also changed:** below 720 px width the navigation now sits above the view
instead of beside it (`style.css`). Before that the sidebar took half a phone
screen and the map was 91 px wide; this helps every view.

### Verification (real browser, fresh port, 2026-09-29)

| Check | Result |
|---|---|
| Half-cell offset | polygon for lat 6.5, lon 66.5 spans 6.375–6.625 and 66.375–66.625; whole-grid bounds 66.375–100.125 E, 6.375–38.625 N |
| Cells vs data | 40 of 40 sampled cells in 8–13 N, 74–79 E: querying the rendered map at each cell's projected centre returns the colour the API's bin gives (dry cells return nothing) |
| Coast | Kochi (9.98 N, 76.3 E) is on the land layer; 1.3° west of it (75.0 E) is sea |
| Every lead (24, 48, 72, 96, 120) | renders; lead change with the map already open takes 50–160 ms |
| Changing lead during the first load | one map instance, correct final lead |
| Toggling, and view switch and back | 1 map instance, 1 WebGL canvas, choice remembered; no errors |
| Tile file missing | canvas plus the caption, no errors |
| WebGL unavailable | same fallback |
| 375 px width | map 343 px wide, no horizontal scroll on any of the four views |

Notes: first map creation takes about a second (style, tiles). The browser
pane used for these checks was hidden, and browsers pause
`requestAnimationFrame` in hidden tabs, so the checks replaced it with a
timer; in a visible tab MapLibre runs normally. In a hidden tab the 10 s load
timeout can trigger the fallback.

## The extreme-probability map on the basemap (step 05)

The extreme-probability view gets the same **Show geography** checkbox (off by
default), using the step-04 component rather than a copy of it. Changes to the
shared component: `setCells` now also accepts a per-cell `opacityAt` and an
optional second `overlayColorAt` layer, and `BasemapMap.shared(key)` holds one
long-lived map per view (creation promise cached, as in step 04). The blended
map was moved onto `BasemapMap.shared` and behaves as before.

**Continuous colours on a square-root scale, per-cell opacity.** Cell colours
are interpolated from the same `probability_colorscale` as the plain grid, but
a probability `p` sits at position `sqrt(p)` along its five stops. On the
linear scale (white at 0, first real colour, light green, at 0.25) every
extreme-rain probability of a few percent fell into the first sliver and the
map read as blank: measured in a real browser on the example grids, 0-8 of
17,415 cells at 115.6 mm and none at 204.5 mm were visibly tinted (channel
distance from white of at least 20), against 4,345-4,885 and 1,067-2,054 now.
The scale is fixed, not stretched to each map's maximum, so one probability is
one colour at every lead and threshold, and the legend labels real
probabilities (0, 1, 5, 10, 25, 50, 100%) at the positions they occupy. In
geography mode opacity rises linearly in that scale position from 0 at p = 0 to
0.85 at the first real colour (p = 6.25%), so solid fills at low probability do
not wash the basemap out with white. Colours are unchanged; only how strongly
they cover the basemap.

**The map is still pale at 204.5 mm, and says why.** The example grids' highest
probabilities are 3.6 / 3.1 / 2.7 / 1.9 / 2.2 % at 115.6 mm and 0.41 / 0.35 /
0.31 / 0.22 / 0.25 % at 204.5 mm (24-120 h). The square-root scale makes the
115.6 mm structure readable; a 0.4% maximum sits 6% of the way along the bar
whatever the scale, and a stronger transform would exaggerate probabilities
that small. The note under the map states the highest probability at the
current lead, so a pale map is not mistaken for missing data.

**Fallback cells stay distinct.** They are a second polygon layer, solid grey
at 0.75 opacity, drawn above the probability layer, with the legend swatch and
the existing warning banner unchanged. Lead 96 h has four (27.5 N 84 E;
27.75 N 83.75, 84, 84.25 E) and they read clearly over the basemap.

### Verification (real browser, fresh port, 2026-09-29)

| Check | Result |
|---|---|
| Colours and opacities | 24 of 24 sampled cells at 120 h, queried on the rendered map at each cell's projected centre, equal an independent recomputation of the colour scale interpolation and of the opacity rule from the API's probabilities |
| Every lead | renders; title and data match the lead; one map instance |
| Fallback overlay | 96 h: 4 cells, grey overlay present at the right cells; other leads have none |
| Switching between the two map views three times, toggles on | 1 map instance and 1 WebGL canvas at the end, no errors |
| Tile file missing | plain grid plus the caption, no errors |
| WebGL unavailable | same fallback |
| 375 px width | map 343 px wide, legend inside the screen, no horizontal scroll |

An earlier test run reported errors after view switches; they came from the
test harness replacing `requestAnimationFrame` without replacing
`cancelAnimationFrame`, so timers survived `map.remove()`. With both replaced
there were none. The same hidden-pane caveat as step 04 applies.

## The official boundary, attribution and the border check (step 06)

### Supplying the official boundary

Put your official-depiction India boundary at
**`data/basemap/india-boundary.geojson`** (a GeoJSON `FeatureCollection`,
`Feature` or bare geometry made of lines or polygons; polygon outlines are
drawn as lines). The path sits under the gitignored `data/`, next to the
tiles, rather than in `dashboard-web/`, so a file whose licence is not yet
confirmed is never committed and never served by the static mount. An optional
top-level `"attribution"` member (any text) is shown in the map's attribution
line; without one a neutral default is used.

`GET /api/basemap/boundary` returns `{"configured": true, "attribution",
"geojson"}`; with no file it returns `404 {"configured": false}`; a file that
exists but is not valid outline GeoJSON is a `500` naming the reason, so a
broken boundary is never mistaken for "not configured".

### What the map shows

| State | Result |
|---|---|
| No file | No outline. A small note inside the map, "No official boundary configured". The grid's own rectangle is never drawn as a border. |
| Valid file | The outline is drawn as a thin dark line as the **topmost layer**, above the basemap and above the forecast cells (`setCells` inserts its layers beneath it). |
| Invalid file or request failure | No outline; the note reads "Boundary unavailable: <reason>". |

**Attribution** is always expanded (never collapsed to an icon), links to
OpenStreetMap's copyright page, and appends the boundary's own attribution
when configured. The boundary text is HTML-escaped before display.

### The no-border check

`python scripts/check_basemap_boundaries.py` (tests in
`tests/test_check_basemap_boundaries.py`). The tile file *does* contain a
`boundaries` layer, so the useful question is what the map can render:

1. The style references only `earth`, `landcover`, `landuse`, `water` and
   `roads` (no `boundaries`, `places`, `pois` or symbol layers); tested in
   `tests/test_dashboard_basemap.py`.
2. In real tiles sampled over **Kashmir and Arunachal Pradesh at zooms 3–10**
   (65 tiles), each style layer must hold only what it should: no line
   features in `earth`, `landcover`, `landuse` (an outline there would be a
   border); `roads` lines of known road kinds only; no feature whose `kind`
   looks like a boundary. The tiles are read with the `pmtiles` CLI and
   decoded with a small protobuf reader inside the script, so **no new
   dependency** was needed.

**Result (2026-09-29): PASSED.** The samples held 232 `boundaries` features,
1,378 `places` and 42 `pois` that the style never draws, against only
polygons in `earth`/`landcover`/`landuse`, road lines in `roads` and
polygons, lines and points in `water`. One thing the first run caught: the
`roads` layer also carries `aeroway` (runways), which the style does not
draw; the allowed-kinds list now includes it and the test checks that the
style's own road filter uses only listed kinds.

**Limit, stated plainly:** the `boundaries` layer is still inside
`india.pmtiles`. Nothing here draws it, but anyone who re-styles the file
could. The `pmtiles` CLI cannot drop a layer. If that matters for the static
deploy, the file would have to be rebuilt with a tool that can filter layers.

### Browser verification (fresh port, 2026-09-29)

| Check | Result |
|---|---|
| No boundary configured, both map views | note visible, attribution visible and expanded, no boundary layer; layers are the basemap layers plus the two forecast layers |
| Every line feature on screen at Kashmir (zoom 5.2), Kashmir wide, Arunachal and the national view | only `roads-major` highway lines; nothing else is a line |
| Synthetic outline configured (removed afterwards) | drawn as the last layer, above the cells; a query at its west edge finds it; note gone; attribution shows `TEST <b>…` escaped, then the OpenStreetMap link |
| Invalid boundary file | API 500 with the reason; map shows "Boundary unavailable: …"; no errors |
| 375 px width, invalid-file state | attribution inside the map, note clear of the attribution and the zoom buttons, legend below the map, no horizontal scroll |

The first mobile run found the note (bottom-left) overlapping the attribution
when its text wrapped; it now sits top-left, clear of the zoom buttons.

The synthetic outline was a test rectangle, not a boundary, and was deleted;
`data/basemap/` holds only the tiles and their manifest.


## Final verification and results (step 07)

### What was built

| Piece | Where |
|---|---|
| Tile builder and provenance manifest | `scripts/build_basemap.py`, `data/basemap/` (gitignored) |
| Vendored MapLibre GL JS 5.24.0 and PMTiles 4.5.0 | `dashboard-web/vendor/` |
| Border-free, label-free style | `dashboard-web/basemap/style.json` |
| Tile and status routes (HTTP Range, 206) | `dashboard/api.py` |
| Reusable map component, shared per view | `dashboard-web/js/charts/basemapMap.js` |
| Toggle on the two map views | `blendedMap.js`, `extremeProbability.js` |
| Official-boundary hook and attribution | `basemapMap.js`, `GET /api/basemap/boundary` |
| No-border check | `scripts/check_basemap_boundaries.py` |
| Tests | `tests/test_build_basemap.py` (16), `test_dashboard_basemap.py` (19), `test_check_basemap_boundaries.py` (11) |

### Measured

| Quantity | Value |
|---|---|
| Tile file | 147,968,833 bytes (148.0 MB), India bounding box, max zoom 10, Protomaps build of 2026-09-28 |
| Build time | about 41 s (155 MB transferred, 51 requests) |
| Lead change with the map already open (rendered and idle) | 0.30–0.32 s on both views, all five leads |
| First map after ticking the box | 0.12–0.17 s in a visible tab with warm caches; about 1.1 s measured earlier with a cold cache |
| Forecast layer size | up to 3,016 non-dry polygons (blended), all 17,415 cells (extreme-probability) |
| New Python dependencies | none (`pyproject.toml` untouched) |
| Files committed for data | none: `data/` is gitignored, checked with `git check-ignore` |

### Verification matrix (real browser, fresh port, 2026-09-29)

Both views, every lead (24, 48, 72, 96, 120), unless stated.

| Check | Result |
|---|---|
| Lead change, map open | renders; title matches the lead; one map instance; 0.30–0.32 s |
| Toggle on/off six times, then switch views back and forth | one map instance and one WebGL canvas at the end, choice remembered, no console or `error` events |
| Pan and zoom at national (zoom 3.6–4.2), state (7.6) and district scale (10) | no errors; basemap features present at every scale |
| Cell alignment, three places, zoom 6 and 10 | Kochi, Mumbai and Chennai each lie inside their 0.25° cell, and the rendered colour there equals the independently recomputed colour (6 of 6) |
| Coastline | land inland and sea offshore at Kochi, Mumbai and Chennai (3 of 3) |
| The four grid corners | 3 px inside each cell edge is hit and 3 px outside is not, for all four corner cells (16 of 16 edge tests); cells are 91×92 px at the south and 91×116 px at the north at zoom 8, the Mercator stretch that a stretched picture would get wrong |
| 375 px width, both views | map 343 px wide, no horizontal scroll, attribution inside the map, note clear of attribution and zoom buttons, controls inside the screen |
| Offline | every request over a session covering both views, all leads and all state changes went to `127.0.0.1` only; a search of the app code finds one `https://` string, the OpenStreetMap copyright link the user can click (not a request). The network could not be physically disconnected from this environment, so "no remote host is ever contacted" rests on that request list plus the code search |

### Failure modes

| Case | Result |
|---|---|
| Tile file missing | plain grid, caption "Basemap unavailable -- showing the plain grid.", both views, no errors |
| Tile file truncated to 300 bytes (present, corrupt) | **found and fixed here.** Before the fix the style loaded fine, the toggle stayed on and the forecast grid floated on an empty background while the note claimed a basemap: an error only arrives when tiles are requested. Now the map records tile-source errors from creation and only counts as ready once the source has loaded; the corrupt file falls back in about 0.5 s on both views with no leftover map |
| WebGL unavailable | same fallback, both views |
| No boundary file | note "No official boundary configured", no outline (step 06) |
| Invalid boundary file | API 500 with the reason; map note "Boundary unavailable: …" (step 06) |

A truncated file at 2 MB still rendered (low-zoom tiles come first in the
archive), which is correct behaviour, not a failure: the test that matters is
a file too broken to read.

### Border check on the final tile file

`python scripts/check_basemap_boundaries.py` **PASSED** on the final file: 65
tiles over Kashmir and Arunachal Pradesh at zooms 3–10; only polygons in
`earth`/`landcover`/`landuse`, road lines in `roads`, polygons/lines/points in
`water`; 232 `boundaries`, 1,378 `places` and 42 `pois` features present but
never drawn. On screen, the only line features at Kashmir, Arunachal and the
national view were highways.

### Known limits

- **Boundary layer still in the file.** `india.pmtiles` contains the
  `boundaries` layer (232 features in the samples). Nothing draws it, but
  someone re-styling the file could; the `pmtiles` tool cannot drop a layer.
  If the static site must not carry it, rebuild with a tool that can filter
  layers.
- **No official boundary is configured.** Until you supply
  `data/basemap/india-boundary.geojson` the map shows no outline; the
  synthetic rectangle used to test drawing was deleted.
- **No place labels.** Labels are off (the `places` layer names disputed
  areas and would need local fonts). Roads and land cover are muted.
- **Max zoom 10.** About district scale; cells are 25 km, so more detail would
  add size without information.
- **Pale extreme-probability map by design.** The example grids peak at
  0.08–5.1 %, well below the 25 % where colour begins; the note says so.
- **Hidden-tab testing.** The browser pane used for verification was often
  hidden, and browsers pause `requestAnimationFrame` in hidden tabs, so most
  checks replaced it with a timer. In a hidden tab the 10 s load timeout can
  trigger the fallback. The last runs were in a visible tab with the real
  timer and matched.
- **CI does not test `dashboard-web`** (JavaScript and CSS are outside its
  test jobs), so the front-end checks above are manual and repeatable, not
  automated.
