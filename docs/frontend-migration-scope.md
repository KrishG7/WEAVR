# Frontend migration scope: HTML/CSS/vanilla-JS + API decision

Step 1 of `workspace/frontend-prompts/` (outside this repo), executing
`workspace/frontendplan.md`'s Phase A groundwork. Locks the tradeoffs that
plan deliberately left open (§6) before any implementation starts.

## 1. `data_loading.py` / `colors.py` reusability — confirmed, checked directly

Grepping both modules' own import lines (not assumed from memory):

```
dashboard/data_loading.py:14:from __future__ import annotations
dashboard/data_loading.py:16:from pathlib import Path
dashboard/data_loading.py:18:import numpy as np
dashboard/data_loading.py:19:import pandas as pd
dashboard/colors.py:11:from __future__ import annotations
dashboard/colors.py:13:import numpy as np
dashboard/colors.py:15:from weavr.rain_bins import RAIN_BIN_LABELS
dashboard/colors.py:16:from weavr.verify import IMD_RAIN_THRESHOLDS_MM
```

Neither module imports `streamlit` or `plotly` (the two mentions of
"Plotly" in `colors.py` are docstring prose, not import statements,
confirmed by grepping specifically for `^import`/`^from` lines). Both are
plain pandas/numpy/pathlib, returning `pd.DataFrame` or plain `dict`
objects. This confirms `frontendplan.md`'s key assumption: the new API
layer (step 2) can import `dashboard.data_loading` and `dashboard.colors`
directly, unchanged, and add only a JSON-serialization layer on top — no
rewrite of either module is needed.

## 2. Charting approach — routed via `AskUserQuestion`

Options presented: Plotly.js (vendored/CDN, closest 1:1 mapping to the
existing `plotly.express`/`graph_objects` calls, fastest migration) vs.
hand-rolled `<canvas>`/SVG (zero dependencies, more code, especially for
`extreme_probability.py`'s two-trace overlay pattern).

**Decision: hand-rolled `<canvas>`/SVG.** No charting library dependency
at all — matches "vanilla JS" literally, not just "no framework."
Consequences locked in for later steps:

- **Weight map** (`weightMap.js`, step 3): hand-drawn SVG grouped bar
  chart (`<rect>` per region×source bar), with an SVG pattern fill
  (`<pattern>` + diagonal-hatch) for `is_fallback` regions, reproducing
  `weight_map.py`'s `pattern_shape="fit_status"` behavior without Plotly.
- **Skill trends** (`skillTrends.js`, step 4): hand-drawn SVG multi-series
  line chart (`<path>` per method, `<circle>` per marker), reproducing
  `px.line(..., markers=True)`.
- **Blended map** (`blendedMap.js`, step 5): `<canvas>`, one `fillRect`
  per grid cell using `rain_bin_index` → `RAIN_BIN_COLORS` directly (no
  colorscale-interpolation library needed at all, since the real bins are
  already discrete integers — this is actually *simpler* without Plotly,
  not harder).
- **Extreme-probability map** (`extremeProbability.js`, step 6): `<canvas>`,
  two draw passes over the same grid — a continuous-probability pass
  (manual colour interpolation across `PROBABILITY_COLORSCALE`'s 5 stops, at square-root position)
  followed by a second pass drawing semi-transparent grey `fillRect`s only
  where `is_fallback` is true, reproducing the two-`go.Heatmap`-trace
  overlay without a second charting-library layer.

No vendored/CDN JS charting dependency is added anywhere in
`dashboard-web/`.

## 3. API framework — routed via `AskUserQuestion`

Options presented: FastAPI (typed, fits this project's existing
`mypy`-gated CI convention — `pyproject.toml`'s `[tool.mypy]` config and
`dev` optional-dependency group already include `mypy`/`ruff`) vs. Flask
(simpler, smaller footprint, no built-in type enforcement).

**Decision: FastAPI**, plus `uvicorn` as the ASGI server. New
optional-dependency group for `pyproject.toml` (added in step 2, not this
step — this step only locks the name and packages):

```toml
dashboard-api = [
    "fastapi",
    "uvicorn[standard]",
]
```

Kept as its own group, separate from the existing `dashboard` group
(`streamlit`, `plotly`) rather than folded into it — `dashboard/app.py`
(Streamlit) stays untouched and installable independently of the new API
per `frontendplan.md` §5, and per §4 Phase E the two are only reconciled
(possibly retiring one) after parity is proven in step 7.

## 4. Grid payload format — decided directly, not ambiguous enough for `AskUserQuestion`

`frontendplan.md` §2.3/§6 flagged this as worth a real decision rather
than a silent default. The real numbers, checked directly:

- Both committed grids (`dashboard/data/example_blend_grid.npz`,
  `example_probability_grid.npz`) are a fixed `129 x 135` shape (17,415
  cells), at 5 fixed lead times, confirmed by the existing
  `tests/test_dashboard_blended_map.py` / `test_dashboard_extreme_probability.py`
  `TestRealCommittedExport` classes.
- A `129x135` array of Python floats serialized as nested JSON arrays is
  on the order of a few hundred KB per grid response (17,415 numbers ×
  roughly 15-20 bytes each as JSON text) — well within normal HTTP
  response sizes, no pagination or compression concern at today's scale.

**Decision: plain nested JSON arrays** (`list[list[float]]` for
`bin_index`/`probability`, `list[bool]`-shaped for `is_fallback`,
`list[float]` for `latitude`/`longitude`). No binary/typed-array encoding.
This is the simplest option and matches every other endpoint's shape
(`weight-map`/`skill-trends` are already row-oriented JSON). Revisit only
if the real grid resolution or lead-time count grows meaningfully beyond
today's `129x135` / 5-lead reality — not assumed now.

## 5. Locked decisions for step 2 to build against

| Decision | Value |
|---|---|
| Charting | Hand-rolled `<canvas>`/SVG, no JS library dependency |
| API framework | FastAPI + uvicorn |
| New dependency group | `dashboard-api = ["fastapi", "uvicorn[standard]"]` |
| Grid payload format | Plain nested JSON arrays |
| `data_loading.py`/`colors.py` | Reused unchanged, confirmed no Streamlit/Plotly coupling |
