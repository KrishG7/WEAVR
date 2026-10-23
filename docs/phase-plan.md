# Phase Plan: Hybrid AI–NWP Multi-Model Forecast Blending

Tracks SIH PS 26081. See the research brief for citations behind each decision.

## Phase 0 — Data access & scoping
Confirm NCMRWF/organiser access to NCUM/NEPS-G archives and the HEPPI dataset.
Stand up access to WeatherBench 2, ECMWF open data, IMD gridded rainfall.
Lock grid (0.25°) and IMD's 03–03 UTC daily accumulation window (labelled by its end date; corrected in #106).
**Exit criterion**: one full monsoon season of aligned forecasts + IMD obs in one store.

## Phase 1 — Baseline pipeline + verification harness
Ingest → align → store (xarray/dask, Zarr).
Implement the verification protocol before training anything: RMSE/bias/ACC,
CRPS/Brier, SEEPS, FSS/pFSS, POD/FAR/CSI/ETS at IMD thresholds
(7.5/64.5/115.6/204.5 mm). Block-by-time splits only. Verify against IMD gauges,
not reanalysis. Run the equal-weight mean as the Tier 0 baseline.

## Phase 2 — Make AI models ensemble-compatible
Turn deterministic AI models (GraphCast, Pangu, ...) into probabilistic
ensembles via lagged ensembles (±4 starts, 12h apart) so they can enter
BMA/EMOS like NWP ensemble members.

## Phase 3 — Static/regional skill weighting (Tier 1)
Fit weights per lead time × region × season via leave-one-year-out CV.
Pool regionally (Neal et al. 30-pattern k-means, or Sreekala & Babu's 6-zone
scheme) given limited AI hindcast years. Clip negative weights to zero.

## Phase 4 — Hierarchical BMA/EMOS stratified by rain regime (Tier 2)
Ji et al. 2025 recipe, re-thresholded to IMD categories: classify each day's
ensemble-mean forecast into a rain-intensity bin, fit a separate combiner per
bin. EMOS-CSG as default combiner (60–90 day rolling window). Also fit
hierarchical BMA and compare per region/season — this tier is the safe,
literature-backed core deliverable.

## Phase 5 — Regime-conditioned + learned-dynamic weighting (Tier 3)
Add regime covariates: monsoon active/break, BSISO/MJO phase, depression/
western-disturbance presence. Optional GBM/ViT-style gating if Phase 4
plateaus. This is the differentiation territory — timebox and evaluate
against Phase 4, don't assume it works.

## Phase 6 — Robustness & operational concerns
Renormalize weights over available members when a source is missing/late.
Decide: run AI models ourselves vs. consume published forecasts.
Daily-refreshed rolling verification and drift detection after model upgrades.

## Phase 7 — Dashboard & deliverables
Blended map, weight-map per model × lead time, skill trend charts,
extreme-probability maps in IMD colour codes. Streamlit/Plotly/Leaflet.
