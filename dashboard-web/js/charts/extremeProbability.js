/**
 * Extreme-probability-map view: reproduces
 * dashboard/views/extreme_probability.py's real behavior with a
 * hand-rolled <canvas> two-layer heatmap (per
 * docs/frontend-migration-scope.md's locked charting decision -- no JS
 * charting library).
 *
 * This view diverges from the blended-map's Tier 1 combiner (step 5):
 * Tier 1's deterministic regional blend has no predictive distribution to
 * compute an exceedance probability from at all. Instead this view uses
 * EMOS-CSG's `ifs_ens` combiner -- fit on the real 50-member IFS ensemble
 * (weavr.emos.fit_emos_csg) -- because the censored-shifted-gamma has a
 * real closed-form survival function
 * (weavr.emos.exceedance_probability_csgd), unlike BMA's mixture, which
 * has no closed form.
 *
 * 204.5mm is IMD's own real "extremely heavy rain" boundary
 * (weavr.verify.IMD_RAIN_THRESHOLDS_MM[-1]); 115.6mm is the "very heavy"
 * boundary below it. Both are offered as a real toggle, read from
 * /api/colors's imd_rain_thresholds_mm rather than hardcoded, and the
 * threshold is always sent explicitly to the API so the label can never
 * disagree with the data.
 * On the two-season base the extremely_heavy bin is never fittable at any
 * lead, and very_heavy not in every fit at 96-120h
 * (docs/v2-evidence-base-results.md section 7, results/bin_fittability.csv).
 * The grid is fitted on the 2018 season and shows a 2020 day, so the map
 * never displays a day its own calibration was trained on (#74).
 * Cells whose own forecast fell in a bin EMOS-CSG could not fit for real
 * are drawn with a second, distinct semi-transparent grey pass -- a
 * fallback cell's ~0 probability must not look identical to a genuinely
 * low-risk, real fitted cell.
 *
 * Step 12 adds a third per-cell method, `csgd+gpd_tail`: the point forecast
 * was too rare to fit a bin, so its exceedance probability comes from a
 * pooled generalised-Pareto tail fitted above the threshold instead. Those
 * numbers come from a different estimator than their neighbours', so they
 * get their own marking -- a centre dot, a texture rather than a colour --
 * leaving the underlying probability colour readable underneath and
 * staying legible in greyscale and for colourblind readers. The `csgd`
 * cells that need no tail stay unmarked, as the plain fitted value.
 *
 * The probability colourscale (GET /api/colors's probability_colorscale,
 * dashboard.colors.PROBABILITY_COLORSCALE server-side) is a continuous
 * 5-stop gradient, unlike the blended map's discrete rain bins -- this
 * view interpolates between stops per real probability value, matching
 * the Streamlit version's go.Heatmap continuous colourscale.
 */

const EXTREME_PROBABILITY_CELL_SIZE = 4;
const EXTREME_PROBABILITY_FALLBACK_RGBA = "rgba(120, 120, 120, 0.75)";

// The pooled-tail method's marking, and the same accent used for its legend
// swatch and for the geography-mode overlay, so one cell reads the same way
// in all three places.
const EXTREME_PROBABILITY_TAIL_METHOD = "csgd+gpd_tail";
const EXTREME_PROBABILITY_TAIL_DOT_RGBA = "rgba(20, 20, 20, 0.85)";
const EXTREME_PROBABILITY_TAIL_DOT_SIZE = 2;
const EXTREME_PROBABILITY_TAIL_GEOGRAPHY_COLOR = "#141414";

// The colour scale is square-root: a probability p sits at position sqrt(p)
// along the server's five stops. Extreme-rain probabilities are a few percent
// at most, and on a linear 0-100% scale every such cell fell inside the first
// sliver between "dry" and "light" and the map read as blank. The scale is
// FIXED (not stretched to each map's maximum), so one probability is one
// colour at every lead and threshold and the maps stay comparable; the legend
// labels real probabilities at the positions they occupy.
function extremeProbabilityScalePosition(probability) {
  return Math.sqrt(Math.min(1, Math.max(0, probability)));
}

// Probabilities labelled on the legend, placed at their scale position.
const EXTREME_PROBABILITY_LEGEND_TICKS = [0, 0.01, 0.05, 0.1, 0.25, 0.5, 1];

// Geography mode. The colour scale's first two stops are dry (white) and
// light (green, at scale position 0.25, i.e. p = 6.25%), so a solid fill at
// low probability would wash the basemap out with white. Opacity therefore
// rises linearly in scale position from 0 at p = 0 to its full value at the
// first non-dry stop; colours are still interpolated from the real scale.
const EXTREME_PROBABILITY_FULL_OPACITY_AT = 0.25;
const EXTREME_PROBABILITY_MAX_OPACITY = 0.85;
const EXTREME_PROBABILITY_GEOGRAPHY_FALLBACK_COLOR = "#787878";

// Remembered across lead changes so the choice survives a re-render.
let extremeProbabilityGeographyOn = false;
// Likewise for the threshold: re-picking a lead must not silently drop the
// reader back to 204.5mm.
let extremeProbabilityThreshold = null;

const EXTREME_PROBABILITY_GEOGRAPHY_UNAVAILABLE =
  "Basemap unavailable -- showing the plain grid.";

/**
 * True when `grid` marks this cell as fitted with the pooled GPD tail:
 * `csgd+gpd_tail` on the EMOS-CSG map, `quantile_avg+gpd_tail` on a grid
 * exported with `--combiner quantile_avg` (#75). Any `+gpd_tail` suffix counts.
 */
function extremeProbabilityIsTailCell(grid, latIndex, lonIndex) {
  if (!Array.isArray(grid.method)) return false;
  const method = String(grid.method[latIndex][lonIndex]);
  return method === EXTREME_PROBABILITY_TAIL_METHOD || method.endsWith("+gpd_tail");
}

/**
 * Real per-method cell counts at this lead, taken from the fetched `method`
 * array -- not hardcoded, and not assumed. A committed grid that predates
 * step 12 has no `method` array at all, in which case the server reports
 * every cell as plain `csgd` and the tail legend entry is reported as
 * unused rather than as a claim that tail cells exist.
 */
function extremeProbabilityMethodCounts(grid) {
  const counts = { csgd: 0, tail: 0, fallback: 0 };
  const latCount = grid.probability.length;
  const lonCount = grid.probability[0].length;
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      if (grid.is_fallback[latIndex][lonIndex]) counts.fallback += 1;
      else if (extremeProbabilityIsTailCell(grid, latIndex, lonIndex)) counts.tail += 1;
      else counts.csgd += 1;
    }
  }
  return counts;
}

/**
 * True when every fitted cell uses the pooled tail -- the case for any
 * threshold above the 64.5mm splice point. A per-cell marker then
 * distinguishes nothing and only hides the probability colours (a dot on
 * every cell of the plain grid, near-black over every cell of the
 * geography map), so neither is drawn and the legend and banner say it once
 * for the whole map instead.
 */
function extremeProbabilityTailIsUniform(counts) {
  return counts.tail > 0 && counts.csgd === 0;
}

/** "115.6mm" -- thresholds render without a trailing .0 anywhere they show. */
function formatThreshold(threshold) {
  return `${Number(threshold)}mm`;
}

/** Fill opacity for a cell of probability `p` in geography mode. */
function extremeProbabilityOpacity(p) {
  const fraction = Math.min(
    1,
    extremeProbabilityScalePosition(p) / EXTREME_PROBABILITY_FULL_OPACITY_AT
  );
  return fraction * EXTREME_PROBABILITY_MAX_OPACITY;
}

/**
 * Why the map is still pale, said in the plain grid as well as the geography
 * view: extreme-rain probabilities are a few percent at most. The colours use
 * a square-root scale to spread them out (1% is 10% of the way along the bar,
 * 4% is 20%), but a map whose highest value is a few percent still sits near
 * the pale end by design -- it is the data, not missing data.
 */
function extremeProbabilityPaleNote(maxProbability) {
  return (
    `The highest probability on this map is ${(maxProbability * 100).toFixed(1)}%. ` +
    "Colours use a fixed square-root scale (1% is 10% of the way along the bar), " +
    "so small probabilities show, and the same value has the same colour at " +
    "every lead and threshold; a mostly pale map is the data -- it is not " +
    "missing data."
  );
}

/** Highest probability on the grid, so the note can say why a map looks pale. */
function extremeProbabilityMax(grid) {
  let max = 0;
  grid.probability.forEach((row) => row.forEach((p) => { if (p > max) max = p; }));
  return max;
}

const EXTREME_PROBABILITY_CAPTION =
  "This map shows EMOS-CSG's real fitted ifs_ens combiner (the " +
  "real 50-member IFS ensemble) -- chosen over BMA because its " +
  "censored-shifted-gamma has a real closed-form exceedance " +
  "probability, unlike BMA's mixture. This diverges from the " +
  "blended map's Tier 1 combiner, which has no predictive " +
  "distribution to compute a probability from. The two thresholds are " +
  "IMD's own real rainfall boundaries, 115.6mm ('very heavy') and " +
  "204.5mm ('extremely heavy'). At both thresholds every cell takes its probability from " +
  "a pooled extreme-value tail rather than from EMOS-CSG directly: above " +
  "the splice point of 64.5mm the probability is extrapolated by a " +
  "generalised-Pareto tail fitted to the pooled upper tail of the fittable " +
  "bins, with one shared shape and per-zone scales. That extrapolation is " +
  "the limit of this map -- it is a tail estimate borrowed from rarer, " +
  "fittable bins, not a probability fitted to those cells' own rain, and " +
  "it should be read as the least certain number here, and as " +
  "experimental: measured on two seasons, the spliced tail removes every " +
  "false zero-probability cell but makes the extremal dependence index " +
  "worse at 5 of 5 leads. In total it forecasts these extremes about as " +
  "often as they occur (0.96x at 115.6mm, 1.15x at 204.5mm), where the " +
  "point-mass control forecast them about half as often " +
  "(docs/tail-repair-results.md). It is " +
  "calibrated for magnitude, not yet for spatial co-occurrence. The tail " +
  "is applied uniformly above 64.5mm rather than only to unfittable cells: " +
  "on the two-season base dry, light and heavy always fit, very_heavy " +
  "fits at 24-72h but not in every fit at 96-120h, and extremely_heavy " +
  "never fits at any lead (results/bin_fittability.csv). EMOS-CSG is " +
  "fitted on the 2018 season and the tail on the IMD archive without " +
  "2020, then applied to the 2020 day shown, so the " +
  "map never shows a day its calibration was trained on. Grey cells are " +
  "the harder fallback: no real probability at all, because EMOS-CSG " +
  "could not fit their bin for real at this lead.";

function hexToRgb(hex) {
  const value = hex.replace("#", "");
  return [
    parseInt(value.substring(0, 2), 16),
    parseInt(value.substring(2, 4), 16),
    parseInt(value.substring(4, 6), 16),
  ];
}

/**
 * Linearly interpolates the scale position of `probability` (0..1; square-root,
 * see extremeProbabilityScalePosition) against `colorscale`'s real
 * [position, hexColor] stops (from /api/colors's probability_colorscale),
 * returning an "rgb(r,g,b)" string. Matches Plotly's own continuous
 * colourscale interpolation between the same stops the Streamlit view
 * uses.
 */
function interpolateProbabilityColor(colorscale, probability) {
  const p = extremeProbabilityScalePosition(probability);
  for (let i = 0; i < colorscale.length - 1; i++) {
    const [p0, hex0] = colorscale[i];
    const [p1, hex1] = colorscale[i + 1];
    if (p >= p0 && p <= p1) {
      const t = p1 === p0 ? 0 : (p - p0) / (p1 - p0);
      const [r0, g0, b0] = hexToRgb(hex0);
      const [r1, g1, b1] = hexToRgb(hex1);
      const r = Math.round(r0 + (r1 - r0) * t);
      const g = Math.round(g0 + (g1 - g0) * t);
      const b = Math.round(b0 + (b1 - b0) * t);
      return `rgb(${r},${g},${b})`;
    }
  }
  return hexToRgb(colorscale[colorscale.length - 1][1]).join(",");
}

/**
 * Draws `grid.probability` as a continuous heatmap (first pass), then
 * `grid.is_fallback` as a second, distinct semi-transparent grey pass on
 * top -- two real layers, not one colour choice, matching
 * dashboard/views/extreme_probability.py's two go.Heatmap traces.
 */
function renderExtremeProbabilityCanvas(container, grid, colorscale) {
  container.innerHTML = "";

  const latCount = grid.probability.length;
  const lonCount = grid.probability[0].length;

  const canvas = document.createElement("canvas");
  canvas.width = lonCount * EXTREME_PROBABILITY_CELL_SIZE;
  canvas.height = latCount * EXTREME_PROBABILITY_CELL_SIZE;
  canvas.style.width = "100%";
  canvas.style.height = "auto";
  canvas.style.imageRendering = "pixelated";
  canvas.setAttribute("role", "img");
  canvas.setAttribute(
    "aria-label",
    `P(rain > ${formatThreshold(grid.threshold)}) map at lead ${grid.lead_hours}h`
  );

  const ctx = canvas.getContext("2d");

  // Layer 1: continuous probability heatmap.
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      const probability = grid.probability[latIndex][lonIndex];
      ctx.fillStyle = interpolateProbabilityColor(colorscale, probability);
      ctx.fillRect(
        lonIndex * EXTREME_PROBABILITY_CELL_SIZE,
        canvasRow * EXTREME_PROBABILITY_CELL_SIZE,
        EXTREME_PROBABILITY_CELL_SIZE,
        EXTREME_PROBABILITY_CELL_SIZE
      );
    }
  }

  // Layer 2: a second, distinct pass -- semi-transparent grey only where
  // is_fallback is true, never blended into layer 1's colour choice.
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      if (grid.is_fallback[latIndex][lonIndex]) {
        ctx.fillStyle = EXTREME_PROBABILITY_FALLBACK_RGBA;
        ctx.fillRect(
          lonIndex * EXTREME_PROBABILITY_CELL_SIZE,
          canvasRow * EXTREME_PROBABILITY_CELL_SIZE,
          EXTREME_PROBABILITY_CELL_SIZE,
          EXTREME_PROBABILITY_CELL_SIZE
        );
      }
    }
  }

  // Layer 3: the pooled-tail cells' own marking -- a centred dot, so the
  // probability colour underneath stays readable. Drawn after the grey
  // fallback pass so a tail cell is never confused with a fallback one.
  // Skipped when every fitted cell is a tail cell: there is nothing to tell
  // apart (see extremeProbabilityTailIsUniform).
  if (extremeProbabilityTailIsUniform(extremeProbabilityMethodCounts(grid))) {
    container.appendChild(canvas);
    return;
  }
  const dotInset =
    (EXTREME_PROBABILITY_CELL_SIZE - EXTREME_PROBABILITY_TAIL_DOT_SIZE) / 2;
  ctx.fillStyle = EXTREME_PROBABILITY_TAIL_DOT_RGBA;
  for (let latIndex = 0; latIndex < latCount; latIndex++) {
    const canvasRow = latCount - 1 - latIndex;
    for (let lonIndex = 0; lonIndex < lonCount; lonIndex++) {
      if (extremeProbabilityIsTailCell(grid, latIndex, lonIndex)) {
        ctx.fillRect(
          lonIndex * EXTREME_PROBABILITY_CELL_SIZE + dotInset,
          canvasRow * EXTREME_PROBABILITY_CELL_SIZE + dotInset,
          EXTREME_PROBABILITY_TAIL_DOT_SIZE,
          EXTREME_PROBABILITY_TAIL_DOT_SIZE
        );
      }
    }
  }

  container.appendChild(canvas);
}

function renderExtremeProbabilityLegend(container, colorscale, grid, counts) {
  const wrapper = document.createElement("div");
  wrapper.style.marginTop = "12px";
  wrapper.style.maxWidth = "400px";

  const label = document.createElement("div");
  label.style.fontSize = "0.85rem";
  label.style.color = "var(--color-text-muted)";
  label.style.marginBottom = "4px";
  label.textContent = `P(rain > ${formatThreshold(grid.threshold)})`;
  wrapper.appendChild(label);

  const gradientStops = colorscale.map(([pos, hex]) => `${hex} ${pos * 100}%`).join(", ");
  const bar = document.createElement("div");
  bar.style.height = "14px";
  bar.style.borderRadius = "3px";
  bar.style.border = "1px solid var(--color-border)";
  bar.style.background = `linear-gradient(to right, ${gradientStops})`;
  wrapper.appendChild(bar);

  // Real probabilities at the positions they occupy on the square-root scale.
  const ticks = document.createElement("div");
  ticks.style.position = "relative";
  ticks.style.height = "1.1em";
  ticks.style.fontSize = "0.75rem";
  ticks.style.color = "var(--color-text-muted)";
  EXTREME_PROBABILITY_LEGEND_TICKS.forEach((tick, index, all) => {
    const span = document.createElement("span");
    span.textContent = `${tick * 100}%`;
    span.style.position = "absolute";
    span.style.left = `${extremeProbabilityScalePosition(tick) * 100}%`;
    span.style.transform =
      index === 0 ? "none" : index === all.length - 1 ? "translateX(-100%)" : "translateX(-50%)";
    ticks.appendChild(span);
  });
  wrapper.appendChild(ticks);

  const tailNote = document.createElement("div");
  tailNote.style.fontSize = "0.8rem";
  tailNote.style.color = "var(--color-text-muted)";
  tailNote.style.marginTop = "8px";
  // The real count, and -- when the committed grid predates step 12 -- an
  // honest "none in this grid" rather than a legend implying cells exist.
  // "Experimental" is in the legend as well as the caption: measured on two
  // seasons the tail removes every false zero but worsens SEDI at 5 of 5
  // leads, and the legend is what a reader looks at while reading the map.
  tailNote.innerHTML =
    '<span class="legend-swatch" style="background:' +
    EXTREME_PROBABILITY_TAIL_GEOGRAPHY_COLOR +
    '"></span>pooled GPD tail <strong>(experimental)</strong>' +
    (extremeProbabilityTailIsUniform(counts)
      ? `: every cell on this map (${counts.tail}), because the threshold lies ` +
        "above the 64.5mm splice point -- so cells are not marked one by one"
      : counts.tail > 0
        ? ` (${counts.tail} cell${counts.tail === 1 ? "" : "s"} at this lead; ` +
          "probability extrapolated from the pooled upper tail)"
        : " (no cells at this threshold use it)");
  wrapper.appendChild(tailNote);

  const fallbackNote = document.createElement("div");
  fallbackNote.style.fontSize = "0.8rem";
  fallbackNote.style.color = "var(--color-text-muted)";
  fallbackNote.style.marginTop = "4px";
  fallbackNote.innerHTML =
    '<span class="legend-swatch" style="background:' +
    EXTREME_PROBABILITY_FALLBACK_RGBA +
    '"></span>fallback cell (not fittable at this lead)';
  wrapper.appendChild(fallbackNote);

  const scaleNote = document.createElement("div");
  scaleNote.style.fontSize = "0.75rem";
  scaleNote.style.color = "var(--color-text-muted)";
  scaleNote.style.marginTop = "4px";
  scaleNote.textContent =
    "Square-root colour scale: fixed, so comparable across leads and thresholds.";
  wrapper.appendChild(scaleNote);

  container.appendChild(wrapper);
}

/**
 * The real threshold control: one radio per threshold the server says the
 * extreme-probability endpoint accepts
 * (/api/colors's extreme_probability_thresholds_mm), so the options come
 * from the server's own contract rather than being restated here. Calls
 * `onChange` with the newly picked value; the caller re-renders.
 */
function buildExtremeProbabilityThresholdControl(
  container,
  thresholds,
  current,
  onChange
) {
  const group = document.createElement("div");
  group.className = "threshold-toggle";
  group.setAttribute("role", "radiogroup");
  group.setAttribute("aria-label", "Rainfall threshold");

  const legend = document.createElement("span");
  legend.className = "threshold-toggle-legend";
  legend.textContent = "Threshold";
  group.appendChild(legend);

  thresholds.forEach((value) => {
    const label = document.createElement("label");
    label.className = "threshold-toggle-option";
    const input = document.createElement("input");
    input.type = "radio";
    input.name = "extreme-probability-threshold";
    input.value = String(value);
    input.checked = value === current;
    input.addEventListener("change", () => {
      if (input.checked) onChange(value);
    });
    label.appendChild(input);
    label.appendChild(
      document.createTextNode(
        Number(value) === thresholds[thresholds.length - 1]
          ? ` ${formatThreshold(value)} (extremely heavy)`
          : ` ${formatThreshold(value)} (very heavy)`
      )
    );
    group.appendChild(label);
  });

  container.appendChild(group);
  return group;
}

const ExtremeProbabilityView = {
  /** Fetches real data for `lead` and renders canvas + captions + legend + warning into `container`. */
  async render(container, lead) {
    container.innerHTML = '<p class="loading-state">Loading extreme-probability map…</p>';

    // Colours and the real IMD threshold list come first: the threshold
    // decides which grid to fetch, and the control can only offer the
    // server's own values.
    let colors;
    try {
      colors = await getColors();
    } catch (err) {
      container.innerHTML = `<p class="error-state">Failed to load extreme-probability map: ${err.message}</p>`;
      return;
    }

    // The control must offer exactly the thresholds
    // /api/extreme-probability accepts. That is a strict subset of
    // /api/colors's imd_rain_thresholds_mm, which carries IMD's lighter
    // boundaries too and has no fitted grid behind it -- offering the whole
    // tuple would offer requests the API rejects with 422.
    const thresholds = (
      colors.extreme_probability_thresholds_mm || colors.imd_rain_thresholds_mm || []
    )
      .slice()
      .sort((a, b) => a - b);
    if (thresholds.length === 0) {
      container.innerHTML =
        '<p class="error-state">The server reported no extreme-probability thresholds, so there is nothing to plot.</p>';
      return;
    }
    // Default to the heaviest real boundary, which is what this view always
    // showed before the toggle existed.
    if (extremeProbabilityThreshold === null) {
      extremeProbabilityThreshold = thresholds[thresholds.length - 1];
    }

    const renderUnavailable = (message) => {
      container.innerHTML = "";
      const note = document.createElement("p");
      note.className = "error-state";
      note.textContent = `No ${formatThreshold(extremeProbabilityThreshold)} data at lead ${lead}h: ${message}`;
      container.appendChild(note);
      // The control stays, so an unavailable threshold is one click away from
      // being changed rather than a dead end.
      buildExtremeProbabilityThresholdControl(
        container,
        thresholds,
        extremeProbabilityThreshold,
        (value) => {
          extremeProbabilityThreshold = value;
          this.render(container, lead);
        }
      );
    };

    let grid;
    try {
      grid = await fetchExtremeProbability(lead, extremeProbabilityThreshold);
    } catch (err) {
      // 404 means this threshold is genuinely not in the committed export --
      // say so in the server's own words instead of quietly falling back to
      // the other threshold under a mismatched label.
      renderUnavailable(err.message);
      return;
    }

    container.innerHTML = "";

    const subheader = document.createElement("h2");
    subheader.className = "view-subheader";
    subheader.textContent =
      `Extreme-probability map: P(rain > ${formatThreshold(grid.threshold)})`;
    container.appendChild(subheader);

    buildExtremeProbabilityThresholdControl(
      container,
      thresholds,
      grid.threshold,
      (value) => {
        extremeProbabilityThreshold = value;
        this.render(container, lead);
      }
    );

    const caption = document.createElement("p");
    caption.className = "view-caption";
    caption.textContent = EXTREME_PROBABILITY_CAPTION;
    container.appendChild(caption);

    const title = document.createElement("p");
    title.className = "chart-title";
    title.textContent =
      `P(rain > ${formatThreshold(grid.threshold)}) at lead ${grid.lead_hours}h ` +
      `(sample: ${grid.sample_time})`;
    container.appendChild(title);

    const toggleLabel = document.createElement("label");
    toggleLabel.className = "map-mode-toggle";
    const toggle = document.createElement("input");
    toggle.type = "checkbox";
    toggle.id = "extreme-probability-geography";
    toggle.checked = extremeProbabilityGeographyOn;
    toggleLabel.appendChild(toggle);
    toggleLabel.appendChild(document.createTextNode("Show geography"));
    container.appendChild(toggleLabel);

    const geographyNote = document.createElement("p");
    geographyNote.className = "view-caption";
    geographyNote.id = "extreme-probability-geography-note";
    container.appendChild(geographyNote);

    const canvasContainer = document.createElement("div");
    container.appendChild(canvasContainer);
    renderExtremeProbabilityCanvas(canvasContainer, grid, colors.probability_colorscale);

    const mapEntry = BasemapMap.shared("extreme-probability");
    const mapContainer = mapEntry.element;
    mapContainer.id = "extreme-probability-geography-map";
    mapContainer.hidden = true;
    container.appendChild(mapContainer);

    const counts = extremeProbabilityMethodCounts(grid);
    const tailUniform = extremeProbabilityTailIsUniform(counts);
    const maxProbability = extremeProbabilityMax(grid);
    const geographyNoteText =
      "Cells are drawn on an OpenStreetMap-derived basemap served from this " +
      "machine, with the same colours as the plain grid. Low probabilities " +
      "fade to transparent so the geography shows through; grey cells are the " +
      "fallback cells" +
      (tailUniform
        ? " (every other cell uses the pooled tail at this threshold). The "
        : ", and near-black cells are the pooled-tail cells. The ") +
      "highest probability at this lead is " +
      `${(maxProbability * 100).toFixed(1)}%, so the map is pale by design ` +
      "-- it is not missing data. The basemap draws no national boundaries.";

    const paleNote = extremeProbabilityPaleNote(maxProbability);
    const showCanvas = (message) => {
      mapContainer.hidden = true;
      canvasContainer.hidden = false;
      geographyNote.textContent = message ? `${message} ${paleNote}` : paleNote;
    };
    showCanvas("");

    const showGeography = async () => {
      if (!(await BasemapMap.isAvailable())) {
        throw new Error("basemap unavailable");
      }
      const handle = await mapEntry.handle();
      handle.setCells({
        latitude: grid.latitude,
        longitude: grid.longitude,
        colorAt: (i, j) => interpolateProbabilityColor(colors.probability_colorscale, grid.probability[i][j]),
        opacityAt: (i, j) => extremeProbabilityOpacity(grid.probability[i][j]),
        // Fallback wins over tail: a fallback cell has no real probability at
        // all, so it must not be dressed up as a merely-uncertain one.
        overlayColorAt: (i, j) => {
          if (grid.is_fallback[i][j]) return EXTREME_PROBABILITY_GEOGRAPHY_FALLBACK_COLOR;
          if (!tailUniform && extremeProbabilityIsTailCell(grid, i, j)) {
            return EXTREME_PROBABILITY_TAIL_GEOGRAPHY_COLOR;
          }
          return null;
        },
      });
      canvasContainer.hidden = true;
      mapContainer.hidden = false;
      handle.resize();
      geographyNote.textContent = geographyNoteText;
    };

    const applyMode = async () => {
      extremeProbabilityGeographyOn = toggle.checked;
      if (!toggle.checked) {
        showCanvas("");
        return;
      }
      try {
        await showGeography();
      } catch (err) {
        // Any failure leaves the plain grid in place, with the reason stated.
        toggle.checked = false;
        extremeProbabilityGeographyOn = false;
        mapContainer.innerHTML = "";
        showCanvas(EXTREME_PROBABILITY_GEOGRAPHY_UNAVAILABLE);
      }
    };
    toggle.addEventListener("change", applyMode);
    if (extremeProbabilityGeographyOn) applyMode();

    renderExtremeProbabilityLegend(
      container,
      colors.probability_colorscale,
      grid,
      counts
    );

    // Both counts are the real fetched per-cell method flags' true counts --
    // not hardcoded -- so a grid that uses no tail cells says so rather than
    // implying any.
    if (counts.tail > 0) {
      const warning = document.createElement("div");
      warning.className = "warning-banner";
      warning.textContent = tailUniform
        ? `Every gridpoint on this map (${counts.tail}) takes its exceedance ` +
          "probability from the pooled extreme-value tail, because the " +
          "threshold lies above the 64.5mm splice point: the whole map is " +
          "an extrapolation, and the least certain kind of value here (see " +
          "caption above)."
        : `${counts.tail} gridpoint(s) at this lead are dotted: their ` +
          "exceedance probability is extrapolated by the pooled " +
          "extreme-value tail rather than fitted to their own bin, so they " +
          "are the least certain values on this map (see caption above).";
      container.appendChild(warning);
    }

    if (counts.fallback > 0) {
      const warning = document.createElement("div");
      warning.className = "warning-banner";
      warning.textContent =
        `${counts.fallback} gridpoint(s) at this lead are shown grey: their ` +
        "forecast fell in a rain-intensity bin with too few real " +
        "training days to fit EMOS-CSG (see caption above).";
      container.appendChild(warning);
    }
  },
};
