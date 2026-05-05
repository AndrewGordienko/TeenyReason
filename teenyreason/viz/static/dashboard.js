const modePalette = ["#2854d8", "#14161a", "#3d8d5e", "#b9821f", "#b85b6c", "#3b6f88", "#7f56d9", "#d96a5e"];
const dashboardState = {
  indexPayload: null,
  latentPayload: null,
  benchmarkPayload: null,
  livePayload: null,
  activeDeck: "live",
  liveArchiveSelection: "live",
  comparisonArchiveMode: "keep",
  selectedLatentMtime: null,
  selectedBenchmarkMtime: null,
};

const liveAnimationState = {
  started: false,
  current: null,
  target: null,
};

function el(id) {
  return document.getElementById(id);
}

function setSelectOptions(selectEl, values, emptyLabel = "") {
  selectEl.innerHTML = "";
  if (emptyLabel) {
    const emptyOption = document.createElement("option");
    emptyOption.value = "";
    emptyOption.textContent = emptyLabel;
    selectEl.appendChild(emptyOption);
  }
  for (const value of values) {
    const option = document.createElement("option");
    option.value = value;
    option.textContent = value;
    selectEl.appendChild(option);
  }
}

function formatInteger(value) {
  return Number(value).toLocaleString();
}

function formatCompactNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return new Intl.NumberFormat(undefined, {notation: "compact", maximumFractionDigits: 1}).format(numeric);
}

function formatSampleSavings(value, positiveLabel = "saves", negativeLabel = "costs") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  return numeric >= 0
    ? `${positiveLabel} ${formatCompactNumber(numeric)}`
    : `${negativeLabel} ${formatCompactNumber(Math.abs(numeric))}`;
}

function formatDateTime(epochSeconds) {
  const numeric = Number(epochSeconds);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return "n/a";
  }
  return new Date(numeric * 1000).toLocaleString();
}

function humanizeArtifactName(name) {
  if (!name) {
    return "none selected";
  }
  return name
    .replace(/\.npz$/i, "")
    .replace(/_/g, " ")
    .replace(/\bppo\b/gi, "PPO")
    .trim();
}

function inferEnvLabelFromName(name) {
  const normalized = (name || "").toLowerCase();
  if (normalized.includes("randomized_continuous_cartpole")) {
    return "Randomized Continuous CartPole";
  }
  if (normalized.includes("randomized_cartpole")) {
    return "Randomized CartPole";
  }
  if (normalized.includes("continuous_cartpole")) {
    return "Continuous CartPole";
  }
  if (normalized.includes("continuous_lunar_lander")) {
    return "Continuous LunarLander";
  }
  if (normalized.includes("bipedal_walker")) {
    return "Bipedal Walker";
  }
  return "";
}

function currentEnvLabel() {
  const contextEnv = dashboardState.indexPayload?.run_context?.env_display_name;
  if (contextEnv) {
    return contextEnv;
  }
  const liveEnv = dashboardState.livePayload?.env_display_name;
  if (liveEnv) {
    return liveEnv;
  }
  const latentEnv = dashboardState.latentPayload?.env_display_name;
  if (latentEnv) {
    return latentEnv;
  }
  const benchmarkEnv = dashboardState.benchmarkPayload?.env_display_name;
  if (benchmarkEnv) {
    return benchmarkEnv;
  }
  const selected = el("benchmarkSelect")?.value || el("latentSelect")?.value || "";
  return inferEnvLabelFromName(selected) || (selected ? humanizeArtifactName(selected) : "Awaiting selection");
}

function solveBadge(value, notRun = false) {
  if (notRun) {
    return "not run";
  }
  if (value == null || value < 0) {
    return "unsolved";
  }
  return formatInteger(value);
}

function hasSolveSummary(summary) {
  return Boolean(summary && !summary.not_run);
}

function solveSummaryLabel(summary) {
  if (!hasSolveSummary(summary)) {
    return "not run";
  }
  return Number(summary.median || 0).toFixed(1);
}

function quantile(values, q) {
  if (!values.length) {
    return 0;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  if (sorted[base + 1] !== undefined) {
    return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  }
  return sorted[base];
}

function computeLatentDerived(payload) {
  if (!payload || !payload.points || !payload.points.length) {
    return null;
  }
  const uncertainties = payload.points.map((point) => point.uncertainty);
  const envParamMeans = payload.points.map((point) => point.env_param_mean);
  const series = payload.series || {};
  const pairwiseBetween = (series.pairwise_between_distance || []).map(Number).filter((value) => Number.isFinite(value));
  const pairwiseBetweenUnit = (series.pairwise_between_distance_unit || []).map(Number).filter((value) => Number.isFinite(value));
  const splitRetrievalRanks = (series.split_retrieval_rank || []).map(Number).filter((value) => Number.isFinite(value));
  const crossFamilySplitRetrievalRanks = (series.cross_family_split_retrieval_rank || []).map(Number).filter((value) => Number.isFinite(value));
  const beliefNorms = payload.points.map((point) => Number(point.belief_norm ?? 0)).filter((value) => Number.isFinite(value));
  const totalModes = payload.mode_counts.reduce((sum, row) => sum + row.count, 0);
  const sortedModes = [...payload.mode_counts].sort((left, right) => right.count - left.count);
  const topMode = sortedModes[0] || null;
  const diagnostics = payload.diagnostics || {};
  return {
    uncertaintyP10: quantile(uncertainties, 0.10),
    uncertaintyMedian: quantile(uncertainties, 0.50),
    uncertaintyP90: quantile(uncertainties, 0.90),
    uncertaintyStd: standardDeviation(uncertainties),
    beliefNormStd: standardDeviation(beliefNorms),
    envMeanMin: Math.min(...envParamMeans),
    envMeanMax: Math.max(...envParamMeans),
    nearestBetweenMedian: quantile(payload.points.map((point) => Number(point.nearest_between_distance ?? 0)), 0.50),
    pairwiseBetweenMean: diagnostics.pairwise_between_mean ?? (pairwiseBetween.length ? pairwiseBetween.reduce((sum, value) => sum + value, 0) / pairwiseBetween.length : 0),
    pairwiseBetweenP10: diagnostics.pairwise_between_p10 ?? (pairwiseBetween.length ? quantile(pairwiseBetween, 0.10) : 0),
    pairwiseBetweenMeanUnit: diagnostics.pairwise_between_mean_unit ?? (pairwiseBetweenUnit.length ? pairwiseBetweenUnit.reduce((sum, value) => sum + value, 0) / pairwiseBetweenUnit.length : 0),
    pcaTotal: payload.summary.pca_explained.reduce((sum, value) => sum + value, 0),
    topModeName: topMode ? topMode.probe_mode : "n/a",
    topModeShare: topMode ? topMode.count / Math.max(totalModes, 1) : 0,
    linearEnvFitR2: diagnostics.linear_env_fit_r2 ?? 0,
    perParamEnvFitR2: diagnostics.per_param_env_fit_r2 ?? [],
    neighborEnvAlignment: diagnostics.neighbor_env_alignment ?? 0,
    windowModeLeakage: diagnostics.window_mode_leakage ?? 0,
    envModeLeakage: diagnostics.env_mode_leakage ?? 0,
    sameEnvSpread: diagnostics.same_env_spread ?? {mean: 0, p90: 0, max: 0},
    splitRetrievalTop1: diagnostics.split_retrieval_top1 ?? 0,
    splitRetrievalTop5: diagnostics.split_retrieval_top5 ?? 0,
    splitRetrievalMrr: diagnostics.split_retrieval_mrr ?? 0,
    splitRetrievalMedianRank: diagnostics.split_retrieval_median_rank ?? 0,
    crossFamilySplitRetrievalTop1: diagnostics.cross_family_split_retrieval_top1 ?? diagnostics.split_retrieval_top1 ?? 0,
    crossFamilySplitRetrievalMrr: diagnostics.cross_family_split_retrieval_mrr ?? diagnostics.split_retrieval_mrr ?? 0,
    crossFamilyGapRatioMean: diagnostics.cross_family_gap_ratio_mean ?? payload.summary.cross_family_gap_ratio_mean ?? 0,
    sameEnvGapRatio: diagnostics.same_env_gap_ratio ?? {mean: 0, p90: 0},
    envParamUncertaintyMean: diagnostics.env_param_uncertainty_mean ?? 0,
    futureProbeErrorMean: diagnostics.future_probe_error_mean ?? 0,
    mechanicsPosteriorStdMean: diagnostics.mechanics_posterior_std_mean ?? 0,
    mechanicsPosteriorEntropyMean: diagnostics.mechanics_posterior_entropy_mean ?? 0,
    supportGroupCountMean: diagnostics.support_group_count_mean ?? payload.summary.support_group_count_mean ?? 0,
    supportGroupRatioMean: diagnostics.support_group_ratio_mean ?? payload.summary.support_group_ratio_mean ?? 0,
    supportTopFamilyShareMean: diagnostics.support_top_family_share_mean ?? payload.summary.support_top_family_share_mean ?? 0,
    supportEffectiveFamilyCountMean: diagnostics.support_effective_family_count_mean ?? payload.summary.support_effective_family_count_mean ?? 0,
    supportFamilyEntropyMean: diagnostics.support_family_entropy_mean ?? payload.summary.support_family_entropy_mean ?? 0,
    supportTiedTopFamilyCountMean: diagnostics.support_tied_top_family_count_mean ?? payload.summary.support_tied_top_family_count_mean ?? 0,
    splitGroupOverlapMean: diagnostics.split_group_overlap_mean ?? payload.summary.split_group_overlap_mean ?? 0,
    crossFamilySplitGroupOverlapMean: diagnostics.cross_family_split_group_overlap_mean ?? payload.summary.cross_family_split_group_overlap_mean ?? 0,
    splitBalancedHalfFraction: diagnostics.split_balanced_half_fraction ?? payload.summary.split_balanced_half_fraction ?? 0,
    splitGroupCountAMean: diagnostics.split_group_count_a_mean ?? 0,
    splitGroupCountBMean: diagnostics.split_group_count_b_mean ?? 0,
    envParamErrorMean: diagnostics.env_param_error_mean ?? 0,
    uncertaintyFeatureImportance: diagnostics.uncertainty_feature_importance ?? [],
    uncertaintyErrorAlignment: diagnostics.uncertainty_error_alignment || {correlation: 0, low_error: 0, high_error: 0, gap: 0},
    failureLift: diagnostics.failure_lift || {low_rate: 0, high_rate: 0, gap: 0, lift: 1},
    pairwiseBetween,
    pairwiseBetweenUnit,
    splitRetrievalRanks,
    crossFamilySplitRetrievalRanks,
  };
}

function renderHero() {
  const payload = dashboardState.indexPayload;
  if (!payload) {
    return;
  }
  const latent = dashboardState.latentPayload;
  const benchmark = dashboardState.benchmarkPayload?.summaries || null;
  const latentSummary = latent?.summary || null;
  const latentDerived = computeLatentDerived(latent);
  const selectedLatent = el("latentSelect")?.value || "";
  const selectedBenchmark = el("benchmarkSelect")?.value || "";

  const chips = [
    `<div class="hero-chip"><strong>Snapshot</strong>${humanizeArtifactName(selectedLatent)}</div>`,
    `<div class="hero-chip"><strong>Benchmark</strong>${humanizeArtifactName(selectedBenchmark)}</div>`,
    `<div class="hero-chip"><strong>Working Env</strong>${currentEnvLabel()}</div>`,
  ];
  if (latent?.artifact_mtime) {
    chips.push(`<div class="hero-chip"><strong>Snapshot Updated</strong>${formatDateTime(latent.artifact_mtime)}</div>`);
  }
  if (dashboardState.benchmarkPayload?.artifact_mtime) {
    chips.push(`<div class="hero-chip"><strong>Benchmark Updated</strong>${formatDateTime(dashboardState.benchmarkPayload.artifact_mtime)}</div>`);
  }
  el("heroContext").innerHTML = chips.join("");

  const rail = [
    {k: "Environment", v: currentEnvLabel()},
    {k: "Env Beliefs", v: latentSummary ? formatCompactNumber(latentSummary.num_envs) : formatInteger(payload.latent_snapshots.length)},
    {k: "Uncertainty", v: latentSummary ? latentSummary.uncertainty_mean.toFixed(3) : "n/a"},
    {k: "Probe Episode Median", v: benchmark ? benchmark.probe_episode.median.toFixed(1) : "n/a"},
  ];

  el("heroRail").innerHTML = rail.map((item) => `
    <div class="hero-stat">
      <div class="k">${item.k}</div>
      <div class="v">${item.v}</div>
    </div>
  `).join("");

  const board = [
    {
      k: "Latent Dim",
      v: latentSummary ? formatInteger(latentSummary.latent_dim) : "n/a",
      n: "posterior mean dimensionality before any 2D projection"
    },
    {
      k: "PCA Coverage",
      v: latentDerived ? `${(100 * latentDerived.pcaTotal).toFixed(1)}%` : "n/a",
      n: "variance captured by the first two axes of the currently selected latent snapshot"
    },
    {
      k: "Mechanics Fit",
      v: latentDerived ? latentDerived.linearEnvFitR2.toFixed(2) : "n/a",
      n: "linear R² when predicting hidden env parameters from the latent coordinates"
    },
    {
      k: "Neighbor Alignment",
      v: latentDerived ? `${(100 * latentDerived.neighborEnvAlignment).toFixed(1)}%` : "n/a",
      n: "how much closer latent neighbors are in env-parameter space than random pairings"
    },
    {
      k: "Paired Retrieval",
      v: latentDerived ? `${(100 * latentDerived.splitRetrievalTop1).toFixed(1)}%` : "n/a",
      n: "top-1 retrieval accuracy when duplicate probe-family evidence can appear on both split halves"
    },
    {
      k: "Cross Retrieval",
      v: latentDerived ? `${(100 * latentDerived.crossFamilySplitRetrievalTop1).toFixed(1)}%` : "n/a",
      n: "stricter top-1 retrieval when the split halves use disjoint probe-family evidence"
    },
    {
      k: "Support Diversity",
      v: latentDerived ? `${(100 * latentDerived.supportGroupRatioMean).toFixed(1)}%` : "n/a",
      n: "share of support windows that come from distinct experiment families instead of repeated copies of one probe"
    },
    {
      k: "Paired Overlap",
      v: latentDerived ? `${(100 * latentDerived.splitGroupOverlapMean).toFixed(1)}%` : "n/a",
      n: "share of paired split-half family coverage that appears on both sides"
    },
    {
      k: "Split Balance",
      v: latentDerived ? `${(100 * latentDerived.splitBalancedHalfFraction).toFixed(1)}%` : "n/a",
      n: "share of env beliefs whose two disjoint support halves stayed balanced in size"
    }
  ];

  el("metricBoard").innerHTML = board.map((item) => `
    <div class="metric-cell">
      <div class="metric-k">${item.k}</div>
      <div class="metric-v">${item.v}</div>
      <div class="metric-note">${item.n}</div>
    </div>
  `).join("");
}

function metricField(metric) {
  if (metric === "env_param_mean") {
    return "env_param_mean";
  }
  if (metric === "uncertainty") {
    return "uncertainty";
  }
  if (metric === "same_env_spread") {
    return "same_env_spread";
  }
  if (metric === "belief_norm") {
    return "belief_norm";
  }
  if (metric === "env_error") {
    return "env_error";
  }
  if (metric === "gap_ratio") {
    return "gap_ratio";
  }
  if (metric === "split_retrieval_margin_deficit") {
    return "split_retrieval_margin_deficit";
  }
  if (metric === "reward_sum") {
    return "reward_sum";
  }
  if (metric === "terminated") {
    return "terminated_numeric";
  }
  return metric;
}

function colorForMetric(point, metric, modeColorMap) {
  if (metric === "terminated") {
    return point.terminated ? "#b85b6c" : "#d8e4ff";
  }
  const value = point.__normalized_metric ?? 0.5;
  const clamped = Math.max(0, Math.min(1, value));
  if (metric === "reward_sum") {
    const red = Math.round(236 - 142 * clamped);
    const green = Math.round(242 - 118 * clamped);
    const blue = Math.round(255 - 18 * clamped);
    return `rgb(${red}, ${green}, ${blue})`;
  }
  if (metric === "env_param_mean") {
    const red = Math.round(235 - 120 * clamped);
    const green = Math.round(244 - 86 * clamped);
    const blue = Math.round(247 - 140 * clamped);
    return `rgb(${red}, ${green}, ${blue})`;
  }
  const red = Math.round(145 + 64 * clamped);
  const green = Math.round(171 - 74 * clamped);
  const blue = Math.round(232 - 108 * clamped);
  return `rgb(${red}, ${green}, ${blue})`;
}

function normalizeMetric(points, metric) {
  if (metric === "terminated") {
    return points.map((point) => ({...point}));
  }
  const field = metricField(metric);
  const values = points.map((point) => Number(point[field]));
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(max - min, 1e-6);
  return points.map((point) => {
    const raw = Number(point[field]);
    return {...point, __normalized_metric: (raw - min) / span};
  });
}

function standardDeviation(values) {
  if (!values.length) {
    return 0;
  }
  const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
  const variance = values.reduce((sum, value) => sum + (value - mean) ** 2, 0) / values.length;
  return Math.sqrt(Math.max(variance, 0));
}

function liveScalar(value, digits = 3, fallback = "n/a") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return numeric.toFixed(digits);
}

function liveInteger(value, fallback = "n/a") {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return fallback;
  }
  return formatInteger(Math.round(numeric));
}

function friendlyPhaseName(phase) {
  if (!phase) {
    return "idle";
  }
  return String(phase).replace(/_/g, " ");
}

function phaseColor(phase) {
  const normalized = String(phase || "").toLowerCase();
  if (normalized.includes("encoder")) {
    return "#7f56d9";
  }
  if (normalized.includes("baseline")) {
    return "#3d8d5e";
  }
  if (normalized.includes("probe_training") || normalized.includes("probe_control") || normalized.includes("probe_reasoning")) {
    return "#2854d8";
  }
  if (normalized.includes("probe") || normalized.includes("collection") || normalized.includes("crawler")) {
    return "#b9821f";
  }
  if (normalized.includes("done")) {
    return "#3d8d5e";
  }
  return "#2854d8";
}

function lineChartSvg(values, label, color) {
  const numericValues = (values || []).map((value) => Number(value)).filter((value) => Number.isFinite(value));
  if (!numericValues.length) {
    return `<div class="empty">No live curve has been recorded for ${label.toLowerCase()} yet.</div>`;
  }
  const width = 440;
  const height = 170;
  const pad = 26;
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);
  const span = Math.max(max - min, 1e-6);
  const xScale = (idx) => pad + (idx / Math.max(numericValues.length - 1, 1)) * (width - 2 * pad);
  const yScale = (value) => height - pad - ((value - min) / span) * (height - 2 * pad);
  const line = numericValues.map((value, idx) => `${idx === 0 ? "M" : "L"} ${xScale(idx).toFixed(2)} ${yScale(value).toFixed(2)}`).join(" ");
  const area = `${line} L ${xScale(numericValues.length - 1).toFixed(2)} ${(height - pad).toFixed(2)} L ${xScale(0).toFixed(2)} ${(height - pad).toFixed(2)} Z`;
  return `
    <div class="live-curve-card">
      <div class="live-curve-title">${label}</div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label}">
        <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.2"></line>
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.2"></line>
        <path d="${area}" fill="${color}" fill-opacity="0.10"></path>
        <path d="${line}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"></path>
        <circle cx="${xScale(numericValues.length - 1).toFixed(2)}" cy="${yScale(numericValues[numericValues.length - 1]).toFixed(2)}" r="4.2" fill="${color}" stroke="#ffffff" stroke-width="1.5"></circle>
        <text x="${pad}" y="${pad - 8}" fill="#6e7685" font-size="12">min ${min.toFixed(3)} · max ${max.toFixed(3)} · latest ${numericValues[numericValues.length - 1].toFixed(3)}</text>
      </svg>
      <div class="live-mini-note">${numericValues.length} observations tracked in the current live run.</div>
    </div>
  `;
}

function comparisonSolveLabel(values) {
  if (!Array.isArray(values) || !values.length) {
    return "pending";
  }
  const solved = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value >= 0);
  if (!solved.length) {
    return "unsolved";
  }
  return `med ${formatInteger(quantile(solved, 0.5))} · ${solved.length}/${values.length}`;
}

function comparisonLatestAvg10(rows) {
  const values = (rows || [])
    .map((row) => Number(row.avg10))
    .filter((value) => Number.isFinite(value));
  return values.length ? values[values.length - 1].toFixed(1) : "pending";
}

function comparisonSolvedValue(values) {
  if (!Array.isArray(values) || !values.length) {
    return null;
  }
  const solved = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value >= 0);
  return solved.length ? Math.min(...solved) : null;
}

function comparisonMedianPositiveValue(values) {
  if (!Array.isArray(values) || !values.length) {
    return null;
  }
  const numeric = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value) && value >= 0)
    .sort((a, b) => a - b);
  if (!numeric.length) {
    return null;
  }
  const mid = Math.floor(numeric.length / 2);
  if (numeric.length % 2 === 1) {
    return numeric[mid];
  }
  return 0.5 * (numeric[mid - 1] + numeric[mid]);
}

function comparisonMedianFiniteValue(values) {
  if (!Array.isArray(values) || !values.length) {
    return null;
  }
  const numeric = values
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);
  if (!numeric.length) {
    return null;
  }
  const mid = Math.floor(numeric.length / 2);
  if (numeric.length % 2 === 1) {
    return numeric[mid];
  }
  return 0.5 * (numeric[mid - 1] + numeric[mid]);
}

function comparisonKeyCandidates(key) {
  return Array.isArray(key) ? key : [key];
}

function comparisonNumericList(summary, key, seedField) {
  for (const candidateKey of comparisonKeyCandidates(key)) {
    const directValues = summary?.[candidateKey];
    if (Array.isArray(directValues) && directValues.length) {
      return directValues
        .map((value) => Number(value))
        .filter((value) => Number.isFinite(value));
    }
  }
  const rows = summary?.seed_results;
  if (Array.isArray(rows) && rows.length) {
    for (const candidateField of comparisonKeyCandidates(seedField)) {
      const values = rows
        .map((row) => Number(row?.[candidateField]))
        .filter((value) => Number.isFinite(value));
      if (values.length) {
        return values;
      }
    }
  }
  return [];
}

function comparisonSolveOrCap(summary, solveKey, totalKey, seedSolveField, seedTotalField) {
  const solvedValue = comparisonSolvedValue(comparisonNumericList(summary, solveKey, seedSolveField));
  if (solvedValue != null) {
    return {value: solvedValue, label: solvedValue >= 1000 ? formatCompactNumber(solvedValue) : solvedValue.toFixed(0)};
  }
  const totalValues = comparisonNumericList(summary, totalKey, seedTotalField)
    .filter((value) => value > 0);
  if (totalValues.length) {
    return {
      value: null,
      label: "not solved",
    };
  }
  return {value: null, label: "not solved"};
}

function comparisonSolveOrCapSeries(summary, solveKey, totalKey, seedSolveField, seedTotalField) {
  const solveValues = comparisonNumericList(summary, solveKey, seedSolveField);
  const rows = [];
  for (let idx = 0; idx < solveValues.length; idx += 1) {
    const solvedValue = Number(solveValues[idx]);
    if (Number.isFinite(solvedValue) && solvedValue >= 0) {
      rows.push({
        value: solvedValue,
        label: solvedValue >= 1000 ? formatCompactNumber(solvedValue) : solvedValue.toFixed(0),
        seedIndex: idx,
      });
    }
  }
  return rows.length ? rows : [{value: null, label: "not solved"}];
}

function comparisonFirstPeakStepFromRows(rows, stepOffset = 0) {
  const points = (rows || [])
    .map((row) => ({
      ret: Number(row.return),
      steps: Number(row.total_env_steps),
    }))
    .filter((row) => Number.isFinite(row.ret) && Number.isFinite(row.steps) && row.steps >= 0);
  if (!points.length) {
    return null;
  }
  const peakReturn = Math.max(...points.map((row) => row.ret));
  const peakRow = points.find((row) => row.ret === peakReturn);
  if (!peakRow) {
    return null;
  }
  return Math.max(0, Number(peakRow.steps) + Number(stepOffset || 0));
}

function comparisonPeakStepValue(summary, histories, directKey, seedField, rowsKey, stepOffset = 0) {
  const directValue = comparisonMedianPositiveValue(comparisonNumericList(summary, directKey, seedField));
  const value = directValue ?? comparisonFirstPeakStepFromRows(histories?.[rowsKey] || [], stepOffset);
  if (value == null) {
    return {value: null, label: "pending"};
  }
  return {
    value,
    label: value >= 1000 ? formatCompactNumber(value) : value.toFixed(0),
  };
}

function comparisonPeakStepSeries(summary, histories, directKey, seedField, rowsKey, stepOffset = 0) {
  const directValues = comparisonNumericList(summary, directKey, seedField)
    .filter((value) => value >= 0);
  const values = directValues.length
    ? directValues
    : [comparisonFirstPeakStepFromRows(histories?.[rowsKey] || [], stepOffset)].filter((value) => value != null);
  if (!values.length) {
    return [{value: null, label: "pending"}];
  }
  return values.map((value, idx) => ({
    value,
    label: value >= 1000 ? formatCompactNumber(value) : value.toFixed(0),
    seedIndex: idx,
  }));
}

function comparisonBestReturnValue(summary, histories, directKey, seedField, rowsKey) {
  const directValue = comparisonMedianFiniteValue(comparisonNumericList(summary, directKey, seedField));
  if (directValue != null) {
    return directValue;
  }
  return comparisonMaxValue(histories?.[rowsKey] || [], "return");
}

function comparisonBestReturnSeries(summary, histories, directKey, seedField, rowsKey) {
  const directValues = comparisonNumericList(summary, directKey, seedField);
  const values = directValues.length
    ? directValues
    : [comparisonMaxValue(histories?.[rowsKey] || [], "return")].filter((value) => value != null);
  if (!values.length) {
    return [{value: null, label: "pending"}];
  }
  return values.map((value, idx) => ({
    value,
    label: value >= 1000 ? formatCompactNumber(value) : value.toFixed(0),
    seedIndex: idx,
  }));
}

function comparisonMaxValue(rows, field) {
  const values = (rows || [])
    .map((row) => Number(row[field]))
    .filter((value) => Number.isFinite(value));
  return values.length ? Math.max(...values) : null;
}

function comparisonEnvShortName(run) {
  const label = run?.env_display_name || run?.env_name || "Env";
  if (label.includes("CartPole")) {
    return "CartPole";
  }
  if (label.includes("Lunar")) {
    return "Lunar";
  }
  if (label.includes("Bipedal")) {
    return "Biped";
  }
  return label.replace("Continuous ", "");
}

function comparisonEncoderOffset(summary) {
  return comparisonMedianPositiveValue(
    comparisonNumericList(summary || {}, "encoder_probe_steps", "encoder_probe_steps")
  ) || 0;
}

function comparisonCurvePoints(rows, xOffset = 0) {
  return (rows || [])
    .map((row, idx) => {
      const rawX = Number(row.total_env_steps);
      return {
        x: (Number.isFinite(rawX) ? rawX : Number(row.episode ?? idx + 1)) + Number(xOffset || 0),
        episode: Number(row.episode ?? idx + 1),
        y: Number(row.avg10),
      };
    })
    .filter((row) => Number.isFinite(row.x) && Number.isFinite(row.y));
}

function niceTicks(minValue, maxValue, count = 5) {
  const min = Number(minValue);
  const max = Number(maxValue);
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [0, 1];
  }
  const span = Math.max(max - min, 1e-6);
  const rawStep = span / Math.max(count - 1, 1);
  const power = Math.pow(10, Math.floor(Math.log10(rawStep)));
  const fraction = rawStep / power;
  const niceFraction = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  const step = niceFraction * power;
  const niceMin = Math.floor(min / step) * step;
  const niceMax = Math.ceil(max / step) * step;
  const ticks = [];
  for (let value = niceMin; value <= niceMax + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(10)));
    if (ticks.length > 8) {
      break;
    }
  }
  return ticks.length >= 2 ? ticks : [niceMin, niceMax];
}

function comparisonFormatAxisValue(value, decimals = 0) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return "n/a";
  }
  if (Math.abs(numeric) >= 1000) {
    return formatCompactNumber(numeric);
  }
  return numeric.toFixed(decimals);
}

function comparisonLineChartSvg(baselineRows, probeRows, summary = {}) {
  const baseline = comparisonCurvePoints(baselineRows, 0);
  const probe = comparisonCurvePoints(probeRows, comparisonEncoderOffset(summary));
  const all = [...baseline, ...probe];
  if (all.length < 2) {
    return `<div class="empty">Waiting for baseline and probe episode summaries.</div>`;
  }
  const width = 520;
  const height = 210;
  const padLeft = 48;
  const padRight = 18;
  const padTop = 34;
  const padBottom = 42;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const minX = 0;
  const maxX = Math.max(...all.map((row) => row.x));
  const minYRaw = Math.min(...all.map((row) => row.y));
  const maxYRaw = Math.max(...all.map((row) => row.y));
  const yTicks = niceTicks(Math.min(0, minYRaw), maxYRaw, 5);
  const xTicks = niceTicks(minX, maxX, 4);
  const minY = Math.min(...yTicks);
  const maxY = Math.max(...yTicks);
  const xScale = (value) => padLeft + ((value - minX) / Math.max(maxX - minX, 1e-6)) * plotWidth;
  const yScale = (value) => padTop + (1 - ((value - minY) / Math.max(maxY - minY, 1e-6))) * plotHeight;
  const pathFor = (rows) => rows
    .map((row, idx) => `${idx === 0 ? "M" : "L"} ${xScale(row.x).toFixed(2)} ${yScale(row.y).toFixed(2)}`)
    .join(" ");
  const yGrid = yTicks.map((tick) => `
    <line x1="${padLeft}" y1="${yScale(tick).toFixed(2)}" x2="${width - padRight}" y2="${yScale(tick).toFixed(2)}" stroke="#edf1f6" stroke-width="1"></line>
    <text x="${padLeft - 8}" y="${(yScale(tick) + 4).toFixed(2)}" text-anchor="end" fill="#6e7685" font-size="10">${comparisonFormatAxisValue(tick, 0)}</text>
  `).join("");
  const xGrid = xTicks.map((tick) => `
    <text x="${xScale(tick).toFixed(2)}" y="${height - 20}" text-anchor="middle" fill="#6e7685" font-size="10">${comparisonFormatAxisValue(tick, 0)}</text>
  `).join("");
  const baselinePath = pathFor(baseline);
  const probePath = pathFor(probe);
  const baselineLast = baseline[baseline.length - 1];
  const probeLast = probe[probe.length - 1];
  return `
    <svg class="comparison-chart" viewBox="0 0 ${width} ${height}" role="img" aria-label="Standard PPO versus probe-conditioned PPO">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      ${yGrid}
      ${xGrid}
      <line x1="${padLeft}" y1="${padTop + plotHeight}" x2="${width - padRight}" y2="${padTop + plotHeight}" stroke="#171b22" stroke-width="1.2"></line>
      <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotHeight}" stroke="#171b22" stroke-width="1.2"></line>
      <text x="${padLeft}" y="${padTop - 13}" fill="#6e7685" font-size="12">avg10 return vs env steps</text>
      <text x="${width - 72}" y="${height - 7}" fill="#6e7685" font-size="11">env steps</text>
      ${baselinePath ? `<path d="${baselinePath}" fill="none" stroke="#3d8d5e" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
      ${probePath ? `<path d="${probePath}" fill="none" stroke="#2854d8" stroke-width="2.8" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
      ${baselineLast ? `<circle cx="${xScale(baselineLast.x).toFixed(2)}" cy="${yScale(baselineLast.y).toFixed(2)}" r="4.2" fill="#3d8d5e" stroke="#ffffff" stroke-width="1.5"></circle>` : ""}
      ${probeLast ? `<circle cx="${xScale(probeLast.x).toFixed(2)}" cy="${yScale(probeLast.y).toFixed(2)}" r="4.2" fill="#2854d8" stroke="#ffffff" stroke-width="1.5"></circle>` : ""}
      <text x="${padLeft}" y="${height - 7}" fill="#3d8d5e" font-size="11">standard PPO</text>
      <text x="${padLeft + 98}" y="${height - 7}" fill="#2854d8" font-size="11">probe-conditioned</text>
    </svg>
  `;
}

function comparisonBarsSvg(runs, config) {
  const width = 760;
  const height = 330;
  const padLeft = 66;
  const padRight = 28;
  const padTop = 34;
  const padBottom = 64;
  const plotWidth = width - padLeft - padRight;
  const plotHeight = height - padTop - padBottom;
  const groups = runs.map((run) => {
    const values = config.values(run);
    const normalizeOne = (item, idx = 0) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        return {
          value: Number.isFinite(Number(item.value)) ? Number(item.value) : null,
          label: item.label || null,
          capped: Boolean(item.capped),
          seedIndex: Number.isFinite(Number(item.seedIndex)) ? Number(item.seedIndex) : idx,
        };
      }
      return {
        value: Number.isFinite(Number(item)) ? Number(item) : null,
        label: null,
        capped: false,
        seedIndex: idx,
      };
    };
    const normalizeSeries = (items) => {
      const source = Array.isArray(items) ? items : [items];
      return source.map((item, idx) => normalizeOne(item, idx));
    };
    return {
      run,
      label: comparisonEnvShortName(run),
      baseline: normalizeSeries(values.baseline),
      probe: normalizeSeries(values.probe),
    };
  });
  const numericValues = groups
    .flatMap((group) => [...group.baseline, ...group.probe].map((item) => item.value))
    .filter((value) => Number.isFinite(value) && (config.allowNegative || value >= 0));
  if (!numericValues.length) {
    return `<div class="empty">Waiting for finished comparison runs.</div>`;
  }
  const minRaw = config.includeZero === false ? Math.min(...numericValues) : Math.min(0, ...numericValues);
  const maxRaw = config.includeZero === false ? Math.max(...numericValues) : Math.max(0, ...numericValues);
  const yTicks = niceTicks(minRaw, maxRaw, 5);
  const minY = Math.min(...yTicks);
  const maxY = Math.max(...yTicks);
  const yScale = (value) => {
    if (!Number.isFinite(value) || (!config.allowNegative && value < 0)) {
      return null;
    }
    return padTop + (1 - ((value - minY) / Math.max(maxY - minY, 1e-6))) * plotHeight;
  };
  const xStep = plotWidth / Math.max(groups.length, 1);
  const groupWidth = Math.min(150, xStep * 0.72);
  const maxBarsPerEnv = Math.max(
    2,
    ...groups.map((group) => group.baseline.length + group.probe.length),
  );
  const barGap = Math.max(2, groupWidth * 0.025);
  const barWidth = Math.max(4, Math.min(12, (groupWidth - barGap * (maxBarsPerEnv - 1)) / maxBarsPerEnv));
  const tickLabel = (tick) => {
    return comparisonFormatAxisValue(tick, config.valueDecimals ?? 0);
  };
  const barFor = (item, x, color, label, showLabel) => {
    const value = item.value;
    const y = yScale(value);
    if (y == null) {
      return `
        <text x="${x}" y="${padTop + plotHeight * 0.46}" text-anchor="middle" fill="#9aa3af" font-size="11">${item.label || "pending"}</text>
      `;
    }
    const zeroY = yScale(0) ?? (padTop + plotHeight);
    const barTop = Math.min(y, zeroY);
    const barHeight = Math.max(Math.abs(zeroY - y), 1);
    const labelY = value >= 0 ? barTop - 7 : y + 13;
    const valueLabel = item.label || (value >= 1000 ? formatCompactNumber(value) : value.toFixed(config.valueDecimals ?? 0));
    const opacity = item.capped ? "0.34" : "1";
    const stroke = item.capped ? `stroke="${color}" stroke-width="1.8" stroke-dasharray="4 3"` : "";
    return `
      <rect x="${(x - barWidth / 2).toFixed(2)}" y="${barTop.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" fill="${color}" fill-opacity="${opacity}" ${stroke}></rect>
      ${showLabel ? `<text x="${x.toFixed(2)}" y="${labelY.toFixed(2)}" text-anchor="middle" fill="#394150" font-size="10">${valueLabel}</text>` : ""}
      <title>${label} seed ${Number(item.seedIndex) + 1}: ${valueLabel}</title>
    `;
  };
  const bars = groups.map((group, idx) => {
    const center = padLeft + xStep * idx + xStep / 2;
    const baselineStart = center - groupWidth / 2 + barWidth / 2;
    const probeStart = center + groupWidth / 2 - (group.probe.length * barWidth + Math.max(0, group.probe.length - 1) * barGap) + barWidth / 2;
    const showLabels = group.baseline.length + group.probe.length <= 2;
    return `
      ${group.baseline.map((item, seedIdx) => barFor(
        item,
        baselineStart + seedIdx * (barWidth + barGap),
        "#3d8d5e",
        `${group.label} standard PPO`,
        showLabels,
      )).join("")}
      ${group.probe.map((item, seedIdx) => barFor(
        item,
        probeStart + seedIdx * (barWidth + barGap),
        "#2854d8",
        `${group.label} probe-conditioned`,
        showLabels,
      )).join("")}
      <text x="${center.toFixed(2)}" y="${height - 31}" text-anchor="middle" fill="#394150" font-size="12">${group.label}</text>
    `;
  }).join("");
  const grid = yTicks.map((tick) => {
    const y = yScale(tick) ?? padTop + plotHeight;
    return `
      <line x1="${padLeft}" y1="${y.toFixed(2)}" x2="${width - padRight}" y2="${y.toFixed(2)}" stroke="#edf1f6" stroke-width="1"></line>
      <text x="${padLeft - 10}" y="${(y + 4).toFixed(2)}" text-anchor="end" fill="#6e7685" font-size="11">${tickLabel(tick)}</text>
    `;
  }).join("");
  return `
    <svg class="paper-figure-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${config.title}">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      ${grid}
      <line x1="${padLeft}" y1="${padTop + plotHeight}" x2="${width - padRight}" y2="${padTop + plotHeight}" stroke="#171b22" stroke-width="1.4"></line>
      <line x1="${padLeft}" y1="${padTop}" x2="${padLeft}" y2="${padTop + plotHeight}" stroke="#171b22" stroke-width="1.4"></line>
      <text x="${padLeft}" y="18" fill="#171b22" font-size="13" font-weight="700">${config.title}</text>
      <text x="${padLeft}" y="${height - 9}" fill="#3d8d5e" font-size="12">standard PPO</text>
      <text x="${padLeft + 104}" y="${height - 9}" fill="#2854d8" font-size="12">probe-conditioned</text>
      <text x="${width - 168}" y="${height - 9}" fill="#6e7685" font-size="11">thin bars are seeds</text>
      <text transform="translate(17 ${padTop + plotHeight / 2}) rotate(-90)" text-anchor="middle" fill="#6e7685" font-size="12">${config.yLabel}</text>
      ${bars}
    </svg>
  `;
}

function comparisonLearningCurvesSvg(runs) {
  const width = 960;
  const height = 420;
  const panelGap = 22;
  const padLeft = 42;
  const padRight = 16;
  const padTop = 38;
  const padBottom = 52;
  const panelWidth = (width - padLeft - padRight - panelGap * 2) / 3;
  const panelHeight = height - padTop - padBottom;
  const panels = runs.map((run, idx) => {
    const histories = run.histories || {};
    const summary = run.summary || {};
    const baseline = comparisonCurvePoints(histories.baseline_returns || [], 0);
    const probe = comparisonCurvePoints(histories.probe_returns || [], comparisonEncoderOffset(summary));
    const all = [...baseline, ...probe];
    const x0 = padLeft + idx * (panelWidth + panelGap);
    const y0 = padTop;
    if (all.length < 2) {
      return `
        <g>
          <rect x="${x0}" y="${y0}" width="${panelWidth}" height="${panelHeight}" fill="#ffffff" stroke="#edf1f6"></rect>
          <text x="${x0 + panelWidth / 2}" y="${y0 + panelHeight / 2}" text-anchor="middle" fill="#9aa3af" font-size="12">pending</text>
          <text x="${x0 + panelWidth / 2}" y="${height - 21}" text-anchor="middle" fill="#394150" font-size="12">${comparisonEnvShortName(run)}</text>
        </g>
      `;
    }
    const minX = 0;
    const maxX = Math.max(...all.map((row) => row.x));
    const minYRaw = Math.min(...all.map((row) => row.y));
    const maxYRaw = Math.max(...all.map((row) => row.y));
    const yTicks = niceTicks(Math.min(0, minYRaw), maxYRaw, 4);
    const xTicks = niceTicks(minX, maxX, 3);
    const minY = Math.min(...yTicks);
    const maxY = Math.max(...yTicks);
    const xScale = (value) => x0 + ((value - minX) / Math.max(maxX - minX, 1e-6)) * panelWidth;
    const yScale = (value) => y0 + (1 - ((value - minY) / Math.max(maxY - minY, 1e-6))) * panelHeight;
    const pathFor = (rows) => rows
      .map((row, rowIdx) => `${rowIdx === 0 ? "M" : "L"} ${xScale(row.x).toFixed(2)} ${yScale(row.y).toFixed(2)}`)
      .join(" ");
    const yGrid = yTicks.map((tick) => `
      <line x1="${x0}" y1="${yScale(tick).toFixed(2)}" x2="${x0 + panelWidth}" y2="${yScale(tick).toFixed(2)}" stroke="#edf1f6" stroke-width="1"></line>
      <text x="${x0 - 6}" y="${(yScale(tick) + 4).toFixed(2)}" text-anchor="end" fill="#6e7685" font-size="9">${comparisonFormatAxisValue(tick, 0)}</text>
    `).join("");
    const xGrid = xTicks.map((tick) => `
      <text x="${xScale(tick).toFixed(2)}" y="${height - 36}" text-anchor="middle" fill="#6e7685" font-size="9">${comparisonFormatAxisValue(tick, 0)}</text>
    `).join("");
    return `
      <g>
        <rect x="${x0}" y="${y0}" width="${panelWidth}" height="${panelHeight}" fill="#ffffff" stroke="#edf1f6"></rect>
        ${yGrid}
        ${xGrid}
        <line x1="${x0}" y1="${y0 + panelHeight}" x2="${x0 + panelWidth}" y2="${y0 + panelHeight}" stroke="#171b22" stroke-width="1.2"></line>
        <line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0 + panelHeight}" stroke="#171b22" stroke-width="1.2"></line>
        <text x="${x0}" y="${y0 - 10}" fill="#6e7685" font-size="11">${comparisonEnvShortName(run)}</text>
        ${baseline.length ? `<path d="${pathFor(baseline)}" fill="none" stroke="#3d8d5e" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
        ${probe.length ? `<path d="${pathFor(probe)}" fill="none" stroke="#2854d8" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"></path>` : ""}
        <text x="${x0 + panelWidth / 2}" y="${height - 21}" text-anchor="middle" fill="#394150" font-size="12">${comparisonEnvShortName(run)}</text>
      </g>
    `;
  }).join("");
  return `
    <svg class="paper-figure-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="Learning curves across environments">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      <text x="${padLeft}" y="19" fill="#171b22" font-size="13" font-weight="700">Avg10 learning curves by environment</text>
      <text x="${padLeft}" y="${height - 7}" fill="#3d8d5e" font-size="12">standard PPO</text>
      <text x="${padLeft + 104}" y="${height - 7}" fill="#2854d8" font-size="12">probe-conditioned</text>
      <text x="${width - 140}" y="${height - 7}" fill="#6e7685" font-size="12">environment steps</text>
      <text transform="translate(17 ${padTop + panelHeight / 2}) rotate(-90)" text-anchor="middle" fill="#6e7685" font-size="12">avg10 return</text>
      ${panels}
    </svg>
  `;
}

function comparisonHasRegularVsCrawlerData(run) {
  if (!run || typeof run !== "object") {
    return false;
  }
  const histories = run.histories || {};
  const summary = run.summary || {};
  const hasCurves =
    Array.isArray(histories.baseline_returns)
    && histories.baseline_returns.length
    && Array.isArray(histories.probe_returns)
    && histories.probe_returns.length;
  const hasSolveSummary =
    comparisonNumericList(summary, ["baseline_env_step_solves", "baseline_step_solves"], "baseline_solved_env_steps").length
    || comparisonNumericList(summary, ["probe_env_step_solves_with_encoder", "probe_step_solves"], "probe_solved_env_steps_with_encoder").length
    || comparisonNumericList(summary, "baseline_episode_solves", "baseline_solved_episode").length
    || comparisonNumericList(summary, "probe_episode_solves", "probe_solved_episode").length;
  return Boolean(hasCurves || hasSolveSummary);
}

function latestComparisonRuns(rawPayload) {
  const rows = [];
  const activeSuiteId = rawPayload?.comparison_suite_id || null;
  const keepArchive = dashboardState.comparisonArchiveMode !== "current";
  if (rawPayload?.available && comparisonHasRegularVsCrawlerData(rawPayload)) {
    rows.push(rawPayload);
  }
  for (const row of rawPayload?.history_runs || []) {
    const sameFreshSuite = activeSuiteId && row.comparison_suite_id === activeSuiteId;
    if (comparisonHasRegularVsCrawlerData(row) && (keepArchive || sameFreshSuite)) {
      rows.push(row);
    }
  }
  const latestByEnv = new Map();
  for (const row of rows) {
    const envKey = row.env_name || row.env_display_name || row.benchmark_tag || row.session_id;
    const previous = latestByEnv.get(envKey);
    const rowTime = Number(row.updated_at || row.finished_at || row.started_at || 0);
    const previousTime = Number(previous?.updated_at || previous?.finished_at || previous?.started_at || 0);
    if (!previous || rowTime >= previousTime) {
      latestByEnv.set(envKey, row);
    }
  }
  const preferredOrder = ["ContinuousCartPole-v0", "LunarLanderContinuous-v3", "BipedalWalker-v3"];
  return [...latestByEnv.values()].sort((a, b) => {
    const ai = preferredOrder.indexOf(a.env_name);
    const bi = preferredOrder.indexOf(b.env_name);
    if (ai !== -1 || bi !== -1) {
      return (ai === -1 ? 99 : ai) - (bi === -1 ? 99 : bi);
    }
    return String(a.env_display_name || a.env_name).localeCompare(String(b.env_display_name || b.env_name));
  });
}

function renderComparisonArchiveControls() {
  const keepButton = el("comparisonKeepArchive");
  const currentButton = el("comparisonCurrentSuite");
  if (!keepButton || !currentButton) {
    return;
  }
  keepButton.classList.toggle("is-active", dashboardState.comparisonArchiveMode !== "current");
  currentButton.classList.toggle("is-active", dashboardState.comparisonArchiveMode === "current");
}

function paperFigureRuns(rawPayload) {
  const latestByEnvName = new Map();
  for (const run of latestComparisonRuns(rawPayload)) {
    if (run.env_name) {
      latestByEnvName.set(run.env_name, run);
    }
  }
  return [
    ["ContinuousCartPole-v0", "Continuous CartPole"],
    ["LunarLanderContinuous-v3", "Continuous LunarLander"],
    ["BipedalWalker-v3", "Bipedal Walker"],
  ].map(([envName, displayName]) => latestByEnvName.get(envName) || {
    env_name: envName,
    env_display_name: displayName,
    benchmark_tag: `${envName}_comparison`,
    histories: {},
    summary: {},
  });
}

function renderPaperFigureBoard(rawPayload) {
  const board = el("paperFigureBoard");
  if (!board) {
    return;
  }
  renderComparisonArchiveControls();
  const runs = paperFigureRuns(rawPayload);
  const hasAnyRun = runs.some((run) => Object.keys(run.histories || {}).length || Object.keys(run.summary || {}).length);
  if (!hasAnyRun) {
    board.innerHTML = `<div class="empty">Run main.py to populate paper-style PPO comparison figures.</div>`;
    return;
  }
  board.innerHTML = `
    <div class="paper-figure-card">
      <div class="paper-figure-title">A. Sample Efficiency</div>
	          <div class="paper-figure-caption">Environment steps until the rolling 10-episode average reaches the Gym-style solved threshold. Each thin bar is one solved seed; unsolved seeds are omitted instead of capped. Lower is better.</div>
		          ${comparisonBarsSvg(runs, {
		            title: "Steps to solve",
		            yLabel: "environment steps",
		            values: (run) => {
	              const summary = run.summary || {};
	              return {
	                baseline: comparisonSolveOrCapSeries(
	                  summary,
	                  ["baseline_env_step_solves", "baseline_step_solves"],
	                  "baseline_total_env_steps",
	                  ["baseline_solved_env_steps", "baseline_step_solves"],
	                  "baseline_total_env_steps",
	                ),
	                probe: comparisonSolveOrCapSeries(
	                  summary,
	                  ["probe_env_step_solves_with_encoder", "probe_step_solves"],
	                  ["probe_total_env_steps_with_encoder", "probe_total_env_steps"],
	                  ["probe_solved_env_steps_with_encoder", "probe_step_solves"],
	                  ["probe_total_env_steps_with_encoder", "probe_total_env_steps"],
            ),
          };
        },
      })}
    </div>
    <div class="paper-figure-card">
      <div class="paper-figure-title">B. Performance Ceiling</div>
	          <div class="paper-figure-caption">Best single-episode return observed during each seed. Higher is better; multiple thin bars show whether one run is carrying the result.</div>
	          ${comparisonBarsSvg(runs, {
	            title: "Peak episode return",
	            yLabel: "return",
	            valueDecimals: 0,
	            allowNegative: true,
        values: (run) => {
	              const summary = run.summary || {};
	              const histories = run.histories || {};
	              return {
	                baseline: comparisonBestReturnSeries(
	                  summary,
	                  histories,
	                  "baseline_best_returns",
	                  "baseline_best_return",
	                  "baseline_returns",
	                ),
	                probe: comparisonBestReturnSeries(
	                  summary,
	                  histories,
	                  "probe_best_returns",
	                  "probe_best_return",
	                  "probe_returns",
            ),
          };
        },
      })}
    </div>
    <div class="paper-figure-card">
      <div class="paper-figure-title">C. Steps To Peak</div>
	          <div class="paper-figure-caption">Environment steps until each seed first reaches its own best observed episode return. Probe bars include encoder probe collection when available. Lower is faster.</div>
		          ${comparisonBarsSvg(runs, {
		            title: "Steps to peak return",
		            yLabel: "environment steps",
		            values: (run) => {
	              const summary = run.summary || {};
          const histories = run.histories || {};
          const encoderOffset = comparisonMedianPositiveValue(
            comparisonNumericList(summary, "encoder_probe_steps", "encoder_probe_steps")
	              ) || 0;
	              return {
	                baseline: comparisonPeakStepSeries(
	                  summary,
	                  histories,
	                  "baseline_peak_env_steps",
	                  "baseline_best_env_steps",
	                  "baseline_returns",
	                  0,
	                ),
	                probe: comparisonPeakStepSeries(
	                  summary,
	                  histories,
	                  "probe_peak_env_steps_with_encoder",
	                  "probe_best_env_steps_with_encoder",
	                  "probe_returns",
              encoderOffset,
            ),
          };
        },
      })}
    </div>
    <div class="paper-figure-card paper-figure-wide">
      <div class="paper-figure-title">D. Learning Dynamics</div>
	          <div class="paper-figure-caption">Smoothed avg10 return curves are plotted against environment steps. Probe curves start after encoder collection so x-axis cost matches the sample-efficiency bars.</div>
      ${comparisonLearningCurvesSvg(runs)}
    </div>
  `;
}

function comparisonStat(label, value) {
  return `
    <div class="comparison-stat">
      <div class="comparison-k">${label}</div>
      <div class="comparison-v">${value}</div>
    </div>
  `;
}

function renderComparisonBoard(rawPayload) {
  const board = el("comparisonBoard");
  if (!board) {
    return;
  }
  renderComparisonArchiveControls();
  const runs = latestComparisonRuns(rawPayload);
  if (!runs.length) {
    board.innerHTML = `<div class="empty">Run main.py to start the CartPole, LunarLander, and Biped PPO comparison suite.</div>`;
    return;
  }
  board.innerHTML = runs.map((run) => {
    const histories = run.histories || {};
    const summary = run.summary || {};
    const baselineRows = histories.baseline_returns || [];
    const probeRows = histories.probe_returns || [];
    const active = Boolean(run.active || (rawPayload?.active && run.session_id === rawPayload.session_id));
    return `
      <div class="comparison-card">
        <div class="comparison-head">
          <div class="comparison-title">${run.env_display_name || run.env_name || "Environment"}</div>
          <div class="comparison-status">${active ? "live" : "archived"}</div>
        </div>
        ${comparisonLineChartSvg(baselineRows, probeRows, summary)}
        <div class="comparison-stats">
          ${comparisonStat("PPO Solve Ep", comparisonSolveLabel(summary.baseline_episode_solves))}
          ${comparisonStat("Probe Solve Ep", comparisonSolveLabel(summary.probe_episode_solves))}
          ${comparisonStat("PPO Solve Steps", comparisonSolveLabel(summary.baseline_env_step_solves || summary.baseline_step_solves))}
          ${comparisonStat("Probe Solve Steps", comparisonSolveLabel(summary.probe_env_step_solves_with_encoder || summary.probe_step_solves))}
          ${comparisonStat("PPO Avg10", comparisonLatestAvg10(baselineRows))}
          ${comparisonStat("Probe Avg10", comparisonLatestAvg10(probeRows))}
        </div>
        <div class="live-mini-note">
          ${run.benchmark_tag || "comparison"} · updated ${formatDateTime(run.updated_at || run.finished_at || run.started_at)}
        </div>
      </div>
    `;
  }).join("");
}

function snapshotFromPayload(payload) {
  const cartpole = payload?.cartpole || {};
  const focus = payload?.focus || {};
  return {
    phase: cartpole.phase || focus.phase || "idle",
    x: Number(cartpole.x) || 0,
    x_dot: Number(cartpole.x_dot) || 0,
    theta: Number(cartpole.theta) || 0,
    theta_dot: Number(cartpole.theta_dot) || 0,
    action_value: Number(cartpole.action_value) || 0,
    reward: Number.isFinite(Number(cartpole.reward)) ? Number(cartpole.reward) : null,
    focus_label: cartpole.focus_label || focus.focus_label || null,
    step_idx: Number.isFinite(Number(cartpole.step_idx)) ? Number(cartpole.step_idx) : null,
    episode_id: Number.isFinite(Number(cartpole.episode_id)) ? Number(cartpole.episode_id) : null,
    env_instance_id: Number.isFinite(Number(cartpole.env_instance_id)) ? Number(cartpole.env_instance_id) : null,
    env_params: cartpole.env_params || null,
  };
}

function updateLiveAnimation(payload) {
  const snapshot = snapshotFromPayload(payload);
  liveAnimationState.target = snapshot;
  if (!liveAnimationState.current) {
    liveAnimationState.current = {...snapshot};
  }
}

function ensureLiveCanvasSize(canvas) {
  if (!canvas) {
    return null;
  }
  const rect = canvas.getBoundingClientRect();
  if (rect.width <= 0 || rect.height <= 0) {
    return null;
  }
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(rect.width * ratio);
  const height = Math.round(rect.height * ratio);
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  return {ctx, width: rect.width, height: rect.height};
}

function drawLiveTheater() {
  const canvas = el("crawlerCanvas");
  const sizing = ensureLiveCanvasSize(canvas);
  if (!sizing) {
    return;
  }

  const {ctx, width, height} = sizing;
  const payload = selectedLiveTracePayload();
  const stage = payload?.stage || {};
  const snapshot = liveAnimationState.target || snapshotFromPayload(payload);

  ctx.clearRect(0, 0, width, height);

  const baseGradient = ctx.createLinearGradient(0, 0, 0, height);
  baseGradient.addColorStop(0, "#ffffff");
  baseGradient.addColorStop(1, "#f5f8fe");
  ctx.fillStyle = baseGradient;
  ctx.fillRect(0, 0, width, height);

  const spotlight = ctx.createRadialGradient(width * 0.22, height * 0.18, 10, width * 0.22, height * 0.18, width * 0.62);
  spotlight.addColorStop(0, "rgba(40, 84, 216, 0.10)");
  spotlight.addColorStop(1, "rgba(40, 84, 216, 0)");
  ctx.fillStyle = spotlight;
  ctx.fillRect(0, 0, width, height);

  const orbit = ctx.createRadialGradient(width * 0.82, height * 0.82, 10, width * 0.82, height * 0.82, width * 0.48);
  orbit.addColorStop(0, "rgba(61, 141, 94, 0.08)");
  orbit.addColorStop(1, "rgba(61, 141, 94, 0)");
  ctx.fillStyle = orbit;
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#edf1f6";
  ctx.lineWidth = 1;
  for (let idx = 1; idx <= 4; idx += 1) {
    const y = 72 + idx * ((height - 136) / 5);
    ctx.beginPath();
    ctx.moveTo(24, y);
    ctx.lineTo(width - 24, y);
    ctx.stroke();
  }

  if (!payload?.available || payload?.env_name !== "ContinuousCartPole-v0") {
    ctx.fillStyle = "#6e7685";
    ctx.font = '600 14px "Avenir Next", "Segoe UI", sans-serif';
    ctx.fillText("Awaiting a live CartPole run.", 34, 52);
    ctx.font = '700 44px "Avenir Next", "Segoe UI", sans-serif';
    ctx.fillStyle = "#171b22";
    ctx.fillText("Crawler Theater", 34, 112);
    ctx.font = '500 18px "Avenir Next", "Segoe UI", sans-serif';
    ctx.fillStyle = "#4a5260";
    ctx.fillText("Start training and this view will begin following the scientist in real time.", 34, 148);
    return;
  }

  if (liveAnimationState.current && liveAnimationState.target) {
    const current = liveAnimationState.current;
    const target = liveAnimationState.target;
    const smooth = 0.12;
    current.x += (target.x - current.x) * smooth;
    current.x_dot += (target.x_dot - current.x_dot) * smooth;
    current.theta += (target.theta - current.theta) * smooth;
    current.theta_dot += (target.theta_dot - current.theta_dot) * smooth;
    current.action_value += (target.action_value - current.action_value) * smooth;
    current.reward = target.reward;
    current.phase = target.phase;
    current.focus_label = target.focus_label;
    current.step_idx = target.step_idx;
    current.episode_id = target.episode_id;
    current.env_instance_id = target.env_instance_id;
    current.env_params = target.env_params;
  }

  const animated = liveAnimationState.current || snapshot;
  const phase = animated?.phase || stage.id || "idle";
  const accent = phaseColor(phase);
  const groundY = height * 0.74;
  const trackLeft = 36;
  const trackRight = width - 36;
  const trackWidth = trackRight - trackLeft;
  const history = payload?.cartpole_history || [];
  const allX = history.map((entry) => Number(entry.x)).filter((value) => Number.isFinite(value));
  allX.push(Number(animated?.x) || 0);
  const xRange = Math.max(1.9, ...allX.map((value) => Math.abs(value)), 0);
  const cartX = width / 2 + ((Number(animated?.x) || 0) / xRange) * (trackWidth * 0.42);
  const cartY = groundY - 30;
  const cartWidth = 88;
  const cartHeight = 34;
  const poleLength = 154;
  const poleTheta = Number(animated?.theta) || 0;
  const poleBaseX = cartX;
  const poleBaseY = cartY - cartHeight * 0.5 + 6;
  const poleTipX = poleBaseX + Math.sin(poleTheta) * poleLength;
  const poleTipY = poleBaseY - Math.cos(poleTheta) * poleLength;
  const clawTargetX = cartX;
  const clawTargetY = cartY - 112;
  const archiveMode = Boolean(payload?.finished && dashboardState.liveArchiveSelection !== "live");

  [
    [72, 18],
    [148, 2],
    [width - 148, 2],
    [width - 72, 18],
  ].forEach(([anchorX, anchorY], idx) => {
    const elbowX = anchorX + (clawTargetX - anchorX) * (idx < 2 ? 0.54 : 0.46);
    const elbowY = anchorY + (clawTargetY - anchorY) * 0.42;
    ctx.strokeStyle = idx < 2 ? "rgba(40, 84, 216, 0.20)" : "rgba(185, 130, 31, 0.18)";
    ctx.lineWidth = 2.6;
    ctx.beginPath();
    ctx.moveTo(anchorX, anchorY);
    ctx.lineTo(elbowX, elbowY);
    ctx.lineTo(clawTargetX + (idx < 2 ? -16 : 16), clawTargetY + 8);
    ctx.stroke();
    ctx.fillStyle = "rgba(23, 27, 34, 0.10)";
    ctx.beginPath();
    ctx.arc(anchorX, anchorY, 5, 0, Math.PI * 2);
    ctx.arc(elbowX, elbowY, 4, 0, Math.PI * 2);
    ctx.fill();
  });

  ctx.strokeStyle = "rgba(23, 27, 34, 0.12)";
  ctx.lineWidth = 1.2;
  ctx.setLineDash([6, 5]);
  ctx.beginPath();
  ctx.arc(clawTargetX, clawTargetY, 36, Math.PI * 0.1, Math.PI * 0.9);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(clawTargetX - 26, clawTargetY + 12);
  ctx.lineTo(clawTargetX - 9, clawTargetY - 5);
  ctx.lineTo(clawTargetX - 1, clawTargetY + 19);
  ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(clawTargetX + 26, clawTargetY + 12);
  ctx.lineTo(clawTargetX + 9, clawTargetY - 5);
  ctx.lineTo(clawTargetX + 1, clawTargetY + 19);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.strokeStyle = "#d1d9e6";
  ctx.lineWidth = 3;
  ctx.beginPath();
  ctx.moveTo(trackLeft, groundY);
  ctx.lineTo(trackRight, groundY);
  ctx.stroke();

  if (history.length) {
    ctx.beginPath();
    history.forEach((entry, idx) => {
      const hx = width / 2 + (Number(entry.x) / xRange) * (trackWidth * 0.42);
      const hy = groundY - 62 - Number(entry.theta || 0) * 18;
      if (idx === 0) {
        ctx.moveTo(hx, hy);
      } else {
        ctx.lineTo(hx, hy);
      }
    });
    ctx.strokeStyle = "rgba(40, 84, 216, 0.22)";
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  ctx.fillStyle = `${accent}1a`;
  ctx.beginPath();
  ctx.arc(cartX, cartY - 18, 62, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "rgba(0, 0, 0, 0.08)";
  ctx.beginPath();
  ctx.ellipse(cartX, groundY + 10, 54, 10, 0, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = "#ffffff";
  ctx.strokeStyle = accent;
  ctx.lineWidth = 2;
  const cartLeft = cartX - cartWidth / 2;
  const cartTop = cartY - cartHeight / 2;
  if (typeof ctx.roundRect === "function") {
    ctx.beginPath();
    ctx.roundRect(cartLeft, cartTop, cartWidth, cartHeight, 12);
    ctx.fill();
    ctx.stroke();
  } else {
    ctx.fillRect(cartLeft, cartTop, cartWidth, cartHeight);
    ctx.strokeRect(cartLeft, cartTop, cartWidth, cartHeight);
  }

  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(cartX - 22, groundY + 2, 10, 0, Math.PI * 2);
  ctx.arc(cartX + 22, groundY + 2, 10, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = "#171b22";
  ctx.lineWidth = 8;
  ctx.beginPath();
  ctx.moveTo(poleBaseX, poleBaseY);
  ctx.lineTo(poleTipX, poleTipY);
  ctx.stroke();

  ctx.fillStyle = "#171b22";
  ctx.beginPath();
  ctx.arc(poleBaseX, poleBaseY, 7, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = accent;
  ctx.beginPath();
  ctx.arc(poleTipX, poleTipY, 11, 0, Math.PI * 2);
  ctx.fill();

  ctx.strokeStyle = archiveMode ? "rgba(61, 141, 94, 0.65)" : "rgba(40, 84, 216, 0.55)";
  ctx.lineWidth = 1.5;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.arc(poleTipX, poleTipY, 20, 0, Math.PI * 2);
  ctx.arc(poleTipX, poleTipY, 34, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(poleTipX - 42, poleTipY);
  ctx.lineTo(poleTipX + 42, poleTipY);
  ctx.moveTo(poleTipX, poleTipY - 42);
  ctx.lineTo(poleTipX, poleTipY + 42);
  ctx.stroke();

  const arrowStrength = Math.max(-1, Math.min(1, (Number(animated?.action_value) || 0) / 2));
  const arrowSpan = 72 * Math.abs(arrowStrength);
  if (arrowSpan > 2) {
    const direction = arrowStrength >= 0 ? 1 : -1;
    const startX = cartX;
    const endX = cartX + direction * arrowSpan;
    const arrowY = cartY + 74;
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(startX, arrowY);
    ctx.lineTo(endX, arrowY);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(endX, arrowY);
    ctx.lineTo(endX - direction * 12, arrowY - 8);
    ctx.lineTo(endX - direction * 12, arrowY + 8);
    ctx.closePath();
    ctx.fillStyle = accent;
    ctx.fill();
  }

  ctx.fillStyle = "#6e7685";
  ctx.font = '600 13px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillText(stage.title || "Live research", 34, 38);
  ctx.fillStyle = "#171b22";
  ctx.font = '700 36px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillText("CartPole Scientist", 34, 82);
  ctx.fillStyle = "#4a5260";
  ctx.font = '500 15px "Avenir Next", "Segoe UI", sans-serif';
  const detail = stage.detail || "Tracing the crawler as it builds a world hypothesis.";
  ctx.fillText(detail.slice(0, 112), 34, 112);

  ctx.fillStyle = accent;
  ctx.fillRect(34, 130, 126, 30);
  ctx.fillStyle = "#ffffff";
  ctx.font = '600 11px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillText(friendlyPhaseName(phase).toUpperCase().slice(0, 28), 44, 149);

  ctx.fillStyle = "#6e7685";
  ctx.font = '600 11px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillText("OPENCLAW SEARCH RIG", width - 174, 32);
  ctx.fillStyle = archiveMode ? "#3d8d5e" : accent;
  ctx.fillText(archiveMode ? "ARCHIVE INSPECT" : "LIVE INSPECT", width - 154, 50);

  ctx.fillStyle = "#171b22";
  ctx.font = '600 13px "Avenir Next", "Segoe UI", sans-serif';
  ctx.fillText(`focus: ${animated?.focus_label || "scanning"}`, 34, 186);
  ctx.fillText(`episode ${liveInteger(animated?.episode_id)} · step ${liveInteger(animated?.step_idx)} · seed ${liveInteger(animated?.env_instance_id)}`, 34, 208);
  ctx.fillText(`x ${liveScalar(animated?.x, 3)} · theta ${liveScalar(animated?.theta, 3)} · action ${liveScalar(animated?.action_value, 3)}`, 34, 230);

  const params = animated?.env_params || {};
  const paramEntries = Object.entries(params);
  if (paramEntries.length) {
    ctx.fillStyle = "#6e7685";
    ctx.font = '600 12px "Avenir Next", "Segoe UI", sans-serif';
    ctx.fillText("hidden mechanics snapshot", width - 248, 38);
    ctx.fillStyle = "#171b22";
    ctx.font = '500 13px "Avenir Next", "Segoe UI", sans-serif';
    paramEntries.slice(0, 4).forEach(([name, value], idx) => {
      ctx.fillText(`${name} ${liveScalar(value, 3)}`, width - 248, 66 + idx * 22);
    });
  }
}

function startLiveAnimationLoop() {
  if (liveAnimationState.started) {
    return;
  }
  liveAnimationState.started = true;
  const tick = () => {
    drawLiveTheater();
    window.requestAnimationFrame(tick);
  };
  window.requestAnimationFrame(tick);
}

function scatterShellSvg(points, xField, yField, xLabel, yLabel, colorField = null) {
  if (!points || !points.length) {
    return `<div class="empty">No diagnostic points are available yet.</div>`;
  }
  const valid = points.filter((point) => Number.isFinite(Number(point[xField])) && Number.isFinite(Number(point[yField])));
  if (!valid.length) {
    return `<div class="empty">No finite diagnostic points are available yet.</div>`;
  }

  const xs = valid.map((point) => Number(point[xField]));
  const ys = valid.map((point) => Number(point[yField]));
  const width = 520;
  const height = 320;
  const pad = 44;
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xScale = (value) => pad + ((value - minX) / Math.max(maxX - minX, 1e-6)) * (width - 2 * pad);
  const yScale = (value) => height - pad - ((value - minY) / Math.max(maxY - minY, 1e-6)) * (height - 2 * pad);
  const colored = colorField ? normalizeMetric(valid, colorField) : valid.map((point) => ({...point, __normalized_metric: 0.5}));
  const identitySegments = xField === "same_env_spread" && yField === "nearest_between_distance"
    ? (() => {
        const start = Math.max(minX, minY);
        const end = Math.min(maxX, maxY);
        if (end <= start) {
          return "";
        }
        return `<line x1="${xScale(start).toFixed(2)}" y1="${yScale(start).toFixed(2)}" x2="${xScale(end).toFixed(2)}" y2="${yScale(end).toFixed(2)}" stroke="#d3dae6" stroke-width="1.25" stroke-dasharray="5 4"></line>`;
      })()
    : "";
  const circles = colored.map((point) => `
    <circle cx="${xScale(Number(point[xField])).toFixed(2)}" cy="${yScale(Number(point[yField])).toFixed(2)}" r="4.4"
      fill="${colorForMetric(point, colorField || "uncertainty", {})}" fill-opacity="0.86" stroke="#ffffff" stroke-width="1">
      <title>${xLabel}=${Number(point[xField]).toFixed(3)} | ${yLabel}=${Number(point[yField]).toFixed(3)} | error=${Number(point.env_error ?? 0).toFixed(3)} | uncert=${Number(point.uncertainty ?? 0).toFixed(3)}</title>
    </circle>
  `).join("");

  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${xLabel} vs ${yLabel}">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      ${identitySegments}
      ${circles}
      <text x="${pad}" y="${height - 12}" fill="#6e7685" font-size="12" letter-spacing="0.08em">${xLabel}</text>
      <text x="14" y="${pad - 10}" fill="#6e7685" font-size="12" letter-spacing="0.08em">${yLabel}</text>
    </svg>
  `;
}

function histogramSvg(values, label, color = "#2854d8", bins = 10) {
  if (!values || !values.length) {
    return `<div class="empty">No diagnostic distribution is available yet.</div>`;
  }
  const valid = values.filter((value) => Number.isFinite(Number(value))).map(Number);
  if (!valid.length) {
    return `<div class="empty">No finite diagnostic distribution is available yet.</div>`;
  }
  const width = 520;
  const height = 240;
  const pad = 40;
  const min = Math.min(...valid);
  const max = Math.max(...valid);
  const span = Math.max(max - min, 1e-6);
  const binCount = Math.min(Math.max(6, bins), 20);
  const counts = Array.from({length: binCount}, () => 0);
  for (const value of valid) {
    const idx = Math.min(binCount - 1, Math.floor(((value - min) / span) * binCount));
    counts[idx] += 1;
  }
  const maxCount = Math.max(...counts, 1);
  const barWidth = (width - 2 * pad) / binCount;
  const bars = counts.map((count, idx) => {
    const barHeight = ((height - 2 * pad) * count) / maxCount;
    const x = pad + idx * barWidth + 2;
    const y = height - pad - barHeight;
    return `<rect x="${x.toFixed(2)}" y="${y.toFixed(2)}" width="${Math.max(barWidth - 4, 2).toFixed(2)}" height="${barHeight.toFixed(2)}" fill="${color}" fill-opacity="0.80"></rect>`;
  }).join("");
  return `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${label} distribution">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      ${bars}
      <text x="${pad}" y="${height - 10}" fill="#6e7685" font-size="12" letter-spacing="0.08em">${label}</text>
      <text x="${pad}" y="${pad - 10}" fill="#6e7685" font-size="12">min ${min.toFixed(3)} · max ${max.toFixed(3)} · std ${standardDeviation(valid).toFixed(3)}</text>
    </svg>
  `;
}

function matrixRow(label, value, note) {
  return `
    <div class="matrix-row">
      <div class="matrix-k">${label}</div>
      <div class="matrix-v">${value}</div>
      <div class="matrix-n">${note}</div>
    </div>
  `;
}

function readoutRow(label, value, note) {
  return `
    <div class="readout-row">
      <div class="readout-label">${label}</div>
      <div class="readout-value">${value}</div>
      <div class="readout-note">${note}</div>
    </div>
  `;
}

function summaryCell(label, value, note) {
  return `
    <div class="summary-cell">
      <div class="summary-label">${label}</div>
      <div class="summary-value">${value}</div>
      <div class="summary-note">${note}</div>
    </div>
  `;
}

function liveSummaryCell(label, value, note) {
  return `
    <div class="live-summary-cell">
      <div class="live-summary-label">${label}</div>
      <div class="live-summary-value">${value}</div>
      <div class="live-summary-note">${note}</div>
    </div>
  `;
}

function setActiveDeck(target, persist = true) {
  dashboardState.activeDeck = target;
  document.querySelectorAll(".deck-button").forEach((button) => {
    button.classList.toggle("is-active", button.dataset.deckTarget === target);
  });
  document.querySelectorAll(".deck-section").forEach((section) => {
    section.classList.toggle("is-active", section.id === `deck-${target}`);
  });
  if (persist) {
    window.localStorage.setItem("teenyreason.activeDeck", target);
  }
  drawLiveTheater();
}

function restoreActiveDeck() {
  const saved = window.localStorage.getItem("teenyreason.activeDeck");
  if (saved === "live" || saved === "latent" || saved === "benchmark" || saved === "comparison") {
    setActiveDeck(saved, false);
  } else {
    setActiveDeck("live", false);
  }
}

function selectedLiveTracePayload(rawPayload = dashboardState.livePayload) {
  if (!rawPayload || !rawPayload.available) {
    return rawPayload;
  }
  if (!dashboardState.liveArchiveSelection || dashboardState.liveArchiveSelection === "live") {
    return rawPayload;
  }
  const archived = (rawPayload.history_runs || []).find((row) => row.session_id === dashboardState.liveArchiveSelection);
  return archived || rawPayload;
}

function historySparklineSvg(points, color) {
  const validPoints = (points || []).filter((row) => Number.isFinite(Number(row.x)) && Number.isFinite(Number(row.theta)));
  if (validPoints.length < 2) {
    return `<div class="live-mini-note">Not enough motion history captured yet.</div>`;
  }
  const width = 280;
  const height = 74;
  const pad = 8;
  const xs = validPoints.map((row) => Number(row.x));
  const ys = validPoints.map((row) => Number(row.theta));
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const xScale = (value) => pad + ((value - minX) / Math.max(maxX - minX, 1e-6)) * (width - 2 * pad);
  const yScale = (value) => height - pad - ((value - minY) / Math.max(maxY - minY, 1e-6)) * (height - 2 * pad);
  const path = validPoints.map((row, idx) => `${idx === 0 ? "M" : "L"} ${xScale(Number(row.x)).toFixed(2)} ${yScale(Number(row.theta)).toFixed(2)}`).join(" ");
  return `
    <svg class="live-archive-mini" viewBox="0 0 ${width} ${height}" role="img" aria-label="Archived crawler motion">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#edf1f6" stroke-width="1"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#edf1f6" stroke-width="1"></line>
      <path d="${path}" fill="none" stroke="${color}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"></path>
    </svg>
  `;
}

function renderLive(payload) {
  if (payload) {
    dashboardState.livePayload = payload;
  }
  const rawPayload = dashboardState.livePayload;
  renderPaperFigureBoard(rawPayload);
  renderComparisonBoard(rawPayload);
  if (
    dashboardState.liveArchiveSelection !== "live" &&
    !(rawPayload?.history_runs || []).some((row) => row.session_id === dashboardState.liveArchiveSelection)
  ) {
    dashboardState.liveArchiveSelection = "live";
  }
  const inspectedPayload = selectedLiveTracePayload(rawPayload);
  updateLiveAnimation(inspectedPayload);

  const hasLivePayload = Boolean(rawPayload?.available);
  const isCartPole = inspectedPayload?.env_name === "ContinuousCartPole-v0";
  if (!hasLivePayload) {
    el("liveFocusWrap").innerHTML = `<div class="empty">Start a training run to stream live PPO curves into this deck.</div>`;
    el("liveFamilyWrap").innerHTML = `<div class="empty">No live probe family scores are available yet.</div>`;
    el("liveArchiveWrap").innerHTML = `<div class="empty">Archived traces will appear here after completed runs.</div>`;
    el("liveWindowsWrap").innerHTML = `<div class="empty">No support windows have been traced yet.</div>`;
    el("liveCurveWrap").innerHTML = `<div class="empty">No live curves are available yet.</div>`;
    el("liveEventsWrap").innerHTML = `<div class="empty">No live events have been recorded yet.</div>`;
    el("liveSummaryWrap").innerHTML = `<div class="empty">Run metadata will appear here once the trace begins.</div>`;
    return;
  }

  const run = inspectedPayload.run || {};
  const stage = inspectedPayload.stage || {};
  const focus = inspectedPayload.focus || {};
  const cartpole = inspectedPayload.cartpole || {};
  const summary = inspectedPayload.summary || {};
  const historyRuns = rawPayload.history_runs || [];
  const focusLabel = focus.focus_label || cartpole.focus_label || "scanning hidden mechanics";
  const envParams = Object.entries(cartpole.env_params || {})
    .map(([name, value]) => `${name} ${liveScalar(value, 3)}`)
    .join(" · ");
  const thetaDegrees = Number.isFinite(Number(cartpole.theta)) ? (Number(cartpole.theta) * 180 / Math.PI) : null;

  el("liveFocusWrap").innerHTML = `
    <div class="live-summary-grid">
      ${liveSummaryCell(
        "Stage",
        stage.title || "Live run",
        stage.detail || "The crawler is progressing through the training pipeline."
      )}
      ${liveSummaryCell(
        "Run",
        `seed ${liveInteger(run.seed)} · ${liveInteger(run.run_index)}/${liveInteger(run.total_runs)}`,
        `variant ${run.variant || "shared"} · env ${payload.env_display_name || currentEnvLabel()}`
      )}
      ${liveSummaryCell(
        "Focus",
        focusLabel,
        `phase ${friendlyPhaseName(focus.phase || cartpole.phase || stage.id)}`
      )}
      ${liveSummaryCell(
        isCartPole ? "Cart Motion" : "Live Theater",
        isCartPole
          ? `x ${liveScalar(cartpole.x, 3)} · dx ${liveScalar(cartpole.x_dot, 3)}`
          : `${payload.env_display_name || currentEnvLabel()} curves active`,
        isCartPole
          ? "Horizontal position and velocity from the current world interaction."
          : "The animated canvas is CartPole-only, but the live training curves and run summary below still update for this environment."
      )}
      ${liveSummaryCell(
        isCartPole ? "Pole Motion" : "Episode",
        isCartPole
          ? `theta ${thetaDegrees == null ? "n/a" : `${thetaDegrees.toFixed(1)}°`}`
          : `episode ${liveInteger(focus.episode)}`,
        isCartPole
          ? `dtheta ${liveScalar(cartpole.theta_dot, 3)} · raw ${liveScalar(cartpole.theta, 3)}`
          : `variant ${focus.variant || run.variant || "training"} · phase ${friendlyPhaseName(focus.phase || stage.id)}`
      )}
      ${liveSummaryCell(
        "Control",
        `action ${liveScalar(cartpole.action_value, 3)}`,
        `reward ${liveScalar(cartpole.reward, 3)} · step ${liveInteger(cartpole.step_idx)}`
      )}
      ${liveSummaryCell(
        "Env Expression",
        `conf ${liveScalar(focus.expression_confidence, 3)} · ready ${focus.expression_ready == null ? "n/a" : focus.expression_ready ? "yes" : "no"}`,
        `${focus.expression_muted_by_policy ? `muted ${focus.expression_mute_reason || "by policy"}` : `uncert ${liveScalar(focus.uncertainty, 3)} · expr-scale ${liveScalar(focus.message_scale, 3)}`} · surprise ${liveScalar(focus.surprise, 3)}`
      )}
      ${liveSummaryCell(
        "Episode",
        `return ${liveScalar(focus.episode_return, 2)}`,
        `avg10 ${liveScalar(focus.avg10, 2)} · probes ${liveInteger(focus.probe_count)}`
      )}
      ${liveSummaryCell(
        "Hidden Mechanics",
        envParams || "not yet decoded",
        "The current hidden-parameter readout attached to the live CartPole snapshot."
      )}
      ${liveSummaryCell(
        "Trace Pulse",
        inspectedPayload.finished ? "archived" : "live",
        `updated ${formatDateTime(inspectedPayload.updated_at || inspectedPayload.finished_at)}`
      )}
    </div>
  `;

  const familyScores = inspectedPayload.family_scores || [];
  if (!familyScores.length) {
    el("liveFamilyWrap").innerHTML = `<div class="empty">The probe planner has not emitted a family scoreboard yet.</div>`;
  } else {
    const maxMagnitude = Math.max(...familyScores.map((row) => Math.abs(Number(row.selection_score || 0))), 1e-6);
    el("liveFamilyWrap").innerHTML = familyScores.map((row) => {
      const score = Number(row.selection_score || 0);
      const width = (100 * Math.abs(score)) / maxMagnitude;
      const isFocus = row.family === focusLabel;
      const barStyle = isFocus
        ? "background: linear-gradient(90deg, rgba(40, 84, 216, 0.95), rgba(61, 141, 94, 0.40)); border-right-color: rgba(40, 84, 216, 0.75);"
        : score < 0
          ? "background: linear-gradient(90deg, rgba(184, 91, 108, 0.80), rgba(184, 91, 108, 0.24)); border-right-color: rgba(184, 91, 108, 0.65);"
          : "";
      return `
        <div class="live-bar-row">
          <div class="live-bar-head">
            <div class="live-bar-label">${row.family}</div>
            <div class="live-bar-value">score ${liveScalar(score, 2)} · value/step ${liveScalar(row.value_per_probe_step, 2)}</div>
          </div>
          <div class="live-bar"><span style="width:${width}%; ${barStyle}"></span></div>
          <div class="live-mini-note">
            predicted value ${liveScalar(row.predicted_marginal_value, 2)} ·
            entropy drop ${liveScalar(row.predicted_entropy_reduction, 2)} ·
            hypothesis separation ${liveScalar(row.predicted_hypothesis_separation, 2)} ·
            raw future ${liveScalar(row.raw_future_error_estimate, 2)} ·
            floored future ${liveScalar(row.future_error_estimate, 2)} ·
            choice future ${liveScalar(row.future_gain_for_choice, 2)} ·
            cost ${liveScalar(row.estimated_probe_cost, 2)} ·
            realized gain ${liveScalar(row.realized_gain, 2)}
          </div>
        </div>
      `;
    }).join("");
  }

  el("liveArchiveWrap").innerHTML = [
    `
      <div class="live-archive-card ${dashboardState.liveArchiveSelection === "live" ? "is-selected" : ""}" data-trace-selection="live">
        <div class="live-archive-head">
          <div class="live-archive-name">Current live trace</div>
          <div class="live-archive-meta">${rawPayload.active ? "active" : "latest snapshot"}</div>
        </div>
        ${historySparklineSvg(rawPayload.cartpole_history || [], "#2854d8")}
        <div class="live-archive-note">The stream you are currently watching. Click to jump back from any archived run.</div>
      </div>
    `,
    ...historyRuns.map((row, idx) => `
      <div class="live-archive-card ${dashboardState.liveArchiveSelection === row.session_id ? "is-selected" : ""}" data-trace-selection="${row.session_id}">
        <div class="live-archive-head">
          <div class="live-archive-name">Archive ${idx + 1} · ${row.benchmark_tag || "trace"}</div>
          <div class="live-archive-meta">${formatDateTime(row.finished_at || row.updated_at)}</div>
        </div>
        ${historySparklineSvg(row.cartpole_history || [], "#3d8d5e")}
        <div class="live-archive-note">
          ${(row.stage?.title || "Completed run")} · seed ${liveInteger(row.run?.seed)} ·
          ${Array.isArray(row.archive_solve_values) && row.archive_solve_values.length
            ? `${row.archive_solve_label || "solves"} ${row.archive_solve_values.join(", ")}`
            : "saved for debugging"}
        </div>
      </div>
    `),
  ].join("");

  const windows = inspectedPayload.recent_windows || [];
  el("liveWindowsWrap").innerHTML = windows.length ? windows.slice(0, 8).map((row) => `
    <div class="live-window-row">
      <div>
        <div class="live-window-label">${row.probe_mode}</div>
        <div class="live-window-detail">
          env ${liveInteger(row.env_instance_id)} · episode ${liveInteger(row.episode_id)} ·
          ${row.terminated ? "terminated" : row.truncated ? "truncated" : "active"}
        </div>
      </div>
      <div class="live-window-value">${liveScalar(row.reward_sum, 2)}</div>
    </div>
  `).join("") : `<div class="empty">No support windows have been recorded yet.</div>`;

  const encoderHistory = (inspectedPayload.histories?.encoder_epochs || []).map((row) => Number(row.total_loss));
  const baselineHistory = (inspectedPayload.histories?.baseline_returns || []).map((row) => Number(row.avg10));
  const probeHistory = (inspectedPayload.histories?.probe_returns || []).map((row) => Number(row.avg10));
  const uncertaintyHistory = (inspectedPayload.histories?.uncertainty || []).map(Number);
  const messageScaleHistory = (inspectedPayload.histories?.message_scale || []).map(Number);
  const curveBlocks = [];
  if (encoderHistory.length && String(stage.id || "").includes("encoder")) {
    curveBlocks.push(lineChartSvg(encoderHistory, "Encoder Total Loss", "#7f56d9"));
  }
  if (baselineHistory.length) {
    curveBlocks.push(lineChartSvg(baselineHistory, "Baseline Avg10 Return", "#3d8d5e"));
  }
  if (probeHistory.length) {
    curveBlocks.push(lineChartSvg(probeHistory, "Probe Avg10 Return", "#2854d8"));
  }
  if (uncertaintyHistory.length) {
    curveBlocks.push(lineChartSvg(uncertaintyHistory, "Uncertainty Trace", "#b9821f"));
  }
  if (messageScaleHistory.length) {
    curveBlocks.push(lineChartSvg(messageScaleHistory, "Expression Scale Trace", "#b85b6c"));
  }
  el("liveCurveWrap").innerHTML = curveBlocks.length ? curveBlocks.join("") : `<div class="empty">Live curves will appear once the trace accumulates enough steps.</div>`;

  const events = inspectedPayload.recent_events || [];
  el("liveEventsWrap").innerHTML = events.length ? events.slice(0, 10).map((row) => `
    <div class="live-event-row">
      <div class="live-event-kind">${row.kind}</div>
      <div>
        <div class="live-event-label">${row.label}</div>
        <div class="live-event-detail">${row.detail} · ${formatDateTime(row.timestamp)}</div>
      </div>
    </div>
  `).join("") : `<div class="empty">No live events have been recorded yet.</div>`;

  const summaryRows = [
    readoutRow(
      "Run State",
      inspectedPayload.finished ? "archived" : "active",
      `benchmark ${inspectedPayload.benchmark_tag || "n/a"} · env ${inspectedPayload.env_display_name || currentEnvLabel()}`
    ),
    readoutRow(
      "Updated",
      formatDateTime(inspectedPayload.updated_at || inspectedPayload.finished_at),
      "The dashboard polls this trace live, so this timestamp should keep moving during training."
    ),
  ];
  if (Array.isArray(summary.baseline_episode_solves) || Array.isArray(summary.probe_episode_solves)) {
    summaryRows.push(
      readoutRow(
        "Solve Episodes",
        `base [${(summary.baseline_episode_solves || []).join(", ")}]`,
        `probe [${(summary.probe_episode_solves || []).join(", ")}]`
      ),
      readoutRow(
        "Solve Env Steps",
        `base [${(summary.baseline_step_solves || []).join(", ")}]`,
        `probe [${(summary.probe_step_solves || []).join(", ")}]`
      ),
    );
  } else {
    summaryRows.push(
      readoutRow(
        "Run Index",
        `${liveInteger(run.run_index)} / ${liveInteger(run.total_runs)}`,
        `seed ${liveInteger(run.seed)}`
      )
    );
  }
  el("liveSummaryWrap").innerHTML = summaryRows.join("");
}

function setPreferredSelection(selectEl, preferredValue, fallbackPrefix) {
  const options = Array.from(selectEl.options).map((option) => option.value);
  if (!options.length) {
    return;
  }
  if (preferredValue && options.includes(preferredValue)) {
    selectEl.value = preferredValue;
    return;
  }
  if (fallbackPrefix) {
    const match = options.find((value) => value.startsWith(fallbackPrefix));
    if (match) {
      selectEl.value = match;
      return;
    }
  }
  selectEl.selectedIndex = 0;
}

function isPlannerControllerStyle(style) {
  return String(style || "").includes("belief_planner");
}

function fullSystemDisplayName(style, oracle = false) {
  if (isPlannerControllerStyle(style)) {
    return oracle ? "Planner Oracle" : "Belief-Planner";
  }
  return oracle ? "Belief Oracle" : "Belief Controller";
}

function fullSystemReferenceName(style, oracle = false) {
  if (isPlannerControllerStyle(style)) {
    return oracle ? "planner oracle" : "belief-planner";
  }
  return oracle ? "belief oracle" : "belief-controller";
}

function renderLatent(payload) {
  dashboardState.latentPayload = payload;
  renderHero();
  const summary = payload.summary;
  const derived = computeLatentDerived(payload);
  const systemId = payload.system_id || {};
  const supportValidity = payload.support_validity || {
    status: "ok",
    headline: "",
    detail: "",
    reasons: [],
    affected_metrics: [],
  };
  const affectedMetrics = (supportValidity.affected_metrics || []).join(", ");
  if (supportValidity.status === "ok") {
    el("latentValidityWrap").innerHTML = "";
  } else {
    const severityClass = supportValidity.status === "invalid" ? "is-invalid" : "is-fragile";
    const reasonItems = (supportValidity.reasons || []).map((reason) => `<li>${reason}</li>`).join("");
    el("latentValidityWrap").innerHTML = `
      <div class="status-banner ${severityClass}">
        <div class="status-banner-head">
          <div>
            <div class="status-banner-label">Snapshot Validity</div>
            <div class="status-banner-title">${supportValidity.headline}</div>
          </div>
          <div class="status-banner-meta">
            watch ${affectedMetrics || "split metrics"} before trusting the headline numbers
          </div>
        </div>
        <div class="status-banner-copy">${supportValidity.detail}</div>
        ${reasonItems ? `<ul class="status-banner-list">${reasonItems}</ul>` : ""}
      </div>
    `;
  }

  el("latentSummary").innerHTML = [
    matrixRow("Env Instances", formatInteger(summary.num_envs), "sampled hidden worlds currently represented in this artifact"),
    matrixRow("Total Windows", formatInteger(summary.num_windows), "probe windows pooled into those env-level beliefs"),
    matrixRow("Windows / Env", summary.window_count_mean.toFixed(1), "average number of probe windows contributing to one env belief"),
    matrixRow("Support / Env", summary.support_count_mean.toFixed(1), "average number of windows actually used to build each env belief"),
    matrixRow("Support Families", `${Number(summary.support_group_count_mean ?? 0).toFixed(1)} · ${(100 * (summary.support_group_ratio_mean ?? 1)).toFixed(1)}%`, "distinct probe families represented per env, followed by the distinct-family share of support windows"),
    matrixRow("Paired Split Overlap", `${(100 * (summary.split_group_overlap_mean ?? 0)).toFixed(1)}%`, "how much split-half family coverage is duplicated on both sides for the paired-repeat diagnostic"),
    matrixRow("Cross Split Overlap", `${(100 * (summary.cross_family_split_group_overlap_mean ?? 0)).toFixed(1)}%`, "how much family coverage overlaps in the deliberately stricter cross-family split"),
    matrixRow("Split Half Balance", `${(100 * (summary.split_balanced_half_fraction ?? 0)).toFixed(1)}%`, "share of env beliefs whose two disjoint split halves stayed balanced in size"),
    matrixRow("Env Param Mean Band", `${derived ? derived.envMeanMin.toFixed(3) : "n/a"} to ${derived ? derived.envMeanMax.toFixed(3) : "n/a"}`, "coarse span of hidden parameter settings covered by the aggregated env beliefs"),
    matrixRow("Uncertainty Mean", summary.uncertainty_mean.toFixed(3), "average disagreement across disjoint small support halves and their decoded mechanics"),
    matrixRow("Uncertainty Std", derived ? derived.uncertaintyStd.toFixed(3) : "n/a", "spread of env-level uncertainty across sampled worlds; near-zero usually means confidence collapse"),
    matrixRow("Posterior Std", derived ? derived.mechanicsPosteriorStdMean.toFixed(3) : "n/a", "mean per-factor mechanics posterior std; lower means the world hypothesis is becoming sharper"),
    matrixRow("Posterior Entropy", derived ? derived.mechanicsPosteriorEntropyMean.toFixed(3) : "n/a", "average entropy of the explicit mechanics posterior carried by the env belief"),
    ...(systemId.available ? [
      matrixRow("System-ID Trust", systemId.trusted ? "yes" : "no", "whether the particle likelihood model cleared the held-out identification gate"),
      matrixRow("SysID Top-1", `${(100 * Number(systemId.validation_top1 || 0)).toFixed(1)}%`, "held-out candidate mechanics retrieval accuracy for the likelihood model"),
      matrixRow("Particle ESS", Number(systemId.particle_ess_ratio_mean || 0).toFixed(3), "effective sample size ratio of the particle posterior; lower means evidence has concentrated on fewer worlds"),
      matrixRow("Particle Leaveout", Number(systemId.particle_leaveout_shift_mean || 0).toFixed(3), "mean posterior shift when one observed probe family is removed"),
    ] : []),
    matrixRow("Belief Norm Std", derived ? derived.beliefNormStd.toFixed(3) : "n/a", "spread of pre-normalization belief magnitudes across sampled worlds; near-zero usually means raw belief collapse"),
    matrixRow("Median Nearest-Between", derived ? derived.nearestBetweenMedian.toFixed(3) : "n/a", "typical distance to the nearest different-world belief; tiny values usually mean the belief cloud has collapsed"),
    matrixRow("Pairwise Between Mean", derived ? derived.pairwiseBetweenMean.toFixed(3) : "n/a", "mean distance across different-world env beliefs; low values usually mean global geometry is still compressed"),
    matrixRow("Unit Pairwise Mean", derived ? derived.pairwiseBetweenMeanUnit.toFixed(3) : "n/a", "mean distance after unit normalization; if raw and unit views disagree, normalization may be hiding useful belief scale"),
    matrixRow("PCA Explained", summary.pca_explained.map((value) => value.toFixed(2)).join(" / "), "variance captured by the first two PCA axes"),
  ].join("");

  el("latentReadout").innerHTML = derived ? [
    readoutRow(
      "Mechanics Fit",
      derived.linearEnvFitR2.toFixed(2),
      "Mean linear R² when regressing true randomized env parameters from the latent coordinates. Higher means the latent is carrying more directly decodable mechanics."
    ),
    readoutRow(
      "Neighbor Alignment",
      `${(100 * derived.neighborEnvAlignment).toFixed(1)}%`,
      "Compares env-parameter distance for nearest latent neighbors against random pairings. Higher is better: nearby latents are describing genuinely similar worlds."
    ),
    readoutRow(
      "Same-env Spread",
      derived.sameEnvSpread.mean.toFixed(3),
      "Average disagreement between env beliefs rebuilt from two disjoint small support halves of the same sampled world. Lower means a few probes already pin down a stable mechanics belief."
    ),
    readoutRow(
      "Gap Ratio",
      `${(100 * derived.sameEnvGapRatio.mean).toFixed(1)}%`,
      "Average same-world split disagreement as a share of the nearest different-world latent distance. Lower is better: same worlds should be much closer than different worlds."
    ),
    readoutRow(
      "Paired Retrieval",
      `${(100 * derived.splitRetrievalTop1).toFixed(1)}%`,
      "Top-1 retrieval accuracy when duplicate probe-family evidence can be split across the two halves. This should improve before strict cross-family transfer does."
    ),
    readoutRow(
      "Cross-Family Retrieval",
      `${(100 * derived.crossFamilySplitRetrievalTop1).toFixed(1)}%`,
      "Top-1 retrieval accuracy when the two halves are forced to use disjoint probe families. This is the stricter transfer check."
    ),
    readoutRow(
      "Paired MRR",
      derived.splitRetrievalMrr.toFixed(2),
      "Mean reciprocal rank for split-half retrieval. Higher is better and usually more stable than top-1 alone."
    ),
    readoutRow(
      "Cross-Family MRR",
      derived.crossFamilySplitRetrievalMrr.toFixed(2),
      "Mean reciprocal rank for the stricter cross-family split retrieval check."
    ),
    readoutRow(
      "Env Param Disagreement",
      derived.envParamUncertaintyMean.toFixed(3),
      "Average decoded-mechanics spread across disjoint support halves and leave-one-goal-out beliefs. This is now the main uncertainty source, not the collapsed posterior std head."
    ),
    readoutRow(
      "Held-out Probe Error",
      derived.futureProbeErrorMean.toFixed(3),
      "Average error when the env belief tries to predict held-out probe-summary evidence. Lower is better: the belief is forecasting what a new probe would reveal."
    ),
    readoutRow(
      "Posterior Std",
      derived.mechanicsPosteriorStdMean.toFixed(3),
      "Mean mechanics-posterior standard deviation. Lower means the crawler is ruling out world hypotheses instead of only spreading the latent cloud."
    ),
    readoutRow(
      "Posterior Entropy",
      derived.mechanicsPosteriorEntropyMean.toFixed(3),
      "Average entropy of the explicit mechanics posterior. This is the new posterior-style uncertainty source for probe planning."
    ),
    readoutRow(
      "Support Families",
      `${derived.supportGroupCountMean.toFixed(1)} · ${(100 * derived.supportGroupRatioMean).toFixed(1)}%`,
      "Distinct probe families represented per env. The percentage falls when each family has paired/repeated windows, so use it with the family count and split overlap."
    ),
    readoutRow(
      "Split Family Overlap",
      `${(100 * derived.splitGroupOverlapMean).toFixed(1)}%`,
      "How often the same probe families show up on both disjoint support halves. Low overlap is intentional for the stricter cross-family split; retrieval then has to match worlds from different probe evidence."
    ),
    readoutRow(
      "Split Half Balance",
      `${(100 * derived.splitBalancedHalfFraction).toFixed(1)}%`,
      `How often the two disjoint support halves stay balanced in size. Mean family counts are ${derived.splitGroupCountAMean.toFixed(2)} and ${derived.splitGroupCountBMean.toFixed(2)}.`
    ),
    readoutRow(
      "Probe Leakage",
      `${(100 * derived.envModeLeakage).toFixed(1)}%`,
      "How much the env-level belief still reveals the dominant support probe family. Lower is better: the belief should encode world mechanics, not probe-style identity."
    ),
    readoutRow(
      "Uncert. vs Error",
      `${derived.uncertaintyErrorAlignment.correlation.toFixed(2)} corr`,
      "Correlation between env-level uncertainty and actual mechanics prediction error. Higher is better if uncertainty is honest."
    ),
    ...derived.perParamEnvFitR2.map((row) => readoutRow(
      `${row.name} R²`,
      Number(row.r2).toFixed(2),
      "Per-parameter linear decode quality from the env-level belief."
    )),
  ].join("") : `<div class="empty">No latent readout is available yet.</div>`;

  const mechanicsSignal =
    !derived ? "n/a" :
    derived.linearEnvFitR2 > 0.65 ? "strong mechanics signal" :
    derived.linearEnvFitR2 > 0.35 ? "partial mechanics signal" :
    "weak mechanics signal";
  const localConsistency =
    !derived ? "n/a" :
    derived.splitRetrievalTop1 > 0.70 ? "support halves retrieve cleanly" :
    derived.splitRetrievalTop1 > 0.40 ? "support halves partly retrieve" :
    derived.neighborEnvAlignment > 0.20 ? "neighbors partially aligned" :
    "neighbors weakly aligned";
  const sameEnvSpread =
    !derived ? "n/a" :
    derived.sameEnvGapRatio.mean < 0.12 ? "few probes already agree" :
    derived.sameEnvGapRatio.mean < 0.25 ? "subset beliefs partially agree" :
    "subset beliefs still disagree";
  const supportDiversity =
    !derived ? "n/a" :
    derived.supportGroupCountMean >= 8.0 && derived.splitGroupOverlapMean > 0.85 ? "support covers all families" :
    derived.supportGroupCountMean >= 5.0 && derived.splitGroupOverlapMean > 0.70 ? "support is paired across families" :
    derived.supportGroupRatioMean > 0.85 ? "support is broadly covered" :
    derived.supportGroupRatioMean > 0.60 ? "support is partly diverse" :
    "support is too narrow";
  const uncertaintySignal =
    !derived ? "n/a" :
    derived.uncertaintyErrorAlignment.correlation > 0.45 ? "uncertainty tracks mechanics error" :
    derived.uncertaintyErrorAlignment.correlation > 0.20 ? "uncertainty weakly tracks mechanics error" :
    "uncertainty mostly flat";
  const supportModeText =
    derived && (derived.topModeName === "mixed" || derived.supportTiedTopFamilyCountMean > 1.5)
      ? `Support families are tied; top share is ${(100 * derived.supportTopFamilyShareMean).toFixed(1)}% across ${derived.supportTiedTopFamilyCountMean.toFixed(1)} top families.`
      : `Top support family is ${derived.topModeName} at ${(100 * derived.topModeShare).toFixed(1)}%.`;

  const guideRows = derived ? [
    ...(supportValidity.status === "ok" ? [] : [readoutRow(
      "Snapshot Validity",
      supportValidity.headline.toLowerCase(),
      `${supportValidity.detail} The most distortion-prone readouts here are ${affectedMetrics || "the split-based metrics"}.`
    )]),
    readoutRow(
      "Hidden Mechanics",
      mechanicsSignal,
      `Current latent-to-env fit is ${derived.linearEnvFitR2.toFixed(2)}. This is the quickest answer to “is the latent encoding the randomized mechanics, or just compressing trajectory texture?”`
    ),
    readoutRow(
      "Local Consistency",
      localConsistency,
      `Nearest-neighbor alignment is ${(100 * derived.neighborEnvAlignment).toFixed(1)}%, paired retrieval is ${(100 * derived.splitRetrievalTop1).toFixed(1)}%, and cross-family retrieval is ${(100 * derived.crossFamilySplitRetrievalTop1).toFixed(1)}%. Paired should move first; cross-family is the harder transfer check.`
    ),
    readoutRow(
      "Same-env Agreement",
      sameEnvSpread,
      `${supportModeText} The more important numbers are split disagreement ${derived.sameEnvSpread.mean.toFixed(3)} and gap ratio ${(100 * derived.sameEnvGapRatio.mean).toFixed(1)}% of nearest-between distance.`
    ),
    readoutRow(
      "Support Coverage",
      supportDiversity,
      `Support covers ${derived.supportGroupCountMean.toFixed(1)} families/env with ${(100 * derived.supportGroupRatioMean).toFixed(1)}% distinct-family share, effective count ${derived.supportEffectiveFamilyCountMean.toFixed(1)}, entropy ${derived.supportFamilyEntropyMean.toFixed(2)}. Paired split overlap is ${(100 * derived.splitGroupOverlapMean).toFixed(1)}%, cross-family overlap is ${(100 * derived.crossFamilySplitGroupOverlapMean).toFixed(1)}%, and split balance is ${(100 * derived.splitBalancedHalfFraction).toFixed(1)}%.`
    ),
    readoutRow(
      "Uncertainty Usefulness",
      uncertaintySignal,
      `Low-uncertainty env beliefs have mechanics error ${derived.uncertaintyErrorAlignment.low_error.toFixed(3)}, while high-uncertainty beliefs have ${derived.uncertaintyErrorAlignment.high_error.toFixed(3)}. This is the main check on whether uncertainty is honest instead of decorative.`
    ),
  ] : [];
  el("latentGuide").innerHTML = derived ? guideRows.join("") : `<div class="empty">No latent interpretation is available yet.</div>`;

  const featureRows = derived?.uncertaintyFeatureImportance || [];
  el("uncertaintyFeatureWrap").innerHTML = featureRows.length ? featureRows.map((row) => readoutRow(
    row.name.replace(/_/g, " "),
    `${(100 * Number(row.weight)).toFixed(1)}%`,
    "Share of the monotone uncertainty head's learned positive weight budget."
  )).join("") : `<div class="empty">No learned uncertainty feature weights are available yet.</div>`;

  const sortedByGap = [...payload.points].sort((left, right) => Number(right.gap_ratio) - Number(left.gap_ratio));
  const sortedByError = [...payload.points].sort((left, right) => Number(right.env_error) - Number(left.env_error));
  const sortedByFutureProbeError = [...payload.points].sort((left, right) => Number(right.future_probe_error || 0) - Number(left.future_probe_error || 0));
  const sortedByUncertainty = [...payload.points].sort((left, right) => Number(right.uncertainty) - Number(left.uncertainty));
  const topGap = sortedByGap[0];
  const topError = sortedByError[0];
  const topFutureProbeError = sortedByFutureProbeError[0];
  const topUncertainty = sortedByUncertainty[0];
  el("outlierGuideWrap").innerHTML = derived ? [
    readoutRow(
      "Worst Gap Ratio",
      topGap ? `${(100 * topGap.gap_ratio).toFixed(1)}%` : "n/a",
      "Highest same-world split disagreement as a share of nearest different-world distance. This is the sharpest local-geometry failure case."
    ),
    readoutRow(
      "Worst Mechanics Error",
      topError ? topError.env_error.toFixed(3) : "n/a",
      "Largest env-parameter prediction error among saved env beliefs. Useful for finding brittle worlds even when median fit looks good."
    ),
    readoutRow(
      "Worst Held-out Probe Error",
      topFutureProbeError ? Number(topFutureProbeError.future_probe_error || 0).toFixed(3) : "n/a",
      "Largest error when an env belief tries to predict held-out probe evidence. Useful when mechanics decode is decent but predictive reuse is weak."
    ),
    readoutRow(
      "Highest Uncertainty",
      topUncertainty ? topUncertainty.uncertainty.toFixed(3) : "n/a",
      "Largest env-level uncertainty score. Compare this to the worst-error case to see whether uncertainty is actually pointing at the right worlds."
    ),
    readoutRow(
      "Median Nearest-Between Distance",
      quantile(payload.points.map((point) => Number(point.nearest_between_distance)), 0.50).toFixed(3),
      "Typical distance to the nearest different-world belief. This gives context for whether same-world split gaps are truly small."
    ),
  ].join("") : `<div class="empty">No outlier readout is available yet.</div>`;

  const metric = el("colorSelect").value;
  const points = normalizeMetric(payload.points, metric);
  if (!points.length) {
    el("latentValidityWrap").innerHTML = "";
    el("latentPlotWrap").innerHTML = `<div class="empty">Run training first so the repo can save a latent snapshot artifact.</div>`;
    el("latentLegend").innerHTML = "";
    el("modeTableWrap").innerHTML = `<div class="empty">No probe-mode summary is available yet.</div>`;
    el("latentGuide").innerHTML = `<div class="empty">No latent interpretation is available yet.</div>`;
    el("uncertaintyErrorPlotWrap").innerHTML = `<div class="empty">No uncertainty/error diagnostic points are available yet.</div>`;
    el("gapRatioPlotWrap").innerHTML = `<div class="empty">No same-world gap diagnostics are available yet.</div>`;
    el("uncertaintyFeatureWrap").innerHTML = `<div class="empty">No learned uncertainty feature weights are available yet.</div>`;
    el("outlierGuideWrap").innerHTML = `<div class="empty">No outlier readout is available yet.</div>`;
    el("uncertaintyDistPlotWrap").innerHTML = `<div class="empty">No uncertainty distribution is available yet.</div>`;
    el("beliefScalePlotWrap").innerHTML = `<div class="empty">No belief-scale distribution is available yet.</div>`;
    el("pairwiseDistPlotWrap").innerHTML = `<div class="empty">No pairwise-distance distribution is available yet.</div>`;
    el("retrievalRankPlotWrap").innerHTML = `<div class="empty">No split-rank distribution is available yet.</div>`;
    el("beliefNormPlotWrap").innerHTML = `<div class="empty">No belief-norm distribution is available yet.</div>`;
    return;
  }

  const xs = points.map((point) => point.x);
  const ys = points.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const width = 840;
  const height = 520;
  const pad = 54;
  const xScale = (value) => pad + ((value - minX) / Math.max(maxX - minX, 1e-6)) * (width - 2 * pad);
  const yScale = (value) => height - pad - ((value - minY) / Math.max(maxY - minY, 1e-6)) * (height - 2 * pad);

  const modeColorMap = {};

  const verticalGrid = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const x = pad + fraction * (width - 2 * pad);
    return `<line x1="${x.toFixed(2)}" y1="${pad}" x2="${x.toFixed(2)}" y2="${height - pad}" stroke="#edf1f6" stroke-width="1"/>`;
  }).join("");

  const horizontalGrid = [0, 0.25, 0.5, 0.75, 1].map((fraction) => {
    const y = pad + fraction * (height - 2 * pad);
    return `<line x1="${pad}" y1="${y.toFixed(2)}" x2="${width - pad}" y2="${y.toFixed(2)}" stroke="#edf1f6" stroke-width="1"/>`;
  }).join("");

  const circles = points.map((point) => {
    const color = colorForMetric(point, metric, modeColorMap);
    return `
      <circle cx="${xScale(point.x).toFixed(2)}" cy="${yScale(point.y).toFixed(2)}" r="4.8" fill="${color}" fill-opacity="0.86" stroke="#ffffff" stroke-width="1">
        <title>windows=${point.window_count} | reward=${point.reward_sum.toFixed(2)} | uncert=${point.uncertainty.toFixed(3)} | spread=${point.same_env_spread.toFixed(3)}</title>
      </circle>
    `;
  }).join("");

  el("latentPlotWrap").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Latent PCA scatter">
      <rect x="0" y="0" width="${width}" height="${height}" fill="#ffffff"></rect>
      ${verticalGrid}
      ${horizontalGrid}
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" stroke="#d7dee8" stroke-width="1.25"></line>
      ${circles}
      <text x="${pad}" y="${height - 18}" fill="#6e7685" font-size="13" letter-spacing="0.14em">PCA AXIS 1</text>
      <text x="18" y="${pad - 14}" fill="#6e7685" font-size="13" letter-spacing="0.14em">PCA AXIS 2</text>
    </svg>
  `;

  if (metric === "terminated") {
    el("latentLegend").innerHTML = `
      <span><i class="swatch" style="background:#b85b6c"></i>terminated or truncated</span>
      <span><i class="swatch" style="background:#d8e4ff"></i>stable env belief</span>
    `;
  } else {
    el("latentLegend").innerHTML = `
      <span>${
        metric === "reward_sum"
          ? "Darker blue points indicate larger average probe reward across the env belief's windows."
          : metric === "env_param_mean"
            ? "Darker teal points indicate larger hidden-parameter mean across the env-parameter vector."
            : metric === "env_error"
              ? "Darker indigo points indicate larger hidden-mechanics prediction error for the env belief."
              : metric === "future_probe_error"
                ? "Darker indigo points indicate larger held-out probe prediction error from the env belief."
              : metric === "gap_ratio"
              ? "Darker indigo points indicate same-world split halves that are too far apart relative to nearby different worlds."
                : metric === "split_retrieval_margin_deficit"
                  ? "Darker indigo points indicate worlds whose two disjoint support halves still look too much like other worlds."
              : metric === "same_env_spread"
                ? "Darker indigo points indicate larger disagreement between env beliefs rebuilt from two disjoint support halves."
                : "Darker indigo points indicate larger ensemble disagreement over hidden env parameters."
      }</span>
    `;
  }

  el("uncertaintyErrorPlotWrap").innerHTML = scatterShellSvg(
    payload.points,
    "uncertainty",
    "env_error",
    "UNCERTAINTY",
    "MECHANICS ERROR",
    "gap_ratio",
  );
  el("gapRatioPlotWrap").innerHTML = scatterShellSvg(
    payload.points,
    "same_env_spread",
    "nearest_between_distance",
    "SAME-WORLD GAP",
    "NEAREST DIFFERENT-WORLD",
    "env_error",
  );
  el("uncertaintyDistPlotWrap").innerHTML = histogramSvg(
    payload.points.map((point) => Number(point.uncertainty)),
    "UNCERTAINTY",
    "#7f56d9",
    10,
  );
  el("beliefScalePlotWrap").innerHTML = histogramSvg(
    payload.points.map((point) => Number(point.nearest_between_distance)),
    "NEAREST DIFFERENT-WORLD",
    "#3d8d5e",
    10,
  );
  el("pairwiseDistPlotWrap").innerHTML = histogramSvg(
    (payload.series?.pairwise_between_distance || []).map((value) => Number(value)),
    "PAIRWISE DIFFERENT-WORLD",
    "#5b7cfa",
    12,
  );
  el("retrievalRankPlotWrap").innerHTML = histogramSvg(
    (payload.series?.split_retrieval_rank || []).map((value) => Number(value)),
    "SPLIT RETRIEVAL RANK",
    "#b85b6c",
    10,
  );
  el("beliefNormPlotWrap").innerHTML = histogramSvg(
    payload.points.map((point) => Number(point.belief_norm ?? 0)),
    "BELIEF NORM",
    "#0f8a84",
    10,
  );
  const compressionBits = payload.compression?.bits || [];
  const compressionFit = payload.compression?.mechanics_fit_r2 || [];
  const compressionTop1 = payload.compression?.split_retrieval_top1 || [];
  const compressionMrr = payload.compression?.split_retrieval_mrr || [];
  const compressionNorm = payload.compression?.message_norm_mean || [];
  el("compressionWrap").innerHTML = compressionBits.length ? compressionBits.map((bits, idx) => readoutRow(
    `${bits === 0 ? "Uncompressed" : `${bits} bits / dim`}`,
    `fit ${Number(compressionFit[idx] || 0).toFixed(2)} · top1 ${(100 * Number(compressionTop1[idx] || 0)).toFixed(1)}% · mrr ${Number(compressionMrr[idx] || 0).toFixed(2)}`,
    `mean expression norm ${Number(compressionNorm[idx] || 0).toFixed(3)}. This is the rate-distortion check on what the solver-facing env expression preserves.`
  )).join("") : `<div class="empty">No env-expression compression summary is available yet.</div>`;

  const totalModeCount = payload.mode_counts.reduce((sum, row) => sum + row.count, 0);
  const sortedModes = [...payload.mode_counts];
  el("modeTableWrap").innerHTML = sortedModes.length ? sortedModes.map((row) => `
    <div class="mode-row">
      <div>
        <div class="mode-name">${row.probe_mode}</div>
        <div class="mode-share">
          ${(100 * row.share).toFixed(1)}% of saved probe windows
          · uncert ${row.uncertainty_mean.toFixed(3)}
          · term ${(100 * row.terminated_rate).toFixed(1)}%
        </div>
      </div>
      <div class="mode-bar"><span style="width:${(100 * row.share)}%"></span></div>
      <div class="mode-count">${formatInteger(row.count)} windows</div>
    </div>
  `).join("") : `<div class="empty">No probe-mode counts are available yet.</div>`;
}

function renderBenchmark(payload) {
  dashboardState.benchmarkPayload = payload;
  renderHero();
  const summaries = payload.summaries;
  const researchMetrics = payload.research_metrics || {};
  const researchArms = researchMetrics.arms || {};
  const researchDeltas = researchMetrics.deltas || {};
  const researchPeak = researchMetrics.peak || {};
  const probeResearchArm = researchArms.probe || {};
  const baselineResearchArm = researchArms.baseline || {};
  const livePayload = dashboardState.livePayload || {};
  const liveSameBenchmark = Boolean(
    livePayload.available
    && livePayload.active
    && livePayload.benchmark_tag
    && payload.benchmark_tag
    && String(livePayload.benchmark_tag) === String(payload.benchmark_tag)
  );
  const liveUpdatedAt = Number(livePayload.updated_at || 0);
  const benchmarkUpdatedAt = Number(payload.artifact_mtime || 0);
  const benchmarkBehindLive = liveSameBenchmark && liveUpdatedAt > benchmarkUpdatedAt + 2;
  const stopReasonTotals = {};
  const episodeStopReasonTotals = {};
  const familySummary = {};
  for (const row of payload.rows) {
    if (row.probe_final_stop_reason) {
      stopReasonTotals[row.probe_final_stop_reason] = (stopReasonTotals[row.probe_final_stop_reason] || 0) + 1;
    }
    for (const [reason, value] of Object.entries(row.probe_stop_reasons || {})) {
      episodeStopReasonTotals[reason] = (episodeStopReasonTotals[reason] || 0) + Number(value || 0);
    }
    for (const [family, metrics] of Object.entries(row.probe_family_expected_gain || {})) {
      if (!familySummary[family]) {
        familySummary[family] = {
          selectionCount: 0,
          secondProbeSelectionCount: 0,
          expectedScore: 0,
          expectedMechanics: 0,
          expectedFuture: 0,
          expectedEntropy: 0,
          expectedHypothesisSeparation: 0,
          expectedRawFutureEstimate: 0,
          expectedFutureEstimate: 0,
          expectedChoiceFuture: 0,
          expectedCost: 0,
          expectedValuePerStep: 0,
          expectedControlUtility: 0,
          expectedStabilityAdjusted: 0,
          expectedSampleEfficiency: 0,
          realizedGain: 0,
          rowCount: 0,
          realizedCount: 0,
        };
      }
      familySummary[family].expectedScore += Number(metrics.score || 0);
      familySummary[family].expectedMechanics += Number(metrics.predicted_mechanics_reduction || 0);
      familySummary[family].expectedFuture += Number(metrics.predicted_future_error_reduction || 0);
      familySummary[family].expectedEntropy += Number(metrics.predicted_entropy_reduction || 0);
      familySummary[family].expectedHypothesisSeparation += Number(metrics.predicted_hypothesis_separation || 0);
      familySummary[family].expectedRawFutureEstimate += Number(metrics.raw_future_error_estimate || metrics.future_error_estimate || 0);
      familySummary[family].expectedFutureEstimate += Number(metrics.future_error_estimate || 0);
      familySummary[family].expectedChoiceFuture += Number(metrics.future_gain_for_choice || 0);
      familySummary[family].expectedCost += Number(metrics.estimated_probe_cost || 0);
      familySummary[family].expectedValuePerStep += Number(metrics.value_per_probe_step || 0);
      familySummary[family].expectedControlUtility += Number(metrics.control_utility_value || 0);
      familySummary[family].expectedStabilityAdjusted += Number(metrics.stability_adjusted_value || 0);
      familySummary[family].expectedSampleEfficiency += Number(metrics.sample_efficiency_score || metrics.value_per_probe_step || 0);
      familySummary[family].rowCount += 1;
    }
    for (const [family, value] of Object.entries(row.probe_family_realized_gain || {})) {
      if (!familySummary[family]) {
        familySummary[family] = {
          selectionCount: 0,
          secondProbeSelectionCount: 0,
          expectedScore: 0,
          expectedMechanics: 0,
          expectedFuture: 0,
          expectedEntropy: 0,
          expectedHypothesisSeparation: 0,
          expectedRawFutureEstimate: 0,
          expectedFutureEstimate: 0,
          expectedChoiceFuture: 0,
          expectedCost: 0,
          expectedValuePerStep: 0,
          realizedGain: 0,
          rowCount: 0,
          realizedCount: 0,
        };
      }
      familySummary[family].realizedGain += Number(value || 0);
      familySummary[family].realizedCount += 1;
    }
    for (const [family, value] of Object.entries(row.probe_family_selection_count || {})) {
      if (!familySummary[family]) {
        familySummary[family] = {
          selectionCount: 0,
          secondProbeSelectionCount: 0,
          expectedScore: 0,
          expectedMechanics: 0,
          expectedFuture: 0,
          expectedEntropy: 0,
          expectedHypothesisSeparation: 0,
          expectedRawFutureEstimate: 0,
          expectedFutureEstimate: 0,
          expectedChoiceFuture: 0,
          expectedCost: 0,
          expectedValuePerStep: 0,
          realizedGain: 0,
          rowCount: 0,
          realizedCount: 0,
        };
      }
      familySummary[family].selectionCount += Number(value || 0);
    }
    for (const [family, value] of Object.entries(row.probe_second_probe_selection_count || {})) {
      if (!familySummary[family]) {
        familySummary[family] = {
          selectionCount: 0,
          secondProbeSelectionCount: 0,
          expectedScore: 0,
          expectedMechanics: 0,
          expectedFuture: 0,
          expectedEntropy: 0,
          expectedHypothesisSeparation: 0,
          expectedRawFutureEstimate: 0,
          expectedFutureEstimate: 0,
          expectedChoiceFuture: 0,
          expectedCost: 0,
          expectedValuePerStep: 0,
          realizedGain: 0,
          rowCount: 0,
          realizedCount: 0,
        };
      }
      familySummary[family].secondProbeSelectionCount += Number(value || 0);
    }
  }
  const readinessReasonCounts = summaries.probe_readiness_reason_counts || {};
  const fairStopBlockerCounts = summaries.probe_fair_stop_blocker_counts || {};
  const shadowBlockerCounts = summaries.probe_shadow_blocker_counts || {};
  const readinessComponentMeans = summaries.probe_readiness_component_means || {};
  const fullSystemStyle = String(payload.full_system_controller_style || "");
  const fullSystemOracleStyle = String(payload.full_system_oracle_controller_style || fullSystemStyle);
  const fullSystemName = fullSystemDisplayName(fullSystemStyle);
  const fullSystemRef = fullSystemReferenceName(fullSystemStyle);
  const fullSystemStateOnlyName = `${fullSystemName} State-Only`;
  const fullSystemOracleName = fullSystemDisplayName(fullSystemOracleStyle, true);
  const fullSystemOracleRef = fullSystemReferenceName(fullSystemOracleStyle, true);
  const dominantReadinessBlocker = Object.entries(readinessReasonCounts)
    .sort((a, b) => Number(b[1]) - Number(a[1]))[0] || ["unknown", 0];
  const dominantFairStopBlocker = Object.entries(fairStopBlockerCounts)
    .sort((a, b) => Number(b[1]) - Number(a[1]))[0] || ["enabled", 0];
  const dominantShadowBlocker = Object.entries(shadowBlockerCounts)
    .sort((a, b) => Number(b[1]) - Number(a[1]))[0] || ["enabled", 0];
  const latentWinGate = payload.latent_win_gate || {};
  const latentWinGateReasons = Array.isArray(payload.latent_win_gate_failure_reasons)
    ? payload.latent_win_gate_failure_reasons
    : [];
  const latentWinGateStatus = latentWinGate.pass ? "PASS" : "BLOCKED";
  const strictUsageStatus = String(payload.probe_strict_usage_status || "unused");
  const honestyHeadline = String(payload.probe_honesty_headline || "");
  const latentSupport = summaries.latent_support_diagnostics || {};
  const supportCenterShare = Number(latentSupport.center_window_share || 0);
  const supportDirectionalShare = Number(latentSupport.directional_window_share || 0);
  const supportMechanicsShare = Number(latentSupport.mechanics_window_share || 0);
  const supportPassiveShare = Number(latentSupport.passive_window_share || 0);
  const supportStressShare = Number(latentSupport.stress_window_share || 0);
  const hasNamedMechanicsSupport = supportMechanicsShare > 0 || supportPassiveShare > 0 || supportStressShare > 0;
  const supportEffectiveFamilies = Number(latentSupport.effective_window_families || 0);
  const supportWindowLeak = Number(latentSupport.window_mode_leakage || 0);
  const supportEnvLeak = Number(latentSupport.env_mode_leakage || 0);
  const latentNearestBetween = Number(latentSupport.nearest_between_median || 0);
  const crossSplitSummary = summaries.latent_cross_split_top1 || summaries.latent_split_top1 || {median: 0, mean: 0};
  const pairedSplitSummary = summaries.latent_paired_split_top1 || {median: 0, mean: 0};
  const systemId = summaries.system_id || {};
  const particleSysidMode = systemId.available && String(systemId.mode || "") === "particle_sysid";
  const particleSubsetWeak =
    particleSysidMode
    && (
      Number(systemId.particle_subset_stability_median || 0) < 0.45
      || Number(systemId.particle_leaveout_shift_median || 0) > 0.25
    );
  const latentGeometryCollapsed =
    !particleSysidMode
    && crossSplitSummary.median < 0.05
    && (summaries.latent_gap_ratio.median > 5.0 || latentNearestBetween < 0.001);
  const nextLatentBottleneck = particleSubsetWeak
    ? "particle subset stability"
    : latentGeometryCollapsed
    ? "belief geometry collapsed"
    : summaries.probe_env_expression_delta.mean <= 0
      ? "expression harmful"
    : hasNamedMechanicsSupport && supportMechanicsShare < 0.80
      ? "mechanics support narrow"
    : !hasNamedMechanicsSupport && supportCenterShare > 0.30
      ? "support center-heavy"
      : supportWindowLeak > 0.35
        ? "window probe leakage"
        : pairedSplitSummary.median < 0.20
          ? "split retrieval flat"
          : summaries.latent_gap_ratio.median > 1.0
            ? "same-world gap too high"
            : "controller attribution";
  el("benchmarkSummary").innerHTML = [
    ...(benchmarkBehindLive ? [
      summaryCell(
        "Artifact State",
        "live run in progress",
        "This deck is showing the last completed benchmark artifact for this tag; the live deck is streaming a newer run that has not written its final benchmark summary yet."
      ),
    ] : []),
    summaryCell("Honesty Check", honestyHeadline || strictUsageStatus, honestyHeadline || "Whether the artifact is actually using the strict fair latent path instead of only winning on the surrounding protocol."),
    summaryCell("Strict Usage", strictUsageStatus, "Derived from the share of strict fair handoffs that really enabled the env expression: unused, intermittent, or active."),
    summaryCell("Expr Delta", summaries.probe_env_expression_delta.mean.toFixed(2), "Matched evaluation return gain from using the learned env expression instead of muting it at control time."),
    summaryCell("Forced Expr Delta", summaries.probe_forced_env_expression_delta.mean.toFixed(2), "Matched evaluation return gain from the small forced diagnostic expression path used to check whether the message is merely under-trusted or actively harmful."),
    summaryCell("Leaveout Stability", Number(readinessComponentMeans.leaveout_stability || 0).toFixed(2), "Mean strict readiness component for leave-one-family-out stability. This is the main honesty blocker when the latent changes too much across support subsets."),
    summaryCell("Baseline Episode Median", summaries.baseline_episode.median.toFixed(1), "median solve episode with unsolved runs capped by completed episodes"),
    summaryCell("Probe Episode Median", summaries.probe_episode.median.toFixed(1), "episode solve speed for the latent-conditioned path"),
    summaryCell("Belief Progress Index", summaries.belief_progress_index.median.toFixed(3), "fast-loop latent progress score combining mechanics fit, geometry, held-out probe prediction, uncertainty honesty, and leakage control"),
    ...(systemId.available ? [
      summaryCell("System-ID Progress", Number(systemId.progress_median || 0).toFixed(3), "particle-system-ID health from held-out sysid quality, posterior sharpness, and leave-one-family stability"),
      summaryCell("SysID Top-1", `${(100 * Number(systemId.validation_top1_median || 0)).toFixed(1)}%`, "held-out candidate mechanics retrieval accuracy for the particle likelihood model"),
      summaryCell("Particle Leaveout", Number(systemId.particle_leaveout_shift_median || 0).toFixed(3), "posterior mean shift when one observed probe family is removed"),
    ] : []),
    summaryCell(
      hasNamedMechanicsSupport ? "Mechanics Support" : "Support Center",
      hasNamedMechanicsSupport ? `${(100 * supportMechanicsShare).toFixed(1)}%` : `${(100 * supportCenterShare).toFixed(1)}%`,
      hasNamedMechanicsSupport
        ? "share of saved windows covered by the named CartPole mechanics probes"
        : "share of saved encoder/support windows that used the center probe family"
    ),
    summaryCell("Effective Families", supportEffectiveFamilies.toFixed(2), "inverse-concentration count of probe families represented in saved support windows"),
    summaryCell("Window Leak", supportWindowLeak.toFixed(2), "linear probe-family leakage measured on individual support windows before env pooling"),
    summaryCell("Env Leak", supportEnvLeak.toFixed(2), "probe-family leakage measured after pooling support windows into env beliefs"),
    summaryCell(
      particleSysidMode ? "Legacy Latent Gate" : "Latent Win Gate",
      latentWinGateStatus,
      particleSysidMode
        ? "The old latent-geometry gate is still shown as the baseline contrast; particle sysid health is tracked separately."
        : "A run only counts as cracked when probe beats baseline, the latent clears the representation floors, and learned context materially beats the matched ablations"
    ),
    ...(payload.probe_shadow_available ? [
      summaryCell("Shadow Episode Median", summaries.probe_shadow_episode.median.toFixed(1), "diagnostic near-ready gate using the same probe protocol without changing headline classification"),
    ] : []),
    summaryCell("Probe No-Expr Median", solveSummaryLabel(summaries.probe_no_expression_episode), "matched probe-control run with the env expression muted during control"),
    ...(payload.full_system_available ? [
      summaryCell(`${fullSystemName} Median`, summaries.full_system_episode.median.toFixed(1), isPlannerControllerStyle(fullSystemStyle) ? "belief-driven planner path that treats the controller context as a planning state instead of a PPO side hint" : "cheap belief-conditioned controller that ranks a small fixed action set instead of running a heavy planner at test time"),
      summaryCell(`${fullSystemName} Success`, `${summaries.full_system_episode.success_rate}/${summaries.full_system_episode.count}`, `how often the breakthrough ${fullSystemRef} track solved across benchmark seeds`),
      ...(payload.full_system_state_only_available ? [
        summaryCell(`${fullSystemRef} Matched Eval`, summaries.full_system_learned_eval.mean_return.mean.toFixed(2), `${fullSystemRef} mean return on the fixed matched fixture set used for controller attribution`),
        summaryCell(`${fullSystemStateOnlyName} Eval`, summaries.full_system_state_only_eval.mean_return.mean.toFixed(2), "matched final-checkpoint evaluation with the belief residual disabled and only the state student active"),
        summaryCell("Belief Over State-Only", summaries.full_system_state_only_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return gain over the matched state-only student from the same trained checkpoint`),
        summaryCell("State-Only Eval Steps", summaries.full_system_state_only_eval.mean_total_env_steps.mean.toFixed(1), "mean total env steps on the shared matched fixture set with the residual disabled"),
      ] : []),
      ...(payload.full_system_zero_context_available ? [
        summaryCell("Zero-Context Eval", summaries.full_system_zero_context_eval.mean_return.mean.toFixed(2), `${fullSystemRef} mean return when the controller context is zeroed on the same matched fixture set`),
        summaryCell("Belief Over Zero", summaries.full_system_zero_context_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return gain over zeroed context on the same fixed fixtures`),
      ] : []),
      ...(payload.full_system_shuffled_context_available ? [
        summaryCell("Shuffled Eval", summaries.full_system_shuffled_context_eval.mean_return.mean.toFixed(2), `${fullSystemRef} mean return when controller context is permuted across matched fixtures`),
        summaryCell("Belief Over Shuffled", summaries.full_system_shuffled_context_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return gain over shuffled controller context on the same fixed fixtures`),
      ] : []),
      ...(payload.full_system_stale_context_available ? [
        summaryCell("Stale Eval", summaries.full_system_stale_context_eval.mean_return.mean.toFixed(2), `${fullSystemRef} mean return when the controller is fed stale previous context on the same matched fixtures`),
        summaryCell("Belief Over Stale", summaries.full_system_stale_context_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return gain over stale controller context on the same fixed fixtures`),
      ] : []),
      summaryCell("Online Refresh Delta", summaries.full_system_online_refinement_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return drop when online context refresh is disabled`),
      ...(payload.full_system_frozen_context_available ? [
        summaryCell("Frozen Eval", summaries.full_system_frozen_context_eval.mean_return.mean.toFixed(2), `${fullSystemRef} mean return when stale controller context is frozen and online refresh stays disabled`),
        summaryCell("Belief Over Frozen", summaries.full_system_frozen_context_ablation_delta.mean.toFixed(2), `${fullSystemRef} eval return gain over stale frozen context on the same fixed fixtures`),
      ] : []),
    ] : []),
    ...(payload.full_system_oracle_available ? [
      summaryCell(`${fullSystemOracleName} Median`, summaries.full_system_oracle_episode.median.toFixed(1), isPlannerControllerStyle(fullSystemOracleStyle) ? "oracle-parameter planner-context track used to separate controller failure from learned-representation failure" : "oracle-parameter cheap-control track used to separate controller failure from learned-representation failure"),
      summaryCell("Oracle Success", `${summaries.full_system_oracle_episode.success_rate}/${summaries.full_system_oracle_episode.count}`, "how often the oracle controller-context track solved across benchmark seeds"),
      summaryCell("Oracle Zero Delta", summaries.full_system_oracle_zero_context_ablation_delta.mean.toFixed(2), "oracle-context eval return drop when the controller context is zeroed during evaluation"),
      summaryCell("Oracle Refresh Delta", summaries.full_system_oracle_online_refinement_ablation_delta.mean.toFixed(2), "oracle-context eval return drop when online refinement is disabled"),
      ...(payload.full_system_oracle_frozen_context_available ? [
        summaryCell("Oracle Frozen Delta", summaries.full_system_oracle_frozen_context_ablation_delta.mean.toFixed(2), "oracle-context eval return drop when stale context is frozen and online refinement stays disabled"),
      ] : []),
    ] : []),
    ...(payload.sim_fanout_available ? [
      summaryCell("Sim-Fanout Median", summaries.sim_fanout_episode.median.toFixed(1), "cheap-sim ceiling baseline that evaluates the same candidate action set directly on the true simulator"),
      summaryCell("Sim-Fanout Success", `${summaries.sim_fanout_episode.success_rate}/${summaries.sim_fanout_episode.count}`, "how often the direct simulator fan-out baseline solved across benchmark seeds"),
    ] : []),
    summaryCell("Post-Expression Steps", summaries.probe_post_expression_steps.median.toFixed(1), "control-only env steps after the final env-expression handoff"),
    summaryCell("Expr Scale Median", summaries.probe_expression_scale_median.median.toFixed(2), "median controller-side env-expression scale during downstream control"),
    ...(payload.probe_shadow_available ? [
      summaryCell("Shadow Enabled", `${(100 * summaries.probe_shadow_expression_enabled_fraction.mean).toFixed(1)}%`, "share of diagnostic shadow handoffs where the near-ready env expression actually got used"),
      summaryCell("Shadow Strict Miss", `${(100 * summaries.probe_shadow_strict_miss_fraction.mean).toFixed(1)}%`, "share of shadow episodes where strict fair would still have muted the latent while the shadow gate exposed it"),
    ] : []),
    summaryCell("Ready Handoffs", `${(100 * summaries.probe_fair_ready_handoff_fraction.mean).toFixed(1)}%`, "share of fair-mode handoffs where the env expression actually cleared the ready bar"),
    summaryCell("Strict Enabled", `${(100 * summaries.probe_fair_expression_enabled_fraction.mean).toFixed(1)}%`, "share of fair-mode handoffs where the stricter fair policy actually enabled the env expression"),
    summaryCell("Muted By Policy", `${(100 * summaries.probe_fair_expression_force_muted_fraction.mean).toFixed(1)}%`, "share of fair-mode handoffs where control force-muted the env expression because it was not ready"),
    summaryCell("Ready But Muted", `${(100 * summaries.probe_expression_ready_but_muted_fraction.mean).toFixed(1)}%`, "share of handoffs where the general ready flag fired but strict fair policy still kept the env expression muted"),
    summaryCell("Top Blocker", `${String(dominantReadinessBlocker[0]).replace(/_/g, " ")}`, "dominant strict readiness failure mode across saved seeds"),
    summaryCell("Fair Stop Blocker", `${String(dominantFairStopBlocker[0]).replace(/_/g, " ")}`, "dominant blocker on the stricter fair-policy stop gate"),
    summaryCell("Second-Probe Choice Gain", summaries.probe_second_probe_choice_future_gain_mean.mean.toFixed(2), "mean choice-time future-gain proxy on the actually chosen fair probe-two family"),
    summaryCell("Baseline Success", `${summaries.baseline_episode.success_rate}/${summaries.baseline_episode.count}`, "how often the baseline actually solved across benchmark seeds"),
    summaryCell("Probe Success", `${summaries.probe_episode.success_rate}/${summaries.probe_episode.count}`, "how often the latent-conditioned branch actually solved across benchmark seeds"),
    summaryCell("Run Class", `${payload.run_classification || "protocol_win"}`, "Whether this artifact looks latent-driven, mostly procedural, or controller-compensated."),
    summaryCell("Benchmark Profile", `${payload.benchmark_profile || "full"}`, "Fast is the cheap local research sweep, full adds oracle and shadow diagnostics, and archived_planner keeps the heavy planner off the default headline path."),
    summaryCell("Benchmark Mode", `${payload.benchmark_mode || "fair"} / ${payload.probe_budget_mode || "fair_two_probe_handoff"}`, "headline mode for this saved benchmark artifact"),
  ].join("");

  el("seedGrid").innerHTML = payload.rows.length ? payload.rows.map((row) => {
    const rowMechanicsShare = Number(row.latent_mechanics_window_share || 0);
    const rowPassiveShare = Number(row.latent_passive_window_share || 0);
    const rowStressShare = Number(row.latent_stress_window_share || 0);
    const rowHasNamedMechanics = rowMechanicsShare > 0 || rowPassiveShare > 0 || rowStressShare > 0;
    const rowSupportMix = rowHasNamedMechanics
      ? `mech ${(100 * rowMechanicsShare).toFixed(1)}% · passive ${(100 * rowPassiveShare).toFixed(1)}% · stress ${(100 * rowStressShare).toFixed(1)}%`
      : `center ${(100 * Number(row.latent_center_window_share || 0)).toFixed(1)}% · dir ${(100 * Number(row.latent_directional_window_share || 0)).toFixed(1)}%`;
    return `
    <div class="seed-row">
      <div>
        <div class="seed-cell-label">Seed</div>
        <div class="seed-cell-value">${row.seed}</div>
        <div class="seed-cell-note">benchmark run id</div>
      </div>
      <div>
        <div class="seed-cell-label">Baseline Ep</div>
        <div class="seed-cell-value">${solveBadge(row.baseline_episode_solve)}</div>
        <div class="seed-cell-note">first solve episode</div>
      </div>
      <div>
        <div class="seed-cell-label">Probe Ep</div>
        <div class="seed-cell-value">${solveBadge(row.probe_episode_solve)}</div>
        <div class="seed-cell-note">latent-conditioned solve</div>
      </div>
      ${payload.probe_shadow_available ? `
      <div>
        <div class="seed-cell-label">Shadow Ep</div>
        <div class="seed-cell-value">${solveBadge(row.probe_shadow_episode_solve)}</div>
        <div class="seed-cell-note">diagnostic near-ready gate</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">No-Expr Ep</div>
        <div class="seed-cell-value">${solveBadge(row.probe_no_expression_episode_solve, summaries.probe_no_expression_episode.not_run)}</div>
        <div class="seed-cell-note">matched probe run with muted expression</div>
      </div>
      ${payload.full_system_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemName} Ep</div>
        <div class="seed-cell-value">${solveBadge(row.full_system_episode_solve)}</div>
        <div class="seed-cell-note">full-system belief-first controller</div>
      </div>
      ` : ""}
      ${payload.full_system_oracle_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemOracleName} Ep</div>
        <div class="seed-cell-value">${solveBadge(row.full_system_oracle_episode_solve)}</div>
        <div class="seed-cell-note">oracle controller-context diagnostic</div>
      </div>
      ` : ""}
      ${payload.sim_fanout_available ? `
      <div>
        <div class="seed-cell-label">Sim-Fanout Ep</div>
        <div class="seed-cell-value">${solveBadge(row.sim_fanout_episode_solve)}</div>
        <div class="seed-cell-note">direct simulator fan-out ceiling</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">Baseline Steps</div>
        <div class="seed-cell-value">${solveBadge(row.baseline_step_solve)}</div>
        <div class="seed-cell-note">env steps to solve</div>
      </div>
      <div>
        <div class="seed-cell-label">Probe Steps</div>
        <div class="seed-cell-value">${solveBadge(row.probe_step_solve)}</div>
        <div class="seed-cell-note">includes probe interaction</div>
      </div>
      ${payload.probe_shadow_available ? `
      <div>
        <div class="seed-cell-label">Shadow Steps</div>
        <div class="seed-cell-value">${solveBadge(row.probe_shadow_step_solve)}</div>
        <div class="seed-cell-note">diagnostic near-ready gate</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">No-Expr Steps</div>
        <div class="seed-cell-value">${solveBadge(row.probe_no_expression_step_solve, summaries.probe_no_expression_steps.not_run)}</div>
        <div class="seed-cell-note">matched probe run with muted expression</div>
      </div>
      ${payload.full_system_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemName} Steps</div>
        <div class="seed-cell-value">${solveBadge(row.full_system_step_solve)}</div>
        <div class="seed-cell-note">full-system controller solve cost</div>
      </div>
      ` : ""}
      ${payload.full_system_oracle_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemOracleName} Steps</div>
        <div class="seed-cell-value">${solveBadge(row.full_system_oracle_step_solve)}</div>
        <div class="seed-cell-note">oracle controller-context solve cost</div>
      </div>
      ` : ""}
      ${payload.sim_fanout_available ? `
      <div>
        <div class="seed-cell-label">Sim-Fanout Steps</div>
        <div class="seed-cell-value">${solveBadge(row.sim_fanout_step_solve)}</div>
        <div class="seed-cell-note">true-simulator candidate-search cost</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">Baseline Total</div>
        <div class="seed-cell-value">${formatInteger(row.baseline_total_env_steps)}</div>
        <div class="seed-cell-note">full run env cost</div>
      </div>
      <div>
        <div class="seed-cell-label">Probe Encoder</div>
        <div class="seed-cell-value">${formatInteger(row.probe_encoder_steps)}</div>
        <div class="seed-cell-note">latent pretraining windows</div>
      </div>
      <div>
        <div class="seed-cell-label">Probe Online</div>
        <div class="seed-cell-value">${formatInteger(row.probe_probe_env_steps)}</div>
        <div class="seed-cell-note">probe interaction cost</div>
      </div>
      <div>
        <div class="seed-cell-label">Post-Expr Steps</div>
        <div class="seed-cell-value">${formatInteger(row.probe_post_expression_env_steps)}</div>
        <div class="seed-cell-note">control-only env steps after handoff</div>
      </div>
      <div>
        <div class="seed-cell-label">Strict Usage</div>
        <div class="seed-cell-value">${row.probe_strict_usage_status || "unused"}</div>
        <div class="seed-cell-note">expr ${Number(row.probe_env_expression_delta || 0).toFixed(2)} · forced ${Number(row.probe_forced_env_expression_delta || 0).toFixed(2)} · scale ${Number(row.probe_forced_env_expression_scale || 0).toFixed(2)}</div>
      </div>
      <div>
        <div class="seed-cell-label">Support Mix</div>
        <div class="seed-cell-value">${rowSupportMix}</div>
        <div class="seed-cell-note">window leak ${Number(row.latent_window_mode_leakage || 0).toFixed(2)} · env leak ${Number(row.latent_env_mode_leakage || 0).toFixed(2)} · eff ${Number(row.latent_support_diagnostics?.effective_window_families || 0).toFixed(2)}</div>
      </div>
      ${payload.full_system_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemName} Post-Ctx</div>
        <div class="seed-cell-value">${formatInteger(row.full_system_post_context_env_steps)}</div>
        <div class="seed-cell-note">${fullSystemRef} control-only steps after the last context refresh</div>
      </div>
      ${payload.full_system_state_only_available ? `
      <div>
        <div class="seed-cell-label">${fullSystemRef} Matched Eval</div>
        <div class="seed-cell-value">${Number(row.full_system_learned_eval_summary?.mean_return || 0).toFixed(2)} return · ${Number(row.full_system_learned_eval_summary?.mean_total_env_steps || 0).toFixed(1)} steps</div>
        <div class="seed-cell-note">selected checkpoint on the fixed matched fixture set</div>
      </div>
      <div>
        <div class="seed-cell-label">${fullSystemStateOnlyName} Eval</div>
        <div class="seed-cell-value">${Number(row.full_system_state_only_eval_summary?.mean_return || 0).toFixed(2)} return · ${Number(row.full_system_state_only_eval_summary?.mean_total_env_steps || 0).toFixed(1)} steps</div>
        <div class="seed-cell-note">same checkpoint, but the residual is disabled and the state student is forced</div>
      </div>
      ` : ""}
      ${row.full_system_zero_context_available ? `
      <div>
        <div class="seed-cell-label">Zero-Context Eval</div>
        <div class="seed-cell-value">${Number(row.full_system_zero_context_eval_summary?.mean_return || 0).toFixed(2)} return · ${Number(row.full_system_zero_context_eval_summary?.mean_total_env_steps || 0).toFixed(1)} steps</div>
        <div class="seed-cell-note">same checkpoint, but the controller context is zeroed on the matched fixture set</div>
      </div>
      ` : ""}
      ${row.full_system_shuffled_context_available ? `
      <div>
        <div class="seed-cell-label">Shuffled Eval</div>
        <div class="seed-cell-value">${Number(row.full_system_shuffled_context_eval_summary?.mean_return || 0).toFixed(2)} return · ${Number(row.full_system_shuffled_context_eval_summary?.mean_total_env_steps || 0).toFixed(1)} steps</div>
        <div class="seed-cell-note">same checkpoint, but controller context is permuted across matched fixtures</div>
      </div>
      ` : ""}
      ${row.full_system_stale_context_available ? `
      <div>
        <div class="seed-cell-label">Stale Eval</div>
        <div class="seed-cell-value">${Number(row.full_system_stale_context_eval_summary?.mean_return || 0).toFixed(2)} return · ${Number(row.full_system_stale_context_eval_summary?.mean_total_env_steps || 0).toFixed(1)} steps</div>
        <div class="seed-cell-note">same checkpoint, but the controller reuses stale previous context on the matched fixture set</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">${fullSystemName} Ablation</div>
        <div class="seed-cell-value">state-only ${Number(row.full_system_state_only_ablation_delta || 0).toFixed(2)} · zero ${Number(row.full_system_zero_context_ablation_delta || 0).toFixed(2)} · shuffled ${Number(row.full_system_shuffled_context_ablation_delta || 0).toFixed(2)} · stale ${Number(row.full_system_stale_context_ablation_delta || 0).toFixed(2)}</div>
        <div class="seed-cell-note">does learned context beat the state student, and does return fall when context is zeroed, permuted, or frozen?</div>
      </div>
      ` : ""}
      ${payload.full_system_oracle_available ? `
      <div>
        <div class="seed-cell-label">Oracle Post-Ctx</div>
        <div class="seed-cell-value">${formatInteger(row.full_system_oracle_post_context_env_steps)}</div>
        <div class="seed-cell-note">oracle control-only steps after the last context handoff</div>
      </div>
      <div>
        <div class="seed-cell-label">Oracle Ablation</div>
        <div class="seed-cell-value">zero ${Number(row.full_system_oracle_zero_context_ablation_delta || 0).toFixed(2)} · stale ${Number(row.full_system_oracle_stale_context_ablation_delta || 0).toFixed(2)}</div>
        <div class="seed-cell-note">if this does not hurt, the controller path is still broken even with true world parameters</div>
      </div>
      ` : ""}
      <div>
        <div class="seed-cell-label">Stop</div>
        <div class="seed-cell-value">${row.probe_final_stop_reason || "n/a"}</div>
        <div class="seed-cell-note">why probing stopped</div>
      </div>
      <div>
        <div class="seed-cell-label">Strict vs Shadow</div>
        <div class="seed-cell-value">${row.probe_strictly_muted_but_shadow_eligible ? "muted but shadow-eligible" : "aligned"}</div>
        <div class="seed-cell-note">strict miss ${(100 * Number(row.probe_shadow_strict_miss_fraction || 0)).toFixed(1)}% · whether strict fair mode muted the latent while the shadow gate would have exposed it</div>
      </div>
      <div>
        <div class="seed-cell-label">Class</div>
        <div class="seed-cell-value">${row.probe_run_classification || "protocol_win"}</div>
        <div class="seed-cell-note">latent vs protocol story for this seed</div>
      </div>
    </div>
  `;
  }).join("") : `<div class="empty">No benchmark summary is available yet.</div>`;

  const stopRows = Object.entries(stopReasonTotals)
    .filter(([, value]) => Number(value) > 0)
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  el("probeStopWrap").innerHTML = stopRows.length ? stopRows.map(([reason, value]) => readoutRow(
    reason.replace(/_/g, " "),
    formatInteger(value),
    `How often this final stop condition ended probing across seeds. Episode-level counts: ${formatInteger(episodeStopReasonTotals[reason] || 0)}.`
  )).join("") : `<div class="empty">No probe-stop summary is available yet.</div>`;

  const familyRows = Object.entries(familySummary)
    .sort((a, b) => Number(b[1].selectionCount) - Number(a[1].selectionCount));
  el("probeFamilyWrap").innerHTML = familyRows.length ? familyRows.map(([family, value]) => {
    const expectedCount = Math.max(Number(value.rowCount || 0), 1);
    const realizedCount = Math.max(Number(value.realizedCount || 0), 1);
    return readoutRow(
      family,
      `picked ${formatInteger(value.selectionCount)} · probe-2 ${formatInteger(value.secondProbeSelectionCount)} · sample-eff ${Number(value.expectedSampleEfficiency / expectedCount).toFixed(2)} · value/step ${Number(value.expectedValuePerStep / expectedCount).toFixed(2)}`,
      `Control utility ${Number(value.expectedControlUtility / expectedCount).toFixed(2)}, stability-adjusted value ${Number(value.expectedStabilityAdjusted / expectedCount).toFixed(2)}, predicted mechanics gain ${Number(value.expectedMechanics / expectedCount).toFixed(2)}, predicted entropy drop ${Number(value.expectedEntropy / expectedCount).toFixed(2)}, hypothesis separation ${Number(value.expectedHypothesisSeparation / expectedCount).toFixed(2)}, predicted future-probe gain ${Number(value.expectedFuture / expectedCount).toFixed(2)}, raw future estimate ${Number(value.expectedRawFutureEstimate / expectedCount).toFixed(2)}, floored future estimate ${Number(value.expectedFutureEstimate / expectedCount).toFixed(2)}, choice-time future gain ${Number(value.expectedChoiceFuture / expectedCount).toFixed(2)}, realized uncertainty drop ${Number(value.realizedGain / realizedCount).toFixed(2)}, estimated probe cost ${Number(value.expectedCost / expectedCount).toFixed(2)}.`
    );
  }).join("") : `<div class="empty">No per-family probe diagnostics are available yet.</div>`;

  const episodeDelta = summaries.baseline_episode.median - summaries.probe_episode.median;
  const stepDelta = summaries.baseline_steps.median - summaries.probe_steps.median;
  const hasNoExpressionSolve = hasSolveSummary(summaries.probe_no_expression_episode);
  const hasFullSystemSolve = payload.full_system_available && hasSolveSummary(summaries.full_system_episode);
  const hasFullSystemOracleSolve = payload.full_system_oracle_available && hasSolveSummary(summaries.full_system_oracle_episode);
  const noExprEpisodeDelta = hasNoExpressionSolve
    ? summaries.probe_no_expression_episode.median - summaries.probe_episode.median
    : 0;
  const strictExpressionActive = strictUsageStatus === "active"
    || Number(summaries.probe_fair_expression_enabled_fraction.mean || 0) > 0.05;
  const latentContributionLabel = strictExpressionActive ? "expression" : "conditioned branch";
  const noExpressionReferenceLabel = strictExpressionActive ? "no-expression" : "no-expression branch";
  const shadowEpisodeDelta = payload.probe_shadow_available && hasNoExpressionSolve
    ? summaries.probe_no_expression_episode.median - summaries.probe_shadow_episode.median
    : 0;
  const fullSystemEpisodeDelta = hasFullSystemSolve && hasNoExpressionSolve
    ? summaries.probe_no_expression_episode.median - summaries.full_system_episode.median
    : 0;
  const fullSystemOracleEpisodeDelta = hasFullSystemOracleSolve && hasNoExpressionSolve
    ? summaries.probe_no_expression_episode.median - summaries.full_system_oracle_episode.median
    : 0;
  const simFanoutEpisodeDelta = payload.sim_fanout_available && payload.full_system_available
    ? summaries.sim_fanout_episode.median - summaries.full_system_episode.median
    : 0;
  const probeSolveStepMedian = probeResearchArm.solve_steps_median ?? NaN;
  const baselineSolveStepMedian = baselineResearchArm.solve_steps_median ?? NaN;
  const probeStepsToPeakMedian = researchPeak.probe_steps_to_peak_median ?? NaN;
  const baselineStepsToPeakMedian = researchPeak.baseline_steps_to_peak_median ?? NaN;
  const probeStepSavingsVsBaseline = researchDeltas.probe_step_savings_vs_baseline ?? NaN;
  const probePeakSavingsVsBaseline = researchPeak.probe_steps_to_peak_savings_vs_baseline ?? NaN;
  el("benchmarkCards").innerHTML = [
    readoutRow(
      "Honesty Check",
      honestyHeadline || `strict usage ${strictUsageStatus}`,
      `Matched env-expression delta ${summaries.probe_env_expression_delta.mean.toFixed(2)}, forced diagnostic delta ${summaries.probe_forced_env_expression_delta.mean.toFixed(2)}, dominant blocker ${String(dominantReadinessBlocker[0]).replace(/_/g, " ")}, leaveout stability ${Number(readinessComponentMeans.leaveout_stability || 0).toFixed(2)}.`
    ),
    readoutRow(
      "Episode Comparison",
      episodeDelta >= 0 ? `probe faster by ${episodeDelta.toFixed(1)}` : `baseline faster by ${Math.abs(episodeDelta).toFixed(1)}`,
      "This is the primary downstream comparison now: how many episodes it took to solve."
    ),
    readoutRow(
      "Research Metrics",
      `solve ${formatCompactNumber(probeSolveStepMedian)} · peak ${formatCompactNumber(probeStepsToPeakMedian)}`,
      `Regular baseline solve ${formatCompactNumber(baselineSolveStepMedian)}, peak ${formatCompactNumber(baselineStepsToPeakMedian)}. Probe solve ${formatSampleSavings(probeStepSavingsVsBaseline)} vs baseline; peak ${formatSampleSavings(probePeakSavingsVsBaseline)} vs baseline.`
    ),
    readoutRow(
      "Solve Reliability",
      `${summaries.probe_episode.success_rate}/${summaries.probe_episode.count} vs ${summaries.baseline_episode.success_rate}/${summaries.baseline_episode.count}`,
      "Probe can still be interesting if it solves more reliably, even when it loses on episode speed."
    ),
    readoutRow(
      "Post-Expression Latency",
      `${summaries.probe_post_expression_steps.median.toFixed(1)} control steps median`,
      "This is the controller-only latency after the final env-expression handoff, before probe cost is added back in."
    ),
    readoutRow(
      "Latent Contribution",
      hasNoExpressionSolve
        ? (
          noExprEpisodeDelta >= 0
            ? `${latentContributionLabel} faster by ${noExprEpisodeDelta.toFixed(1)}${strictExpressionActive ? "" : " · strict unused"}`
            : `${noExpressionReferenceLabel} faster by ${Math.abs(noExprEpisodeDelta).toFixed(1)}`
        )
        : "not run",
      hasNoExpressionSolve && strictExpressionActive
        ? "This compares the matched probe run against the same protocol with the env expression muted."
        : hasNoExpressionSolve
        ? "Strict fair expression was not active here, so this compares branches rather than proving learned message contribution."
        : "No matched no-expression solve arm is available in this artifact, so this page will not infer latent contribution from a zero placeholder."
    ),
    readoutRow(
      particleSysidMode ? "Legacy Latent Gate" : "Latent Gate",
      latentWinGateStatus,
      latentWinGateReasons.length
        ? `Still blocked by: ${latentWinGateReasons.join(", ")}.`
        : "This artifact clears the cracked-it gate instead of only looking good on one downstream view."
    ),
    ...(systemId.available ? [
      readoutRow(
        "System-ID Gate",
        `${(100 * Number(systemId.trusted_fraction || 0)).toFixed(1)}% trusted`,
        `Top-1 ${(100 * Number(systemId.validation_top1_median || 0)).toFixed(1)}%, margin ${Number(systemId.validation_margin_median || 0).toFixed(2)}, ESS ${Number(systemId.particle_ess_ratio_median || 0).toFixed(3)}, leaveout ${Number(systemId.particle_leaveout_shift_median || 0).toFixed(3)}.`
      ),
    ] : []),
    readoutRow(
      "Representation Floor",
      particleSysidMode
        ? `sysid ${Number(systemId.progress_median || 0).toFixed(3)} · top1 ${(100 * Number(systemId.validation_top1_median || 0)).toFixed(1)}% · ess ${Number(systemId.particle_ess_ratio_median || 0).toFixed(3)} · leaveout ${Number(systemId.particle_leaveout_shift_median || 0).toFixed(3)}`
        : `bpi ${summaries.belief_progress_index.median.toFixed(3)} · fit ${summaries.latent_mechanics_fit.median.toFixed(2)} · neighbor ${(100 * summaries.latent_neighbor_alignment.median).toFixed(1)}% · paired ${(100 * pairedSplitSummary.median).toFixed(1)}% · cross ${(100 * crossSplitSummary.median).toFixed(1)}%`,
      particleSysidMode
        ? `Trusted ${(100 * Number(systemId.trusted_fraction || 0)).toFixed(1)}%, margin ${Number(systemId.validation_margin_median || 0).toFixed(2)}, entropy ${Number(systemId.particle_entropy_median || 0).toFixed(2)}, subset stability ${Number(systemId.particle_subset_stability_median || 0).toFixed(2)}. Legacy latent gap is still ${summaries.latent_gap_ratio.median.toFixed(2)}.`
        : `Gap ratio ${summaries.latent_gap_ratio.median.toFixed(2)}, held-out probe error ${summaries.latent_heldout_probe_error.median.toFixed(2)}, leakage ${(100 * summaries.latent_probe_leakage.median).toFixed(1)}%, uncertainty/error corr ${summaries.latent_uncert_error_corr.median.toFixed(2)}.`
    ),
    readoutRow(
      "Next Bottleneck",
      nextLatentBottleneck,
      particleSysidMode
        ? `Expression delta ${summaries.probe_env_expression_delta.mean.toFixed(2)}, sysid trusted ${(100 * Number(systemId.trusted_fraction || 0)).toFixed(1)}%, particle leaveout ${Number(systemId.particle_leaveout_shift_median || 0).toFixed(3)}, subset ${Number(systemId.particle_subset_stability_median || 0).toFixed(2)}.`
        : `Expression delta ${summaries.probe_env_expression_delta.mean.toFixed(2)}, ${hasNamedMechanicsSupport ? `mechanics support ${(100 * supportMechanicsShare).toFixed(1)}%` : `center support ${(100 * supportCenterShare).toFixed(1)}%`}, paired retrieval ${(100 * pairedSplitSummary.median).toFixed(1)}%, cross-family retrieval ${(100 * crossSplitSummary.median).toFixed(1)}%, gap ratio ${summaries.latent_gap_ratio.median.toFixed(2)}.`
    ),
    readoutRow(
      "Support Mix",
      hasNamedMechanicsSupport
        ? `mechanics ${(100 * supportMechanicsShare).toFixed(1)}% · passive ${(100 * supportPassiveShare).toFixed(1)}% · stress ${(100 * supportStressShare).toFixed(1)}%`
        : `center ${(100 * supportCenterShare).toFixed(1)}% · directional ${(100 * supportDirectionalShare).toFixed(1)}%`,
      `Support/env ${Number(latentSupport.support_count_mean || 0).toFixed(1)}, split-overlap ${(100 * Number(latentSupport.split_group_overlap_mean || 0)).toFixed(1)}%, top-family share ${(100 * Number(latentSupport.dominant_window_share || 0)).toFixed(1)}%.`
    ),
    readoutRow(
      "Leakage Split",
      `window ${supportWindowLeak.toFixed(2)} · env ${supportEnvLeak.toFixed(2)}`,
      `Nearest-between ${Number(latentSupport.nearest_between_median || 0).toFixed(4)}, pairwise-between ${Number(latentSupport.pairwise_between_mean || 0).toFixed(4)}, norm std ${Number(latentSupport.belief_norm_std || 0).toFixed(4)}.`
    ),
    ...(payload.probe_shadow_available ? [
      readoutRow(
        "Shadow Diagnostic",
        hasNoExpressionSolve
          ? (shadowEpisodeDelta >= 0 ? `shadow faster than no-expression by ${shadowEpisodeDelta.toFixed(1)}` : `no-expression faster than shadow by ${Math.abs(shadowEpisodeDelta).toFixed(1)}`)
          : "no-expression arm not run",
        hasNoExpressionSolve
          ? "This is the non-headline check for whether near-ready latents contain any downstream value before they clear the strict fair gate."
          : "Shadow is available, but the matched muted-expression solve arm is missing, so the episode-speed comparison is withheld."
      ),
    ] : []),
    ...(payload.full_system_available ? [
      ...(payload.full_system_state_only_available ? [
        readoutRow(
          "Belief Over State-Only",
          `${summaries.full_system_state_only_ablation_delta.mean.toFixed(2)} mean return gain`,
          "This is the matched attribution check: same trained checkpoint, but the state-only student path is forced and the belief residual is disabled."
        ),
      ] : []),
      readoutRow(
        `${fullSystemName} Outcome`,
        hasNoExpressionSolve && hasFullSystemSolve
          ? (fullSystemEpisodeDelta >= 0 ? `${fullSystemRef} faster than no-expression by ${fullSystemEpisodeDelta.toFixed(1)}` : `no-expression faster than ${fullSystemRef} by ${Math.abs(fullSystemEpisodeDelta).toFixed(1)}`)
          : `${fullSystemRef} median ${solveSummaryLabel(summaries.full_system_episode)}`,
        hasNoExpressionSolve
          ? (
            isPlannerControllerStyle(fullSystemStyle)
              ? "This is the breakthrough-path comparison: whether the richer controller context beats the matched probe protocol with no controller-side belief input."
              : "This is the breakthrough-path comparison: whether the cheap belief-conditioned controller beats the matched probe protocol with no controller-side belief input."
          )
          : "No matched no-expression solve arm is available, so this reports the controller solve median without claiming a breakthrough-path delta."
      ),
      readoutRow(
        "Belief Dependence",
        `state-only ${summaries.full_system_state_only_ablation_delta.mean.toFixed(2)} · zero ${summaries.full_system_zero_context_ablation_delta.mean.toFixed(2)} · shuffled ${summaries.full_system_shuffled_context_ablation_delta.mean.toFixed(2)} · stale ${summaries.full_system_stale_context_ablation_delta.mean.toFixed(2)} · frozen ${summaries.full_system_frozen_context_ablation_delta.mean.toFixed(2)} · actor ${summaries.full_system_actor_only_ablation_delta.mean.toFixed(2)}`,
        "A real belief-first controller should lose return when the context is removed, permuted, or frozen."
      ),
      readoutRow(
        "Online Refinement",
        `no-refresh ${summaries.full_system_online_refinement_ablation_delta.mean.toFixed(2)} · frozen ${summaries.full_system_frozen_context_ablation_delta.mean.toFixed(2)}`,
        "Positive values here mean the controller benefits from refreshing the belief online; a large frozen drop means stale context identity also matters once refresh can no longer heal it."
      ),
    ] : []),
    ...(payload.full_system_oracle_available ? [
      readoutRow(
        "Oracle Bridge",
        hasNoExpressionSolve && hasFullSystemOracleSolve
          ? (fullSystemOracleEpisodeDelta >= 0 ? `${fullSystemOracleRef} faster than no-expression by ${fullSystemOracleEpisodeDelta.toFixed(1)}` : `no-expression faster than ${fullSystemOracleRef} by ${Math.abs(fullSystemOracleEpisodeDelta).toFixed(1)}`)
          : `${fullSystemOracleRef} median ${solveSummaryLabel(summaries.full_system_oracle_episode)}`,
        hasNoExpressionSolve
          ? "This is the controller-identification gate: if oracle context still loses, the controller/training loop is the blocker before we blame the learned representation."
          : "No matched no-expression solve arm is available, so this reports the oracle-context median without claiming a delta."
      ),
      readoutRow(
        "Oracle Dependence",
        `zero ${summaries.full_system_oracle_zero_context_ablation_delta.mean.toFixed(2)} · shuffled ${summaries.full_system_oracle_shuffled_context_ablation_delta.mean.toFixed(2)} · stale ${summaries.full_system_oracle_stale_context_ablation_delta.mean.toFixed(2)} · actor ${summaries.full_system_oracle_actor_only_ablation_delta.mean.toFixed(2)}`,
        "Oracle context should clearly beat zeroed, shuffled, and stale ablations if the recurrent controller can actually use structured belief."
      ),
    ] : []),
    ...(payload.sim_fanout_available && payload.full_system_available ? [
      readoutRow(
        "Sim Ceiling",
        simFanoutEpisodeDelta >= 0 ? `${fullSystemRef} faster than sim-fanout by ${simFanoutEpisodeDelta.toFixed(1)}` : `sim-fanout faster than ${fullSystemRef} by ${Math.abs(simFanoutEpisodeDelta).toFixed(1)}`,
        "This is the cheap-sim reality check: if direct simulator fan-out wins easily, the learned controller still has not earned its extra machinery."
      ),
    ] : []),
    readoutRow(
      "Step Reality Check",
      stepDelta >= 0 ? `probe cheaper by ${stepDelta.toFixed(1)}` : `baseline cheaper by ${Math.abs(stepDelta).toFixed(1)}`,
      "Environment-step cost stays here as a secondary check so we do not confuse extra probing with real efficiency."
    ),
    readoutRow(
      "Probe Cost Split",
      `encoder ${formatCompactNumber(payload.rows.reduce((sum, row) => sum + Number(row.probe_encoder_steps || 0), 0) / Math.max(payload.rows.length, 1))} · online ${formatCompactNumber(payload.rows.reduce((sum, row) => sum + Number(row.probe_probe_env_steps || 0), 0) / Math.max(payload.rows.length, 1))} · control ${formatCompactNumber(payload.rows.reduce((sum, row) => sum + Number(row.probe_control_env_steps || 0), 0) / Math.max(payload.rows.length, 1))}`,
      "This keeps the crawler, online probing, and downstream control costs separated instead of hiding them in one step total."
    ),
    readoutRow(
      "Expression Usage",
      payload.probe_shadow_available
        ? `strict active ${(100 * summaries.probe_fair_expression_enabled_fraction.mean).toFixed(1)}% · shadow active ${(100 * summaries.probe_shadow_expression_enabled_fraction.mean).toFixed(1)}% · strict miss ${(100 * summaries.probe_shadow_strict_miss_fraction.mean).toFixed(1)}%`
        : `strict active ${(100 * summaries.probe_fair_expression_enabled_fraction.mean).toFixed(1)}% · shadow not run`,
      "This separates strict fair usage from the softer diagnostic shadow gate, so we can see whether shadow is truly selective or just overriding the headline mute policy."
    ),
    readoutRow(
      "Handoff Honesty",
      `ready ${(100 * summaries.probe_fair_ready_handoff_fraction.mean).toFixed(1)}% · strict ${(100 * summaries.probe_fair_expression_enabled_fraction.mean).toFixed(1)}% · ready-but-muted ${(100 * summaries.probe_expression_ready_but_muted_fraction.mean).toFixed(1)}%`,
      "Precision mode should separate the general ready flag from the stricter fair-policy enablement instead of treating them as the same event."
    ),
    readoutRow(
      "Confidence Split",
      `ready ${summaries.probe_fair_ready_confidence_median.median.toFixed(2)} · muted ${summaries.probe_fair_muted_confidence_median.median.toFixed(2)}`,
      "These are the raw controller-side confidence levels on ready handoffs versus handoffs that were force-muted."
    ),
    readoutRow(
      "Readiness Blockers",
      `${String(dominantReadinessBlocker[0]).replace(/_/g, " ")} · fair ${String(dominantFairStopBlocker[0]).replace(/_/g, " ")} · shadow ${String(dominantShadowBlocker[0]).replace(/_/g, " ")}`,
      `Strict component means: future ${Number(readinessComponentMeans.future_probe_quality || 0).toFixed(2)}, subset ${Number(readinessComponentMeans.subset_stability || 0).toFixed(2)}, leaveout ${Number(readinessComponentMeans.leaveout_stability || 0).toFixed(2)}, diversity ${Number(readinessComponentMeans.support_diversity || 0).toFixed(2)}. Live geometry means: subset ${Number(summaries.probe_online_subset_stability_mean?.mean || 0).toFixed(2)}, complete ${(100 * Number(summaries.probe_online_geometry_complete_fraction?.mean || 0)).toFixed(1)}%, split gap ${Number(summaries.probe_online_split_latent_disagreement_mean?.mean || 0).toFixed(3)}, retrieval deficit ${Number(summaries.probe_online_split_retrieval_margin_deficit_mean?.mean || 0).toFixed(3)}.`
    ),
    readoutRow(
      "Second-Probe Signal",
      `raw ${summaries.probe_second_probe_raw_future_gain_mean.mean.toFixed(2)} · floored ${summaries.probe_second_probe_future_estimate_mean.mean.toFixed(2)} · choice ${summaries.probe_second_probe_choice_future_gain_mean.mean.toFixed(2)}`,
      "These are the benchmark-level means for the actually chosen fair probe-two family, so we can see whether the second ranking field is still collapsing toward zero."
    ),
    readoutRow(
      "Probe Selection Regime",
      `coverage ${(100 * summaries.probe_family_coverage_satisfied_fraction.mean).toFixed(1)}% · value-driven ${(100 * summaries.probe_second_probe_value_driven_fraction.mean).toFixed(1)}% · uniformity ${(100 * summaries.probe_uniformity_pressure_active_fraction.mean).toFixed(1)}%`,
      "Coverage should only be a floor. After that, the second probe should look value-driven instead of collapsing toward a flat family rotation."
    ),
  ].join("");

  el("dashboardNote").textContent = benchmarkBehindLive
    ? `Viewing the last completed benchmark artifact from ${formatDateTime(payload.artifact_mtime)} while a newer live run for ${payload.benchmark_tag || "this tag"} is active. The benchmark deck will refresh after that run writes its final summary.`
    : `Viewing benchmark artifact updated ${formatDateTime(payload.artifact_mtime)}. The benchmark section stays intentionally flatter and lower in the hierarchy. If these results move but the latent diagnostics remain muddy, the representation story still is not strong enough yet.`;
}

async function fetchJson(path) {
  const separator = path.includes("?") ? "&" : "?";
  const response = await fetch(`${path}${separator}ts=${Date.now()}`, {cache: "no-store"});
  return response.json();
}

async function loadLatent() {
  const name = el("latentSelect").value;
  if (!name) {
    dashboardState.latentPayload = null;
    renderHero();
    el("latentValidityWrap").innerHTML = "";
    el("latentSummary").innerHTML = "";
    el("latentReadout").innerHTML = `<div class="empty">Run training first so the repo can save a latent snapshot artifact.</div>`;
    el("latentPlotWrap").innerHTML = `<div class="empty">Run training first so the repo can save a latent snapshot artifact.</div>`;
    el("latentLegend").innerHTML = "";
    el("modeTableWrap").innerHTML = `<div class="empty">No probe-mode summary is available yet.</div>`;
    el("latentGuide").innerHTML = `<div class="empty">No latent interpretation is available yet.</div>`;
    el("uncertaintyErrorPlotWrap").innerHTML = `<div class="empty">No uncertainty/error diagnostic points are available yet.</div>`;
    el("gapRatioPlotWrap").innerHTML = `<div class="empty">No same-world gap diagnostics are available yet.</div>`;
    el("uncertaintyFeatureWrap").innerHTML = `<div class="empty">No learned uncertainty feature weights are available yet.</div>`;
    el("outlierGuideWrap").innerHTML = `<div class="empty">No outlier readout is available yet.</div>`;
    el("uncertaintyDistPlotWrap").innerHTML = `<div class="empty">No uncertainty distribution is available yet.</div>`;
    el("beliefScalePlotWrap").innerHTML = `<div class="empty">No belief-scale distribution is available yet.</div>`;
    el("pairwiseDistPlotWrap").innerHTML = `<div class="empty">No pairwise-distance distribution is available yet.</div>`;
    el("retrievalRankPlotWrap").innerHTML = `<div class="empty">No split-rank distribution is available yet.</div>`;
    el("beliefNormPlotWrap").innerHTML = `<div class="empty">No belief-norm distribution is available yet.</div>`;
    el("compressionWrap").innerHTML = `<div class="empty">No env-expression compression summary is available yet.</div>`;
    return;
  }
  renderLatent(await fetchJson(`/api/latent/${encodeURIComponent(name)}`));
  dashboardState.selectedLatentMtime = Number(
    (dashboardState.indexPayload?.latent_snapshot_mtimes || {})[name] || 0
  );
}

async function loadBenchmark() {
  const name = el("benchmarkSelect").value;
  if (!name) {
    dashboardState.benchmarkPayload = null;
    renderHero();
    el("benchmarkSummary").innerHTML = "";
    el("seedGrid").innerHTML = `<div class="empty">Run a benchmark first so the repo can save a summary artifact.</div>`;
    el("probeStopWrap").innerHTML = `<div class="empty">No probe-stop summary is available yet.</div>`;
    el("probeFamilyWrap").innerHTML = `<div class="empty">No per-family probe diagnostics are available yet.</div>`;
    el("benchmarkCards").innerHTML = "";
    el("dashboardNote").textContent = "";
    return;
  }
  renderBenchmark(await fetchJson(`/api/benchmark/${encodeURIComponent(name)}`));
  dashboardState.selectedBenchmarkMtime = Number(
    (dashboardState.indexPayload?.benchmark_summary_mtimes || {})[name] || 0
  );
}

async function loadLive() {
  renderLive(await fetchJson("/api/live"));
}

async function clearComparisonArchive() {
  await fetch("/api/live/history/clear", {method: "POST", cache: "no-store"});
  dashboardState.liveArchiveSelection = "live";
  await loadLive();
}

async function refreshArtifacts() {
  const payload = await fetchJson("/api/index");
  dashboardState.indexPayload = payload;
  setSelectOptions(el("latentSelect"), payload.latent_snapshots);
  setSelectOptions(el("benchmarkSelect"), payload.benchmark_summaries, "No matching benchmark yet");
  const context = payload.run_context || {};
  const desiredLatent = context.default_latent_snapshot || "";
  const desiredBenchmark = context.default_benchmark_summary || "";
  const latentChanged = desiredLatent && desiredLatent !== el("latentSelect").value;
  const benchmarkChanged = desiredBenchmark && desiredBenchmark !== el("benchmarkSelect").value;
  if (latentChanged) {
    el("latentSelect").value = desiredLatent;
    await loadLatent();
  }
  if (benchmarkChanged) {
    el("benchmarkSelect").value = desiredBenchmark;
    await loadBenchmark();
  }
  const currentLatent = el("latentSelect").value;
  const currentBenchmark = el("benchmarkSelect").value;
  const latestLatentMtime = Number((payload.latent_snapshot_mtimes || {})[currentLatent] || 0);
  const latestBenchmarkMtime = Number((payload.benchmark_summary_mtimes || {})[currentBenchmark] || 0);
  if (
    currentLatent &&
    !latentChanged &&
    latestLatentMtime > Number(dashboardState.selectedLatentMtime || 0)
  ) {
    await loadLatent();
  }
  if (
    currentBenchmark &&
    !benchmarkChanged &&
    latestBenchmarkMtime > Number(dashboardState.selectedBenchmarkMtime || 0)
  ) {
    await loadBenchmark();
  }
}

async function boot() {
  const payload = await fetchJson("/api/index");
  dashboardState.indexPayload = payload;
  setSelectOptions(el("latentSelect"), payload.latent_snapshots);
  setSelectOptions(el("benchmarkSelect"), payload.benchmark_summaries, "No matching benchmark yet");
  const context = payload.run_context || {};
  setPreferredSelection(
    el("latentSelect"),
    context.default_latent_snapshot,
    context.benchmark_tag ? `${context.benchmark_tag}_seed_` : "",
  );
  setPreferredSelection(
    el("benchmarkSelect"),
    context.default_benchmark_summary,
    context.benchmark_tag || "",
  );
  document.querySelectorAll(".deck-button").forEach((button) => {
    button.addEventListener("click", () => setActiveDeck(button.dataset.deckTarget));
  });
  el("latentSelect").addEventListener("change", loadLatent);
  el("benchmarkSelect").addEventListener("change", loadBenchmark);
  el("colorSelect").addEventListener("change", loadLatent);
  el("comparisonKeepArchive").addEventListener("click", () => {
    dashboardState.comparisonArchiveMode = "keep";
    renderPaperFigureBoard(dashboardState.livePayload);
    renderComparisonBoard(dashboardState.livePayload);
  });
  el("comparisonCurrentSuite").addEventListener("click", () => {
    dashboardState.comparisonArchiveMode = "current";
    renderPaperFigureBoard(dashboardState.livePayload);
    renderComparisonBoard(dashboardState.livePayload);
  });
  el("comparisonClearArchive").addEventListener("click", clearComparisonArchive);
  el("liveArchiveWrap").addEventListener("click", (event) => {
    const card = event.target.closest("[data-trace-selection]");
    if (!card) {
      return;
    }
    dashboardState.liveArchiveSelection = card.dataset.traceSelection || "live";
    renderLive();
  });
  window.addEventListener("resize", drawLiveTheater);
  restoreActiveDeck();
  renderHero();
  startLiveAnimationLoop();
  await loadLive();
  await loadLatent();
  await loadBenchmark();
  window.setInterval(loadLive, 650);
  window.setInterval(refreshArtifacts, 2000);
}

boot();
