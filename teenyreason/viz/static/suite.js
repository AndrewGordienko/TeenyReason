(function () {
  const domainOrder = ["cartpole", "language", "image", "board"];
  const domainFallbackTitles = {
    cartpole: "CartPole RL",
    language: "Shakespeare LM",
    image: "MNIST Vision",
    board: "Tic-Tac-Toe Rules",
  };

  function el(id) {
    return document.getElementById(id);
  }

  function number(value, digits) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "0";
    }
    if (Math.abs(numeric) >= 1000) {
      return numeric.toLocaleString(undefined, {maximumFractionDigits: 0});
    }
    return numeric.toFixed(digits);
  }

  function optionalNumber(value, digits) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) {
      return "n/a";
    }
    return number(numeric, digits);
  }

  function metricValue(metric) {
    if (!metric || typeof metric !== "object") {
      return "0";
    }
    const name = String(metric.name || "");
    const digits = name.includes("accuracy") || name.includes("bpc") ? 3 : 1;
    return number(metric.value, digits);
  }

  function rowsForSpark(domain) {
    const rows = Array.isArray(domain.rows) ? domain.rows : [];
    if (domain.domain === "language") {
      return rows.map((row) => Number(row.belief_bpc || row.probe_bpc || 0));
    }
    if (domain.domain === "image") {
      return rows.map((row) => Number(row.belief_accuracy || row.probe_accuracy || 0));
    }
    if (domain.domain === "board") {
      return rows.map((row) => Number(row.belief_move_accuracy || 0));
    }
    return rows.map((row) => Number(row.probe_solve_episode || row.probe_solve_env_steps || 0));
  }

  function sparkline(values) {
    const safe = values.filter((value) => Number.isFinite(value));
    if (safe.length < 2) {
      return "";
    }
    const min = Math.min(...safe);
    const max = Math.max(...safe);
    const span = Math.max(max - min, 1e-6);
    const points = safe.map((value, index) => {
      const x = 4 + (index / Math.max(safe.length - 1, 1)) * 92;
      const y = 36 - ((value - min) / span) * 28;
      return `${x.toFixed(2)},${y.toFixed(2)}`;
    });
    return `<svg class="suite-spark" viewBox="0 0 100 42" preserveAspectRatio="none">
      <polyline points="${points.join(" ")}" fill="none" stroke="#2854d8" stroke-width="2" vector-effect="non-scaling-stroke"></polyline>
    </svg>`;
  }

  function renderLane(domainName, domain) {
    const active = domain && Object.keys(domain).length > 0;
    const title = active ? domain.title || domainFallbackTitles[domainName] : domainFallbackTitles[domainName];
    const metric = active ? domain.headline_metric || {} : {};
    const metricName = String(metric.name || "metric").replaceAll("_", " ");
    return `<article class="suite-lane">
      <div class="suite-lane-title">
        <h3>${title}</h3>
        <span class="suite-status">${active ? domain.readiness || "loaded" : "idle"}</span>
      </div>
      <div class="suite-main-number">${metricValue(metric)}</div>
      <div class="suite-main-label">${metricName}</div>
      <div class="suite-mini-grid">
        <div class="suite-mini"><div class="k">Solver Gain</div><div class="v">${number(active ? domain.belief_contribution_margin : 0, 3)}</div></div>
        <div class="suite-mini"><div class="k">Ablation Gap</div><div class="v">${number(active ? domain.ablation_gap : 0, 3)}</div></div>
        <div class="suite-mini"><div class="k">Trust</div><div class="v">${number(active ? domain.trust : 0, 3)}</div></div>
        <div class="suite-mini"><div class="k">Evidence Cost</div><div class="v">${number(active ? domain.evidence_cost : 0, 0)}</div></div>
      </div>
      ${sparkline(active ? rowsForSpark(domain) : [])}
    </article>`;
  }

  function renderCrossTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-table-row"><div>No suite metrics yet</div></div>`;
    }
    const body = rows.map((row) => `<div class="suite-table-row">
      <div>${row.domain || ""}</div>
      <div>${number(row.subset_consistency, 3)}</div>
      <div>${number(row.solver_gain, 3)}</div>
      <div>${number(row.content_lift, 3)}</div>
      <div>${number(row.belief_bitrate, 0)}</div>
    </div>`).join("");
    return `<div class="suite-table-row suite-table-head">
      <div>Domain</div><div>Subset</div><div>Gain</div><div>Content</div><div>Bits</div>
    </div>${body}`;
  }

  function renderCausalTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-causal-row"><div>No causal message checks yet</div></div>`;
    }
    const body = rows.map((row) => {
      const causal = row.content_causal ? "pass" : "blocked";
      return `<div class="suite-causal-row">
        <div>${row.domain || ""}</div>
        <div>${number(row.learned, 3)}</div>
        <div>${number(row.zero, 3)}</div>
        <div>${number(row.shuffled, 3)}</div>
        <div>${number(row.stale, 3)}</div>
        <div>${number(row.content_lift, 3)}</div>
        <div><span class="suite-gate-state ${causal}">${row.content_causal ? "causal" : "blocked"}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-causal-row suite-table-head">
      <div>Domain</div><div>Learned</div><div>Zero</div><div>Shuffle</div><div>Stale</div><div>Lift</div><div>Use</div>
    </div>${body}`;
  }

  function renderMechanismTable(rows) {
    const activeRows = Array.isArray(rows) ? rows.filter((row) => row.hidden_target) : [];
    if (activeRows.length === 0) {
      return `<div class="suite-mechanism-row"><div>No controlled hidden-target checks yet</div></div>`;
    }
    const body = activeRows.map((row) => {
      const causal = row.content_causal ? "pass" : "blocked";
      return `<div class="suite-mechanism-row">
        <div>${row.domain || ""}</div>
        <div>${row.hidden_rule || ""}</div>
        <div>${row.decoded_rule || ""}</div>
        <div>${number(row.decode_accuracy, 3)}</div>
        <div>${number(row.subset_agreement, 3)}</div>
        <div>${number(row.baseline_accuracy, 3)}</div>
        <div>${number(row.belief_accuracy, 3)}</div>
        <div>${number(row.content_lift, 3)}</div>
        <div><span class="suite-gate-state ${causal}">${row.content_causal ? "causal" : "blocked"}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-mechanism-row suite-table-head">
      <div>Domain</div><div>Hidden</div><div>Decoded</div><div>Decode</div><div>Subset</div><div>Base</div><div>Belief</div><div>Lift</div><div>Use</div>
    </div>${body}`;
  }

  function renderTransferTable(rows) {
    const activeRows = Array.isArray(rows) ? rows.filter((row) => row.hidden_target) : [];
    if (activeRows.length === 0) {
      return `<div class="suite-transfer-row"><div>No mechanism-transfer checks yet</div></div>`;
    }
    const body = activeRows.map((row) => {
      const causal = row.real_causal ? "pass" : "blocked";
      return `<div class="suite-transfer-row">
        <div>${row.domain || ""}</div>
        <div>${number(row.decode_accuracy, 3)}</div>
        <div>${number(row.subset_agreement, 3)}</div>
        <div>${number(row.mechanism_content_lift, 3)}</div>
        <div>${number(row.bridge_content_lift, 3)}</div>
        <div>${number(row.real_content_lift, 3)}</div>
        <div>${number(row.bridge_to_real_gap, 3)}</div>
        <div><span class="suite-gate-state ${causal}">${row.real_causal ? "real" : "gap"}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-transfer-row suite-table-head">
      <div>Domain</div><div>Decode</div><div>Subset</div><div>Mechanism</div><div>Bridge</div><div>Real</div><div>Drop</div><div>State</div>
    </div>${body}`;
  }

  function renderHandoffTable(rows) {
    const activeRows = Array.isArray(rows) ? rows.filter((row) => row.hidden_target) : [];
    if (activeRows.length === 0) {
      return `<div class="suite-handoff-row"><div>No latent handoff economics yet</div></div>`;
    }
    const body = activeRows.map((row) => `<div class="suite-handoff-row">
      <div>${row.domain || ""}</div>
      <div>${number(row.cheap_decode_accuracy, 3)}</div>
      <div>${number(row.cheap_content_lift, 3)}</div>
      <div>${number(row.action_change_fraction, 3)}</div>
      <div>${number(row.value_delta_correct_vs_shuffled, 3)}</div>
      <div>${number(row.cheap_dedicated_probe_steps, 0)}</div>
      <div>${number(row.expensive_dedicated_probe_steps, 0)}</div>
      <div>${number(row.dual_use_probe_fraction, 3)}</div>
    </div>`).join("");
    return `<div class="suite-handoff-row suite-table-head">
      <div>Domain</div><div>Cheap Decode</div><div>Lift</div><div>Action Change</div><div>Value Delta</div><div>Cheap Probe</div><div>Old Probe</div><div>Dual Use</div>
    </div>${body}`;
  }

  function renderUtilityTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-utility-row"><div>No latent utility rows yet</div></div>`;
    }
    const body = rows.map((row) => `<div class="suite-utility-row">
      <div>${row.domain || ""}</div>
      <div>${number(row.real_gain, 3)}</div>
      <div>${number(row.real_content_lift, 3)}</div>
      <div>${number(row.bridge_to_real_gap, 3)}</div>
      <div>${number(row.low_budget_gain, 3)}</div>
      <div>${number(row.high_budget_gain, 3)}</div>
      <div>${number(row.budget_gate_mean_gain, 3)}</div>
      <div>${number(row.gain_per_1k_bits, 3)}</div>
      <div>${row.bottleneck || ""}</div>
    </div>`).join("");
    return `<div class="suite-utility-row suite-table-head">
      <div>Domain</div><div>Real Gain</div><div>Real Lift</div><div>Bridge Drop</div><div>Low</div><div>High</div><div>Gate Mean</div><div>Gain/kbit</div><div>Bottleneck</div>
    </div>${body}`;
  }

  function renderWakeTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-wake-row"><div>No wake-up gate rows yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.wake_expensive_probe ? "blocked" : "pass";
      return `<div class="suite-wake-row">
        <div>${row.domain || ""}</div>
        <div><span class="suite-gate-state ${state}">${row.wake_expensive_probe ? "wake" : "sleep"}</span></div>
        <div>${row.reason || ""}</div>
        <div>${row.confidence_low ? "yes" : "no"}</div>
        <div>${row.bridge_gap_high ? "yes" : "no"}</div>
        <div>${row.high_budget_regression ? "yes" : "no"}</div>
        <div>${number(row.fallback_probe_roi, 4)}</div>
      </div>`;
    }).join("");
    return `<div class="suite-wake-row suite-table-head">
      <div>Domain</div><div>Gate</div><div>Reason</div><div>Conf</div><div>Gap</div><div>Regress</div><div>ROI</div>
    </div>${body}`;
  }

  function renderWorldTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-world-row"><div>No world-understanding diagnostics yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.passes_target ? "pass" : "blocked";
      return `<div class="suite-world-row">
        <div>${row.domain || ""}</div>
        <div>${number(row.score, 3)}</div>
        <div><span class="suite-gate-state ${state}">${row.passes_target ? "pass" : "blocked"}</span></div>
        <div>${number(row.factor_decode, 3)}</div>
        <div>${number(row.counterfactual, 3)}</div>
        <div>${number(row.intervention_lift, 3)}</div>
        <div>${number(row.compression, 3)}</div>
        <div>${number(row.transfer, 3)}</div>
        <div>${row.verdict || ""}</div>
        <div>${row.next_test || ""}</div>
      </div>`;
    }).join("");
    return `<div class="suite-world-row suite-table-head">
      <div>Domain</div><div>Score</div><div>Gate</div><div>Factor</div><div>Counter</div><div>Intervene</div><div>Compress</div><div>Transfer</div><div>Verdict</div><div>Next</div>
    </div>${body}`;
  }

  function renderBeliefHandoffTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-belief-handoff-row"><div>No shared handoff contract yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.claim_allowed ? "pass" : "blocked";
      return `<div class="suite-belief-handoff-row">
        <div>${row.domain || ""}</div>
        <div><span class="suite-gate-state ${state}">${row.claim_allowed ? "claim" : "blocked"}</span></div>
        <div>${row.all_ablation_arms_pass ? "yes" : "no"}</div>
        <div>${row.positive_gain_per_sample ? "yes" : "no"}</div>
        <div>${number(row.gain_per_sample, 4)}</div>
        <div>${number(row.net_sample_savings, 3)}</div>
        <div>${number(row.compression_bits, 0)}</div>
        <div>${number(row.counterfactual_score, 3)}</div>
        <div>${row.failure_reasons || ""}</div>
      </div>`;
    }).join("");
    return `<div class="suite-belief-handoff-row suite-table-head">
      <div>Domain</div><div>Claim</div><div>Abl</div><div>Gain/Sample</div><div>Gain/S</div><div>Net Samples</div><div>Bits</div><div>Counter</div><div>Why Blocked</div>
    </div>${body}`;
  }

  function renderRateTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-rate-row"><div>No rate-distortion rows yet</div></div>`;
    }
    const body = rows.map((row) => `<div class="suite-rate-row">
      <div>${row.domain || ""}</div>
      <div>${number(row.current_bits, 0)}</div>
      <div>${number(row.current_content_lift, 4)}</div>
      <div>${number(row.current_lift_per_1k_bits, 4)}</div>
      <div>${number(row.current_transfer_retained, 3)}</div>
      <div>${number(row.target_bits, 0)}</div>
      <div>${number(row.target_content_lift, 4)}</div>
      <div>${row.next_test || ""}</div>
    </div>`).join("");
    return `<div class="suite-rate-row suite-table-head">
      <div>Domain</div><div>Bits</div><div>Lift</div><div>Lift/kbit</div><div>Retain</div><div>Target Bits</div><div>Target Lift</div><div>Next</div>
    </div>${body}`;
  }

  function renderHandoffRepairTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-handoff-repair-row"><div>No handoff repair rows yet</div></div>`;
    }
    const body = rows.map((row) => `<div class="suite-handoff-repair-row">
      <div>${row.domain || ""}</div>
      <div>${row.current_arm || ""}</div>
      <div>${row.best_arm || ""}</div>
      <div>${number(row.best_gain, 4)}</div>
      <div>${number(row.best_content_lift, 4)}</div>
      <div>${number(row.best_gain_per_cost, 6)}</div>
      <div>${number(row.decision_delta_correct_vs_shuffled, 4)}</div>
      <div>${number(row.decision_delta_correct_vs_best_ablation, 4)}</div>
      <div>${number(row.fallback_probe_roi, 6)}</div>
      <div>${row.blocker || ""}</div>
      <div>${row.next_action || ""}</div>
    </div>`).join("");
    return `<div class="suite-handoff-repair-row suite-table-head">
      <div>Domain</div><div>Current</div><div>Best</div><div>Gain</div><div>Lift</div><div>Gain/Cost</div><div>ShufΔ</div><div>BestΔ</div><div>ROI</div><div>Blocker</div><div>Next</div>
    </div>${body}`;
  }

  function renderDecisionUtilityTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-decision-utility-row"><div>No decision-utility rows yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.decision === "use_belief" ? "pass" : "blocked";
      return `<div class="suite-decision-utility-row">
        <div>${row.domain || ""}</div>
        <div><span class="suite-gate-state ${state}">${row.decision || ""}</span></div>
        <div>${number(row.raw_solver_gain, 4)}</div>
        <div>${number(row.gated_solver_gain, 4)}</div>
        <div>${number(row.accepted_fraction, 3)}</div>
        <div>${number(row.gain_when_accepted, 4)}</div>
        <div>${number(row.decision_delta_correct_vs_shuffled, 4)}</div>
        <div>${number(row.avoided_harm, 4)}</div>
        <div>${number(row.net_utility, 4)}</div>
        <div>${row.next_action || ""}</div>
      </div>`;
    }).join("");
    return `<div class="suite-decision-utility-row suite-table-head">
      <div>Domain</div><div>Decision</div><div>Raw Gain</div><div>Gated Gain</div><div>Accept</div><div>Gain/Accept</div><div>Delta</div><div>Avoided</div><div>Net</div><div>Next</div>
    </div>${body}`;
  }

  function renderDecisionLocalTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-decision-local-row"><div>No decision-local belief rows yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.state === "decision_local_belief_ready" ? "pass" : "blocked";
      return `<div class="suite-decision-local-row">
        <div>${row.domain || ""}</div>
        <div><span class="suite-gate-state ${state}">${row.state || ""}</span></div>
        <div>${number(row.score, 3)}</div>
        <div>${number(row.accepted_fraction, 3)}</div>
        <div>${number(row.mean_decision_delta, 4)}</div>
        <div>${number(row.positive_delta_fraction, 3)}</div>
        <div>${number(row.counterfactual_score, 3)}</div>
        <div>${number(row.action_sensitivity, 3)}</div>
        <div>${number(row.useful_bits, 0)}</div>
        <div>${number(row.utility_per_1k_bits, 4)}</div>
        <div>${number(row.value_of_information, 4)}</div>
        <div>${row.blocker || ""}</div>
        <div>${row.next_action || ""}</div>
      </div>`;
    }).join("");
    return `<div class="suite-decision-local-row suite-table-head">
      <div>Domain</div><div>State</div><div>Score</div><div>Accept</div><div>Delta</div><div>PosΔ</div><div>Counter</div><div>Sens</div><div>Bits</div><div>Util/kbit</div><div>VOI</div><div>Blocker</div><div>Next</div>
    </div>${body}`;
  }

  function renderDecisionLocalCrawlerTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-decision-crawler-row"><div>No decision-local crawler rows yet</div></div>`;
    }
    const activeRows = rows.filter((row) => row.hidden_target);
    if (activeRows.length === 0) {
      return `<div class="suite-decision-crawler-row"><div>No decision-local crawler rows yet</div></div>`;
    }
    const body = activeRows.map((row) => {
      const verdict = String(row.verdict || "");
      const state = verdict.includes("wins") ? "pass" : "blocked";
      return `<div class="suite-decision-crawler-row">
        <div>${row.domain || ""}</div>
        <div>${row.modality || ""}</div>
        <div>${number(row.baseline_decision_score, 4)}</div>
        <div>${number(row.crawler_decision_score, 4)}</div>
        <div>${number(row.regret_reduction, 4)}</div>
        <div>${number(row.content_lift, 4)}</div>
        <div>${number(row.voi, 6)}</div>
        <div>${number(row.intervention_cost, 1)}</div>
        <div>${number(row.net_sample_savings, 4)}</div>
        <div>${number(row.entropy_reduction, 4)}</div>
        <div>${number(row.claim_allowed_rate, 3)}</div>
        <div><span class="suite-gate-state ${state}">${verdict}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-decision-crawler-row suite-table-head">
      <div>Domain</div><div>Modality</div><div>Base</div><div>Crawler</div><div>Regret Δ</div><div>Lift</div><div>VOI</div><div>Cost</div><div>Net</div><div>Entropy Δ</div><div>Claim</div><div>Verdict</div>
    </div>${body}`;
  }

  function renderAffordanceCrawlerTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-affordance-row"><div>No persistent affordance rows yet</div></div>`;
    }
    const activeRows = rows.filter((row) => row.hidden_target);
    if (activeRows.length === 0) {
      return `<div class="suite-affordance-row"><div>No persistent affordance rows yet</div></div>`;
    }
    const body = activeRows.map((row) => {
      const verdict = String(row.verdict || "");
      const state = verdict.includes("positive") ? "pass" : "blocked";
      return `<div class="suite-affordance-row">
        <div>${row.domain || ""}</div>
        <div>${number(row.reuse_horizon, 0)}</div>
        <div>${number(row.baseline_decision_score, 4)}</div>
        <div>${number(row.affordance_decision_score, 4)}</div>
        <div>${number(row.regret_reduction, 4)}</div>
        <div>${number(row.total_probe_cost, 2)}</div>
        <div>${number(row.amortized_probe_cost, 4)}</div>
        <div>${number(row.net_value_after_reuse, 4)}</div>
        <div>${optionalNumber(row.break_even_reuse_count, 2)}</div>
        <div>${number(row.passive_update_fraction, 3)}</div>
        <div>${number(row.dedicated_probe_fraction, 3)}</div>
        <div>${number(row.surprise_mean, 4)}</div>
        <div><span class="suite-gate-state ${state}">${verdict}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-affordance-row suite-table-head">
      <div>Domain</div><div>Reuse</div><div>Base</div><div>Afford</div><div>Regret Δ</div><div>Cost</div><div>Amort</div><div>Net</div><div>Break Even</div><div>Passive</div><div>Dedicated</div><div>Surprise</div><div>Verdict</div>
    </div>${body}`;
  }

  function renderSamplePerformanceTable(rows) {
    if (!Array.isArray(rows) || rows.length === 0) {
      return `<div class="suite-sample-row"><div>No sample-performance rows yet</div></div>`;
    }
    const body = rows.map((row) => {
      const state = row.state && String(row.state).includes("win") ? "pass" : "blocked";
      const solveAvailable = Boolean(row.solve_available);
      return `<div class="suite-sample-row">
        <div>${row.domain || ""}</div>
        <div>${row.sample_axis || ""}</div>
        <div>${optionalNumber(row.baseline_samples_to_peak, 0)}</div>
        <div>${optionalNumber(row.crawler_samples_to_peak, 0)}</div>
        <div>${optionalNumber(row.peak_sample_savings, 0)}</div>
        <div>${optionalNumber(row.peak_score_delta, 4)}</div>
        <div>${solveAvailable ? optionalNumber(row.baseline_samples_to_solve, 0) : "n/a"}</div>
        <div>${solveAvailable ? optionalNumber(row.crawler_samples_to_solve, 0) : "n/a"}</div>
        <div>${solveAvailable ? optionalNumber(row.solve_sample_savings, 0) : "n/a"}</div>
        <div>${optionalNumber(row.best_solver_gain, 4)}</div>
        <div><span class="suite-gate-state ${state}">${row.state || ""}</span></div>
        <div>${row.next_action || ""}</div>
      </div>`;
    }).join("");
    return `<div class="suite-sample-row suite-table-head">
      <div>Domain</div><div>Axis</div><div>Base Peak</div><div>Crawler Peak</div><div>Peak Δ</div><div>Score Δ</div><div>Base Solve</div><div>Crawler Solve</div><div>Solve Δ</div><div>Best Gain</div><div>State</div><div>Next</div>
    </div>${body}`;
  }

  function renderPlannerComparisonTable(rows) {
    const activeRows = Array.isArray(rows) ? rows.filter((row) => row.profile) : [];
    if (activeRows.length === 0) {
      return `<div class="suite-planner-comparison-row"><div>No planner-comparison rows yet</div></div>`;
    }
    const body = activeRows.map((row) => {
      const verdict = String(row.verdict || "");
      const state = verdict.includes("wins") ? "pass" : "blocked";
      return `<div class="suite-planner-comparison-row">
        <div>${row.domain || ""}</div>
        <div>${row.profile || ""}</div>
        <div>${optionalNumber(row.ppo_samples_to_solve, 0)}</div>
        <div>${optionalNumber(row.no_belief_mpc_samples_to_solve, 0)}</div>
        <div>${optionalNumber(row.crawler_belief_mpc_samples_to_solve, 0)}</div>
        <div>${optionalNumber(row.persistent_affordance_samples_to_solve, 0)}</div>
        <div>${optionalNumber(row.persistent_affordance_amortized_samples_to_solve, 1)}</div>
        <div>${optionalNumber(row.oracle_mpc_samples_to_solve, 0)}</div>
        <div>${optionalNumber(row.crawler_vs_ppo_sample_savings, 0)}</div>
        <div>${optionalNumber(row.crawler_vs_no_belief_mpc_sample_savings, 0)}</div>
        <div>${optionalNumber(row.persistent_affordance_amortized_vs_no_belief_mpc_sample_savings, 1)}</div>
        <div>${number(row.planner_return_gain, 4)}</div>
        <div>${number(row.action_regret_reduction, 4)}</div>
        <div>${number(row.persistent_affordance_regret_reduction, 4)}</div>
        <div>${number(row.probe_roi, 6)}</div>
        <div>${number(row.persistent_affordance_probe_roi, 6)}</div>
        <div>${number(row.belief_beats_no_belief_fraction, 3)}</div>
        <div>${number(row.belief_beats_all_ablation_fraction, 3)}</div>
        <div><span class="suite-gate-state ${state}">${verdict}</span></div>
      </div>`;
    }).join("");
    return `<div class="suite-planner-comparison-row suite-table-head">
      <div>Domain</div><div>Profile</div><div>PPO Solve</div><div>No-Belief Solve</div><div>Crawler Solve</div><div>Persist Solve</div><div>Persist Amort</div><div>Oracle Solve</div><div>vs PPO</div><div>vs MPC</div><div>Persist vs MPC</div><div>Return Gain</div><div>Regret Δ</div><div>Persist Regret</div><div>ROI</div><div>Persist ROI</div><div>Beat No</div><div>Beat Abl</div><div>Verdict</div>
    </div>${body}`;
  }

  function renderAcceptance(acceptance) {
    const entries = Object.entries(acceptance || {});
    if (entries.length === 0) {
      return `<div class="suite-gate"><span>No acceptance gates yet</span><span class="suite-gate-state blocked">idle</span></div>`;
    }
    return entries.map(([key, value]) => {
      const state = value ? "pass" : "blocked";
      return `<div class="suite-gate">
        <span>${key.replaceAll("_", " ")}</span>
        <span class="suite-gate-state ${state}">${value ? "pass" : "blocked"}</span>
      </div>`;
    }).join("");
  }

  async function fetchJson(path) {
    const response = await fetch(`${path}?ts=${Date.now()}`, {cache: "no-store"});
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    return response.json();
  }

  async function loadSuite() {
    const lanes = el("suiteLanes");
    const runLabel = el("suiteRunLabel");
    const cross = el("suiteCrossTable");
    const causal = el("suiteCausalityTable");
    const mechanism = el("suiteMechanismTable");
    const transfer = el("suiteTransferTable");
    const handoff = el("suiteHandoffTable");
    const utility = el("suiteUtilityTable");
    const wake = el("suiteWakeTable");
    const world = el("suiteWorldTable");
    const beliefHandoff = el("suiteBeliefHandoffTable");
    const rate = el("suiteRateTable");
    const handoffRepair = el("suiteHandoffRepairTable");
    const decisionUtility = el("suiteDecisionUtilityTable");
    const decisionLocal = el("suiteDecisionLocalTable");
    const decisionCrawler = el("suiteDecisionLocalCrawlerTable");
    const affordanceCrawler = el("suiteAffordanceCrawlerTable");
    const samplePerformance = el("suiteSamplePerformanceTable");
    const plannerComparison = el("suitePlannerComparisonTable");
    const gates = el("suiteAcceptance");
    if (!lanes || !runLabel || !cross || !causal || !mechanism || !transfer || !handoff || !utility || !wake || !world || !beliefHandoff || !rate || !handoffRepair || !decisionUtility || !decisionLocal || !decisionCrawler || !affordanceCrawler || !samplePerformance || !plannerComparison || !gates) {
      return;
    }
    try {
      const payload = await fetchJson("/api/suite/latest");
      const domains = payload.domains || {};
      lanes.innerHTML = domainOrder.map((name) => renderLane(name, domains[name] || {domain: name})).join("");
      runLabel.textContent = payload.available ? payload.run_id || payload.artifact_name || "suite loaded" : "No suite artifact yet";
      const crossDomain = payload.cross_domain || {};
      cross.innerHTML = renderCrossTable(crossDomain.metric_rows || []);
      causal.innerHTML = renderCausalTable(crossDomain.causal_rows || []);
      mechanism.innerHTML = renderMechanismTable(crossDomain.mechanism_rows || []);
      transfer.innerHTML = renderTransferTable(crossDomain.transfer_rows || []);
      handoff.innerHTML = renderHandoffTable(crossDomain.handoff_rows || []);
      utility.innerHTML = renderUtilityTable(crossDomain.latent_utility_rows || []);
      wake.innerHTML = renderWakeTable(crossDomain.wake_up_rows || []);
      world.innerHTML = renderWorldTable(crossDomain.world_understanding_rows || []);
      beliefHandoff.innerHTML = renderBeliefHandoffTable(crossDomain.belief_handoff_rows || []);
      rate.innerHTML = renderRateTable(crossDomain.rate_distortion_rows || []);
      handoffRepair.innerHTML = renderHandoffRepairTable(crossDomain.handoff_repair_rows || []);
      decisionUtility.innerHTML = renderDecisionUtilityTable(crossDomain.decision_utility_rows || []);
      decisionLocal.innerHTML = renderDecisionLocalTable(crossDomain.decision_local_rows || []);
      decisionCrawler.innerHTML = renderDecisionLocalCrawlerTable(crossDomain.decision_local_crawler_rows || []);
      affordanceCrawler.innerHTML = renderAffordanceCrawlerTable(crossDomain.affordance_crawler_rows || []);
      samplePerformance.innerHTML = renderSamplePerformanceTable(crossDomain.sample_performance_rows || []);
      plannerComparison.innerHTML = renderPlannerComparisonTable(crossDomain.planner_comparison_rows || []);
      gates.innerHTML = renderAcceptance(crossDomain.acceptance || {});
    } catch (error) {
      lanes.innerHTML = domainOrder.map((name) => renderLane(name, {domain: name})).join("");
      runLabel.textContent = "Suite endpoint unavailable";
      cross.innerHTML = renderCrossTable([]);
      causal.innerHTML = renderCausalTable([]);
      mechanism.innerHTML = renderMechanismTable([]);
      transfer.innerHTML = renderTransferTable([]);
      handoff.innerHTML = renderHandoffTable([]);
      utility.innerHTML = renderUtilityTable([]);
      wake.innerHTML = renderWakeTable([]);
      world.innerHTML = renderWorldTable([]);
      beliefHandoff.innerHTML = renderBeliefHandoffTable([]);
      rate.innerHTML = renderRateTable([]);
      handoffRepair.innerHTML = renderHandoffRepairTable([]);
      decisionUtility.innerHTML = renderDecisionUtilityTable([]);
      decisionLocal.innerHTML = renderDecisionLocalTable([]);
      decisionCrawler.innerHTML = renderDecisionLocalCrawlerTable([]);
      affordanceCrawler.innerHTML = renderAffordanceCrawlerTable([]);
      samplePerformance.innerHTML = renderSamplePerformanceTable([]);
      plannerComparison.innerHTML = renderPlannerComparisonTable([]);
      gates.innerHTML = renderAcceptance({});
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadSuite();
    window.setInterval(loadSuite, 5000);
  });
})();
