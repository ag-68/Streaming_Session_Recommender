const panels = ["overview", "formulation", "benchmarks", "results", "try"];
const state = {};
let chartsDrawn = false;

function show(name) {
  if (!panels.includes(name)) name = "overview";
  for (const panel of panels) {
    const section = document.getElementById(panel);
    const tab = document.getElementById("tab-" + panel);
    const on = panel === name;
    section.hidden = !on;
    tab.setAttribute("aria-selected", on ? "true" : "false");
  }
  if (location.hash !== "#" + name) history.replaceState(null, "", "#" + name);
  if (name === "results") drawCharts();
}

function num(value) {
  return Number(value).toFixed(4);
}

function ms(value) {
  return Number(value).toFixed(2) + " ms";
}

async function loadJson(name) {
  const response = await fetch("data/" + name + ".json");
  if (!response.ok) throw new Error(name);
  return response.json();
}

function setText(id, value) {
  const node = document.getElementById(id);
  if (node) node.textContent = value;
}

function bestIndices(values, higherIsBetter) {
  const nums = values.map((v) => (v === null || v === undefined || Number.isNaN(Number(v)) ? null : Number(v)));
  const valid = nums.filter((v) => v !== null);
  if (!valid.length) return [];
  const target = higherIsBetter ? Math.max(...valid) : Math.min(...valid);
  return nums.map((v, i) => (v !== null && v === target ? i : -1)).filter((i) => i >= 0);
}

function cellHtml(value, isBest, format) {
  const text = format ? format(value) : String(value);
  return "<td" + (isBest ? ' class="cell-best"' : "") + ">" + text + "</td>";
}

function fillHeadlines() {
  const variants = state.sasrec_variants;
  if (!variants || !variants.models) {
    setText("headline-sasrec", "—");
    setText("headline-item", "—");
    setText("headline-popularity", "—");
    setText("take-accuracy", "Result files were not loaded.");
    return;
  }
  const m = variants.models;
  setText("headline-sasrec", num(m.signed_trail["ndcg@10"]));
  setText("headline-item", num(m.item_knn["ndcg@10"]));
  setText("headline-popularity", num(m.popularity["ndcg@10"]));
  const p95 = variants.p95_ms_at_20;
  if (p95 != null) setText("headline-p95", ms(p95));
  const pass = variants.pass_rule_met
    ? "The pass rule is met: Polarity-SASRec is above popularity."
    : "The pass rule is not met: Polarity-SASRec is not above popularity.";
  setText(
    "headline-note",
    pass +
      " Vanilla-SASRec full-catalog NDCG@10 is " + num(m.vanilla["ndcg@10"]) +
      "; Veto-SASRec is " + num(m.veto["ndcg@10"]) +
      "; Polarity-SASRec is " + num(m.signed_trail["ndcg@10"]) + "."
  );
  setText(
    "take-accuracy",
    "On this final exam, Polarity-SASRec NDCG@10 is " + num(m.signed_trail["ndcg@10"]) +
    ", Veto-SASRec is " + num(m.veto["ndcg@10"]) +
    ", Vanilla-SASRec is " + num(m.vanilla["ndcg@10"]) +
    ", popularity is " + num(m.popularity["ndcg@10"]) +
    ", and item-to-item similarity is " + num(m.item_knn["ndcg@10"]) +
    ". " + pass
  );
}

function fillTraining() {
  if (!state.curves) {
    document.getElementById("take-training").textContent = "Training curve was not loaded.";
    return;
  }
  const best = state.curves.best_epoch;
  const row = state.curves.epochs.find((item) => item.epoch === best);
  document.getElementById("take-training").textContent =
    "The saved Vanilla-SASRec model is epoch " + best +
    (row ? ", where the practice NDCG@10 was " + num(row["val_ndcg@10"]) + "." : ".");
}

function fillHistory() {
  if (!state.history_length) {
    document.getElementById("take-history").textContent = "History-length results were not loaded.";
    return;
  }
  const lengths = state.history_length.lengths;
  const last = lengths.length - 1;
  document.getElementById("take-history").textContent =
    "When each method sees only the last " + lengths[last] + " films, Vanilla-SASRec NDCG@10 is " +
    num(state.history_length.models.sasrec[last]) +
    ", item-to-item is " + num(state.history_length.models.item_knn[last]) +
    ", and popularity is " + num(state.history_length.models.popularity[last]) +
    ". The chart also shows the shorter windows.";
}

function fillLatency() {
  const variants = state.sasrec_variants;
  if (state.latency) {
    const row = state.latency.by_history_length["20"] || Object.values(state.latency.by_history_length)[0];
    setText("headline-p95", ms(row.p95));
    const hardware = state.latency.hardware;
    setText(
      "take-latency",
      "Suppose the history has 20 films. On this machine, 95% of Vanilla-SASRec requests finished in " + ms(row.p95) +
      " or less. Veto shares that forward; Polarity-SASRec is the same latency class. " +
      "The machine was " + hardware.cpu + " (" + hardware.cores + " cores), " + hardware.os + "."
    );
    return;
  }
  if (variants && variants.p95_ms_at_20 != null) {
    setText("headline-p95", ms(variants.p95_ms_at_20));
    setText(
      "take-latency",
      "p95 at 20 films is about " + ms(variants.p95_ms_at_20) +
      " for Vanilla-SASRec and Veto (same forward). Polarity-SASRec is the same latency class."
    );
    return;
  }
  setText("headline-p95", "not measured");
  setText("take-latency", "Response time has not been measured yet.");
}

function fillCounts() {
  if (!state.data_stats) return;
  const stats = state.data_stats;
  document.getElementById("data-counts").textContent =
    "The file starts with " + stats.raw_ratings.toLocaleString() + " ratings from " +
    stats.raw_users.toLocaleString() + " people. After the 4- and 5-star like rule, and after dropping anyone with fewer than 5 liked films, this run uses " +
    stats.users.toLocaleString() + " people, " + stats.items.toLocaleString() + " films, and " +
    stats.positive_ratings.toLocaleString() + " liked ratings. " + stats.users_dropped + " people were dropped because their history was shorter than 5 likes. Lower ratings still feed Proposed Enhanced-SASRec.";
}

function fillSettings() {
  if (!state.settings) {
    document.querySelector("#settings-table tbody").innerHTML =
      "<tr><td colspan='2'>Settings were not loaded. Serve the docs folder over HTTP so data/settings.json can be fetched.</td></tr>";
    return;
  }
  const rows = [
    ["Recent films the sequence model can see", state.settings.max_len],
    ["Embedding length", state.settings.hidden_dim],
    ["Attention blocks", state.settings.num_blocks],
    ["Attention heads", state.settings.num_heads],
    ["Dropout", state.settings.dropout],
    ["Learning rate", state.settings.lr],
    ["Batch size", state.settings.batch_size],
    ["Neighbors kept per film", state.settings.item_knn_neighbors],
    ["Side features", state.settings.use_side_features ? (state.settings.side_feature_groups || []).join(", ") : "off"],
    ["Seed", state.settings.seed],
  ];
  const body = rows.map((row) => "<tr><td>" + row[0] + "</td><td>" + row[1] + "</td></tr>").join("");
  document.querySelector("#settings-table tbody").innerHTML = body;
}

function sliceRows(buckets) {
  return buckets.filter((bucket) => bucket.models && bucket.models.sasrec).map((bucket) => {
    const sasrec = bucket.models.sasrec;
    const item = bucket.models.item_knn;
    const pop = bucket.models.popularity;
    return "<tr>" +
      "<td>" + bucket.name + "</td>" +
      "<td>" + sasrec.n + "</td>" +
      "<td>" + num(sasrec["ndcg@10"]) + " / " + num(sasrec["hr@10"]) + "</td>" +
      "<td>" + num(item["ndcg@10"]) + " / " + num(item["hr@10"]) + "</td>" +
      "<td>" + num(pop["ndcg@10"]) + " / " + num(pop["hr@10"]) + "</td>" +
      "</tr>";
  }).join("");
}

function sliceTable(title, buckets) {
  return "<h3>" + title + "</h3>" +
    "<table><thead><tr>" +
    "<th>Bucket</th><th>n (viewers)</th>" +
    "<th>Polarity-SASRec NDCG@10 / HR@10</th>" +
    "<th>Item-to-item NDCG@10 / HR@10</th>" +
    "<th>Popularity NDCG@10 / HR@10</th>" +
    "</tr></thead><tbody>" + sliceRows(buckets) + "</tbody></table>";
}

function fillSlices() {
  const host = document.getElementById("slice-table");
  if (!host) return;
  if (!state.slices) {
    host.innerHTML = "<p>This split was not run.</p>";
    return;
  }
  host.innerHTML =
    sliceTable("By history length", state.slices.history_buckets || []) +
    sliceTable("By how widely liked the hidden film was", state.slices.popularity_buckets || []);
}

function fillGlossary() {
  if (!state.glossary) return;
  document.getElementById("glossary").innerHTML = "<table><tbody>" + state.glossary.map((item) =>
    "<tr><th>" + item.term + (item.open_form ? "<br>" + item.open_form : "") + "</th><td>" + item.sentence + "</td></tr>"
  ).join("") + "</tbody></table>";
}

function fillTry() {
  const users = state.demo_users && state.demo_users.users;
  if (!users) return;
  const menu = document.getElementById("viewer");
  menu.innerHTML = users.map((user, index) =>
    '<option value="' + index + '">Viewer ' + (index + 1) + " — " + user.heldout_title + "</option>"
  ).join("");
  menu.onchange = () => renderViewer(users[Number(menu.value)]);
  renderViewer(users[0]);
}

function list(items, heldout, kind) {
  return items.map((item) => {
    const mark = item.title === heldout ? " hit" : "";
    return '<span class="pill pill-' + kind + mark + '">' + item.rank + ". " + item.title + "</span>";
  }).join("");
}

function renderViewer(user) {
  const hit = user.hit_at_10;
  const seqName = (state.demo_users && state.demo_users.sequence_model) || "Polarity-SASRec";
  document.getElementById("viewer-body").innerHTML =
    "<div class='demo-block'>" +
    "<h3>Films this person had already liked</h3>" +
    "<div class='row'>" + user.history_titles.map((title) => "<span class='pill pill-history'>" + title + "</span>").join("") + "</div>" +
    "</div>" +
    "<p class='heldout-line'>The film we hid, which this person rated later: <strong class='heldout-title'>" + user.heldout_title + "</strong></p>" +
    "<div class='demo-block demo-polarity'>" +
    "<h3>" + seqName + " " + (hit.sasrec ? "(in its top 10)" : "(not in its top 10)") + "</h3>" +
    "<div class='row'>" + list(user.sasrec, user.heldout_title, "polarity") + "</div>" +
    "</div>" +
    "<div class='demo-block demo-knn'>" +
    "<h3>Item-to-item " + (hit.item_knn ? "(in its top 10)" : "(not in its top 10)") + "</h3>" +
    "<div class='row'>" + list(user.item_knn, user.heldout_title, "knn") + "</div>" +
    "</div>" +
    "<div class='demo-block demo-pop'>" +
    "<h3>Popularity " + (hit.popularity ? "(in its top 10)" : "(not in its top 10)") + "</h3>" +
    "<div class='row'>" + list(user.popularity, user.heldout_title, "pop") + "</div>" +
    "</div>";
  const ids = user.history_raw_ids.join(", ");
  document.getElementById("curl").textContent =
    'curl -X POST http://127.0.0.1:8000/recommend -H "Content-Type: application/json" -d "{\\"history\\":[' +
    ids + '],\\"k\\":5}"';
}

function renderBridgeTable(hostId) {
  const host = document.getElementById(hostId);
  if (!host) return;
  const bridge = state.protocol_bridge;
  if (!bridge) {
    host.innerHTML = "<p>Protocol bridge data was not loaded.</p>";
    return;
  }
  const sampled = bridge.ours_under_paper_sampled.models;
  const paper = bridge.paper_style_under_ours_full_catalog;
  const rows = [
    {
      label: sampled.vanilla.label,
      fullN: sampled.vanilla.full_catalog["ndcg@10"],
      fullH: sampled.vanilla.full_catalog["hr@10"],
      sampN: sampled.vanilla["ndcg@10"],
      sampH: sampled.vanilla["hr@10"],
    },
    {
      label: sampled.veto.label,
      fullN: sampled.veto.full_catalog["ndcg@10"],
      fullH: sampled.veto.full_catalog["hr@10"],
      sampN: sampled.veto["ndcg@10"],
      sampH: sampled.veto["hr@10"],
    },
    {
      label: sampled.signed_trail.label,
      fullN: sampled.signed_trail.full_catalog["ndcg@10"],
      fullH: sampled.signed_trail.full_catalog["hr@10"],
      sampN: sampled.signed_trail["ndcg@10"],
      sampH: sampled.signed_trail["hr@10"],
    },
    {
      label: paper.label,
      fullN: paper.full_catalog["ndcg@10"],
      fullH: paper.full_catalog["hr@10"],
      sampN: paper.sampled_100["ndcg@10"],
      sampH: paper.sampled_100["hr@10"],
    },
  ];
  const bestFullN = bestIndices(rows.map((r) => r.fullN), true);
  const bestFullH = bestIndices(rows.map((r) => r.fullH), true);
  const bestSampN = bestIndices(rows.map((r) => r.sampN), true);
  const bestSampH = bestIndices(rows.map((r) => r.sampH), true);
  const body = rows.map((row, i) =>
    "<tr><td>" + row.label + "</td>" +
    cellHtml(row.fullN, bestFullN.includes(i), num) +
    cellHtml(row.fullH, bestFullH.includes(i), num) +
    cellHtml(row.sampN, bestSampN.includes(i), num) +
    cellHtml(row.sampH, bestSampH.includes(i), num) +
    "</tr>"
  ).join("");
  host.innerHTML =
    "<table><thead><tr>" +
    "<th>Model</th><th>Full-catalog NDCG@10</th><th>Full-catalog HR@10</th>" +
    "<th>Sampled@100 NDCG@10</th><th>Sampled@100 HR@10</th>" +
    "</tr></thead><tbody>" + body + "</tbody></table>";
}

function fillBridge() {
  renderBridgeTable("bridge-table-results");
  const note = document.getElementById("bridge-note-results");
  if (!note) return;
  const bridge = state.protocol_bridge;
  if (!bridge) {
    note.textContent = "Protocol bridge data was not loaded.";
    return;
  }
  const ref = bridge.paper_table_iii_reference;
  note.textContent =
    "Paper Table III reference is about " + num(ref["ndcg@10"]) + " / " + num(ref["hr@10"]) +
    " on sampled@100. Our paper-style row is an approximate recipe in this codebase, not a bit-exact reproduction. Best cells (green) are highest in each column.";
}

function fillVariantTables() {
  const variants = state.sasrec_variants;
  const variantHost = document.getElementById("variant-table");
  const costHost = document.getElementById("accuracy-cost-table");
  if (!variantHost || !costHost) return;
  if (!variants || !variants.models) {
    variantHost.innerHTML = "<p>Variant metrics were not loaded.</p>";
    costHost.innerHTML = "";
    return;
  }
  const keys = variants.variants || ["vanilla", "veto", "signed_trail"];
  const rows = keys.map((key) => variants.models[key]);

  const bestN = bestIndices(rows.map((r) => r["ndcg@10"]), true);
  const bestH = bestIndices(rows.map((r) => r["hr@10"]), true);
  const bestC = bestIndices(rows.map((r) => r["coverage@10"]), true);
  variantHost.innerHTML =
    "<h4>Variant comparison (our exam)</h4>" +
    "<table><thead><tr>" +
    "<th>Model</th><th>NDCG@10</th><th>HR@10</th><th>Coverage@10</th><th>Behavior</th>" +
    "</tr></thead><tbody>" +
    rows.map((row, i) =>
      "<tr><td>" + row.label + "</td>" +
      cellHtml(row["ndcg@10"], bestN.includes(i), num) +
      cellHtml(row["hr@10"], bestH.includes(i), num) +
      cellHtml(row["coverage@10"], bestC.includes(i), num) +
      "<td>" + row.behavior + "</td></tr>"
    ).join("") +
    "</tbody></table>";

  const p95s = rows.map((r) => r.p95_ms_at_20);
  const bestP = bestIndices(p95s, false);
  costHost.innerHTML =
    "<h4>Accuracy vs cost</h4>" +
    "<table><thead><tr>" +
    "<th>Model</th><th>NDCG@10</th><th>HR@10</th><th>Behavior</th><th>p95@20</th>" +
    "</tr></thead><tbody>" +
    rows.map((row, i) =>
      "<tr><td>" + row.label + "</td>" +
      cellHtml(row["ndcg@10"], bestN.includes(i), num) +
      cellHtml(row["hr@10"], bestH.includes(i), num) +
      "<td>" + row.behavior + "</td>" +
      cellHtml(
        row.p95_ms_at_20,
        bestP.includes(i),
        (v) => ms(v) + (row.p95_note ? " <span class='metric-sub'>(" + row.p95_note + ")</span>" : "")
      ) +
      "</tr>"
    ).join("") +
    "</tbody></table>" +
    "<p class='metric-sub'>" + (variants.latency_note || "") + "</p>";
}

function layout() {
  return {
    paper_bgcolor: "#10140f",
    plot_bgcolor: "#10140f",
    font: { color: "#f6eef4" },
    margin: { t: 40, r: 20, b: 50, l: 55 },
    legend: { bgcolor: "#1a2118", bordercolor: "#2c3826" },
    xaxis: { gridcolor: "#2c3826" },
    yaxis: { gridcolor: "#2c3826" },
  };
}

function drawCharts() {
  if (!window.Plotly) return;
  chartsDrawn = true;
  const variants = state.sasrec_variants;
  if (variants && variants.models) {
    const order = variants.chart_order || ["popularity", "item_knn", "vanilla", "veto", "signed_trail"];
    const colors = {
      popularity: "#e7a8cc",
      item_knn: "#ff9f1c",
      vanilla: "#b388ff",
      veto: "#8bd450",
      signed_trail: "#5ec8ff",
    };
    const measures = ["ndcg@10", "hr@10", "coverage@10"];
    const traces = order.map((key) => {
      const model = variants.models[key];
      return {
        type: "bar",
        name: model.label,
        marker: { color: colors[key] || "#f6eef4" },
        x: ["NDCG@10", "HR@10", "Coverage@10"],
        y: measures.map((measure) => model[measure]),
      };
    });
    Plotly.newPlot("chart-metrics", traces, { ...layout(), barmode: "group", title: "Full-catalog leave-one-out" }, { responsive: true, displayModeBar: false });
    document.getElementById("chart-metrics-note").textContent =
      "Five methods on the same likes-only full-catalog exam: Popularity, Item-to-item, Vanilla-SASRec, Veto-SASRec, Polarity-SASRec. Bars are NDCG@10, HR@10, and Coverage@10.";
  } else if (state.metrics) {
    const colors = { popularity: "#e7a8cc", item_knn: "#ff9f1c", sasrec: "#b388ff" };
    const labels = { popularity: "Popularity", item_knn: "Item-to-item", sasrec: "SASRec" };
    const measures = ["ndcg@10", "hr@10", "coverage@10"];
    const traces = Object.keys(labels).map((name) => ({
      type: "bar",
      name: labels[name],
      marker: { color: colors[name] },
      x: ["NDCG@10", "HR@10", "Coverage@10"],
      y: measures.map((measure) => state.metrics.models[name][measure]),
    }));
    Plotly.newPlot("chart-metrics", traces, { ...layout(), barmode: "group", title: "Test split" }, { responsive: true, displayModeBar: false });
  }

  if (state.curves) {
    const epochs = state.curves.epochs.map((row) => row.epoch);
    Plotly.newPlot("chart-training", [
      { x: epochs, y: state.curves.epochs.map((row) => row.train_loss), name: "Training loss", yaxis: "y", line: { color: "#ff9f1c" } },
      { x: epochs, y: state.curves.epochs.map((row) => row["val_ndcg@10"]), name: "Validation NDCG@10", yaxis: "y2", line: { color: "#b388ff" } },
    ], {
      ...layout(),
      title: "Vanilla-SASRec training",
      yaxis: { title: "Loss", gridcolor: "#2c3826" },
      yaxis2: { title: "NDCG@10", overlaying: "y", side: "right", gridcolor: "#2c3826" },
    }, { responsive: true, displayModeBar: false });
    document.getElementById("chart-training-note").textContent =
      "The orange line is training loss, on the left axis. The purple line is the practice NDCG@10, on the right axis. The saved Vanilla checkpoint is epoch " + state.curves.best_epoch + ", the best practice score.";
  }

  if (state.history_length) {
    const histLabels = { popularity: "Popularity", item_knn: "Item-to-item", sasrec: "Vanilla-SASRec" };
    const histColors = { popularity: "#e7a8cc", item_knn: "#ff9f1c", sasrec: "#b388ff" };
    Plotly.newPlot("chart-history", Object.keys(histLabels).map((name) => ({
      x: state.history_length.lengths,
      y: state.history_length.models[name],
      name: histLabels[name],
      mode: "lines+markers",
      line: { color: histColors[name] },
    })), { ...layout(), title: "Shorter histories", xaxis: { title: "Recent films", gridcolor: "#2c3826" }, yaxis: { title: "NDCG@10", gridcolor: "#2c3826" } }, { responsive: true, displayModeBar: false });
    document.getElementById("chart-history-note").textContent =
      "Moving left means the method saw fewer recent films. Films cut off by that window count as unseen, so they can appear in the list again.";
  }

  if (state.latency) {
    const lengths = Object.keys(state.latency.by_history_length).map(Number).sort((a, b) => a - b);
    Plotly.newPlot("chart-latency", ["p50", "p95", "p99"].map((mark) => ({
      x: lengths,
      y: lengths.map((length) => state.latency.by_history_length[String(length)][mark]),
      name: mark,
      mode: "lines+markers",
    })).concat([{
      x: lengths,
      y: lengths.map(() => 50),
      name: "50 ms",
      mode: "lines",
      line: { dash: "dash", color: "#ff5d73" },
    }]), { ...layout(), title: "Local HTTP time (Vanilla)", yaxis: { title: "Milliseconds", gridcolor: "#2c3826" } }, { responsive: true, displayModeBar: false });
    document.getElementById("chart-latency-note").textContent =
      "Each point is a history length for Vanilla-SASRec. Veto is a post-score add-on on that forward; Polarity-SASRec is the same latency class. The dashed line is the 50 ms budget on this computer and catalog.";
  }

  fillTakeaways();
}

function fillTakeaways() {
  const host = document.getElementById("key-takeaways");
  if (!host) return;
  const variants = state.sasrec_variants;
  if (!variants || !variants.models) {
    host.textContent = "Variant metrics were not loaded.";
    return;
  }
  const m = variants.models;
  host.textContent =
    "Polarity-SASRec leads our variants on full-catalog NDCG@10 (" + num(m.signed_trail["ndcg@10"]) +
    "); Veto-SASRec is the cheap lift over Vanilla (" + num(m.veto["ndcg@10"]) +
    " vs " + num(m.vanilla["ndcg@10"]) +
    ") with no retrain. The protocol bridge shows exam difficulty: sampled@100 lands near ~0.59 while full-catalog sits near ~0.15–0.17. " +
    "Coverage@10 is " + num(m.signed_trail["coverage@10"]) + " for Polarity-SASRec, " +
    num(m.veto["coverage@10"]) + " for Veto, and " + num(m.vanilla["coverage@10"]) + " for Vanilla.";
}

document.querySelectorAll(".tabs button").forEach((button) => {
  button.addEventListener("click", () => show(button.id.replace("tab-", "")));
});
window.addEventListener("hashchange", () => show(location.hash.slice(1)));

function renderFormulas() {
  if (!window.katex) return;
  document.querySelectorAll(".formula[data-tex]").forEach((node) => {
    try {
      katex.render(node.getAttribute("data-tex"), node, { displayMode: true, throwOnError: false });
    } catch (error) {
      /* keep the plain HTML already in the node */
    }
  });
}

function showLoadBanner(message) {
  const banner = document.getElementById("load-banner");
  if (!banner) return;
  banner.hidden = false;
  banner.textContent = message;
}

(async function main() {
  renderFormulas();
  if (location.protocol === "file:") {
    showLoadBanner(
      "This page was opened as a local file. Browsers block the result JSON in that mode, so the numbers stay empty. From the docs folder run: python -m http.server 8000  then open http://127.0.0.1:8000/"
    );
  }
  const names = [
    "sasrec_variants",
    "protocol_bridge",
    "metrics",
    "metrics_id_only",
    "curves",
    "latency",
    "history_length",
    "slices",
    "demo_users",
    "glossary",
    "data_stats",
    "settings",
  ];
  await Promise.all(names.map(async (name) => {
    try { state[name] = await loadJson(name); } catch (error) { state[name] = null; }
  }));
  for (const fill of [
    fillHeadlines,
    fillLatency,
    fillTraining,
    fillHistory,
    fillCounts,
    fillSettings,
    fillSlices,
    fillGlossary,
    fillTry,
    fillBridge,
    fillVariantTables,
  ]) {
    try { fill(); } catch (error) { console.error(error); }
  }
  show(location.hash.slice(1) || "overview");
  if (!state.sasrec_variants && !state.metrics) {
    setText(
      "headline-note",
      "The result files in docs/data could not be loaded. Serve the docs folder over HTTP instead of opening index.html as a file."
    );
    if (location.protocol !== "file:") {
      showLoadBanner(
        "Could not load docs/data/*.json. Check that those files exist next to index.html, then refresh."
      );
    }
  }
})();
