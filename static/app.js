/* PostmortemEnv · live console
   Vanilla JS, no framework. Streams agent events over SSE and renders them
   into the research-lab UI. The new flow:

     1. loadModels()  → populates the combobox (free + paid groups).
     2. user picks a model → applyModelSelection(id) toggles the credential
        card and free-tier note.
     3. user picks a scenario, presses Run → POST /api/runs with
        { agent: "hf", model_id, hf_token? , task_id }.
     4. SSE handlers (onReset/onStep/onThought/onUsage/onDone/onCurriculum)
        unchanged in spirit — they render into the new layout.
*/

const $ = (id) => document.getElementById(id);

const state = {
  // run/scenario
  tasks: [],
  selectedTaskId: null,
  // model
  models: [],
  freeTierAvailable: true,
  selectedModelId: null,
  // streaming
  currentRun: null,
  eventSource: null,
  scenarioMeta: null,
  maxSteps: 40,
  factsSeen: new Set(),
  chain: [],
  cause: null,
  groundTruthCause: null,
  tokensIn: 0,
  tokensCached: 0,
};

const STORAGE_KEYS = {
  theme: "pme.theme",
  model: "pme.model_id",
};

// --------------------------------------------------------------------- init

document.addEventListener("DOMContentLoaded", async () => {
  bindThemeToggle();
  bindRunForm();
  bindHfTokenReveal();
  renderSegments($("scoreFill"), 0);
  renderSegments($("budgetFill"), 0);

  await Promise.all([loadModels(), loadTasks(), loadCurriculum(), loadHistory()]);
});

// --------------------------------------------------------------------- theme

function bindThemeToggle() {
  const btn = $("themeToggle");
  if (!btn) return;
  syncThemeToggleState();
  btn.addEventListener("click", () => {
    const cur = document.documentElement.getAttribute("data-theme") || "light";
    const next = cur === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem(STORAGE_KEYS.theme, next); } catch {}
    syncThemeToggleState();
  });
}

function syncThemeToggleState() {
  const btn = $("themeToggle");
  if (!btn) return;
  const isDark = document.documentElement.getAttribute("data-theme") === "dark";
  btn.setAttribute("aria-pressed", String(isDark));
  btn.setAttribute("aria-label", isDark ? "Switch to light theme" : "Switch to dark theme");
}

// --------------------------------------------------------------------- data

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    if (!res.ok) throw new Error(`${res.status}`);
    const data = await res.json();
    state.models = data.models || [];
    state.freeTierAvailable = !!data.free_tier_available;
    if (!state.freeTierAvailable) $("freeTierBanner").hidden = false;
    renderModelListbox();
    bindCombobox();

    // Restore previous selection or pick the first free-tier model.
    let preferred = null;
    try { preferred = localStorage.getItem(STORAGE_KEYS.model); } catch {}
    const initial =
      state.models.find((m) => m.id === preferred) ||
      state.models.find((m) => m.tier === "free" && m.free_tier_available) ||
      state.models[0];
    if (initial) applyModelSelection(initial.id);
  } catch (e) {
    setStatus(`Failed to load models: ${e.message}`, "error");
  }
}

async function loadTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) throw new Error(`${res.status}`);
    state.tasks = await res.json();
    renderTaskSelect();
    if (state.tasks.length && !state.selectedTaskId) {
      selectTask(state.tasks[0].task_id);
    }
  } catch (e) {
    setStatus(`Failed to load tasks: ${e.message}`, "error");
  }
}

async function loadCurriculum() {
  try {
    const res = await fetch("/api/curriculum");
    const data = await res.json();
    renderCurriculum(data);
  } catch { /* silent */ }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    renderHistory(data);
  } catch { /* silent */ }
}

// --------------------------------------------------------------------- model combobox

function renderModelListbox() {
  const ul = $("modelListbox");
  ul.innerHTML = "";

  const groups = [
    { tier: "free", label: "Free tier · runs on HF credits" },
    { tier: "paid", label: "Paid tier · bring your own HF token" },
  ];

  for (const g of groups) {
    const items = state.models.filter((m) => m.tier === g.tier);
    if (!items.length) continue;
    const header = document.createElement("li");
    header.className = "combobox-group-label";
    header.setAttribute("role", "presentation");
    header.textContent = g.label;
    ul.appendChild(header);

    for (const m of items) {
      const li = document.createElement("li");
      li.className = "model-row";
      li.setAttribute("role", "option");
      li.id = `opt-${m.id}`;
      li.dataset.modelId = m.id;
      li.setAttribute("aria-selected", "false");
      const disabled = m.tier === "free" && !m.free_tier_available;
      if (disabled) li.setAttribute("aria-disabled", "true");

      const chip = m.tier === "free"
        ? `<span class="chip chip-free">Free · ~$${m.est_cost_usd.toFixed(2)}</span>`
        : `<span class="chip chip-paid">~$${m.est_cost_usd.toFixed(2)} / run</span>`;

      li.innerHTML = `
        <span class="model-logo" aria-hidden="true">🤗</span>
        <span class="model-name">
          ${escapeHtml(m.display_name)}
          <span class="model-meta">${m.params_b}B · ${formatCtx(m.context_window)} ctx · ${escapeHtml(m.license)}</span>
        </span>
        <span class="model-blurb">${escapeHtml(m.blurb)}</span>
        <span class="model-chip-cell">${chip}</span>
      `;
      li.addEventListener("click", () => {
        if (li.getAttribute("aria-disabled") === "true") return;
        applyModelSelection(m.id);
        closeListbox();
        $("modelTrigger").focus();
      });
      ul.appendChild(li);
    }
  }
}

function bindCombobox() {
  const trigger = $("modelTrigger");
  const listbox = $("modelListbox");

  trigger.addEventListener("click", () => {
    const open = trigger.getAttribute("aria-expanded") === "true";
    open ? closeListbox() : openListbox();
  });

  trigger.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      openListbox();
      moveActive(0);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      openListbox();
      moveActive(-1, /* fromEnd */ true);
    }
  });

  listbox.addEventListener("keydown", (e) => {
    const options = optionEls();
    const cur = options.findIndex((el) => el.classList.contains("active"));
    if (e.key === "ArrowDown") {
      e.preventDefault();
      moveActive(cur < 0 ? 0 : (cur + 1) % options.length);
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      moveActive(cur < 0 ? options.length - 1 : (cur - 1 + options.length) % options.length);
    } else if (e.key === "Home") {
      e.preventDefault(); moveActive(0);
    } else if (e.key === "End") {
      e.preventDefault(); moveActive(options.length - 1);
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      const active = options[cur];
      if (active && active.getAttribute("aria-disabled") !== "true") {
        applyModelSelection(active.dataset.modelId);
        closeListbox();
        trigger.focus();
      }
    } else if (e.key === "Escape" || e.key === "Tab") {
      closeListbox();
      if (e.key === "Escape") trigger.focus();
    } else if (e.key.length === 1 && /\S/.test(e.key)) {
      // Type-ahead by first letter
      const letter = e.key.toLowerCase();
      const start = (cur + 1) % options.length;
      for (let i = 0; i < options.length; i++) {
        const idx = (start + i) % options.length;
        const name = options[idx].querySelector(".model-name")?.textContent.trim().toLowerCase() || "";
        if (name.startsWith(letter)) { moveActive(idx); break; }
      }
    }
  });

  document.addEventListener("click", (e) => {
    if (!$("modelCombobox").contains(e.target)) closeListbox();
  });
}

function optionEls() {
  return Array.from($("modelListbox").querySelectorAll(".model-row"));
}

function moveActive(idx, fromEnd = false) {
  const options = optionEls();
  if (!options.length) return;
  if (fromEnd) idx = options.length - 1;
  options.forEach((el) => el.classList.remove("active"));
  const target = options[idx];
  if (!target) return;
  target.classList.add("active");
  target.scrollIntoView({ block: "nearest" });
  $("modelTrigger").setAttribute("aria-activedescendant", target.id);
}

function openListbox() {
  const trigger = $("modelTrigger");
  const listbox = $("modelListbox");
  trigger.setAttribute("aria-expanded", "true");
  listbox.hidden = false;
  // Move focus inside the listbox so arrow keys are scoped here
  listbox.focus();
  // Highlight current selection
  const sel = optionEls().findIndex((el) => el.dataset.modelId === state.selectedModelId);
  if (sel >= 0) moveActive(sel);
}

function closeListbox() {
  $("modelTrigger").setAttribute("aria-expanded", "false");
  $("modelListbox").hidden = true;
  $("modelTrigger").removeAttribute("aria-activedescendant");
}

function applyModelSelection(modelId) {
  const m = state.models.find((x) => x.id === modelId);
  if (!m) return;
  state.selectedModelId = m.id;
  try { localStorage.setItem(STORAGE_KEYS.model, m.id); } catch {}

  // Update trigger content
  const chip = m.tier === "free"
    ? `<span class="chip chip-free">Free · ~$${m.est_cost_usd.toFixed(2)}</span>`
    : `<span class="chip chip-paid">~$${m.est_cost_usd.toFixed(2)} / run</span>`;
  $("modelTriggerContent").innerHTML = `
    <span class="trigger-row">
      <span class="model-logo" aria-hidden="true">🤗</span>
      <span class="model-name">
        ${escapeHtml(m.display_name)}
        <span class="model-meta">${m.params_b}B · ${formatCtx(m.context_window)} ctx</span>
      </span>
      <span class="model-chip-cell">${chip}</span>
    </span>
  `;

  // Update aria-selected on options
  optionEls().forEach((el) => {
    el.setAttribute("aria-selected", String(el.dataset.modelId === m.id));
  });

  // Toggle credential card vs free-tier note
  const isPaid = m.tier === "paid";
  $("credCard").hidden = !isPaid;
  $("freeTierNote").hidden = isPaid || !state.freeTierAvailable;
  $("credCardCost").textContent = `~$${m.est_cost_usd.toFixed(2)} / run`;
  $("credCardTitle").textContent = isPaid ? "HF token required" : "HF token";

  // Free-tier disabled, even free models need a user token → show cred card
  if (m.tier === "free" && !state.freeTierAvailable) {
    $("credCard").hidden = false;
    $("credCardTitle").textContent = "HF token required (server has no free tier)";
    $("freeTierNote").hidden = true;
  }
}

function formatCtx(n) {
  if (n >= 1024) return `${Math.round(n / 1024)}K`;
  return String(n);
}

// --------------------------------------------------------------------- task select

function renderTaskSelect() {
  const sel = $("taskSelect");
  sel.innerHTML = "";
  for (const t of state.tasks) {
    const opt = document.createElement("option");
    opt.value = t.task_id;
    opt.textContent = `${t.name}  ·  ${t.difficulty}  ·  ${t.max_steps} steps`;
    sel.appendChild(opt);
  }
  sel.addEventListener("change", () => selectTask(sel.value));
}

function selectTask(id) {
  state.selectedTaskId = id;
  const t = state.tasks.find((x) => x.task_id === id);
  if (t) {
    $("streamTitle").textContent = t.name;
    $("streamSubtitle").textContent = t.description || "Frozen telemetry awaits investigation.";
    if ($("taskSelect").value !== id) $("taskSelect").value = id;
  }
}

function renderCurriculum(data) {
  const el = $("curriculumPanel");
  if (!el) return;
  el.innerHTML = "";
  const tasks = data.tasks || {};
  for (const [tid, info] of Object.entries(tasks)) {
    const solveRate = info.attempts ? Math.round((100 * info.solves) / info.attempts) : 0;
    const row = document.createElement("div");
    row.className = "curriculum-row";
    const mult = info.difficulty_multiplier || 1;
    row.innerHTML = `
      <div class="top">
        <span class="name">${escapeHtml(shortTaskName(tid))}</span>
        <span class="elo">ELO ${Math.round(info.elo || 0)}</span>
      </div>
      <div class="bot">
        <span>×${mult.toFixed(2)}</span>
        <div class="mult-bar"><div style="width:${Math.min(100, (mult - 0.6) / 1.9 * 100)}%"></div></div>
        <span>${info.attempts || 0} runs · ${solveRate}% solved</span>
      </div>
    `;
    el.appendChild(row);
  }
}

function renderHistory(runs) {
  const ul = $("historyList");
  if (!ul) return;
  ul.innerHTML = "";
  if (!runs.length) {
    ul.innerHTML = '<li class="muted history-item">No runs yet.</li>';
    return;
  }
  for (const r of runs.slice(0, 12)) {
    const li = document.createElement("li");
    li.className = "history-item";
    const scoreCls = (r.final_score ?? 0) >= 0.70 ? "win" : "loss";
    const scoreTxt = r.final_score !== null && r.final_score !== undefined
      ? r.final_score.toFixed(2) : "—";
    li.innerHTML = `
      <span class="task">${escapeHtml(shortTaskName(r.task_id))} · ${escapeHtml(r.agent_type)}</span>
      <span class="score ${scoreCls}">${scoreTxt}</span>
    `;
    ul.appendChild(li);
  }
}

// --------------------------------------------------------------------- credential card

function bindHfTokenReveal() {
  const btn = $("hfTokenReveal");
  const inp = $("hfToken");
  if (!btn || !inp) return;
  btn.addEventListener("click", () => {
    const showing = inp.type === "text";
    inp.type = showing ? "password" : "text";
    btn.textContent = showing ? "Show" : "Hide";
    btn.setAttribute("aria-pressed", String(!showing));
  });
}

// --------------------------------------------------------------------- run

function bindRunForm() {
  const form = $("runPanel");
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    startRun();
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      startRun();
    }
  });
}

async function startRun() {
  if (!state.selectedTaskId) {
    setStatus("Pick a scenario first.", "error");
    return;
  }
  if (!state.selectedModelId) {
    setStatus("Pick a model first.", "error");
    return;
  }
  const model = state.models.find((m) => m.id === state.selectedModelId);
  if (!model) {
    setStatus("Selected model is no longer available.", "error");
    return;
  }
  const needsToken = model.tier === "paid" || (model.tier === "free" && !state.freeTierAvailable);
  const hfToken = $("hfToken").value.trim();
  if (needsToken && !hfToken) {
    setStatus("This model needs your HuggingFace token.", "error");
    $("hfToken").focus();
    return;
  }

  const body = {
    agent: "hf",
    task_id: state.selectedTaskId,
    model_id: state.selectedModelId,
  };
  if (needsToken) body.hf_token = hfToken;

  // reset UI
  clearStream();
  state.factsSeen = new Set();
  state.chain = [];
  state.cause = null;
  state.groundTruthCause = null;
  state.tokensIn = 0;
  state.tokensCached = 0;
  updateStats({ facts: 0, wrong: 0, tokensIn: 0, cached: 0 });
  updateScore(null, null);
  $("causeSlot").innerHTML = '<span class="muted">—</span>';
  $("causeSlot").classList.remove("matched");
  $("chainList").innerHTML = '<li class="muted">—</li>';

  $("runBtn").disabled = true;
  $("runBtn").querySelector(".btn-label").textContent = "Running…";
  setStatus(`Starting run with ${model.display_name}…`, "ok");

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    const { run_id } = await res.json();
    state.currentRun = run_id;
    subscribe(run_id);
  } catch (e) {
    setStatus(`Failed to start: ${e.message}`, "error");
    resetRunButton();
  }
}

function subscribe(runId) {
  if (state.eventSource) state.eventSource.close();
  const es = new EventSource(`/api/stream/${runId}`);
  state.eventSource = es;

  es.addEventListener("reset", (e) => onReset(JSON.parse(e.data)));
  es.addEventListener("step", (e) => onStep(JSON.parse(e.data)));
  es.addEventListener("thought", (e) => onThought(JSON.parse(e.data)));
  es.addEventListener("usage", (e) => onUsage(JSON.parse(e.data)));
  es.addEventListener("curriculum", (e) => onCurriculum(JSON.parse(e.data)));
  es.addEventListener("done", (e) => onDone(JSON.parse(e.data)));
  es.addEventListener("error", (e) => {
    try {
      const data = JSON.parse(e.data);
      const msg = data.message || "stream error";
      const isKeyIssue = /token|api key|unauthor|401|403/i.test(msg);
      const hint = isKeyIssue
        ? "Check that your HuggingFace token has Inference permission."
        : "Try a different model or scenario.";
      setStatus(msg, "error");
      showStreamBanner(msg, hint);
      resetRunButton();
      es.close();
      state.eventSource = null;
    } catch { /* heartbeat */ }
  });
  es.addEventListener("_eof", () => {
    es.close();
    state.eventSource = null;
  });

  es.onerror = () => {
    // EventSource auto-reconnects; only treat as fatal if the server closes.
  };
}

// --------------------------------------------------------------------- events

function onReset(ev) {
  state.scenarioMeta = ev.scenario;
  state.maxSteps = ev.scenario.max_steps;
  renderGraph(ev.scenario.service_graph);
  $("streamMeta").innerHTML = `
    <span class="meta-chip">budget <strong>${ev.scenario.max_steps}</strong></span>
    <span class="meta-chip">services <strong>${Object.keys(ev.scenario.service_graph).length}</strong></span>
    <span class="meta-chip">commits <strong>${ev.scenario.available_commits.length}</strong></span>
    <span class="meta-chip">traces <strong>${ev.scenario.available_trace_ids.length}</strong></span>
  `;
}

function onStep(ev) {
  const { step, action, reason, observation } = ev;
  const card = document.createElement("div");
  card.className = "step-card";
  card.tabIndex = 0;
  card.setAttribute("aria-label",
    `Step ${step}, action ${action.action_type}, reward ${observation.reward.toFixed(3)}`);
  if (action.action_type === "submit") card.classList.add("terminal");

  const newFacts = diffFacts(observation.known_facts);
  const paramsHtml = renderActionParams(action);

  card.innerHTML = `
    <div class="step-index">#${String(step).padStart(2, "0")}</div>
    <div class="step-main">
      <div class="step-head">
        <span class="action-badge" data-kind="${action.action_type}">${action.action_type}</span>
        <span class="action-params">${paramsHtml}</span>
      </div>
      ${reason ? `<div class="reason">${escapeHtml(reason)}</div>` : ""}
      ${observation.query_result ? renderObservation(observation.query_result) : ""}
      ${newFacts.length ? `<div class="facts-new">${newFacts.map(f => `<span class="fact-pill">+ ${escapeHtml(f.slice(0, 120))}</span>`).join("")}</div>` : ""}
    </div>
    <div class="step-reward ${observation.reward >= 0.15 ? "ok" : observation.reward <= 0.02 ? "warn" : ""}">${observation.reward.toFixed(3)}</div>
  `;

  const body = $("streamBody");
  clearEmptyState();
  body.appendChild(card);
  body.scrollTop = body.scrollHeight;

  const obsEl = card.querySelector(".observation");
  const toggle = card.querySelector(".observation-expand");
  if (toggle && obsEl) {
    toggle.addEventListener("click", () => {
      obsEl.classList.toggle("collapsed");
      toggle.textContent = obsEl.classList.contains("collapsed") ? "EXPAND" : "COLLAPSE";
    });
  }

  const queriedSvc = action.service;
  if (queriedSvc) highlightGraphNode(queriedSvc, "queried");

  const stepsUsed = state.maxSteps - observation.remaining_budget;
  updateBudget(stepsUsed, state.maxSteps);
  updateStats({
    facts: observation.known_facts.length,
    wrong: observation.wrong_hypotheses,
  });

  if (action.action_type === "submit") {
    state.cause = action.final_cause;
    state.chain = action.final_chain || [];
    renderCauseAndChain();
  } else if (action.action_type === "hypothesize" && observation.query_result?.includes("CORRECT")) {
    state.cause = action.cause_entity_id;
    renderCauseAndChain();
  } else if (action.action_type === "explain_chain" && action.chain) {
    state.chain = action.chain;
    renderCauseAndChain();
  }
}

function onThought(ev) {
  const div = document.createElement("div");
  div.className = "thought";
  div.textContent = `> ${ev.text}`;
  clearEmptyState();
  const body = $("streamBody");
  body.appendChild(div);
  body.scrollTop = body.scrollHeight;
}

function onUsage(ev) {
  state.tokensIn = ev.input_tokens || 0;
  state.tokensCached = ev.cache_read_tokens || 0;
  updateStats({ tokensIn: state.tokensIn, cached: state.tokensCached });
}

function onCurriculum(ev) {
  const sign = ev.agent_elo_delta >= 0 ? "+" : "";
  setStatus(
    `Curriculum: agent ELO ${sign}${ev.agent_elo_delta}, difficulty × ${ev.difficulty_multiplier.toFixed(2)}`,
    ev.solved ? "ok" : ""
  );
  loadCurriculum();
}

function onDone(ev) {
  state.groundTruthCause = ev.cause;
  const matched = state.cause && state.cause === ev.cause;
  if (matched) $("causeSlot").classList.add("matched");
  updateScore(ev.score, ev.steps);
  resetRunButton();
  setStatus(
    `Done — score ${ev.score.toFixed(3)} in ${ev.steps} steps.`,
    ev.score >= 0.70 ? "ok" : "error"
  );

  if (ev.chain && ev.chain.length) showCascadeOnGraph(ev.chain, ev.cause);

  const summary = document.createElement("div");
  summary.className = "step-card terminal";
  summary.tabIndex = 0;
  summary.innerHTML = `
    <div class="step-index">END</div>
    <div class="step-main">
      <div class="step-head">
        <span class="action-badge" data-kind="submit">episode complete</span>
        <span class="action-params">score: ${ev.score.toFixed(4)} · steps: ${ev.steps}</span>
      </div>
      <div class="reason">
        ground-truth cause: <span class="action-params">${escapeHtml(ev.cause)}</span>
        ${ev.usage ? ` · tokens in: ${ev.usage.input_tokens}, out: ${ev.usage.output_tokens}` : ""}
      </div>
    </div>
    <div class="step-reward ${ev.score >= 0.70 ? "ok" : "warn"}">${ev.score.toFixed(3)}</div>
  `;
  $("streamBody").appendChild(summary);
  $("streamBody").scrollTop = $("streamBody").scrollHeight;

  loadHistory();
}

// --------------------------------------------------------------------- renderers

function renderActionParams(action) {
  const { action_type, ...rest } = action;
  if (!Object.keys(rest).length) return "<span class='muted'>—</span>";
  const parts = [];
  for (const [k, v] of Object.entries(rest)) {
    if (v === null || v === undefined) continue;
    if (Array.isArray(v)) {
      parts.push(`${k}=[${v.length} step${v.length === 1 ? "" : "s"}]`);
    } else if (typeof v === "object") {
      parts.push(`${k}=${escapeHtml(JSON.stringify(v))}`);
    } else {
      parts.push(`${k}=${escapeHtml(String(v))}`);
    }
  }
  return parts.join(" · ");
}

function renderObservation(text) {
  const collapsed = text.length > 400;
  return `
    <div class="observation ${collapsed ? "collapsed" : ""}">${escapeHtml(text)}</div>
    ${collapsed ? '<span class="observation-expand" tabindex="0" role="button">EXPAND</span>' : ""}
  `;
}

function renderCauseAndChain() {
  const slot = $("causeSlot");
  if (state.cause) {
    const ok = state.groundTruthCause && state.cause === state.groundTruthCause;
    slot.innerHTML = `<span>${escapeHtml(state.cause)}${ok ? '<span class="match-tag">MATCH</span>' : ""}</span>`;
    slot.classList.toggle("matched", !!ok);
  } else {
    slot.innerHTML = '<span class="muted">—</span>';
  }

  const ol = $("chainList");
  ol.innerHTML = "";
  if (!state.chain.length) {
    ol.innerHTML = '<li class="muted">—</li>';
    return;
  }
  for (const link of state.chain) {
    const li = document.createElement("li");
    li.className = "chain-item";
    li.innerHTML = `
      <span class="svc">${escapeHtml(link.service)}</span>
      <span class="arrow" aria-hidden="true">→</span>
      <span class="effect">${escapeHtml(link.effect)}</span>
    `;
    ol.appendChild(li);
  }
}

function renderGraph(graph) {
  const el = $("graph");
  el.innerHTML = "";
  const services = Object.keys(graph);
  if (!services.length) {
    el.innerHTML = '<span class="muted">no graph</span>';
    return;
  }

  const W = 280, H = 170;
  const diamond = {
    frontend: { x: W/2, y: 20 },
    auth:     { x: W-50, y: H/2 },
    data:     { x: W/2, y: H-20 },
    batch:    { x: 50, y: H/2 },
  };
  const isCanonical =
    services.length === 4 && services.every((s) => s in diamond);
  const positions = {};
  if (isCanonical) {
    Object.assign(positions, diamond);
  } else {
    const cx = W / 2, cy = H / 2;
    const radius = Math.min(W, H) / 2 - 30;
    services.forEach((svc, i) => {
      const theta = (2 * Math.PI * i) / services.length - Math.PI / 2;
      positions[svc] = {
        x: cx + radius * Math.cos(theta),
        y: cy + radius * Math.sin(theta),
      };
    });
  }
  const fallback = { x: W/2, y: H/2 };

  const svgNS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(svgNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.classList.add("graph-svg");
  svg.setAttribute("role", "img");
  const desc = document.createElementNS(svgNS, "desc");
  desc.textContent = "Service dependency graph showing " +
    services.join(", ") + ".";
  svg.appendChild(desc);

  const defs = document.createElementNS(svgNS, "defs");
  const marker = document.createElementNS(svgNS, "marker");
  marker.setAttribute("id", "arrowhead");
  marker.setAttribute("viewBox", "0 0 10 7");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "3.5");
  marker.setAttribute("markerWidth", "8");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto");
  const arrowPath = document.createElementNS(svgNS, "polygon");
  arrowPath.setAttribute("points", "0 0, 10 3.5, 0 7");
  arrowPath.setAttribute("fill", "currentColor");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  for (const svc of services) {
    const deps = graph[svc] || [];
    const from = positions[svc] || fallback;
    for (const dep of deps) {
      const to = positions[dep] || fallback;
      const line = document.createElementNS(svgNS, "line");
      line.setAttribute("x1", from.x);
      line.setAttribute("y1", from.y);
      line.setAttribute("x2", to.x);
      line.setAttribute("y2", to.y);
      line.classList.add("svc-edge");
      line.dataset.from = svc;
      line.dataset.to = dep;
      svg.appendChild(line);
    }
  }

  const nodeW = 70, nodeH = 24;
  for (const svc of services) {
    const pos = positions[svc] || fallback;
    const g = document.createElementNS(svgNS, "g");
    g.classList.add("svc-group");
    g.dataset.svc = svc;

    const rect = document.createElementNS(svgNS, "rect");
    rect.setAttribute("x", pos.x - nodeW/2);
    rect.setAttribute("y", pos.y - nodeH/2);
    rect.setAttribute("width", nodeW);
    rect.setAttribute("height", nodeH);
    rect.setAttribute("rx", "4");
    rect.classList.add("svc-node");
    g.appendChild(rect);

    const label = document.createElementNS(svgNS, "text");
    label.setAttribute("x", pos.x);
    label.setAttribute("y", pos.y);
    label.classList.add("svc-label");
    label.textContent = svc;
    g.appendChild(label);

    svg.appendChild(g);
  }

  el.appendChild(svg);
}

function highlightGraphNode(serviceName, type) {
  const svg = document.querySelector(".graph-svg");
  if (!svg) return;
  const group = svg.querySelector(`.svc-group[data-svc="${serviceName}"]`);
  if (group && !group.classList.contains("root-cause")) {
    group.classList.add(type);
  }
}

function highlightGraphEdge(from, to) {
  const svg = document.querySelector(".graph-svg");
  if (!svg) return;
  const edge = svg.querySelector(`.svc-edge[data-from="${from}"][data-to="${to}"]`);
  if (edge) edge.classList.add("active");
  const rev = svg.querySelector(`.svc-edge[data-from="${to}"][data-to="${from}"]`);
  if (rev) rev.classList.add("active");
}

function showCascadeOnGraph(chain, cause) {
  if (cause && chain.length > 0) {
    highlightGraphNode(chain[0].service, "root-cause");
  }
  chain.forEach((link, i) => {
    setTimeout(() => {
      highlightGraphNode(link.service, "in-chain");
      if (i > 0) highlightGraphEdge(chain[i-1].service, link.service);
    }, i * 300);
  });
}

// --------------------------------------------------------------------- helpers

function diffFacts(currentFacts) {
  const out = [];
  for (const f of currentFacts) {
    if (!state.factsSeen.has(f)) {
      state.factsSeen.add(f);
      out.push(f);
    }
  }
  return out;
}

function renderSegments(el, value, { over = false } = {}) {
  const v = Math.max(0, Math.min(1, value));
  const count = Math.round(v * 20);
  el.classList.toggle("over", over);
  el.innerHTML = "";
  for (let i = 0; i < 20; i++) {
    const seg = document.createElement("span");
    seg.className = "seg" + (i < count ? " on" : "");
    el.appendChild(seg);
  }
}

function updateScore(score) {
  if (score === null || score === undefined) {
    renderSegments($("scoreFill"), 0);
    $("scoreValue").textContent = "—";
    return;
  }
  renderSegments($("scoreFill"), score);
  $("scoreValue").textContent = score.toFixed(3);
}

function updateBudget(used, max) {
  const pct = used / max;
  renderSegments($("budgetFill"), pct, { over: pct >= 0.9 });
  $("budgetValue").textContent = `${used} / ${max}`;
}

function updateStats({ facts, wrong, tokensIn, cached }) {
  if (facts !== undefined) $("statFacts").textContent = facts;
  if (wrong !== undefined) $("statWrong").textContent = wrong;
  if (tokensIn !== undefined) $("statTokensIn").textContent = tokensIn;
  if (cached !== undefined) $("statCached").textContent = cached;
}

function setStatus(msg, kind = "") {
  const el = $("inlineStatus");
  if (!el) return;
  el.dataset.kind = kind;
  el.textContent = msg;
  clearTimeout(setStatus._timer);
  if (kind === "error") return; // sticky on error
  setStatus._timer = setTimeout(() => {
    if (el.textContent === msg) {
      el.textContent = "";
      el.dataset.kind = "";
    }
  }, 6000);
}

function resetRunButton() {
  $("runBtn").disabled = false;
  $("runBtn").querySelector(".btn-label").textContent = "Run investigation";
}

function clearStream() {
  $("streamBody").innerHTML = "";
}

function clearEmptyState() {
  const empty = $("streamBody").querySelector(".empty-state");
  if (empty) empty.remove();
}

function showStreamBanner(message, hint) {
  const body = $("streamBody");
  if (!body) return;
  body.innerHTML = "";
  const banner = document.createElement("div");
  banner.className = "stream-banner";
  banner.innerHTML = `
    <div class="stream-banner-label">ERROR</div>
    <div class="stream-banner-msg">${escapeHtml(message)}</div>
    ${hint ? `<div class="stream-banner-hint">${escapeHtml(hint)}</div>` : ""}
  `;
  body.appendChild(banner);
}

function shortTaskName(tid) {
  return tid.replace(/^task\d+_/, "").replace(/_/g, " ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
