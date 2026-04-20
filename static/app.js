/* PostmortemEnv live console — vanilla JS, no framework.
   Streams agent events over SSE and renders them with Perplexity-style polish.
*/

const $ = (id) => document.getElementById(id);

const state = {
  tasks: [],
  selectedTaskId: null,
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

// --------------------------------------------------------------------- init

document.addEventListener("DOMContentLoaded", async () => {
  bindAgentRow();
  bindRunButton();
  renderSegments($("scoreFill"), 0);
  renderSegments($("budgetFill"), 0);
  await loadTasks();
  await loadCurriculum();
  await loadHistory();
});

// --------------------------------------------------------------------- data

async function loadTasks() {
  try {
    const res = await fetch("/api/tasks");
    if (!res.ok) throw new Error(`${res.status}`);
    state.tasks = await res.json();
    renderTaskList();
    if (state.tasks.length && !state.selectedTaskId) {
      selectTask(state.tasks[0].task_id);
    }
  } catch (e) {
    toast(`Failed to load tasks: ${e.message}`, "error");
  }
}

async function loadCurriculum() {
  try {
    const res = await fetch("/api/curriculum");
    const data = await res.json();
    renderCurriculum(data);
  } catch {
    // silent
  }
}

async function loadHistory() {
  try {
    const res = await fetch("/api/runs");
    const data = await res.json();
    renderHistory(data);
  } catch {
    // silent
  }
}

// --------------------------------------------------------------------- render: sidebar

function renderTaskList() {
  const ul = $("taskList");
  ul.innerHTML = "";
  for (const t of state.tasks) {
    const li = document.createElement("li");
    li.className = "task-item" + (t.task_id === state.selectedTaskId ? " selected" : "");
    li.innerHTML = `
      <span class="name">${escapeHtml(t.name)}</span>
      <span class="meta">
        <span class="difficulty ${t.difficulty}">${t.difficulty}</span>
        <span>${t.max_steps} steps</span>
      </span>
    `;
    li.addEventListener("click", () => selectTask(t.task_id));
    ul.appendChild(li);
  }
}

function selectTask(id) {
  state.selectedTaskId = id;
  renderTaskList();
  const t = state.tasks.find((x) => x.task_id === id);
  if (t) {
    $("streamTitle").textContent = t.name;
    $("streamSubtitle").textContent = t.description;
  }
}

function renderCurriculum(data) {
  const el = $("curriculumPanel");
  el.innerHTML = "";
  const tasks = data.tasks || {};
  for (const [tid, info] of Object.entries(tasks)) {
    const solveRate = info.attempts ? Math.round((100 * info.solves) / info.attempts) : 0;
    const row = document.createElement("div");
    row.className = "curriculum-row";
    row.innerHTML = `
      <div class="top">
        <span class="name">${escapeHtml(shortTaskName(tid))}</span>
        <span class="elo">ELO ${Math.round(info.elo || 0)}</span>
      </div>
      <div class="bot">
        <span>×${(info.difficulty_multiplier || 1).toFixed(2)}</span>
        <div class="mult-bar"><div style="width:${Math.min(100, ((info.difficulty_multiplier || 1) - 0.6) / 1.9 * 100)}%"></div></div>
        <span>${info.attempts || 0} runs · ${solveRate}% solved</span>
      </div>
    `;
    el.appendChild(row);
  }
}

function renderHistory(runs) {
  const ul = $("historyList");
  ul.innerHTML = "";
  if (!runs.length) {
    ul.innerHTML = '<li class="muted empty">no runs yet</li>';
    return;
  }
  for (const r of runs.slice(0, 12)) {
    const li = document.createElement("li");
    li.className = "history-item";
    const scoreCls = (r.final_score ?? 0) >= 0.70 ? "win" : "loss";
    const scoreTxt = r.final_score !== null && r.final_score !== undefined
      ? r.final_score.toFixed(2)
      : "—";
    li.innerHTML = `
      <span class="task">${escapeHtml(shortTaskName(r.task_id))} · ${escapeHtml(r.agent_type)}</span>
      <span class="score ${scoreCls}">${scoreTxt}</span>
    `;
    ul.appendChild(li);
  }
}

// --------------------------------------------------------------------- agent picker

function bindAgentRow() {
  const row = $("agentRow");
  row.addEventListener("change", (e) => {
    const v = e.target.value;
    for (const pill of row.querySelectorAll(".radio-pill")) {
      pill.classList.toggle("selected", pill.querySelector("input").value === v);
    }
    $("llmPanel").classList.toggle("hidden", v !== "llm");
  });
}

// --------------------------------------------------------------------- run

function bindRunButton() {
  const btn = $("runBtn");
  btn.addEventListener("click", startRun);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      startRun();
    }
  });
}

async function startRun() {
  if (!state.selectedTaskId) {
    toast("Pick a scenario first", "error");
    return;
  }
  const agent = document.querySelector('input[name="agent"]:checked').value;

  const body = {
    agent,
    task_id: state.selectedTaskId,
  };
  if (agent === "llm") {
    body.provider = $("llmProvider").value;
    const key = $("llmKey").value.trim();
    const model = $("llmModel").value.trim();
    if (!key) {
      toast("LLM mode needs an API key (not stored, sent to provider only)", "error");
      return;
    }
    body.api_key = key;
    if (model) body.model = model;
  }

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

  setStatus("running");
  $("runBtn").disabled = true;
  $("runBtn").querySelector(".btn-label").textContent = "RUNNING…";

  try {
    const res = await fetch("/api/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${await res.text()}`);
    const { run_id } = await res.json();
    state.currentRun = run_id;
    subscribe(run_id);
  } catch (e) {
    toast(`Failed to start: ${e.message}`, "error");
    setStatus("error");
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
      toast(data.message || "stream error", "error");
      setStatus("error");
      resetRunButton();
    } catch {
      /* heartbeat */
    }
  });
  es.addEventListener("_eof", () => {
    es.close();
    state.eventSource = null;
  });

  es.onerror = () => {
    // EventSource auto-reconnects; only treat as fatal if the run is closed server-side.
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
  if (action.action_type === "submit") card.classList.add("terminal");

  const newFacts = diffFacts(observation.known_facts);
  const rewardChip = renderReward(observation.reward);
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

  // Wire observation expand toggle
  const obsEl = card.querySelector(".observation");
  const toggle = card.querySelector(".observation-expand");
  if (toggle && obsEl) {
    toggle.addEventListener("click", () => {
      obsEl.classList.toggle("collapsed");
      toggle.textContent = obsEl.classList.contains("collapsed") ? "EXPAND" : "COLLAPSE";
    });
  }

  // Side panel updates
  const stepsUsed = state.maxSteps - observation.remaining_budget;
  updateBudget(stepsUsed, state.maxSteps);
  updateStats({
    facts: observation.known_facts.length,
    wrong: observation.wrong_hypotheses,
  });

  // Capture cause / chain from action payload
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
  toast(`Curriculum: agent ELO ${sign}${ev.agent_elo_delta}, difficulty × ${ev.difficulty_multiplier.toFixed(2)}`,
        ev.solved ? "ok" : "");
  loadCurriculum();
}

function onDone(ev) {
  state.groundTruthCause = ev.cause;
  const matched = state.cause && state.cause === ev.cause;
  if (matched) {
    $("causeSlot").classList.add("matched");
  }
  updateScore(ev.score, ev.steps);
  setStatus(ev.score >= 0.70 ? "done" : "error");
  resetRunButton();

  const summary = document.createElement("div");
  summary.className = "step-card terminal";
  summary.innerHTML = `
    <div class="step-index">END</div>
    <div class="step-main">
      <div class="step-head">
        <span class="action-badge" data-kind="submit">episode complete</span>
        <span class="action-params">score: ${ev.score.toFixed(4)} · steps: ${ev.steps}</span>
      </div>
      <div class="reason">
        ground-truth cause: <span class="action-params">${escapeHtml(ev.cause)}</span>
        ${ev.usage ? ` · tokens in: ${ev.usage.input_tokens}, cached: ${ev.usage.cache_read_tokens}` : ""}
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
    ${collapsed ? '<span class="observation-expand">EXPAND</span>' : ""}
  `;
}

function renderReward(r) {
  return `<span class="step-reward">${r.toFixed(3)}</span>`;
}

function renderCauseAndChain() {
  const slot = $("causeSlot");
  if (state.cause) {
    const ok = state.groundTruthCause && state.cause === state.groundTruthCause;
    slot.innerHTML = `<span>${escapeHtml(state.cause)}${ok ? '<span class="match-tag">[ MATCH ]</span>' : ""}</span>`;
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
      <span class="arrow">→</span>
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
  for (const svc of services) {
    const n = document.createElement("span");
    n.className = "node";
    n.textContent = svc;
    n.dataset.svc = svc;
    el.appendChild(n);
    const deps = graph[svc] || [];
    if (deps.length) {
      const arrow = document.createElement("span");
      arrow.className = "edge";
      arrow.textContent = " → ";
      el.appendChild(arrow);
      el.appendChild(document.createTextNode(deps.join(", ")));
    }
    el.appendChild(document.createElement("br"));
  }
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

function updateScore(score, steps) {
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

function setStatus(s) {
  const pill = $("statusPill");
  pill.dataset.status = s;
  pill.querySelector(".status-text").textContent = `[ ${s.toUpperCase()} ]`;
}

function resetRunButton() {
  $("runBtn").disabled = false;
  $("runBtn").querySelector(".btn-label").textContent = "START INVESTIGATION";
}

function clearStream() {
  $("streamBody").innerHTML = "";
}

function clearEmptyState() {
  const empty = $("streamBody").querySelector(".empty-state");
  if (empty) empty.remove();
}

function toast(msg, kind = "") {
  const el = $("inlineStatus");
  if (!el) return;
  const prefix = kind === "error" ? "ERROR" : kind === "ok" ? "OK" : "INFO";
  el.dataset.kind = kind || "info";
  el.textContent = `[${prefix}] ${msg}`;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    if (el.textContent.includes(msg)) {
      el.textContent = "";
      el.dataset.kind = "";
    }
  }, 6000);
}

function shortTaskName(tid) {
  return tid.replace(/^task\d+_/, "").replace(/_/g, " ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}
