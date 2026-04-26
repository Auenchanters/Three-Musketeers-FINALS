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
  bindTrainingPanel();
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
    // The model trigger button still says "Loading models…" by default.
    // Replace it with a visible failure state so the user can't silently
    // sit on a dead dropdown wondering why nothing is selectable.
    const trigger = $("modelTriggerContent");
    if (trigger) {
      trigger.innerHTML =
        '<span class="model-load-error">Couldn\'t load models — refresh the page</span>';
    }
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
    // Same problem as loadModels: both task selectors will sit on
    // "Loading scenarios…" forever if /api/tasks errors. Replace the
    // placeholder with a visible failure option in both selects.
    for (const id of ["taskSelect", "trainingTaskSelect"]) {
      const sel = $(id);
      if (sel) {
        sel.innerHTML =
          '<option value="" disabled selected>Couldn\'t load scenarios — refresh the page</option>';
      }
    }
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
      // Surface the requires_provider flag so the user knows up-front that
      // this model only routes through one specific HF Inference Provider
      // (e.g. Featherless-ai). Without this badge they only learn it after
      // a 400 from the router with model_not_supported.
      const providerBadge = m.requires_provider
        ? `<span class="chip chip-warn" title="Enable ${escapeHtml(m.requires_provider)} at huggingface.co/settings/inference-providers">Needs ${escapeHtml(m.requires_provider)}</span>`
        : "";

      li.innerHTML = `
        <span class="model-logo" aria-hidden="true">🤗</span>
        <span class="model-name">
          ${escapeHtml(m.display_name)}
          <span class="model-meta">${m.params_b}B · ${formatCtx(m.context_window)} ctx · ${escapeHtml(m.license)}</span>
        </span>
        <span class="model-blurb">${escapeHtml(m.blurb)}</span>
        <span class="model-chip-cell">${chip}${providerBadge}</span>
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

  // Provider-specific banner for free models that only route through one
  // HF Inference Provider. Failing to enable that provider is the most
  // common reason a "free" model still 400s for a fresh HF account.
  const note = $("providerRequirementNote");
  if (note) {
    if (m.requires_provider) {
      note.hidden = false;
      note.innerHTML = `Heads up: <strong>${escapeHtml(m.display_name)}</strong> only routes through <strong>${escapeHtml(m.requires_provider)}</strong>. Enable it at <a href="https://huggingface.co/settings/inference-providers" target="_blank" rel="noopener">huggingface.co/settings/inference-providers</a> first, otherwise the run will 400 with <code>model_not_supported</code>.`;
    } else {
      note.hidden = true;
      note.textContent = "";
    }
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

  // Mirror the same task list into the live-training panel's selector.
  // Default it to a task where REINFORCE shows a clear lift in <500 episodes.
  const trainSel = $("trainingTaskSelect");
  if (trainSel) {
    trainSel.innerHTML = "";
    const PREFERRED_DEFAULT = "task5_data_corruption_cascade";
    for (const t of state.tasks) {
      const opt = document.createElement("option");
      opt.value = t.task_id;
      opt.textContent = `${t.name}  ·  ${t.difficulty}`;
      trainSel.appendChild(opt);
    }
    const def = state.tasks.find((t) => t.task_id === PREFERRED_DEFAULT)
      || state.tasks[0];
    if (def) trainSel.value = def.task_id;
  }
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
    showStreamBanner(
      "Pick a scenario before running.",
      "Use the Scenario dropdown above to choose a frozen incident.",
    );
    return;
  }
  if (!state.selectedModelId) {
    setStatus("Pick a model first.", "error");
    showStreamBanner(
      "Pick a model before running.",
      "Use the Model dropdown above; the default Qwen2.5 1.5B is a good starting point.",
    );
    return;
  }
  const model = state.models.find((m) => m.id === state.selectedModelId);
  if (!model) {
    setStatus("Selected model is no longer available.", "error");
    showStreamBanner("Selected model is no longer available.", "Refresh the page and pick another.");
    return;
  }
  const needsToken = model.tier === "paid" || (model.tier === "free" && !state.freeTierAvailable);
  const hfToken = $("hfToken").value.trim();
  if (needsToken && !hfToken) {
    const reason = model.tier === "paid"
      ? `${model.display_name} is a paid-tier model (~$${model.est_cost_usd.toFixed(2)}/run) so we never charge it to the shared free tier.`
      : `This deployment doesn't have a server-side HuggingFace token configured, so even free-tier models need yours.`;
    setStatus("This model needs your HuggingFace token.", "error");
    showStreamBanner(
      "HuggingFace token required to start this run.",
      `${reason} Paste a token from huggingface.co/settings/tokens (with Inference permission) into the field above, then press Run again. Tokens are used once and never stored.`,
    );
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
    // Surface the failure inside the live investigation panel too, so a
    // user who scrolled past the run-config never sees it sitting empty
    // wondering what's happening. Most common failure modes here are:
    //   - 401/403 (token rejected)
    //   - 400 with "Model X is paid-tier" (server-side validation)
    //   - 503/504 (HF Inference cold-start or queue saturation)
    const m = String(e.message || "");
    const isAuth = /401|403|unauthor|invalid token|token/i.test(m);
    const is5xx = /50[0-9]/.test(m);
    const hint = isAuth
      ? "Double-check your HuggingFace token has Inference permission, then try again."
      : is5xx
        ? "Hugging Face Inference is throttled or cold-starting; wait ~30s and retry, or pick a different free-tier model."
        : "Try a different model or scenario, or check the browser console for details.";
    showStreamBanner(`Failed to start run: ${m}`, hint);
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
      // Prefer the structured hint the backend now attaches for HF
      // Inference failures (HFInferenceError._hint maps 400/401/403/404/
      // 429/503 to a tailored next step). Fall back to keyword sniffing
      // for older error shapes.
      let hint = data.hint;
      if (!hint) {
        const isKeyIssue = /token|api key|unauthor|401|403/i.test(msg);
        hint = isKeyIssue
          ? "Check that your HuggingFace token has Inference permission."
          : "Try a different model or scenario.";
      }
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

    const dot = document.createElementNS(svgNS, "circle");
    dot.setAttribute("cx", pos.x + nodeW/2 - 6);
    dot.setAttribute("cy", pos.y - nodeH/2 + 6);
    dot.setAttribute("r", "3");
    dot.classList.add("outage-dot");
    g.appendChild(dot);

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
  // Prefer the human-readable `task_name` from the scenario JSON
  // (loaded into state.tasks at startup). Falls back to a slug-cleanup
  // for safety when state.tasks isn't populated yet (e.g. during the
  // very first SSE event before /api/tasks resolves).
  const t = state.tasks.find((x) => x.task_id === tid);
  if (t && t.name && t.name !== tid) return t.name;
  return tid.replace(/^task\d+_/, "").replace(/_/g, " ");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// =============================================================================
// Live training panel
// =============================================================================
//
// Subscribes to /api/training/stream/{id} and incrementally draws an SVG chart
// of (rolling-mean reward vs. episode) against the random baseline. Every
// "Start training" click is a *fresh* on-policy REINFORCE run — there's no
// pre-recorded curve, the points fill in left-to-right as the policy actually
// learns. See web/runner.py and web/training_loop.py for the backend.

const SVG_NS = "http://www.w3.org/2000/svg";

const trainingState = {
  sessionId: null,
  eventSource: null,
  nEpisodes: 500,
  randomBaseline: null,
  // Each metric event: { episode, score, rolling_mean, lift_over_random, ... }
  points: [],
  // Cached chart geometry so we can redraw on resize without re-laying-out.
  geom: null,
};

function bindTrainingPanel() {
  const btn = $("trainingStartBtn");
  if (!btn) return;
  btn.addEventListener("click", () => startTraining());

  // Shift+Enter triggers training (matches the run panel's ⌘+Enter UX).
  document.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && e.shiftKey && !(e.metaKey || e.ctrlKey)) {
      const target = e.target;
      if (target && target.matches("input,textarea,select,[contenteditable=true]")) {
        return;
      }
      e.preventDefault();
      startTraining();
    }
  });

  // Re-paint the chart on resize so it stays sharp on viewport changes.
  let resizeRaf = 0;
  window.addEventListener("resize", () => {
    if (!trainingState.points.length) return;
    cancelAnimationFrame(resizeRaf);
    resizeRaf = requestAnimationFrame(() => paintTrainingChart());
  });
}

async function startTraining() {
  const sel = $("trainingTaskSelect");
  const epsInput = $("trainingEpsInput");
  const taskId = sel?.value;
  if (!taskId) {
    setTrainingStatus("Pick a task first.", "error");
    showTrainingError(
      "Pick a task before starting training.",
      "Use the Task dropdown above to choose a scenario.",
    );
    return;
  }
  const nEpisodes = clamp(parseInt(epsInput?.value, 10) || 500, 50, 600);
  if (epsInput) epsInput.value = String(nEpisodes);

  // Tear down any previous run.
  if (trainingState.eventSource) {
    trainingState.eventSource.close();
    trainingState.eventSource = null;
  }
  trainingState.points = [];
  trainingState.randomBaseline = null;
  trainingState.nEpisodes = nEpisodes;
  trainingState.sessionId = null;

  resetTrainingSummary();
  resetTrainingChart();
  clearTrainingError();
  const rubricHost = $("trainingRubrics");
  if (rubricHost) rubricHost.hidden = true;
  setTrainingStatus("Estimating random baseline…", "");

  const btn = $("trainingStartBtn");
  if (btn) {
    btn.disabled = true;
    const lbl = btn.querySelector(".btn-label");
    if (lbl) lbl.textContent = "Training…";
  }

  try {
    const res = await fetch("/api/training/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task_id: taskId, n_episodes: nEpisodes }),
    });
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    const meta = await res.json();
    trainingState.sessionId = meta.session_id;
    trainingState.nEpisodes = meta.n_episodes;
    const runtimeEl = $("trainingRuntime");
    if (runtimeEl) runtimeEl.textContent = meta.runtime || "local";
    subscribeTraining(meta.session_id);
  } catch (e) {
    // Don't leave the chart sitting on its placeholder ("Press Start
    // training to begin a fresh run.") when the run actually failed —
    // that's the same silent-failure pattern as the live-investigation
    // panel had before we added showStreamBanner.
    const msg = String(e.message || e);
    const isAuth = /401|403|unauthor|token/i.test(msg);
    const is5xx = /50[0-9]/.test(msg);
    const hint = isAuth
      ? "The training endpoint rejected the request. Reload the page and try again."
      : is5xx
        ? "Server failed to start training. Try a smaller episode count, or wait a moment and retry."
        : "Try a different scenario or fewer episodes, or check the browser console for details.";
    setTrainingStatus(`Failed to start: ${msg}`, "error");
    showTrainingError(`Failed to start training: ${msg}`, hint);
    resetTrainingButton();
  }
}

function subscribeTraining(sessionId) {
  const es = new EventSource(`/api/training/stream/${sessionId}`);
  trainingState.eventSource = es;

  es.addEventListener("baselines", (e) => onTrainingBaselines(JSON.parse(e.data)));
  es.addEventListener("metric",    (e) => onTrainingMetric(JSON.parse(e.data)));
  es.addEventListener("done",      (e) => onTrainingDone(JSON.parse(e.data)));
  es.addEventListener("error",     (e) => {
    try {
      const data = JSON.parse(e.data);
      const msg = data.message || "stream error";
      setTrainingStatus(msg, "error");
      showTrainingError(
        `Training stream failed: ${msg}`,
        "The server closed the SSE connection. Press Start training again to retry.",
      );
    } catch {/* heartbeat */}
    resetTrainingButton();
    es.close();
    trainingState.eventSource = null;
  });
  es.addEventListener("_eof", () => {
    es.close();
    trainingState.eventSource = null;
  });
  // EventSource auto-reconnects on transient network errors; only the
  // server-emitted "error" event above is fatal.
}

function onTrainingBaselines(ev) {
  trainingState.randomBaseline = ev.random;
  trainingState.nEpisodes = ev.n_episodes || trainingState.nEpisodes;
  $("trainingBaseline").textContent = formatScore(ev.random);
  // policy_kind ("neural" | "tabular") was added when the in-browser
  // learner was upgraded from the original 64-state tabular softmax to
  // a numpy linear policy + value baseline over a 31-dim feature vector.
  // Show it in the status so judges can see which policy is running.
  const kind = ev.policy_kind || "policy";
  setTrainingStatus(
    `Training on ${shortTaskName(ev.task_id)} · ${ev.action_menu_size} actions · ${ev.n_episodes} episodes · ${kind}`,
    "ok",
  );
  paintTrainingChart();
}

function onTrainingMetric(ev) {
  trainingState.points.push(ev);
  $("trainingEpisodeNow").textContent = `${ev.episode} / ${trainingState.nEpisodes}`;
  $("trainingRolling").textContent = formatScore(ev.rolling_mean);
  const lift = ev.lift_over_random;
  const liftEl = $("trainingLift");
  liftEl.textContent = `${lift >= 0 ? "+" : ""}${lift.toFixed(3)}`;
  liftEl.classList.toggle("negative", lift < 0);
  paintTrainingChart();
}

function onTrainingDone(ev) {
  const lift = ev.lift_over_random;
  const sign = lift >= 0 ? "+" : "";
  setTrainingStatus(
    `Done — final rolling mean ${ev.final_rolling_mean.toFixed(3)} (${sign}${lift.toFixed(3)} vs random) over ${ev.n_episodes} episodes.`,
    lift > 0.05 ? "ok" : "",
  );
  resetTrainingButton();
  paintTrainingChart(/* terminal */ true);
  renderRubricBreakdown(ev.rubric_breakdown);
}

// Show the rubric decomposition once an episode completes. The breakdown
// makes the ~0.66 ceiling tangible: judges can see e.g. cause=0.95 but
// chain=0.0 and immediately understand the structural limitation
// (chain accuracy needs NLP-level extraction of effect strings).
function renderRubricBreakdown(rubrics) {
  const host = $("trainingRubrics");
  const body = $("trainingRubricsBody");
  if (!host || !body || !Array.isArray(rubrics) || rubrics.length === 0) return;
  const fmt = (n) => (typeof n === "number" ? n.toFixed(3) : "—");
  body.innerHTML = "";
  for (const r of rubrics) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.rubric}</td>` +
      `<td class="num">${fmt(r.weight)}</td>` +
      `<td class="num">${fmt(r.mean_raw_score)}</td>` +
      `<td class="num">${fmt(r.mean_weighted_score)}</td>`;
    body.appendChild(tr);
  }
  host.hidden = false;
}

// ---------- chart ----------------------------------------------------------

function resetTrainingChart() {
  const host = $("trainingChart");
  if (!host) return;
  host.classList.remove("has-data");
  // Keep the empty-state node, drop any prior svg.
  Array.from(host.querySelectorAll("svg")).forEach((s) => s.remove());
}

// Render an in-place error overlay inside the training chart container so a
// failed start (HTTP 5xx, missing task, dropped SSE) is impossible to miss.
// Without this, the chart still showed "Press Start training to begin a
// fresh run" even after the run had blown up — same silent-failure pattern
// we already fixed for the live-investigation panel.
function showTrainingError(message, hint) {
  const host = $("trainingChart");
  if (!host) return;
  // Drop any prior svg + the default empty-state node so the error is
  // the only visible thing in the chart container.
  host.classList.remove("has-data");
  Array.from(host.querySelectorAll("svg")).forEach((s) => s.remove());
  const empty = host.querySelector(".training-chart-empty");
  if (empty) empty.hidden = true;
  let err = host.querySelector(".training-chart-error");
  if (!err) {
    err = document.createElement("div");
    err.className = "training-chart-error";
    err.setAttribute("role", "alert");
    host.appendChild(err);
  }
  err.innerHTML =
    `<div class="training-chart-error-label">TRAINING ERROR</div>` +
    `<div class="training-chart-error-msg">${escapeHtml(message)}</div>` +
    (hint ? `<div class="training-chart-error-hint">${escapeHtml(hint)}</div>` : "");
  err.hidden = false;
}

function clearTrainingError() {
  const host = $("trainingChart");
  if (!host) return;
  const err = host.querySelector(".training-chart-error");
  if (err) err.remove();
  const empty = host.querySelector(".training-chart-empty");
  if (empty) empty.hidden = false;
}

function resetTrainingSummary() {
  $("trainingEpisodeNow").textContent = `0 / ${trainingState.nEpisodes}`;
  $("trainingBaseline").textContent = "—";
  $("trainingRolling").textContent = "—";
  const liftEl = $("trainingLift");
  liftEl.textContent = "—";
  liftEl.classList.remove("negative");
}

function paintTrainingChart(terminal = false) {
  const host = $("trainingChart");
  if (!host) return;
  const points = trainingState.points;
  const baseline = trainingState.randomBaseline;
  if (!points.length && baseline === null) return;

  host.classList.add("has-data");

  // Layout: pad room for axis labels on left + bottom.
  const rect = host.getBoundingClientRect();
  const W = Math.max(420, Math.floor(rect.width - 36));
  const H = 320;
  const pad = { top: 18, right: 24, bottom: 36, left: 44 };
  const innerW = W - pad.left - pad.right;
  const innerH = H - pad.top - pad.bottom;

  const xMax = Math.max(trainingState.nEpisodes, 1);
  // Y-axis fixed 0..1 — env reward is bounded, and a fixed scale lets the
  // viewer see the curve climb toward the ceiling instead of having the
  // axis auto-rescale and hide progress.
  const yMin = 0, yMax = 1;

  const xOf = (ep) => pad.left + (ep / xMax) * innerW;
  const yOf = (s) => pad.top + (1 - (s - yMin) / (yMax - yMin)) * innerH;

  // (Re)build the SVG from scratch each repaint. The chart is small (≤500
  // points) so this is cheaper than a diff renderer and keeps the code
  // legible.
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.classList.add("training-chart-svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "none");
  svg.setAttribute("role", "img");

  appendGridAndAxes(svg, pad, innerW, innerH, xMax, yMin, yMax);

  if (baseline !== null && Number.isFinite(baseline)) {
    const yb = yOf(baseline);
    const line = document.createElementNS(SVG_NS, "line");
    line.classList.add("training-baseline-line");
    line.setAttribute("x1", pad.left);
    line.setAttribute("x2", pad.left + innerW);
    line.setAttribute("y1", yb);
    line.setAttribute("y2", yb);
    svg.appendChild(line);

    const lbl = document.createElementNS(SVG_NS, "text");
    lbl.classList.add("training-baseline-label");
    lbl.setAttribute("x", pad.left + innerW - 6);
    lbl.setAttribute("y", yb - 6);
    lbl.setAttribute("text-anchor", "end");
    lbl.textContent = `random ${baseline.toFixed(3)}`;
    svg.appendChild(lbl);
  }

  if (points.length) {
    // Raw per-episode dots (low-opacity primary).
    const rawG = document.createElementNS(SVG_NS, "g");
    rawG.classList.add("training-raw-points");
    for (const p of points) {
      const c = document.createElementNS(SVG_NS, "circle");
      c.setAttribute("cx", xOf(p.episode));
      c.setAttribute("cy", yOf(p.score));
      c.setAttribute("r", 1.6);
      rawG.appendChild(c);
    }
    svg.appendChild(rawG);

    // Trained-policy rolling-mean line + soft fill underneath.
    const linePts = points
      .map((p) => `${xOf(p.episode).toFixed(2)},${yOf(p.rolling_mean).toFixed(2)}`)
      .join(" ");
    const last = points[points.length - 1];

    const area = document.createElementNS(SVG_NS, "polygon");
    area.classList.add("training-trained-area");
    const xFirst = xOf(points[0].episode).toFixed(2);
    const xLast = xOf(last.episode).toFixed(2);
    const yFloor = (pad.top + innerH).toFixed(2);
    area.setAttribute("points", `${xFirst},${yFloor} ${linePts} ${xLast},${yFloor}`);
    svg.appendChild(area);

    const line = document.createElementNS(SVG_NS, "polyline");
    line.classList.add("training-trained-line");
    line.setAttribute("points", linePts);
    svg.appendChild(line);

    // Current-position marker — small primary dot at the latest rolling mean.
    const dot = document.createElementNS(SVG_NS, "circle");
    dot.classList.add("training-current-marker");
    dot.setAttribute("cx", xOf(last.episode));
    dot.setAttribute("cy", yOf(last.rolling_mean));
    dot.setAttribute("r", terminal ? 4 : 3);
    svg.appendChild(dot);
  }

  // Replace any prior svg in-place.
  Array.from(host.querySelectorAll("svg")).forEach((s) => s.remove());
  host.appendChild(svg);
}

function appendGridAndAxes(svg, pad, innerW, innerH, xMax, yMin, yMax) {
  const grid = document.createElementNS(SVG_NS, "g");
  grid.classList.add("training-grid");

  const Y_TICKS = [0, 0.25, 0.5, 0.75, 1.0];
  for (const t of Y_TICKS) {
    const y = pad.top + (1 - (t - yMin) / (yMax - yMin)) * innerH;
    const ln = document.createElementNS(SVG_NS, "line");
    ln.setAttribute("x1", pad.left);
    ln.setAttribute("x2", pad.left + innerW);
    ln.setAttribute("y1", y);
    ln.setAttribute("y2", y);
    grid.appendChild(ln);
  }
  svg.appendChild(grid);

  const axis = document.createElementNS(SVG_NS, "g");
  axis.classList.add("training-axis");

  const yAx = document.createElementNS(SVG_NS, "line");
  yAx.setAttribute("x1", pad.left);
  yAx.setAttribute("x2", pad.left);
  yAx.setAttribute("y1", pad.top);
  yAx.setAttribute("y2", pad.top + innerH);
  axis.appendChild(yAx);

  for (const t of Y_TICKS) {
    const y = pad.top + (1 - (t - yMin) / (yMax - yMin)) * innerH;
    const tx = document.createElementNS(SVG_NS, "text");
    tx.setAttribute("x", pad.left - 8);
    tx.setAttribute("y", y + 3);
    tx.setAttribute("text-anchor", "end");
    tx.textContent = t.toFixed(2);
    axis.appendChild(tx);
  }

  const xAx = document.createElementNS(SVG_NS, "line");
  xAx.setAttribute("x1", pad.left);
  xAx.setAttribute("x2", pad.left + innerW);
  xAx.setAttribute("y1", pad.top + innerH);
  xAx.setAttribute("y2", pad.top + innerH);
  axis.appendChild(xAx);

  // ~5-6 evenly spaced episode ticks.
  const N = 5;
  for (let i = 0; i <= N; i++) {
    const ep = Math.round((xMax * i) / N);
    const x = pad.left + (ep / xMax) * innerW;
    const tx = document.createElementNS(SVG_NS, "text");
    tx.setAttribute("x", x);
    tx.setAttribute("y", pad.top + innerH + 16);
    tx.setAttribute("text-anchor", "middle");
    tx.textContent = String(ep);
    axis.appendChild(tx);
  }

  // Axis labels.
  const yLabel = document.createElementNS(SVG_NS, "text");
  yLabel.setAttribute("x", 12);
  yLabel.setAttribute("y", pad.top + innerH / 2);
  yLabel.setAttribute("text-anchor", "middle");
  yLabel.setAttribute("transform", `rotate(-90, 12, ${pad.top + innerH / 2})`);
  yLabel.textContent = "score";
  axis.appendChild(yLabel);

  const xLabel = document.createElementNS(SVG_NS, "text");
  xLabel.setAttribute("x", pad.left + innerW / 2);
  xLabel.setAttribute("y", pad.top + innerH + 30);
  xLabel.setAttribute("text-anchor", "middle");
  xLabel.textContent = "episode";
  axis.appendChild(xLabel);

  svg.appendChild(axis);
}

// ---------- helpers --------------------------------------------------------

function setTrainingStatus(msg, kind = "") {
  const el = $("trainingStatus");
  if (!el) return;
  el.dataset.kind = kind;
  el.textContent = msg;
}

function resetTrainingButton() {
  const btn = $("trainingStartBtn");
  if (!btn) return;
  btn.disabled = false;
  const lbl = btn.querySelector(".btn-label");
  if (lbl) lbl.textContent = "Start training";
}

function formatScore(v) {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(3);
}

function clamp(v, lo, hi) {
  if (!Number.isFinite(v)) return lo;
  return Math.max(lo, Math.min(hi, v));
}
