// ============================================================================
// COVID-19 Analytics Platform - Frontend
// ============================================================================

const byId = (id) => document.getElementById(id);

// App State
let activeJobId = null;
let pollTimer = null;
let authToken = null;
let currentDetailRunId = null;
let _assistantIndexed = false;

// Utilities
function apiUrl(path) {
  if (/^https?:\/\//i.test(path)) return path;
  return path.startsWith("/") ? path : `/${path}`;
}

function formatNumber(value, digits = 0) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  if (!Number.isFinite(num)) return String(value);
  return num.toLocaleString(undefined, { maximumFractionDigits: digits });
}

function formatMetric(value) {
  if (value === null || value === undefined || value === "") return "-";
  const num = Number(value);
  return Number.isFinite(num) ? num.toFixed(2) : String(value);
}

function formatDate(dateString) {
  if (!dateString) return "-";
  try {
    const date = new Date(dateString);
    return date.toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit"
    });
  } catch {
    return dateString;
  }
}

function setStatus(message, kind = "muted") {
  const el = byId("runStatus");
  el.className = `status-message ${kind}`;
  el.textContent = message;
}

function apiMessage(data, fallback) {
  return data?.error?.message || data?.detail || data?.message || fallback;
}

// Authentication
async function ensureAuthenticated() {
  if (authToken) return true;

  try {
    const response = await fetch(apiUrl("/auth/login"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: "analyst", password: "analyst123" }),
    });

    if (!response.ok) throw new Error("Authentication failed");

    const data = await response.json();
    authToken = data.access_token;
    return true;
  } catch (error) {
    console.error("Auth error:", error);
    setStatus("Authentication failed", "error");
    return false;
  }
}

// API Calls
async function api(url, options = {}) {
  await ensureAuthenticated();

  const headers = { ...(options.headers || {}) };
  if (authToken) {
    headers["Authorization"] = `Bearer ${authToken}`;
  }
  if (options.body && !headers["Content-Type"]) {
    headers["Content-Type"] = "application/json";
  }

  const response = await fetch(apiUrl(url), { ...options, headers });
  const contentType = response.headers.get("content-type") || "";
  const data = contentType.includes("application/json") ? await response.json() : await response.text();

  if (!response.ok) {
    throw new Error(apiMessage(data, `Request failed (${response.status})`));
  }
  return data;
}

// Country Loading
async function fillCountries() {
  const select = byId("country");
  select.disabled = true;

  try {
    const data = await api("/api/countries");
    select.innerHTML = "";
    data.countries.forEach((country) => {
      const option = document.createElement("option");
      option.value = country;
      option.textContent = country;
      option.selected = country === data.default_country;
      select.appendChild(option);
    });
  } catch (error) {
    select.innerHTML = '<option value="India">India</option>';
  } finally {
    select.disabled = false;
  }
}

// Table Rendering
function renderEmptyRow(tableBody, columnCount, message) {
  tableBody.innerHTML = "";
  const row = document.createElement("tr");
  row.innerHTML = `<td colspan="${columnCount}" class="empty-state">${message}</td>`;
  tableBody.appendChild(row);
}

function formatTimestamp(dateString) {
  if (!dateString) return "-";
  const date = new Date(dateString);
  return date.toLocaleString("en-US", {
    month: "short",
    day: "2-digit",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

// Load dashboard summary cards
async function loadSummary() {
  try {
    const data = await api('/api/summary');
    const trackedValue = byId('trackedCountriesValue');
    const horizonValue = byId('forecastHorizonValue');
    const runsValue = byId('forecastRunsValue');
    const analyticsValue = byId('liveAnalyticsValue');

    if (trackedValue) trackedValue.textContent = String(data.tracked_countries);
    if (horizonValue) horizonValue.textContent = `${data.default_horizon}d`;
    if (runsValue) runsValue.textContent = String(data.total_runs);
    if (analyticsValue) analyticsValue.textContent = data.live_analytics || 'Ready';

    const stats = {
      latestRun: data.latest_run,
    };
    window.__dashboardStats = stats;
  } catch (error) {
    console.warn('Summary load failed', error);
  }
}

// Load and Display Runs
async function loadRuns() {
  const body = byId("runsBody");
  renderEmptyRow(body, 8, "Loading analysis runs...");

  try {
    const data = await api("/api/runs?limit=20");
    body.innerHTML = "";

    if (!data.runs?.length) {
      renderEmptyRow(body, 8, "No analysis runs yet. Start one using the pipeline controls above.");
      return [];
    }

    data.runs.forEach((run) => {
      const row = document.createElement("tr");
      const statusClass = run.status.toLowerCase();
      row.innerHTML = `
        <td><button type="button" class="run-action rid" data-id="${run.id}" title="View details">${run.id}</button></td>
        <td>${formatTimestamp(run.created_at)}</td>
        <td><span class="badge ${statusClass}">${run.status}</span></td>
        <td>${run.country}</td>
        <td>${run.horizon}d</td>
        <td><strong>${formatMetric(run.mape)}</strong></td>
        <td>${formatMetric(run.rmse)}</td>
        <td>${formatMetric(run.mae)}</td>
      `;
      body.appendChild(row);
    });

    document.querySelectorAll(".rid").forEach((button) => {
      button.addEventListener("click", () => loadRunDetail(button.dataset.id));
    });

    if (!currentDetailRunId && data.runs.length) {
      loadRunDetail(data.runs[0].id);
    }

    // cache runs for local assistant lookup
    window.__runs = data.runs;

    return data.runs;
  } catch (error) {
    renderEmptyRow(body, 8, `Error loading runs: ${error.message}`);
    return [];
  }
}

// Load Run Details
async function loadRunDetail(id) {
  currentDetailRunId = id;
  const header = byId("detailHeader");
  const subtitle = byId("detailSubtitle");
  const img = byId("dashImg");
  const placeholder = byId("dashPlaceholder");
  header.textContent = "Loading details...";
  subtitle.textContent = "";
  if (img) {
    img.style.display = "none";
    img.src = "";
  }
  if (placeholder) {
    placeholder.textContent = "Loading dashboard preview...";
    placeholder.classList.remove("hidden");
  }

  try {
    const data = await api(`/api/runs/${id}`);
    const run = data.run;
    header.textContent = `Run #${run.id}`;
    subtitle.textContent = `${run.country} • ${run.horizon}-day forecast • Status: ${run.status.toUpperCase()}`;

    // CFR Table
    const cfrBody = byId("cfrBody");
    if (!data.top_cfr?.length) {
      renderEmptyRow(cfrBody, 4, "No CFR data for this run");
    } else {
      cfrBody.innerHTML = "";
      data.top_cfr.forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.rank_no}</td>
          <td>${item.country}</td>
          <td><strong>${formatMetric(item.cfr_pct)}%</strong></td>
          <td>${formatNumber(item.total_cases)}</td>
        `;
        cfrBody.appendChild(row);
      });
    }

    // Forecast Table
    const forecastBody = byId("forecastBody");
    if (!data.forecast?.length) {
      renderEmptyRow(forecastBody, 4, "No forecast data available");
    } else {
      forecastBody.innerHTML = "";
      data.forecast.slice(0, 15).forEach((item) => {
        const row = document.createElement("tr");
        row.innerHTML = `
          <td>${item.forecast_date}</td>
          <td><strong>${formatNumber(item.forecast_value)}</strong></td>
          <td>${formatNumber(item.lower_bound)}</td>
          <td>${formatNumber(item.upper_bound)}</td>
        `;
        forecastBody.appendChild(row);
      });
    }

    // Dashboard Image
    if (run.dashboard_path && img) {
      if (placeholder) placeholder.classList.add("hidden");
      if (img._dashboardObjectUrl) {
        URL.revokeObjectURL(img._dashboardObjectUrl);
        img._dashboardObjectUrl = null;
      }
      img.style.display = "none";
      img.onerror = () => {
        if (img) img.style.display = "none";
        if (placeholder) {
          placeholder.textContent = "Dashboard image could not be loaded.";
          placeholder.classList.remove("hidden");
        }
      };
      img.onload = () => {
        img.style.display = "block";
        if (placeholder) placeholder.classList.add("hidden");
      };

      try {
        await ensureAuthenticated();
        const response = await fetch(apiUrl(`/api/runs/${id}/dashboard?ts=${Date.now()}`), {
          headers: authToken ? { Authorization: `Bearer ${authToken}` } : {},
        });
        if (!response.ok) {
          throw new Error(`Dashboard request failed (${response.status})`);
        }
        const blob = await response.blob();
        const objectUrl = URL.createObjectURL(blob);
        img._dashboardObjectUrl = objectUrl;
        img.src = objectUrl;
      } catch (error) {
        if (img) img.style.display = "none";
        if (placeholder) {
          placeholder.textContent = "Dashboard image could not be loaded.";
          placeholder.classList.remove("hidden");
        }
        console.error("Dashboard load failed", error);
      }
    } else if (placeholder) {
      placeholder.textContent = "No dashboard available for this run.";
      placeholder.classList.remove("hidden");
    }
  } catch (error) {
    header.textContent = `Error: ${error.message}`;
    subtitle.textContent = "Failed to load run details";
    renderEmptyRow(byId("cfrBody"), 4, "Error loading data");
    renderEmptyRow(byId("forecastBody"), 4, "Error loading data");
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function setAssistantMessage(role, text) {
  const history = byId('assistantHistory');
  if (!history) return;
  const message = document.createElement('div');
  message.className = `assistant-message ${role}`;
  message.innerHTML = escapeHtml(text);

  const placeholder = history.querySelector('.assistant-empty');
  if (placeholder) placeholder.remove();

  history.appendChild(message);
  history.scrollTop = history.scrollHeight;
}

function renderAssistantContexts(contexts) {
  const container = byId('assistantContexts');
  if (!container) return;
  if (!contexts?.length) {
    container.classList.add('hidden');
    container.innerHTML = '';
    return;
  }

  const html = [`<h4>Retrieved contexts</h4>`].concat(contexts.map((context) => `
      <pre><strong>${escapeHtml(context.source)}</strong>\n${escapeHtml(context.content)}</pre>
    `));
  container.innerHTML = html.join('');
  container.classList.remove('hidden');
}

async function indexAssistantDocs() {
  try {
    setStatus('Indexing assistant documents...', 'loading');
    const data = await api('/assistant/index', { method: 'POST' });
    _assistantIndexed = true;
    setStatus(`Indexed ${data.documents} runs for the assistant.`, 'success');
  } catch (error) {
    _assistantIndexed = false;
    setStatus(`Assistant indexing failed: ${error.message}`, 'error');
    throw error;
  }
}

async function askAssistant() {
  const questionEl = byId('assistantQuestion');
  if (!questionEl) return;
  const question = questionEl.value.trim();
  if (!question) {
    setStatus('Please type a question for the assistant.', 'error');
    return;
  }

  setAssistantMessage('user', question);
  questionEl.value = '';
  renderAssistantContexts([]);
  setStatus('Getting assistant answer...', 'loading');

  try {
    // try lightweight local handlers for common questions first
    const qLower = question.toLowerCase();
    if (qLower.includes("latest run") && qLower.includes("mape")) {
      const runs = window.__runs || [];
      if (runs.length) {
        const latest = runs[0];
        const ans = `Latest run (#${latest.id}) MAPE = ${latest.mape ?? 'N/A'}`;
        setAssistantMessage('bot', ans);
        setStatus('Assistant (local) response ready.', 'success');
        return;
      }
    }

    if (qLower.includes('highest cfr') || qLower.includes('highest case fatality') || qLower.includes('highest fatality')) {
      // Check top CFR across recent runs (limit 10)
      const runs = (window.__runs || []).slice(0, 10);
      if (!runs.length) {
        setAssistantMessage('bot', 'No runs available to inspect.');
        setStatus('Assistant (local) finished.', 'muted');
        return;
      }
      let best = { country: null, cfr: -1, run_id: null };
      for (const r of runs) {
        try {
          const details = await api(`/api/runs/${r.id}`);
          const top = (details.top_cfr || [])[0];
          if (top && typeof top.cfr_pct === 'number' && top.cfr_pct > best.cfr) {
            best = { country: top.country, cfr: top.cfr_pct, run_id: r.id };
          }
        } catch (e) {
          // ignore per-run failures
        }
      }
      if (best.country) {
        setAssistantMessage('bot', `Highest CFR across recent runs: ${best.country} (${best.cfr.toFixed(2)}%) in run #${best.run_id}`);
        setStatus('Assistant (local) response ready.', 'success');
      } else {
        setAssistantMessage('bot', 'Could not determine highest CFR from recent runs.');
        setStatus('Assistant (local) finished.', 'muted');
      }
      return;
    }

    if (qLower.includes('summarize') && qLower.includes('forecast')) {
      const runs = window.__runs || [];
      if (!runs.length) {
        setAssistantMessage('bot', 'No runs available to summarize.');
        setStatus('Assistant (local) finished.', 'muted');
        return;
      }
      const latest = runs[0];
      try {
        const det = await api(`/api/runs/${latest.id}`);
        const fc = (det.forecast || []).slice(0, 5).map(f => `${f.forecast_date}:${Math.round(f.forecast_value)}`).join(', ');
        setAssistantMessage('bot', `Run #${latest.id} forecast sample: ${fc || 'No forecast points available.'}`);
        setStatus('Assistant (local) response ready.', 'success');
        return;
      } catch (e) {
        // fall through to server-based query
      }
    }

    // fallback to server-side RAG
    if (!_assistantIndexed) {
      try { await indexAssistantDocs(); } catch { }
    }
    const response = await api('/assistant/query', {
      method: 'POST',
      body: JSON.stringify({ question, top_k: 3 }),
    });
    if (response.answer?.includes('No indexed documents')) {
      try { await indexAssistantDocs(); } catch { }
      return askAssistant();
    }
    setAssistantMessage('bot', response.answer || 'No answer returned.');
    renderAssistantContexts(response.contexts || []);
    setStatus('Assistant response received.', 'success');
  } catch (error) {
    setAssistantMessage('bot', `Error: ${error.message}`);
    setStatus(`Assistant failed: ${error.message}`, 'error');
  }
}

// Job Status Polling
function setRunning(isRunning) {
  byId("runBtn").disabled = isRunning;
  byId("cancelBtn").disabled = !isRunning;
}

async function pollJob(jobId) {
  try {
    const data = await api(`/api/jobs/${jobId}`);
    const job = data.job;

    if (job.status === "running" || job.status === "queued") {
      const progress = job.progress || 0;
      const message = job.message || `${job.status}...`;
      setStatus(`${message} (${progress}%)`, "loading");
      pollTimer = setTimeout(() => pollJob(jobId), 1200);
    } else if (["completed", "failed", "cancelled"].includes(job.status)) {
      clearTimeout(pollTimer);
      activeJobId = null;
      setRunning(false);

      if (job.status === "completed") {
        setStatus("✓ Pipeline completed successfully!", "success");
        await loadRuns();
        if (job.result?.run_id) {
          setTimeout(() => loadRunDetail(job.result.run_id), 500);
        }
      } else if (job.status === "failed") {
        setStatus(`✗ Pipeline failed: ${job.error || "Unknown error"}`, "error");
      } else {
        setStatus("⊘ Pipeline cancelled", "error");
      }
    }
  } catch (error) {
    clearTimeout(pollTimer);
    activeJobId = null;
    setRunning(false);
    setStatus(`✗ Error: ${error.message}`, "error");
  }
}

// Run Pipeline
async function runPipeline() {
  clearTimeout(pollTimer);
  const country = byId("country").value;
  const horizon = Number(byId("horizon").value || 30);

  if (!country) {
    setStatus("Please select a country", "error");
    return;
  }

  if (horizon < 7 || horizon > 90) {
    setStatus("Forecast horizon must be between 7 and 90 days", "error");
    return;
  }

  setRunning(true);
  setStatus("Preparing pipeline...", "loading");

  try {
    const payload = {
      country,
      horizon,
      with_dashboard: byId("withDashboard").checked,
    };
    const data = await api("/api/jobs", { method: "POST", body: JSON.stringify(payload) });
    activeJobId = data.job_id;
    await pollJob(activeJobId);
  } catch (error) {
    activeJobId = null;
    setRunning(false);
    setStatus(`✗ Error: ${error.message}`, "error");
  }
}

// Cancel Pipeline
async function cancelPipeline() {
  if (!activeJobId) return;
  try {
    await api(`/api/jobs/${activeJobId}/cancel`, { method: "POST" });
    setStatus("Cancellation requested...", "loading");
  } catch (error) {
    setStatus(`✗ Error: ${error.message}`, "error");
  }
}

// Export runs as CSV
async function exportRunsCSV() {
  try {
    const resp = await api('/api/runs?limit=200');
    const runs = resp.runs || [];
    if (!runs.length) {
      setStatus('No runs to export', 'error');
      return;
    }
    const headers = ['id', 'created_at', 'status', 'country', 'horizon', 'mape', 'rmse', 'mae'];
    const rows = runs.map(r => headers.map(h => JSON.stringify(r[h] ?? '')).join(','));
    const csv = [headers.join(','), ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `runs_export_${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    setStatus('Exported runs CSV', 'success');
  } catch (err) {
    setStatus(`Export failed: ${err.message}`, 'error');
  }
}

// Auto-refresh
let _autoRefreshTimer = null;
function setAutoRefresh(enabled) {
  if (_autoRefreshTimer) {
    clearInterval(_autoRefreshTimer);
    _autoRefreshTimer = null;
  }
  if (enabled) {
    _autoRefreshTimer = setInterval(loadRuns, 30_000);
    setStatus('Auto-refresh enabled (30s)', 'muted');
  } else {
    setStatus('Auto-refresh disabled', 'muted');
  }
}

// Theme toggle
let currentTheme = localStorage.getItem('ui_theme') || 'light';

function applyTheme(theme) {
  if (theme === 'dark') document.documentElement.classList.add('theme-dark');
  else document.documentElement.classList.remove('theme-dark');
  localStorage.setItem('ui_theme', theme);
}

function updateThemeToggleLabel(theme) {
  const toggleBtn = byId('toggleTheme');
  const headerBtn = byId('themeToggleHeader');
  const label = theme === 'dark' ? 'Light Mode' : 'Dark Mode';
  if (toggleBtn) toggleBtn.textContent = label;
  if (headerBtn) headerBtn.textContent = label;
}

function setTheme(theme) {
  currentTheme = theme;
  applyTheme(theme);
  updateThemeToggleLabel(theme);
}

// Dashboard placeholder handling and image fallback
function setupDashboardFallback() {
  const img = byId('dashImg');
  const placeholder = byId('dashPlaceholder');
  if (!img || !placeholder) return;
  img.onload = () => { placeholder.classList.add('hidden'); img.style.display = 'block'; };
  img.onerror = () => { img.style.display = 'none'; placeholder.classList.remove('hidden'); };
}

// Update Header Time
function updateHeaderTime() {
  const timeEl = byId("headerTime");
  if (timeEl) {
    const now = new Date();
    timeEl.textContent = now.toLocaleTimeString("en-US", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
  }
}

// Event Listeners
byId("runBtn").addEventListener("click", runPipeline);
byId("cancelBtn").addEventListener("click", cancelPipeline);

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  setRunning(false);
  fillCountries();
  const refreshBtn = byId("refreshBtn");
  if (refreshBtn) {
    refreshBtn.addEventListener("click", loadRuns);
  }
  loadSummary();
  const runsPromise = loadRuns();
  updateHeaderTime();
  setInterval(updateHeaderTime, 1000);
  // Wire up extra UI options
  const exportBtn = byId('exportCsv');
  if (exportBtn) exportBtn.addEventListener('click', exportRunsCSV);
  const autoCb = byId('autoRefresh');
  if (autoCb) autoCb.addEventListener('change', (e) => setAutoRefresh(e.target.checked));
  const toggleThemeBtn = byId('toggleTheme');
  const headerThemeBtn = byId('themeToggleHeader');
  if (toggleThemeBtn) toggleThemeBtn.addEventListener('click', (event) => { event.preventDefault(); setTheme(currentTheme === 'light' ? 'dark' : 'light'); });
  if (headerThemeBtn) headerThemeBtn.addEventListener('click', (event) => { event.preventDefault(); setTheme(currentTheme === 'light' ? 'dark' : 'light'); });

  const assistantSend = byId('assistantSend');
  const assistantQuestion = byId('assistantQuestion');
  const indexDocsBtn = byId('indexDocsBtn');
  if (assistantSend) assistantSend.addEventListener('click', askAssistant);
  if (assistantQuestion) {
    assistantQuestion.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        askAssistant();
      }
    });
  }
  if (indexDocsBtn) indexDocsBtn.addEventListener('click', indexAssistantDocs);
  document.querySelectorAll('.sample-question').forEach((button) => {
    button.addEventListener('click', () => {
      const question = button.textContent.trim();
      const questionEl = byId('assistantQuestion');
      if (questionEl) questionEl.value = question;
      askAssistant();
    });
  });
  runsPromise.then(() => indexAssistantDocs()).catch(() => { });

  const trackedBtn = byId('statTrackedCountries');
  const horizonBtn = byId('statForecastHorizon');
  const runsBtn = byId('statForecastRuns');
  const liveBtn = byId('statLiveAnalytics');

  if (trackedBtn) trackedBtn.addEventListener('click', () => {
    const countrySelect = byId('country');
    if (countrySelect) {
      countrySelect.focus();
      setStatus('Select a country to run analysis.', 'muted');
    }
  });
  if (horizonBtn) horizonBtn.addEventListener('click', () => {
    const horizonInput = byId('horizon');
    if (horizonInput) {
      horizonInput.focus();
      setStatus('Adjust your forecast horizon and run the pipeline.', 'muted');
    }
  });
  if (runsBtn) runsBtn.addEventListener('click', () => {
    const runsSection = byId('runsBody');
    if (runsSection) runsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  });
  if (liveBtn) liveBtn.addEventListener('click', () => {
    if (window.__dashboardStats?.latestRun) {
      loadRunDetail(window.__dashboardStats.latestRun.id);
      setStatus('Showing latest run details.', 'muted');
    } else {
      setStatus('No run details available yet.', 'error');
    }
  });

  // More menu behavior
  const moreBtn = byId('moreBtn');
  const moreOptions = byId('moreOptions');
  if (moreBtn && moreOptions) {
    moreBtn.addEventListener('click', (event) => {
      event.stopPropagation();
      moreOptions.classList.toggle('hidden');
    });
    moreOptions.addEventListener('click', (event) => event.stopPropagation());
    document.addEventListener('click', (ev) => {
      if (!moreBtn.contains(ev.target) && !moreOptions.contains(ev.target)) {
        moreOptions.classList.add('hidden');
      }
    });
  }
  // apply saved theme
  setTheme(currentTheme);
  setupDashboardFallback();
});
