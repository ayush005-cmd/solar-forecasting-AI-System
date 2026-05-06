/* ============================================================
   Solar Power Prediction — Frontend Script
   Chart.js + Vanilla JS SPA
   ============================================================ */

'use strict';

// ── Navigation ──────────────────────────────────────────────
function navigate(id) {
  document.querySelectorAll('section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.nav-links a').forEach(a => a.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelector(`.nav-links a[data-section="${id}"]`)?.classList.add('active');

  if (id === 'analytics')  loadAnalytics();
  if (id === 'models')     loadModels();
  if (id === 'clusters')   loadClusters();
  if (id === 'dashboard')  loadSummary();
  if (id === 'evaluation') loadEvaluation();
}

document.querySelectorAll('.nav-links a').forEach(a => {
  a.addEventListener('click', e => { e.preventDefault(); navigate(a.dataset.section); });
});

// ── Dark / Light mode ────────────────────────────────────────
const darkToggle = document.getElementById('dark-toggle');

function applyTheme(isLight) {
  document.body.classList.toggle('light', isLight);
  darkToggle.textContent = isLight ? '◑ Dark mode' : '☀ Light mode';
  try { localStorage.setItem('theme', isLight ? 'light' : 'dark'); } catch(e) {}
}

(function() {
  let saved = 'dark';
  try { saved = localStorage.getItem('theme') || 'dark'; } catch(e) {}
  applyTheme(saved === 'light');
})();

darkToggle.addEventListener('click', () => {
  applyTheme(!document.body.classList.contains('light'));
});

// ── Toast ────────────────────────────────────────────────────
function showToast(msg, type = 'success') {
  const container = document.getElementById('toast-container');
  const t = document.createElement('div');
  t.className = `toast ${type}`;
  t.innerHTML = `<span>${type === 'success' ? '✓' : '✕'}</span> ${msg}`;
  container.appendChild(t);
  setTimeout(() => t.remove(), 4000);
}

// ── Prediction Form ──────────────────────────────────────────
const predictForm = document.getElementById('predict-form');
const predictBtn  = document.getElementById('predict-btn');
const btnText     = document.getElementById('btn-text');
const spinner     = document.getElementById('spinner');
const outputCard  = document.getElementById('output-card');

// Guard: if any element is missing, JS won't crash silently
if (!predictForm || !predictBtn || !outputCard) {
  console.error('SolarAI: Required DOM elements not found. Check that index.html is served by Flask.');
}

predictForm && predictForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const temperature = document.getElementById('temperature').value;
  const irradiance  = document.getElementById('irradiance').value;
  const humidity    = document.getElementById('humidity').value;
  const windSpeed   = document.getElementById('wind-speed').value;
  const hour        = document.getElementById('hour').value;

  // FIX: use .trim() === '' so that 0 is treated as a valid value (not falsy)
  if (temperature.trim() === '' || irradiance.trim() === '' ||
      humidity.trim()    === '' || windSpeed.trim()   === '' ||
      hour.trim()        === '') {
    showToast('Please fill all fields', 'error');
    return;
  }

  const body = {
    temperature: parseFloat(temperature),
    irradiance:  parseFloat(irradiance),
    humidity:    parseFloat(humidity),
    windSpeed:   parseFloat(windSpeed),
    hour:        parseInt(hour, 10),
  };

  if (Object.values(body).some(v => isNaN(v))) {
    showToast('Invalid numeric input', 'error');
    return;
  }

  predictBtn.disabled = true;
  btnText.textContent = 'Predicting\u2026';
  spinner.style.display = 'block';
  outputCard.style.display = 'none';
  document.getElementById('placeholder-card').style.display = 'none';

  try {
    const res = await fetch('/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });

    // Safe JSON parse: if body is empty/HTML, show the raw text instead
    let data;
    const rawText = await res.text();
    try {
      data = JSON.parse(rawText);
    } catch (_) {
      throw new Error(
        res.ok
          ? 'Server returned non-JSON response. Is Flask running on port 5000?'
          : 'Server error ' + res.status + ': ' + rawText.slice(0, 120)
      );
    }

    if (!res.ok) {
      throw new Error(data.error || ('Server error ' + res.status));
    }

    renderPrediction(data);
    showToast('Prediction complete!', 'success');

  } catch (err) {
    console.error('Prediction error:', err);
    showToast(err.message, 'error');
    showErrorCard(err.message);
  } finally {
    predictBtn.disabled = false;
    btnText.textContent  = 'Predict Output';
    spinner.style.display = 'none';
  }
});

function showErrorCard(msg) {
  outputCard.innerHTML =
    '<div class="card-title">Prediction Results</div>' +
    '<div style="text-align:center;padding:2rem 1rem;">' +
      '<div style="font-size:2rem;margin-bottom:1rem;">\u26a0</div>' +
      '<div style="color:var(--danger);font-weight:600;margin-bottom:.5rem;">Prediction Failed</div>' +
      '<div style="color:var(--text-dim);font-size:.875rem;">' + msg + '</div>' +
      '<div style="color:var(--text-muted);font-size:.78rem;margin-top:1rem;">' +
        'Make sure Flask is running: <code style="color:var(--gold)">python app.py</code>' +
      '</div>' +
    '</div>';
  outputCard.style.display = 'block';
}

function getOutputCardHTML() {
  return (
    '<div class="card-title">Prediction Results</div>' +
    '<div style="text-align:center;padding:1rem 0 .5rem;">' +
      '<div style="font-size:.8rem;color:var(--text-muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:.5rem;">Ensemble Forecast</div>' +
      '<div class="ensemble-value" id="val-ensemble">\u2014</div>' +
      '<div class="ensemble-unit">kW predicted output</div>' +
    '</div>' +
    '<div class="model-breakdown">' +
      '<div class="model-value-card"><div class="label">SVM</div><div class="value val-svm" id="val-svm">\u2014</div><div style="font-size:.72rem;color:var(--text-muted)">kW</div></div>' +
      '<div class="model-value-card"><div class="label">Bagging</div><div class="value val-bag" id="val-bag">\u2014</div><div style="font-size:.72rem;color:var(--text-muted)">kW</div></div>' +
      '<div class="model-value-card"><div class="label">Time-Series</div><div class="value val-ts" id="val-ts">\u2014</div><div style="font-size:.72rem;color:var(--text-muted)">kW</div></div>' +
    '</div>' +
    '<div class="confidence-bar-wrap">' +
      '<div class="confidence-label"><span>Model Confidence</span><span id="conf-pct">\u2014</span></div>' +
      '<div class="confidence-bar"><div class="confidence-fill" id="conf-fill" style="width:0%"></div></div>' +
    '</div>' +
    '<div id="outlier-badge"></div>'
  );
}

function renderPrediction(d) {
  outputCard.innerHTML = getOutputCardHTML();

  document.getElementById('val-ensemble').textContent = d.ensemblePrediction.toFixed(2);
  document.getElementById('val-svm').textContent      = d.svmPrediction.toFixed(2);
  document.getElementById('val-bag').textContent      = d.baggingPrediction.toFixed(2);
  document.getElementById('val-ts').textContent       = d.timeSeriesPrediction.toFixed(2);

  const pct = Math.round(d.confidence * 100);
  document.getElementById('conf-pct').textContent  = pct + '%';
  document.getElementById('conf-fill').style.width = pct + '%';

  const badge = document.getElementById('outlier-badge');
  if (d.outlierDetected) {
    badge.className   = 'outlier-badge outlier-yes';
    badge.textContent = '\u26a0 Outlier detected in input';
  } else {
    badge.className   = 'outlier-badge outlier-no';
    badge.textContent = '\u2713 Input values within normal range';
  }

  outputCard.style.display = 'block';
}

// ── Summary KPIs ─────────────────────────────────────────────
let summaryLoaded = false;
async function loadSummary() {
  if (summaryLoaded) return;
  try {
    const r = await fetch('/api/summary');
    const d = await r.json();
    document.getElementById('kpi-predictions').textContent = d.totalPredictions.toLocaleString();
    document.getElementById('kpi-output').textContent      = d.avgDailyOutput + ' kW';
    document.getElementById('kpi-accuracy').textContent    = d.bestModelAccuracy + '%';
    document.getElementById('kpi-sites').textContent       = d.sitesMonitored;
    document.getElementById('kpi-peak').textContent        = d.peakPower + ' kW';
    document.getElementById('kpi-confidence').textContent  = Math.round(d.avgConfidence * 100) + '%';
    summaryLoaded = true;
  } catch(e) { console.warn('Summary load failed', e); }
}

// ── Analytics Charts ─────────────────────────────────────────
let analyticsLoaded = false;
let tsChart, fiChart;

async function loadAnalytics() {
  if (analyticsLoaded) return;
  analyticsLoaded = true;

  const [tsRes, fiRes] = await Promise.all([
    fetch('/api/timeseries'),
    fetch('/api/feature-importance'),
  ]);
  const tsData = await tsRes.json();
  const fiData = await fiRes.json();

  const pts    = tsData.points.slice(-48);
  const labels = pts.map(p => {
    const d = new Date(p.timestamp);
    return String(d.getDate()).padStart(2,'0') + '/' + String(d.getHours()).padStart(2,'0') + 'h';
  });

  const baseLineOpts = { tension: 0.4, pointRadius: 0, borderWidth: 2, fill: false };

  tsChart = new Chart(document.getElementById('ts-chart'), {
    type: 'line',
    data: {
      labels,
      datasets: [
        { label: 'Actual',     data: pts.map(p=>p.actual),              borderColor:'#fbbf24', ...baseLineOpts },
        { label: 'SVM',        data: pts.map(p=>p.svmPredicted),        borderColor:'#22d3ee', ...baseLineOpts, borderDash:[4,3] },
        { label: 'Bagging',    data: pts.map(p=>p.baggingPredicted),    borderColor:'#34d399', ...baseLineOpts, borderDash:[6,3] },
        { label: 'TimeSeries', data: pts.map(p=>p.timeSeriesPredicted), borderColor:'#a78bfa', ...baseLineOpts, borderDash:[2,2] },
      ],
    },
    options: chartOptions('Power Output (kW)'),
  });

  fiChart = new Chart(document.getElementById('fi-chart'), {
    type: 'bar',
    data: {
      labels: fiData.items.map(i => i.feature),
      datasets: [{
        label: 'Importance',
        data: fiData.items.map(i => i.importance),
        backgroundColor: ['#fbbf24','#22d3ee','#34d399','#a78bfa','#f87171'],
        borderRadius: 6,
        borderSkipped: false,
      }],
    },
    options: {
      ...chartOptions('Importance Score'),
      indexAxis: 'y',
      plugins: { legend: { display: false }, ...chartOptions('').plugins },
    },
  });
}

// ── Model Comparison ─────────────────────────────────────────
let modelsLoaded = false;

async function loadModels() {
  if (modelsLoaded) return;
  modelsLoaded = true;

  const res  = await fetch('/api/model-comparison');
  const data = await res.json();
  const tbody = document.getElementById('model-tbody');
  tbody.innerHTML = '';

  data.models.forEach(m => {
    const isBest = m.model === data.bestModel;
    const r2pct  = Math.round(m.r2 * 100);
    const badgeHTML = isBest ? '<span class="best-badge">Best</span>' : '';
    tbody.innerHTML +=
      '<tr class="' + (isBest ? 'best-row' : '') + '">' +
        '<td>' +
          '<div class="model-name">' + m.model + ' ' + badgeHTML + '</div>' +
          '<div class="model-desc">' + m.description + '</div>' +
        '</td>' +
        '<td>' +
          '<div class="metric-val">' + m.mae + '</div>' +
          '<div class="metric-bar-wrap"><div class="metric-bar"><div class="metric-bar-fill" style="width:' + (100-Math.min(m.mae*10,100)) + '%"></div></div></div>' +
        '</td>' +
        '<td>' +
          '<div class="metric-val">' + m.rmse + '</div>' +
          '<div class="metric-bar-wrap"><div class="metric-bar"><div class="metric-bar-fill" style="width:' + (100-Math.min(m.rmse*8,100)) + '%"></div></div></div>' +
        '</td>' +
        '<td>' +
          '<div class="metric-val">' + r2pct + '%</div>' +
          '<div class="metric-bar-wrap"><div class="metric-bar"><div class="metric-bar-fill" style="width:' + r2pct + '%"></div></div></div>' +
        '</td>' +
      '</tr>';
  });
}

// ── Clustering ───────────────────────────────────────────────
let clustersLoaded = false;
let clusterChart;

const CLUSTER_COLORS = ['#fbbf24','#22d3ee','#34d399','#a78bfa'];
const CLUSTER_BORDER = ['#d97706','#0891b2','#059669','#7c3aed'];

async function loadClusters() {
  if (clustersLoaded) return;
  clustersLoaded = true;

  const res  = await fetch('/api/clustering');
  const data = await res.json();

  const byCluster = Array.from({length: data.numClusters}, () => []);
  data.points.forEach(p => byCluster[p.cluster].push({ x: p.irradiance, y: p.temperature, r: 5 }));

  const centroidDataset = {
    label: 'Centroids',
    data: data.centroids.map(c => ({ x: c.irradiance, y: c.temperature })),
    backgroundColor: 'rgba(255,255,255,0.9)',
    borderColor: '#fff',
    pointStyle: 'star',
    pointRadius: 10,
    order: 0,
  };

  const datasets = byCluster.map((pts, i) => ({
    label: (data.centroids[i] && data.centroids[i].label) ? data.centroids[i].label : ('Cluster ' + i),
    data: pts,
    backgroundColor: CLUSTER_COLORS[i] + '55',
    borderColor: CLUSTER_BORDER[i],
    pointRadius: 5,
    borderWidth: 1.5,
    order: 1,
  }));

  clusterChart = new Chart(document.getElementById('cluster-chart'), {
    type: 'scatter',
    data: { datasets: [centroidDataset, ...datasets] },
    options: {
      ...chartOptions('Temperature (\u00b0C)'),
      scales: {
        x: { ...scaleOpts(), title: { display: true, text: 'Solar Irradiance (W/m\u00b2)', color: '#64748b', font:{size:11} } },
        y: { ...scaleOpts(), title: { display: true, text: 'Temperature (\u00b0C)',         color: '#64748b', font:{size:11} } },
      },
    },
  });

  const legend = document.getElementById('cluster-legend');
  legend.innerHTML = data.centroids.map((c,i) =>
    '<div class="legend-item">' +
      '<div class="legend-dot" style="background:' + CLUSTER_COLORS[i] + '"></div>' +
      c.label +
    '</div>'
  ).join('');
}

// ── Chart.js shared options ───────────────────────────────────
function scaleOpts() {
  return {
    grid:  { color: 'rgba(255,255,255,0.05)' },
    ticks: { color: '#64748b', font: { size: 10 } },
  };
}

function chartOptions(yLabel) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 600, easing: 'easeOutQuart' },
    plugins: {
      legend: {
        labels: { color: '#94a3b8', font: { size: 11 }, boxWidth: 12, padding: 16 },
      },
      tooltip: {
        backgroundColor: 'rgba(15,22,36,0.95)',
        borderColor:     'rgba(255,255,255,0.08)',
        borderWidth:     1,
        titleColor:      '#e2e8f0',
        bodyColor:       '#94a3b8',
        padding:         10,
      },
    },
    scales: {
      x: scaleOpts(),
      y: { ...scaleOpts(), title: { display: !!yLabel, text: yLabel, color: '#64748b', font:{size:11} } },
    },
  };
}

// ── Model Health Banner ──────────────────────────────────────
async function checkModelHealth() {
  try {
    const res = await fetch('/api/health');
    const d   = await res.json();
    if (!d.models_ready) {
      const banner = document.createElement('div');
      banner.style.cssText =
        'background:#7c3aed22;border:1px solid #7c3aed;color:#c4b5fd;' +
        'padding:.75rem 1.25rem;text-align:center;font-size:.85rem;' +
        'position:sticky;top:64px;z-index:999;';
      banner.innerHTML =
        '⚠ Models not loaded. Run <code style="color:#fbbf24;background:#0f162488;' +
        'padding:.1em .4em;border-radius:4px">python train.py</code> then restart Flask.';
      document.body.insertBefore(banner, document.querySelector('main'));
    }
  } catch(e) { /* Flask not running */ }
}

// ── Outlier Stats Visualisation ──────────────────────────────
let outlierLoaded = false;
async function loadOutlierStats() {
  if (outlierLoaded) return;
  outlierLoaded = true;
  try {
    const res  = await fetch('/api/outlier-stats');
    const data = await res.json();

    const container = document.getElementById('outlier-stats-grid');
    if (!container) return;

    const colourMap = {
      irradiance:   '#fbbf24',
      temperature:  '#f87171',
      wind_speed:   '#22d3ee',
      power_output: '#34d399',
    };
    const labels = {
      irradiance:   'Solar Irradiance (W/m²)',
      temperature:  'Temperature (°C)',
      wind_speed:   'Wind Speed (m/s)',
      power_output: 'Power Output (kW)',
    };

    container.innerHTML = Object.entries(data).map(([key, s]) => {
      const colour = colourMap[key] || '#94a3b8';
      const label  = labels[key]   || key;
      const pct    = s.outliers_removed || 0;
      return `
        <div class="outlier-stat-card">
          <div class="outlier-feature-name" style="color:${colour}">${label}</div>
          <div class="outlier-bounds">
            IQR bounds: <strong>${s.lower_bound}</strong> – <strong>${s.upper_bound}</strong>
            &nbsp;|&nbsp; IQR = ${s.iqr}
          </div>
          <div class="outlier-removed">
            <span style="color:${colour}">●</span>
            ${pct} outlier rows removed
          </div>
        </div>`;
    }).join('');
  } catch(e) {
    const container = document.getElementById('outlier-stats-grid');
    if (container) container.innerHTML = '<div style="color:var(--text-muted);padding:1rem">Run train.py to see real outlier statistics.</div>';
  }
}

// Patch navigate to load outlier stats when analytics page is opened
const _origNavigate = navigate;
window.navigate = function(id) {
  _origNavigate(id);
  if (id === 'analytics') loadOutlierStats();
};

// ── Evaluation Charts ────────────────────────────────────────
let evaluationLoaded = false;
let avpChart = null;
let resChart = null;
let residualData = null;
let cvData = null;

const MODEL_COLORS = {
  ensemble:   '#fbbf24',
  bagging:    '#34d399',
  svm:        '#22d3ee',
  timeseries: '#a78bfa',
};

const MODEL_KEYS = {
  ensemble:   { pred: 'ensemblePred',    res: 'ensembleResidual'   },
  bagging:    { pred: 'baggingPred',     res: 'baggingResidual'    },
  svm:        { pred: 'svmPred',         res: 'svmResidual'        },
  timeseries: { pred: 'timeSeriesPred',  res: 'timeSeriesResidual' },
};

const MODEL_LABELS = {
  ensemble:   'Ensemble',
  bagging:    'Bagging (RF)',
  svm:        'SVM (SVR)',
  timeseries: 'Time-Series (Ridge)',
};

async function loadEvaluation() {
  if (evaluationLoaded) return;
  evaluationLoaded = true;

  try {
    const [resRes, cvRes] = await Promise.all([
      fetch('/api/residuals'),
      fetch('/api/cv-results'),
    ]);
    residualData = await resRes.json();
    cvData       = await cvRes.json();

    // Wire up tab clicks
    document.querySelectorAll('#avp-tabs .eval-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#avp-tabs .eval-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderAvpChart(btn.dataset.model);
      });
    });
    document.querySelectorAll('#res-tabs .eval-tab').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('#res-tabs .eval-tab').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        renderResChart(btn.dataset.model);
      });
    });

    renderAvpChart('ensemble');
    renderResChart('ensemble');
    renderCvResults();

  } catch(e) {
    console.warn('Evaluation load failed:', e);
  }
}

// ── Actual vs Predicted scatter ───────────────────────────────
function renderAvpChart(modelKey) {
  const pts    = residualData.points;
  const color  = MODEL_COLORS[modelKey];
  const keys   = MODEL_KEYS[modelKey];
  const actual = pts.map(p => p.actual);
  const preds  = pts.map(p => p[keys.pred]);

  // Compute stats
  const n      = actual.length;
  const mae    = actual.reduce((s,a,i) => s + Math.abs(preds[i]-a), 0) / n;
  const mse    = actual.reduce((s,a,i) => s + (preds[i]-a)**2,       0) / n;
  const rmse   = Math.sqrt(mse);
  const mean_a = actual.reduce((s,v) => s+v,0)/n;
  const ss_tot = actual.reduce((s,a) => s+(a-mean_a)**2, 0);
  const ss_res = actual.reduce((s,a,i) => s+(preds[i]-a)**2, 0);
  const r2     = 1 - ss_res/ss_tot;

  // Perfect-fit line
  const allVals = [...actual, ...preds];
  const lo = Math.min(...allVals) - 1;
  const hi = Math.max(...allVals) + 1;

  const scatterData = pts.map((p,i) => ({ x: actual[i], y: preds[i] }));

  const config = {
    type: 'scatter',
    data: {
      datasets: [
        {
          label: MODEL_LABELS[modelKey],
          data:  scatterData,
          backgroundColor: color + '55',
          borderColor:     color,
          borderWidth:     0.5,
          pointRadius:     3,
          order: 1,
        },
        {
          label: 'Perfect fit (y = x)',
          data: [{ x: lo, y: lo }, { x: hi, y: hi }],
          type: 'line',
          borderColor: 'rgba(255,255,255,0.6)',
          borderWidth: 1.5,
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
          order: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: {
          ...scaleOpts(),
          title: { display: true, text: 'Actual Power (kW)', color: '#64748b', font:{size:11} },
          min: lo, max: hi,
        },
        y: {
          ...scaleOpts(),
          title: { display: true, text: 'Predicted Power (kW)', color: '#64748b', font:{size:11} },
          min: lo, max: hi,
        },
      },
      plugins: {
        legend: { labels: { color: '#94a3b8', font:{size:10}, boxWidth:10, padding:12 } },
        tooltip: {
          backgroundColor: 'rgba(15,22,36,0.95)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          callbacks: {
            label: ctx => `Actual: ${ctx.raw.x.toFixed(2)} kW  Predicted: ${ctx.raw.y.toFixed(2)} kW`,
          },
        },
      },
    },
  };

  if (avpChart) { avpChart.destroy(); avpChart = null; }
  avpChart = new Chart(document.getElementById('avp-chart'), config);

  document.getElementById('avp-stats').innerHTML =
    `<div class="eval-stat">R² <strong>${r2.toFixed(4)}</strong></div>` +
    `<div class="eval-stat">MAE <strong>${mae.toFixed(3)} kW</strong></div>` +
    `<div class="eval-stat">RMSE <strong>${rmse.toFixed(3)} kW</strong></div>` +
    `<div class="eval-stat">n = <strong>${n}</strong> test samples</div>`;
}

// ── Residual Histogram ────────────────────────────────────────
function renderResChart(modelKey) {
  const pts    = residualData.points;
  const color  = MODEL_COLORS[modelKey];
  const resKey = MODEL_KEYS[modelKey].res;
  const resVals = pts.map(p => p[resKey]);

  const mean = resVals.reduce((s,v)=>s+v,0)/resVals.length;
  const std  = Math.sqrt(resVals.reduce((s,v)=>s+(v-mean)**2,0)/resVals.length);

  // Build histogram bins
  const min = Math.min(...resVals), max = Math.max(...resVals);
  const nBins = 35;
  const step  = (max - min) / nBins;
  const bins  = Array.from({length: nBins}, (_,i) => min + i * step);
  const counts = new Array(nBins).fill(0);
  resVals.forEach(v => {
    const idx = Math.min(Math.floor((v - min) / step), nBins - 1);
    counts[idx]++;
  });
  const density = counts.map(c => c / (resVals.length * step));
  const labels  = bins.map(b => b.toFixed(2));

  const config = {
    type: 'bar',
    data: {
      labels,
      datasets: [{
        label: `Residuals — ${MODEL_LABELS[modelKey]}`,
        data: density,
        backgroundColor: color + '66',
        borderColor:     color,
        borderWidth:     1,
        borderRadius:    2,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: {
          ...scaleOpts(),
          title: { display: true, text: 'Residual (Predicted − Actual)  [kW]', color:'#64748b', font:{size:11} },
          ticks: { ...scaleOpts().ticks, maxTicksLimit: 10 },
        },
        y: {
          ...scaleOpts(),
          title: { display: true, text: 'Density', color:'#64748b', font:{size:11} },
        },
      },
      plugins: {
        legend: { labels: { color:'#94a3b8', font:{size:10} } },
        tooltip: {
          backgroundColor: 'rgba(15,22,36,0.95)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#e2e8f0',
          bodyColor: '#94a3b8',
          callbacks: {
            title: ctx => `Residual ≈ ${ctx[0].label} kW`,
            label: ctx => `Density: ${ctx.raw.toFixed(4)}`,
          },
        },
        annotation: undefined,
      },
    },
  };

  if (resChart) { resChart.destroy(); resChart = null; }
  resChart = new Chart(document.getElementById('res-chart'), config);

  const skewSign = mean > 0.05 ? '⬆ slight over-prediction' :
                   mean < -0.05 ? '⬇ slight under-prediction' : '✓ unbiased';
  document.getElementById('res-stats').innerHTML =
    `<div class="eval-stat">Mean residual <strong>${mean.toFixed(4)} kW</strong> — ${skewSign}</div>` +
    `<div class="eval-stat">Std dev <strong>${std.toFixed(4)} kW</strong> (≈ RMSE)</div>` +
    `<div class="eval-stat">n = <strong>${resVals.length}</strong> test samples</div>`;
}

// ── Cross-Validation Results ──────────────────────────────────
function renderCvResults() {
  const container = document.getElementById('cv-grid');
  if (!cvData) return;

  const entries = [
    { key: 'svm',        label: 'SVM (SVR)',             color: MODEL_COLORS.svm },
    { key: 'bagging',    label: 'Bagging (RF)',           color: MODEL_COLORS.bagging },
    { key: 'timeseries', label: 'Time-Series (Ridge)',    color: MODEL_COLORS.timeseries },
  ];

  container.innerHTML = entries.map(({ key, label, color }) => {
    const d = cvData[key];
    if (!d) return '';
    const foldChips = d.folds.map((v,i) =>
      `<span class="cv-fold-chip">Fold ${i+1}: ${v}</span>`
    ).join('');
    const r2pct = Math.round(d.mean * 100);
    return `
      <div class="cv-model-row">
        <div class="cv-model-name" style="color:${color}">${label}</div>
        <div class="cv-folds">
          ${foldChips}
          <span class="cv-mean-chip" style="background:${color}22;color:${color};border:1px solid ${color}55">
            Mean R² ${d.mean} ± ${d.std}
          </span>
        </div>
      </div>`;
  }).join('');
}

// ── Init ──────────────────────────────────────────────────────
checkModelHealth();
navigate('landing');