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

// ── Init ──────────────────────────────────────────────────────
checkModelHealth();
navigate('landing');