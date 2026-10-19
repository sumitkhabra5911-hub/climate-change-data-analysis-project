/**
 * TerraPulse - Global Climate Intelligence & Prediction Platform
 * Modern ES6 Application Logic & Interactive Visualizations
 */

document.addEventListener('DOMContentLoaded', () => {
  // Ensure dataset is loaded
  if (!window.CLIMATE_DATA) {
    console.error('CLIMATE_DATA bundle not found!');
    return;
  }

  const data = window.CLIMATE_DATA;

  // Global State
  
  // Check Live REST API Status
  const apiStatusText = document.getElementById('api-status-text');
  async function checkApiHealth() {
    try {
      const res = await fetch('/api/v1/health');
      if (res.ok) {
        const data = await res.json();
        if (apiStatusText) {
          apiStatusText.textContent = `API Live (v${data.version}) | ${data.records_indexed} Indexed`;
        }
      }
    } catch (e) {
      if (apiStatusText) {
        apiStatusText.textContent = 'Client Edge Simulation (Local)';
      }
    }
  }
  checkApiHealth();

  const state = {
    theme: localStorage.getItem('theme') || 'dark',
    activeTab: 'overview',
    selectedCountry: 'United States',
    charts: {},
    table: {
      searchTerm: '',
      riskFilter: 'all',
      currentPage: 1,
      rowsPerPage: 15
    }
  };

  // Initialize Theme
  applyTheme(state.theme);

  // Setup Theme Toggle
  const themeToggleBtn = document.getElementById('theme-toggle-btn');
  if (themeToggleBtn) {
    themeToggleBtn.addEventListener('click', () => {
      state.theme = state.theme === 'dark' ? 'light' : 'dark';
      localStorage.setItem('theme', state.theme);
      applyTheme(state.theme);
      updateChartThemes();
    });
  }

  // Setup Navigation Tabs
  const navTabs = document.querySelectorAll('.nav-tab-btn');
  navTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      const target = tab.dataset.target;
      switchTab(target);
    });
  });

  // Initialize Modules
  initOverview();
  initEDA();
  initCountryExplorer();
  initMLSimulator();
  initForecastSimulator();
  initDataTable();

  // If Lucide icons script is present
  if (window.lucide) {
    window.lucide.createIcons();
  }

  /* ==========================================================================
     Theme Management
     ========================================================================== */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const icon = document.getElementById('theme-icon');
    if (icon) {
      icon.setAttribute('data-lucide', theme === 'dark' ? 'sun' : 'moon');
      if (window.lucide) window.lucide.createIcons();
    }
  }

  function updateChartThemes() {
    const isDark = state.theme === 'dark';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

    Object.values(state.charts).forEach(chart => {
      if (chart && chart.options) {
        if (chart.options.scales) {
          if (chart.options.scales.x) {
            chart.options.scales.x.ticks.color = textColor;
            chart.options.scales.x.grid.color = gridColor;
          }
          if (chart.options.scales.y) {
            chart.options.scales.y.ticks.color = textColor;
            chart.options.scales.y.grid.color = gridColor;
          }
        }
        chart.update();
      }
    });
  }

  /* ==========================================================================
     Tab Switching
     ========================================================================== */
  function switchTab(tabId) {
    state.activeTab = tabId;
    navTabs.forEach(tab => {
      tab.classList.toggle('active', tab.dataset.target === tabId);
    });
    document.querySelectorAll('.tab-section').forEach(sec => {
      sec.classList.toggle('active', sec.id === `section-${tabId}`);
    });

    // Resize Chart.js canvases when made visible
    setTimeout(() => {
      Object.values(state.charts).forEach(chart => {
        if (chart) chart.resize();
      });
    }, 50);
  }

  /* ==========================================================================
     Section 1: Overview
     ========================================================================== */
  function initOverview() {
    // 1. Overview Global Anomaly Chart
    const ctxOverview = document.getElementById('chart-overview-anomaly');
    if (ctxOverview) {
      const years = data.global_annual.map(d => d.Year);
      const anomalies = data.global_annual.map(d => d.Temp_Anomaly);
      const rolling5 = computeRollingAverage(anomalies, 5);

      state.charts.overviewAnomaly = new Chart(ctxOverview, {
        type: 'line',
        data: {
          labels: years,
          datasets: [
            {
              label: 'Annual Mean Anomaly (°C)',
              data: anomalies,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.08)',
              borderWidth: 1.5,
              pointRadius: 2,
              pointHoverRadius: 5,
              tension: 0.2,
              fill: true
            },
            {
              label: '5-Year Moving Average',
              data: rolling5,
              borderColor: '#f59e0b',
              borderWidth: 2.8,
              pointRadius: 0,
              tension: 0.3
            }
          ]
        },
        options: getChartOptions('Temperature Anomaly (°C)')
      });
    }

    // 2. Overview Sea Level Rise Chart
    const ctxSLR = document.getElementById('chart-overview-slr');
    if (ctxSLR) {
      const years = data.global_annual.map(d => d.Year);
      const slr = data.global_annual.map(d => d.Sea_Level_Rise_mm);

      state.charts.overviewSLR = new Chart(ctxSLR, {
        type: 'line',
        data: {
          labels: years,
          datasets: [{
            label: 'Global Mean Sea Level Rise (mm)',
            data: slr,
            borderColor: '#06b6d4',
            backgroundColor: 'rgba(6, 182, 212, 0.15)',
            borderWidth: 2.5,
            pointRadius: 0,
            fill: true,
            tension: 0.25
          }]
        },
        options: getChartOptions('Sea Level Rise (mm)')
      });
    }

    // 3. Top Emitters Mini Chart
    const ctxTopEmitters = document.getElementById('chart-overview-emitters');
    if (ctxTopEmitters) {
      const labels = data.top_emitters.slice(0, 7).map(d => d.Country);
      const values = data.top_emitters.slice(0, 7).map(d => d.CO2_GT);

      state.charts.overviewEmitters = new Chart(ctxTopEmitters, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Cumulative CO2 (GT)',
            data: values,
            backgroundColor: [
              '#ef4444', '#f97316', '#f59e0b', '#eab308',
              '#10b981', '#06b6d4', '#8b5cf6'
            ],
            borderRadius: 6
          }]
        },
        options: {
          ...getChartOptions('Gigatons (GT)'),
          indexAxis: 'y'
        }
      });
    }
  }

  /* ==========================================================================
     Section 2: Exploratory Data Analysis (8 Charts)
     ========================================================================== */
  function initEDA() {
    const years = data.global_annual.map(d => d.Year);

    // Chart 1: Global Temperature Anomaly Trend
    const ctx1 = document.getElementById('chart-eda-1');
    if (ctx1) {
      const anomalies = data.global_annual.map(d => d.Temp_Anomaly);
      const rolling = computeRollingAverage(anomalies, 5);
      state.charts.eda1 = new Chart(ctx1, {
        type: 'bar',
        data: {
          labels: years,
          datasets: [
            {
              type: 'bar',
              label: 'Annual Anomaly (°C)',
              data: anomalies,
              backgroundColor: anomalies.map(v => v >= 0 ? 'rgba(239, 68, 68, 0.65)' : 'rgba(59, 130, 246, 0.65)'),
              borderRadius: 2
            },
            {
              type: 'line',
              label: '5-Year Moving Avg',
              data: rolling,
              borderColor: '#800026',
              borderWidth: 2.5,
              pointRadius: 0,
              tension: 0.3
            }
          ]
        },
        options: getChartOptions('Anomaly (°C)')
      });
    }

    // Chart 2: Top 10 Cumulative CO2 Emitters
    const ctx2 = document.getElementById('chart-eda-2');
    if (ctx2) {
      const labels = data.top_emitters.map(d => d.Country).reverse();
      const vals = data.top_emitters.map(d => d.CO2_GT).reverse();
      state.charts.eda2 = new Chart(ctx2, {
        type: 'bar',
        data: {
          labels: labels,
          datasets: [{
            label: 'Cumulative Emissions (GT)',
            data: vals,
            backgroundColor: '#f97316',
            borderRadius: 6
          }]
        },
        options: {
          ...getChartOptions('Gigatons (GT)'),
          indexAxis: 'y'
        }
      });
    }

    // Chart 3: Correlation Matrix Table
    renderCorrelationMatrix();

    // Chart 4: Sea Level Rise Acceleration
    const ctx4 = document.getElementById('chart-eda-4');
    if (ctx4) {
      const slr = data.global_annual.map(d => d.Sea_Level_Rise_mm);
      state.charts.eda4 = new Chart(ctx4, {
        type: 'line',
        data: {
          labels: years,
          datasets: [{
            label: 'Cumulative Sea Level Rise (mm)',
            data: slr,
            borderColor: '#0284c7',
            backgroundColor: 'rgba(2, 132, 199, 0.2)',
            fill: true,
            borderWidth: 2.5,
            pointRadius: 0
          }]
        },
        options: getChartOptions('mm above 1950 base')
      });
    }

    // Chart 5: Decadal Shift in Anomalies
    const ctx5 = document.getElementById('chart-eda-5');
    if (ctx5) {
      const decades = Object.keys(data.decades);
      const means = decades.map(d => data.decades[d].mean);
      const q25s = decades.map(d => data.decades[d].q25);
      const q75s = decades.map(d => data.decades[d].q75);

      state.charts.eda5 = new Chart(ctx5, {
        type: 'bar',
        data: {
          labels: decades,
          datasets: [
            {
              label: 'Decade Mean Anomaly (°C)',
              data: means,
              backgroundColor: ['#fed976', '#feb24c', '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026', '#4a0011'],
              borderRadius: 6
            }
          ]
        },
        options: getChartOptions('Temperature Anomaly (°C)')
      });
    }

    // Chart 6: CO2 per Capita vs Temperature Anomaly Scatter
    const ctx6 = document.getElementById('chart-eda-6');
    if (ctx6) {
      // Sample 300 points across years for clean scatter rendering
      const scatterPoints = data.table_rows
        .filter((_, idx) => idx % 7 === 0)
        .map(row => ({
          x: row.CO2_per_Capita,
          y: row.Temp_Anomaly
        }));

      state.charts.eda6 = new Chart(ctx6, {
        type: 'scatter',
        data: {
          datasets: [{
            label: 'Country-Year Observations',
            data: scatterPoints,
            backgroundColor: 'rgba(139, 92, 246, 0.65)',
            pointRadius: 3.5,
            pointHoverRadius: 6
          }]
        },
        options: {
          ...getChartOptions('Temp Anomaly (°C)'),
          scales: {
            x: {
              title: { display: true, text: 'CO2 Emissions per Capita (Metric Tons)', color: '#94a3b8' },
              grid: { color: 'rgba(255, 255, 255, 0.06)' },
              ticks: { color: '#94a3b8' }
            },
            y: {
              title: { display: true, text: 'Temperature Anomaly (°C)', color: '#94a3b8' },
              grid: { color: 'rgba(255, 255, 255, 0.06)' },
              ticks: { color: '#94a3b8' }
            }
          }
        }
      });
    }

    // Chart 7: Precipitation Trends
    const ctx7 = document.getElementById('chart-eda-7');
    if (ctx7) {
      const precip = data.global_annual.map(d => d.Precipitation_mm);
      const precipMA = computeRollingAverage(precip, 5);

      state.charts.eda7 = new Chart(ctx7, {
        type: 'line',
        data: {
          labels: years,
          datasets: [
            {
              label: 'Global Mean Precipitation (mm)',
              data: precip,
              borderColor: '#22c55e',
              borderWidth: 1.5,
              pointRadius: 2,
              tension: 0.2
            },
            {
              label: '5-Year Rolling Avg',
              data: precipMA,
              borderColor: '#15803d',
              borderWidth: 2.5,
              pointRadius: 0
            }
          ]
        },
        options: getChartOptions('Precipitation (mm/year)')
      });
    }

    // Chart 8: Regional Divergence (Multi-Country)
    const ctx8 = document.getElementById('chart-eda-8');
    if (ctx8) {
      const focus = ['Canada', 'Norway', 'United States', 'China', 'India', 'Brazil', 'Australia'];
      const colors = ['#38bdf8', '#818cf8', '#f59e0b', '#ef4444', '#10b981', '#ec4899', '#a855f7'];

      const datasets = focus.map((c, i) => {
        const hist = data.country_profiles[c].history;
        const anomalies = hist.map(h => h.Temp_Anomaly);
        const smoothed = computeRollingAverage(anomalies, 5);
        return {
          label: c,
          data: smoothed,
          borderColor: colors[i],
          borderWidth: 2.2,
          pointRadius: 0,
          tension: 0.3
        };
      });

      state.charts.eda8 = new Chart(ctx8, {
        type: 'line',
        data: {
          labels: years,
          datasets: datasets
        },
        options: getChartOptions('Smoothed Anomaly (°C)')
      });
    }
  }

  function renderCorrelationMatrix() {
    const container = document.getElementById('corr-table-container');
    if (!container) return;

    const cols = data.correlation.columns;
    const vals = data.correlation.values;

    let html = '<table class="corr-table"><thead><tr><th>Indicator</th>';
    cols.forEach(c => {
      const short = c.replace('_C', '').replace('_MT', '').replace('_mm', '').replace('_pct', '');
      html += `<th>${short}</th>`;
    });
    html += '</tr></thead><tbody>';

    cols.forEach((rowName, rIdx) => {
      const short = rowName.replace('_C', '').replace('_MT', '').replace('_mm', '').replace('_pct', '');
      html += `<tr><th>${short}</th>`;
      vals[rIdx].forEach((val, cIdx) => {
        const color = getCorrColor(val);
        html += `<td class="corr-cell" style="background-color: ${color}; color: #ffffff;" title="${rowName} vs ${cols[cIdx]}: ${val}">${val.toFixed(2)}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  function getCorrColor(val) {
    if (val >= 0) {
      const alpha = Math.min(1, Math.max(0.1, val));
      return `rgba(239, 68, 68, ${alpha * 0.75})`;
    } else {
      const alpha = Math.min(1, Math.max(0.1, Math.abs(val)));
      return `rgba(59, 130, 246, ${alpha * 0.75})`;
    }
  }

  /* ==========================================================================
     Section 3: Country Intelligence
     ========================================================================== */
  function initCountryExplorer() {
    const select = document.getElementById('country-select');
    if (!select) return;

    // Populate options
    select.innerHTML = '';
    data.countries.forEach(c => {
      const opt = document.createElement('option');
      opt.value = c;
      opt.textContent = c;
      if (c === state.selectedCountry) opt.selected = true;
      select.appendChild(opt);
    });

    select.addEventListener('change', (e) => {
      state.selectedCountry = e.target.value;
      updateCountryView();
    });

    updateCountryView();
  }

  function updateCountryView() {
    const profile = data.country_profiles[state.selectedCountry];
    if (!profile) return;

    document.getElementById('meta-country-name').textContent = state.selectedCountry;
    document.getElementById('meta-baseline').textContent = `${profile.baseline_temp}°C`;
    document.getElementById('meta-temp-1950').textContent = `${profile.temp_1950}°C`;
    document.getElementById('meta-temp-2023').textContent = `${profile.temp_2023}°C`;
    document.getElementById('meta-anomaly').textContent = `+${profile.current_anomaly}°C`;
    document.getElementById('meta-co2').textContent = `${profile.total_co2_gt} GT`;
    document.getElementById('meta-pop').textContent = (profile.pop_2023 / 1e6).toFixed(1) + 'M';

    // Update Country Chart
    const ctx = document.getElementById('chart-country-history');
    if (ctx) {
      const hist = profile.history;
      const years = hist.map(h => h.Year);
      const temps = hist.map(h => h.Average_Temperature_C);
      const anomalies = hist.map(h => h.Temp_Anomaly);

      if (state.charts.countryChart) {
        state.charts.countryChart.destroy();
      }

      state.charts.countryChart = new Chart(ctx, {
        type: 'line',
        data: {
          labels: years,
          datasets: [
            {
              label: 'Average Temperature (°C)',
              data: temps,
              borderColor: '#f59e0b',
              borderWidth: 2,
              pointRadius: 2,
              tension: 0.2
            },
            {
              label: 'Temperature Anomaly (°C)',
              data: anomalies,
              borderColor: '#ef4444',
              backgroundColor: 'rgba(239, 68, 68, 0.1)',
              fill: true,
              borderWidth: 2,
              pointRadius: 0,
              tension: 0.2
            }
          ]
        },
        options: getChartOptions('Temperature (°C)')
      });
    }
  }

  /* ==========================================================================
     Section 4: AI Climate Risk Simulator (ML Inference)
     ========================================================================== */
  /* ==========================================================================
     Section 4: AI Climate Risk Simulator (ML Inference)
     ========================================================================== */
  function initMLSimulator() {
    const inputs = [
      'sim-year', 'sim-co2', 'sim-pop', 'sim-slr',
      'sim-precip', 'sim-humidity', 'sim-base-temp', 'sim-temp-change'
    ];

    inputs.forEach(id => {
      const el = document.getElementById(id);
      if (el) {
        el.addEventListener('input', () => {
          // Update display badge
          const badge = document.getElementById(`${id}-val`);
          if (badge) badge.textContent = el.value;
          runMLInference();
        });
      }
    });

    const modelSelect = document.getElementById('sim-model-select');
    if (modelSelect) {
      modelSelect.addEventListener('change', runMLInference);
    }

    const runBtn = document.getElementById('btn-run-simulation');
    if (runBtn) {
      runBtn.addEventListener('click', runMLInference);
    }

    runMLInference();
  }

  async function runMLInference() {
    const ml = data.ml_model;
    if (!ml) return;

    // Read form values
    const year = parseFloat(document.getElementById('sim-year')?.value || 2024);
    const co2 = parseFloat(document.getElementById('sim-co2')?.value || 1200);
    const pop = parseFloat(document.getElementById('sim-pop')?.value || 85000000);
    const slr = parseFloat(document.getElementById('sim-slr')?.value || 215);
    const precip = parseFloat(document.getElementById('sim-precip')?.value || 850);
    const humidity = parseFloat(document.getElementById('sim-humidity')?.value || 68);
    const baseTemp = parseFloat(document.getElementById('sim-base-temp')?.value || 12.5);
    const tempChange = parseFloat(document.getElementById('sim-temp-change')?.value || 0.45);
    const modelName = document.getElementById('sim-model-select')?.value || 'Logistic Regression';

    // Derived features
    const co2PerCapita = (co2 * 1_000_000) / pop;
    const cumulativeCO2 = co2 * (year - 1950) * 0.55;

    // Try live API call
    let apiSuccess = false;
    try {
      const payload = {
        model_name: modelName,
        features: {
          Year: Math.round(year),
          CO2_Emissions_MT: co2,
          CO2_per_Capita: co2PerCapita,
          Cumulative_CO2_MT: cumulativeCO2,
          Sea_Level_Rise_mm: slr,
          Precipitation_mm: precip,
          Humidity_pct: humidity,
          Population: pop,
          Country_Baseline_Temp: baseTemp,
          Temp_Change_YoY: tempChange
        }
      };

      const resp = await fetch('/api/v1/predict/risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });

      if (resp.ok) {
        const resData = await resp.json();
        apiSuccess = true;
        updateVerdictDisplay(
          resData.confidence_percentage,
          resData.risk_prediction === 1,
          resData.risk_label,
          `Server-side ${resData.model_used} model calculated ${resData.confidence_percentage}% confidence based on 10 standardized variables.`
        );
        renderApiFeatureContributions(resData.top_risk_factors);
        const engineBadge = document.getElementById('sim-engine-badge');
        if (engineBadge) {
          engineBadge.textContent = `API: ${resData.model_used}`;
          engineBadge.style.color = 'var(--emerald)';
        }
      }
    } catch (e) {
      // Offline fallback
    }

    if (!apiSuccess) {
      // Client-side fallback computation
      const x = [
        year, co2, co2PerCapita, cumulativeCO2, slr,
        precip, humidity, pop, baseTemp, tempChange
      ];

      let logit = ml.intercept;
      const contributions = [];

      for (let i = 0; i < x.length; i++) {
        const mean = ml.scaler_mean[i];
        const scale = ml.scaler_scale[i];
        const weight = ml.weights[i];
        const z = (x[i] - mean) / scale;
        const effect = z * weight;
        logit += effect;
        contributions.push({
          feature: ml.features[i],
          effect: effect,
          absEffect: Math.abs(effect)
        });
      }

      const probability = 1 / (1 + Math.exp(-logit));
      const percent = Math.min(99.9, Math.max(0.1, probability * 100));
      const isHigh = percent >= 50;

      updateVerdictDisplay(
        percent,
        isHigh,
        isHigh ? 'HIGH CLIMATE RISK ACCELERATION' : 'NORMAL / CONTROLLED RISK',
        isHigh
          ? 'Local client engine predicts extreme warming exceeding the historical 75th percentile threshold.'
          : 'Local client engine indicates climate variables remain within historical variance bounds.'
      );
      renderFeatureContributions(contributions);

      const engineBadge = document.getElementById('sim-engine-badge');
      if (engineBadge) {
        engineBadge.textContent = 'Edge Client Fallback';
        engineBadge.style.color = 'var(--amber)';
      }
    }
  }

  function updateVerdictDisplay(percent, isHigh, tagText, descText) {
    const verdictBox = document.getElementById('sim-verdict-box');
    const verdictProb = document.getElementById('sim-verdict-prob');
    const verdictTag = document.getElementById('sim-verdict-tag');
    const verdictDesc = document.getElementById('sim-verdict-desc');

    if (verdictProb) verdictProb.textContent = `${percent.toFixed(1)}%`;
    if (verdictBox) {
      verdictBox.className = `verdict-box ${isHigh ? 'high-risk' : 'normal-risk'}`;
    }
    if (verdictTag) verdictTag.textContent = tagText;
    if (verdictDesc) verdictDesc.textContent = descText;
  }

  function renderApiFeatureContributions(topFactors) {
    const container = document.getElementById('feature-contrib-container');
    if (!container || !topFactors) return;

    const maxVal = Math.max(...topFactors.map(s => s.impact_score), 0.01);
    let html = '';
    topFactors.forEach(f => {
      const widthPct = Math.min(100, Math.round((f.impact_score / maxVal) * 100));
      const isRiskIncrease = f.direction === 'increases_risk';
      const color = isRiskIncrease ? 'var(--coral)' : 'var(--emerald)';
      const friendlyName = f.feature.replace(/_/g, ' ');

      html += `
        <div class="contrib-item">
          <div class="contrib-header">
            <span style="font-weight: 600; color: var(--text-main);">${friendlyName}</span>
            <span style="font-family: var(--font-mono); color: ${color};">${isRiskIncrease ? '+' : '-'}${f.impact_score.toFixed(2)}</span>
          </div>
          <div class="contrib-bar-track">
            <div class="contrib-bar-fill" style="width: ${widthPct}%; background: ${color};"></div>
          </div>
        </div>
      `;
    });
    container.innerHTML = html;
  }

  function renderFeatureContributions(contributions) {
    const container = document.getElementById('feature-contrib-container');
    if (!container) return;

    // Sort by absolute influence
    const sorted = [...contributions].sort((a, b) => b.absEffect - a.absEffect).slice(0, 5);
    const maxVal = Math.max(...sorted.map(s => s.absEffect), 0.01);

    let html = '';
    sorted.forEach(c => {
      const widthPct = Math.min(100, Math.round((c.absEffect / maxVal) * 100));
      const isPositive = c.effect >= 0;
      const color = isPositive ? 'var(--coral)' : 'var(--emerald)';
      const friendlyName = c.feature.replace(/_/g, ' ');

      html += `
        <div class="contrib-item">
          <div class="contrib-header">
            <span style="font-weight: 600; color: var(--text-main);">${friendlyName}</span>
            <span style="font-family: var(--font-mono); color: ${color};">${isPositive ? '+' : ''}${c.effect.toFixed(2)}</span>
          </div>
          <div class="contrib-bar-track">
            <div class="contrib-bar-fill" style="width: ${widthPct}%; background: ${color};"></div>
          </div>
        </div>
      `;
    });

    container.innerHTML = html;
  }

  /* ==========================================================================
     Section 5: 2050 Climate Forecasting Simulator
     ========================================================================== */
  function initForecastSimulator() {
    const ctx = document.getElementById('chart-forecast-2050');
    if (!ctx) return;

    const histYears = data.global_annual.map(d => d.Year);
    const histAnomalies = data.global_annual.map(d => d.Temp_Anomaly);

    const fYears = data.forecast.map(d => d.year);
    const fPreds = data.forecast.map(d => d.predicted_anomaly);
    const fLower = data.forecast.map(d => d.lower_95);
    const fUpper = data.forecast.map(d => d.upper_95);
    const fParis = data.forecast.map(d => d.paris_target);
    const fNetZero = data.forecast.map(d => d.net_zero_path);

    // Combine timeline
    const allLabels = [...histYears, ...fYears];
    const histSeries = [...histAnomalies, ...new Array(fYears.length).fill(null)];
    const predSeries = [...new Array(histYears.length).fill(null), ...fPreds];
    const upperSeries = [...new Array(histYears.length).fill(null), ...fUpper];
    const lowerSeries = [...new Array(histYears.length).fill(null), ...fLower];
    const parisSeries = [...new Array(histYears.length).fill(null), ...fParis];
    const netZeroSeries = [...new Array(histYears.length).fill(null), ...fNetZero];

    state.charts.forecast = new Chart(ctx, {
      type: 'line',
      data: {
        labels: allLabels,
        datasets: [
          {
            label: 'Historical Observation (1950–2023)',
            data: histSeries,
            borderColor: '#38bdf8',
            borderWidth: 2,
            pointRadius: 1.5,
            tension: 0.15
          },
          {
            label: 'Business as Usual Projection (2024–2050)',
            data: predSeries,
            borderColor: '#ef4444',
            borderWidth: 2.8,
            borderDash: [5, 5],
            pointRadius: 0,
            tension: 0.1
          },
          {
            label: '95% Upper CI',
            data: upperSeries,
            borderColor: 'rgba(239, 68, 68, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: '+1',
            backgroundColor: 'rgba(239, 68, 68, 0.12)'
          },
          {
            label: '95% Lower CI',
            data: lowerSeries,
            borderColor: 'rgba(239, 68, 68, 0.25)',
            borderWidth: 1,
            pointRadius: 0,
            fill: false
          },
          {
            label: '1.5°C Paris Agreement Limit',
            data: parisSeries,
            borderColor: '#10b981',
            borderWidth: 1.8,
            borderDash: [4, 4],
            pointRadius: 0
          },
          {
            label: 'Net-Zero Mitigation Scenario',
            data: netZeroSeries,
            borderColor: '#8b5cf6',
            borderWidth: 2.2,
            pointRadius: 0
          }
        ]
      },
      options: getChartOptions('Global Temp Anomaly (°C)')
    });

    // Milestone Cards & Scenario Updates
    const slider = document.getElementById('forecast-year-slider');
    const scenarioSelect = document.getElementById('forecast-scenario-select');

    async function updateForecastMilestones() {
      const yr = parseInt(slider?.value || 2035);
      const scenario = scenarioSelect?.value || 'business_as_usual';
      const sliderValBadge = document.getElementById('forecast-slider-val');
      if (sliderValBadge) sliderValBadge.textContent = yr;

      // Try calling live API
      let apiDone = false;
      try {
        const resp = await fetch('/api/v1/predict/forecast', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ target_year: yr, scenario: scenario })
        });
        if (resp.ok) {
          const fRes = await resp.json();
          apiDone = true;
          document.getElementById('forecast-val-temp').textContent = `+${fRes.projected_anomaly_celsius}°C`;
          document.getElementById('forecast-val-ci').textContent = `[+${fRes.lower_bound_95}°C, +${fRes.upper_bound_95}°C]`;
          document.getElementById('forecast-val-slr').textContent = `+${fRes.estimated_sea_level_rise_mm} mm`;
        }
      } catch (e) {
        // Fallback
      }

      if (!apiDone) {
        const pt = data.forecast.find(f => f.year === yr) || data.forecast[data.forecast.length - 1];
        let proj = pt.predicted_anomaly;
        const yearsAhead = Math.max(0, yr - 2023);
        if (scenario === 'paris_agreement') {
          proj = Math.min(proj, +(1.45 + 0.05 * (1 - Math.exp(-yearsAhead / 15.0))).toFixed(3));
        } else if (scenario === 'net_zero') {
          proj = Math.min(proj, +(1.30 + 0.15 * (1 - Math.exp(-yearsAhead / 12.0))).toFixed(3));
        }

        document.getElementById('forecast-val-temp').textContent = `+${proj.toFixed(2)}°C`;
        document.getElementById('forecast-val-ci').textContent = `[+${(proj - 0.52).toFixed(2)}°C, +${(proj + 0.52).toFixed(2)}°C]`;
        document.getElementById('forecast-val-slr').textContent = `+${Math.round(218 + (yr - 2023) * 4.2)} mm`;
      }
    }

    if (slider) slider.addEventListener('input', updateForecastMilestones);
    if (scenarioSelect) scenarioSelect.addEventListener('change', updateForecastMilestones);
  }

  /* ==========================================================================
     Section 6: Searchable Data Table & CSV Export
     ========================================================================== */
  function initDataTable() {
    const searchInput = document.getElementById('table-search');
    const riskSelect = document.getElementById('table-risk-filter');
    const exportBtn = document.getElementById('btn-export-csv');

    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        state.table.searchTerm = e.target.value.toLowerCase();
        state.table.currentPage = 1;
        renderTableRows();
      });
    }

    if (riskSelect) {
      riskSelect.addEventListener('change', (e) => {
        state.table.riskFilter = e.target.value;
        state.table.currentPage = 1;
        renderTableRows();
      });
    }

    if (exportBtn) {
      exportBtn.addEventListener('click', exportTableToCSV);
    }

    renderTableRows();
  }

  function getFilteredRows() {
    return data.table_rows.filter(r => {
      const matchSearch = state.table.searchTerm === '' ||
        r.Country.toLowerCase().includes(state.table.searchTerm) ||
        r.Year.toString().includes(state.table.searchTerm);

      const matchRisk = state.table.riskFilter === 'all' ||
        (state.table.riskFilter === 'high' && r.High_Climate_Risk === 1) ||
        (state.table.riskFilter === 'normal' && r.High_Climate_Risk === 0);

      return matchSearch && matchRisk;
    });
  }

  function renderTableRows() {
    const tbody = document.getElementById('climate-table-body');
    const infoSpan = document.getElementById('table-info');
    if (!tbody) return;

    const filtered = getFilteredRows();
    const total = filtered.length;
    const start = (state.table.currentPage - 1) * state.table.rowsPerPage;
    const end = Math.min(start + state.table.rowsPerPage, total);
    const pageRows = filtered.slice(start, end);

    if (pageRows.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" style="text-align: center; padding: 2rem; color: var(--text-muted);">No records matching filters.</td></tr>';
      if (infoSpan) infoSpan.textContent = 'Showing 0 of 0 records';
      return;
    }

    let html = '';
    pageRows.forEach(r => {
      const isHigh = r.High_Climate_Risk === 1;
      const tagClass = isHigh ? 'tag-risk high' : 'tag-risk normal';
      const tagText = isHigh ? 'High Risk' : 'Normal';

      html += `
        <tr>
          <td><strong>${r.Country}</strong></td>
          <td>${r.Year}</td>
          <td>${r.Average_Temperature_C}°C</td>
          <td style="color: ${r.Temp_Anomaly >= 0 ? 'var(--coral)' : 'var(--cyan)'}; font-weight: 600;">
            ${r.Temp_Anomaly >= 0 ? '+' : ''}${r.Temp_Anomaly}°C
          </td>
          <td>${r.CO2_Emissions_MT.toLocaleString()}</td>
          <td>${r.CO2_per_Capita.toFixed(2)}</td>
          <td>${r.Sea_Level_Rise_mm} mm</td>
          <td><span class="${tagClass}">${tagText}</span></td>
        </tr>
      `;
    });

    tbody.innerHTML = html;
    if (infoSpan) {
      infoSpan.textContent = `Showing ${start + 1} to ${end} of ${total.toLocaleString()} records`;
    }

    renderPagination(total);
  }

  function renderPagination(totalRows) {
    const container = document.getElementById('pagination-buttons');
    if (!container) return;

    const totalPages = Math.ceil(totalRows / state.table.rowsPerPage) || 1;
    let html = '';

    html += `<button class="page-btn" id="page-prev" ${state.table.currentPage <= 1 ? 'disabled' : ''}>← Prev</button>`;

    // Page indicator
    html += `<span style="padding: 0.35rem 0.75rem; font-size: 0.8rem; font-weight: 600;">Page ${state.table.currentPage} of ${totalPages}</span>`;

    html += `<button class="page-btn" id="page-next" ${state.table.currentPage >= totalPages ? 'disabled' : ''}>Next →</button>`;

    container.innerHTML = html;

    document.getElementById('page-prev')?.addEventListener('click', () => {
      if (state.table.currentPage > 1) {
        state.table.currentPage--;
        renderTableRows();
      }
    });

    document.getElementById('page-next')?.addEventListener('click', () => {
      if (state.table.currentPage < totalPages) {
        state.table.currentPage++;
        renderTableRows();
      }
    });
  }

  function exportTableToCSV() {
    const rows = getFilteredRows();
    if (!rows.length) return;

    const headers = ['Country', 'Year', 'Average_Temperature_C', 'Temp_Anomaly', 'CO2_Emissions_MT', 'CO2_per_Capita', 'Sea_Level_Rise_mm', 'High_Climate_Risk'];
    let csv = headers.join(',') + '\n';

    rows.forEach(r => {
      csv += `"${r.Country}",${r.Year},${r.Average_Temperature_C},${r.Temp_Anomaly},${r.CO2_Emissions_MT},${r.CO2_per_Capita},${r.Sea_Level_Rise_mm},${r.High_Climate_Risk}\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = `climate_data_filtered_${Date.now()}.csv`;
    link.click();
  }

  /* ==========================================================================
     Helper Utilities
     ========================================================================== */
  function computeRollingAverage(arr, windowSize = 5) {
    const result = [];
    for (let i = 0; i < arr.length; i++) {
      const start = Math.max(0, i - windowSize + 1);
      const slice = arr.slice(start, i + 1);
      const avg = slice.reduce((a, b) => a + b, 0) / slice.length;
      result.push(Number(avg.toFixed(3)));
    }
    return result;
  }

  function getChartOptions(yAxisLabel = '') {
    const isDark = state.theme === 'dark';
    const textColor = isDark ? '#94a3b8' : '#475569';
    const gridColor = isDark ? 'rgba(255, 255, 255, 0.06)' : 'rgba(0, 0, 0, 0.06)';

    return {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: textColor,
            font: { family: "'Plus Jakarta Sans', sans-serif", size: 11, weight: '600' },
            boxWidth: 12,
            usePointStyle: true
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleColor: '#ffffff',
          bodyColor: '#e2e8f0',
          borderColor: 'rgba(255, 255, 255, 0.1)',
          borderWidth: 1,
          padding: 10,
          cornerRadius: 8
        }
      },
      scales: {
        x: {
          grid: { color: gridColor },
          ticks: { color: textColor, font: { size: 10 } }
        },
        y: {
          title: yAxisLabel ? { display: true, text: yAxisLabel, color: textColor, font: { size: 11 } } : undefined,
          grid: { color: gridColor },
          ticks: { color: textColor, font: { size: 10 } }
        }
      }
    };
  }
});
