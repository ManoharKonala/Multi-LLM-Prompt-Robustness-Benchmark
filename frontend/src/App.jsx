import { useState, useCallback, useEffect } from 'react';
import { RobustnessTable } from './components/RobustnessTable';
import { RankChart } from './components/RankChart';
import { CoverageGrid } from './components/CoverageGrid';
import { RunLog } from './components/RunLog';
import { MODEL_META, ATTACKS, DATASETS, now } from './constants';

// ── Parse CSV text into array of objects ──────────────────────────────────────
function parseCSV(text) {
  const lines = text.trim().split('\n');
  if (lines.length < 2) return [];
  const headers = lines[0].split(',').map(h => h.trim());
  return lines.slice(1).map(line => {
    const vals = line.split(',');
    const obj = {};
    headers.forEach((h, i) => {
      const raw = (vals[i] || '').trim();
      obj[h] = isNaN(raw) || raw === '' ? raw : Number(raw);
    });
    return obj;
  });
}

// ── Derive summary metrics ────────────────────────────────────────────────────
function getSummary(results, selectedModels) {
  if (!results || results.length === 0)
    return { modelsCount: 0, avgCleanAcc: null, avgRobustness: null, bestModel: null };

  const filtered = results.filter(r => selectedModels.includes(r.Model));
  if (!filtered.length)
    return { modelsCount: 0, avgCleanAcc: null, avgRobustness: null, bestModel: null };

  const models = [...new Set(filtered.map(r => r.Model))];
  const cleanVals = filtered.map(r => r['Clean Accuracy']).filter(v => v != null);
  const robVals   = filtered.map(r => r['Robustness Score']).filter(v => v != null);

  const avgClean = cleanVals.length
    ? (cleanVals.reduce((s, v) => s + v, 0) / cleanVals.length).toFixed(1)
    : null;
  const avgRob = robVals.length
    ? (robVals.reduce((s, v) => s + v, 0) / robVals.length).toFixed(1)
    : null;

  // Best model = highest avg robustness
  const modelScores = {};
  const modelCounts = {};
  for (const r of filtered) {
    const m = r.Model;
    modelScores[m] = (modelScores[m] || 0) + r['Robustness Score'];
    modelCounts[m] = (modelCounts[m] || 0) + 1;
  }
  let best = null, bestScore = -Infinity;
  for (const m of models) {
    const avg = modelScores[m] / modelCounts[m];
    if (avg > bestScore) { bestScore = avg; best = m; }
  }

  return {
    modelsCount: models.length,
    avgCleanAcc: avgClean,
    avgRobustness: avgRob,
    bestModel: best ? (MODEL_META[best]?.label || best) : null,
  };
}

// ── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [activeTab, setActiveTab] = useState('benchmark');

  // Controls
  const [selectedModels, setSelectedModels] = useState(
    Object.keys(MODEL_META)
  );
  const [dataset, setDataset]       = useState('sst2');
  const [attackLevel, setAttackLevel] = useState('character');
  const [samples, setSamples]         = useState(3);
  const [intensity, setIntensity]     = useState('0.3');

  // Results
  const [results, setResults]   = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [runStatus, setRunStatus] = useState('idle'); // idle | running | done | error

  // Log lines
  const [logLines, setLogLines] = useState([]);

  const addLog = useCallback((text, type = '') => {
    setLogLines(prev => [...prev, { time: now(), text, type }]);
  }, []);

  // On mount — try to load results from the CSV the Python script produced
  useEffect(() => {
    fetchResults();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function fetchResults() {
    try {
      // Vite will serve files from /public — we copy the CSV there via the run
      const res = await fetch('/results.csv');
      if (!res.ok) return;
      const text = await res.text();
      const parsed = parseCSV(text);
      if (parsed.length) {
        setResults(parsed);
        addLog('PRB initialized — v0.1.0', 'info');
        addLog(`✓ Loaded ${parsed.length} result rows from results.csv`, 'success');
        setRunStatus('done');
      }
    } catch {
      // no file yet — that's fine
    }
  }

  function toggleModel(m) {
    setSelectedModels(prev =>
      prev.includes(m) ? prev.filter(x => x !== m) : [...prev, m]
    );
  }

  // Simulate a run (real integration would call a backend endpoint)
  async function handleRun() {
    if (isRunning) return;
    setIsRunning(true);
    setRunStatus('running');
    setLogLines([]);
    addLog('PRB initialized — v0.1.0', 'info');
    addLog(`Starting benchmark — dataset=${dataset}, samples=${samples}, intensity=${intensity}`, '');

    // Poll for the CSV file every 3s (Python script writes it progressively)
    const maxWait = 300; // seconds
    let waited = 0;
    addLog(`Waiting for Python benchmark to write results…`, 'warn');

    const poll = setInterval(async () => {
      waited += 3;
      try {
        const res = await fetch('/results.csv?t=' + Date.now());
        if (res.ok) {
          const text = await res.text();
          const parsed = parseCSV(text);
          if (parsed.length > (results?.length || 0)) {
            const models = [...new Set(parsed.map(r => r.Model))];
            addLog(`✓ ${parsed.length} rows loaded (models: ${models.join(', ')})`, 'success');
            setResults(parsed);
          }
        }
      } catch { /* ignore */ }

      if (waited >= maxWait) {
        clearInterval(poll);
        setIsRunning(false);
        setRunStatus('done');
        addLog('Polling complete.', '');
      }
    }, 3000);

    // Also auto-stop after ~5 min
    setTimeout(() => {
      clearInterval(poll);
      setIsRunning(false);
      setRunStatus('done');
    }, maxWait * 1000);
  }

  const summary = getSummary(results, selectedModels);

  return (
    <div className="app-wrapper">
      {/* ── Top Nav ── */}
      <nav className="top-nav">
        <div className="nav-logo">PRB</div>
        <div className="version-badge">v0.1.0</div>
        <div className="nav-tabs">
          <button
            id="tab-benchmark"
            className={`nav-tab ${activeTab === 'benchmark' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmark')}
          >
            benchmark
          </button>
          <button
            id="tab-custom"
            className={`nav-tab ${activeTab === 'custom' ? 'active' : ''}`}
            onClick={() => setActiveTab('custom')}
          >
            custom prompt
          </button>
        </div>
      </nav>

      {/* ── Control Panel ── */}
      <section className="control-panel section-gap">
        <div className="control-row">
          {/* Models */}
          <div className="control-group">
            <div className="control-label">models</div>
            <div className="model-chips">
              {Object.entries(MODEL_META).map(([key, meta]) => (
                <button
                  key={key}
                  id={`chip-${key}`}
                  className={`model-chip ${selectedModels.includes(key) ? `active ${meta.cls}` : ''}`}
                  onClick={() => toggleModel(key)}
                >
                  <span className="chip-dot" style={{ background: meta.color }} />
                  {meta.label}
                </button>
              ))}
            </div>
          </div>

          {/* Dataset */}
          <div className="control-group">
            <div className="control-label">dataset</div>
            <select
              id="select-dataset"
              className="ctrl-select"
              value={dataset}
              onChange={e => setDataset(e.target.value)}
            >
              {DATASETS.map(d => <option key={d} value={d}>{d}</option>)}
            </select>
          </div>

          {/* Attack level */}
          <div className="control-group">
            <div className="control-label">attack level</div>
            <select
              id="select-attack"
              className="ctrl-select"
              value={attackLevel}
              onChange={e => setAttackLevel(e.target.value)}
            >
              <option value="character">character</option>
              <option value="word">word</option>
              <option value="sentence">sentence</option>
              <option value="semantic">semantic</option>
            </select>
          </div>

          {/* Samples */}
          <div className="control-group">
            <div className="control-label">samples</div>
            <input
              id="input-samples"
              type="number"
              className="ctrl-input"
              min={1} max={100}
              value={samples}
              onChange={e => setSamples(Number(e.target.value))}
            />
          </div>

          {/* Intensity */}
          <div className="control-group">
            <div className="control-label">intensity</div>
            <select
              id="select-intensity"
              className="ctrl-select"
              value={intensity}
              onChange={e => setIntensity(e.target.value)}
            >
              <option value="0.1">0.1</option>
              <option value="0.2">0.2</option>
              <option value="0.3">0.3</option>
              <option value="0.5">0.5</option>
              <option value="0.8">0.8</option>
            </select>
          </div>

          {/* Run button */}
          <div className="run-section">
            <button
              id="btn-run"
              className={`run-btn ${isRunning ? 'running' : ''}`}
              onClick={handleRun}
              disabled={isRunning}
            >
              {isRunning && <span className="spinner" />}
              {isRunning ? 'running…' : 'run benchmark ↗'}
            </button>
            <div className="run-status">
              <span className={`status-dot ${runStatus}`} />
              {runStatus}
            </div>
          </div>
        </div>
      </section>

      {/* ── Metric Cards ── */}
      <div className="metric-cards section-gap">
        <div className="metric-card">
          <div className="metric-card-label">models tested</div>
          <div className="metric-card-value">
            {summary.modelsCount || '—'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card-label">avg clean acc</div>
          <div className="metric-card-value highlight">
            {summary.avgCleanAcc != null ? `${summary.avgCleanAcc}%` : '—'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card-label">avg robustness</div>
          <div className="metric-card-value highlight">
            {summary.avgRobustness != null ? `${summary.avgRobustness}%` : '—'}
          </div>
        </div>
        <div className="metric-card">
          <div className="metric-card-label">best model</div>
          <div className="metric-card-value" style={{ fontSize: '18px', paddingTop: '6px' }}>
            {summary.bestModel || '—'}
          </div>
        </div>
      </div>

      {/* ── Main Grid: Robustness Table + Rank Chart ── */}
      <div className="main-grid section-gap">
        <div className="panel">
          <div className="panel-header">
            robustness score (%) — model × attack type
          </div>
          <div className="panel-body">
            <RobustnessTable
              results={results}
              attacks={ATTACKS}
              selectedModels={selectedModels}
            />
          </div>
        </div>

        <div className="panel">
          <div className="panel-header">overall robustness rank</div>
          <div className="panel-body">
            <RankChart results={results} selectedModels={selectedModels} />
          </div>
        </div>
      </div>

      {/* ── Coverage Grid ── */}
      <div className="panel coverage-section section-gap">
        <div className="panel-header">
          perturbation coverage — {4} levels (roadmap)
        </div>
        <CoverageGrid />
      </div>

      {/* ── Run Log ── */}
      <div className="panel run-log-section">
        <div className="panel-header">run log</div>
        <RunLog lines={logLines} />
      </div>

      {/* ── Footer ── */}
      <footer className="footer">
        PRB v0.1.0 — Multi-LLM Prompt Robustness Benchmark
      </footer>
    </div>
  );
}
