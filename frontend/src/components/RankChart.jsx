import { MODEL_META } from '../constants';

export function RankChart({ results, selectedModels }) {
  if (!results || results.length === 0) {
    return (
      <div className="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="3" y="13" width="4" height="8" rx="1"/><rect x="10" y="9" width="4" height="12" rx="1"/><rect x="17" y="5" width="4" height="16" rx="1"/>
        </svg>
        <div>No data yet.</div>
      </div>
    );
  }

  // aggregate avg robustness per model
  const sums = {};
  const cnts = {};
  for (const row of results) {
    const m = row.Model;
    if (!selectedModels.includes(m)) continue;
    sums[m] = (sums[m] || 0) + row['Robustness Score'];
    cnts[m] = (cnts[m] || 0) + 1;
  }

  const entries = Object.keys(sums).map(m => ({
    model: m,
    score: Math.round(sums[m] / cnts[m]),
  })).sort((a, b) => b.score - a.score);

  if (!entries.length) return <div className="empty-state">No selected models in results.</div>;

  return (
    <div className="rank-list">
      {entries.map(({ model, score }) => {
        const meta = MODEL_META[model] || { label: model, color: '#888' };
        return (
          <div key={model} className="rank-item">
            <div className="rank-model-row">
              <div className="rank-model-name">
                <span className="model-dot" style={{ background: meta.color }} />
                {meta.label}
              </div>
              <span className="rank-score">{score}%</span>
            </div>
            <div className="rank-bar-track">
              <div
                className="rank-bar-fill"
                style={{ width: `${score}%`, background: meta.color }}
              >
                {score > 20 && (
                  <span className="rank-bar-label">{score}%</span>
                )}
              </div>
            </div>
          </div>
        );
      })}
      <div className="rank-axis">
        <span>0%</span><span>50%</span><span>100%</span>
      </div>
    </div>
  );
}
