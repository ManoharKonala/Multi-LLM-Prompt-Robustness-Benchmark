import { MODEL_META, scoreClass } from '../constants';

export function RobustnessTable({ results, attacks, selectedModels }) {
  if (!results || results.length === 0) {
    return (
      <div className="empty-state">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
          <rect x="3" y="3" width="18" height="18" rx="2"/><path d="M9 9h6M9 13h6M9 17h4"/>
        </svg>
        <div>No results yet — run the benchmark to populate the table.</div>
      </div>
    );
  }

  // aggregate: for each model, compute average per attack
  const modelRows = {};
  for (const row of results) {
    const m = row.Model;
    if (!selectedModels.includes(m)) continue;
    if (!modelRows[m]) modelRows[m] = { _cleanAccSum: 0, _count: 0 };
    const attack = row['Attack Type'];
    modelRows[m][attack] = (modelRows[m][attack] || 0) + row['Robustness Score'];
    modelRows[m]['_cnt_' + attack] = (modelRows[m]['_cnt_' + attack] || 0) + 1;
    modelRows[m]._cleanAccSum += row['Clean Accuracy'];
    modelRows[m]._count += 1;
  }

  const models = Object.keys(modelRows);

  // compute avg robustness per model
  const modelAvg = {};
  for (const m of models) {
    const vals = attacks.map(a => {
      const cnt = modelRows[m]['_cnt_' + a.id] || 1;
      return (modelRows[m][a.id] || 0) / cnt;
    });
    modelAvg[m] = vals.length ? (vals.reduce((s, v) => s + v, 0) / vals.length).toFixed(0) : 'N/A';
  }

  return (
    <table className="robustness-table">
      <thead>
        <tr>
          <th>Model</th>
          {attacks.map(a => <th key={a.id}>{a.label}</th>)}
          <th>avg</th>
        </tr>
      </thead>
      <tbody>
        {models.map(m => {
          const meta = MODEL_META[m] || { label: m, color: '#888', cls: '' };
          return (
            <tr key={m}>
              <td>
                <div className="model-cell">
                  <span className="model-dot" style={{ background: meta.color }} />
                  {meta.label}
                </div>
              </td>
              {attacks.map(a => {
                const cnt = modelRows[m]['_cnt_' + a.id] || 1;
                const val = modelRows[m][a.id] != null
                  ? Math.round(modelRows[m][a.id] / cnt)
                  : null;
                return (
                  <td key={a.id}>
                    <span className={`score-badge ${scoreClass(val)}`}>
                      {val != null ? `${val}%` : '—'}
                    </span>
                  </td>
                );
              })}
              <td>
                <span className={`score-badge ${scoreClass(Number(modelAvg[m]))}`}>
                  {modelAvg[m] !== 'N/A' ? `${modelAvg[m]}%` : '—'}
                </span>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
