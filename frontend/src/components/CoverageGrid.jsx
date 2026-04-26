import { COVERAGE_ITEMS } from '../constants';

export function CoverageGrid() {
  return (
    <div className="coverage-grid">
      {COVERAGE_ITEMS.map(item => (
        <div key={item.id} className="coverage-card">
          <div className="coverage-card-title">{item.label}</div>
          <div className="coverage-card-desc">{item.desc}</div>
          <div className={`coverage-status status-${item.status}`}>
            {item.status}
          </div>
        </div>
      ))}
    </div>
  );
}
