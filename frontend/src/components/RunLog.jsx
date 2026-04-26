import { useEffect, useRef } from 'react';

export function RunLog({ lines }) {
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [lines]);

  return (
    <div className="log-body">
      {lines.length === 0 && (
        <div style={{ color: 'var(--text-muted)' }}>Waiting to start...</div>
      )}
      {lines.map((line, i) => (
        <div key={i} className="log-line">
          <span className="log-time">{line.time}</span>
          <span className={`log-text ${line.type || ''}`}>{line.text}</span>
        </div>
      ))}
      <div ref={endRef} />
    </div>
  );
}
