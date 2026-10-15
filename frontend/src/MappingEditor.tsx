import { useState } from "react";

interface Props {
  mapping: Record<string, number>;
  onChange: (mapping: Record<string, number>) => void;
}

export function MappingEditor({ mapping, onChange }: Props) {
  const [currentMapping, setCurrentMapping] = useState(mapping);
  const [rows, setRows] = useState(() => {
    const entries = Object.entries(mapping).map(([key, value]) => ({ key, value: String(value) }));
    return entries.length ? entries : [{ key: "", value: "" }];
  });
  if (mapping !== currentMapping) {
    setCurrentMapping(mapping);
    const entries = Object.entries(mapping).map(([key, value]) => ({ key, value: String(value) }));
    setRows(entries.length ? entries : [{ key: "", value: "" }]);
  }
  function update(next: typeof rows) {
    setRows(next);
    const nextMapping = Object.fromEntries(next.map(({ key, value }) => [key, Number(value)]));
    setCurrentMapping(nextMapping);
    onChange(nextMapping);
  }
  return (
    <div>
      <p className="muted">Map exact text to numbers. Every value must have a pair. Use this to choose numeric values explicitly; one hot also accepts strings directly.</p>
      {rows.map((row, index) => (
        <div className="transform-heading" key={index}>
          <label>
            String
            <input value={row.key} onChange={(event) => {
              const key = event.target.value;
              event.target.setCustomValidity(rows.some((other, i) => i !== index && other.key === key) ? "Each string must be unique." : "");
              update(rows.map((item, i) => i === index ? { ...item, key } : item));
            }} ref={(input) => {
              input?.setCustomValidity(rows.some((other, i) => i !== index && other.key === row.key) ? "Each string must be unique." : "");
            }} />
          </label>
          <label>
            Number
            <input required type="number" step="any" value={row.value} onChange={(event) =>
              update(rows.map((item, i) => i === index ? { ...item, value: event.target.value } : item))
            } />
          </label>
          <button type="button" aria-label={`Remove mapping pair ${index + 1}`} disabled={rows.length === 1}
            onClick={() => update(rows.filter((_, i) => i !== index))}>×</button>
        </div>
      ))}
      <button type="button" onClick={() => update([...rows, { key: "", value: "" }])}>+ Add pair</button>
    </div>
  );
}
