import { useEffect, useRef, useState, type FormEvent, type ReactNode } from 'react';
import { connectionError, nextPosition, type ProjectSpec, type StoredFile, type DatasetSpec, type TransformSpec } from './network';

export type DialogState =
  | { kind: 'source'; id?: string; position?: [number, number]; file?: StoredFile }
  | { kind: 'layer'; id?: string; parent?: string }
  | { kind: 'connection'; source: string; target?: string; id?: string }
  | { kind: 'transforms'; id: string }
  | { kind: 'confirm'; title: string; description: string; confirm: () => Promise<boolean> };
interface Props {
  dialog: DialogState;
  project: ProjectSpec;
  files: StoredFile[];
  busy: boolean;
  error: string;
  save: (project: ProjectSpec) => Promise<boolean>;
  upload: (files: FileList) => Promise<StoredFile[]>;
  close: () => void;
}
const makeId = (prefix: string) => `${prefix}-${crypto.randomUUID().slice(0, 6)}`;
const basename = (path: string) => path.split(/[\\/]/).pop() ?? path;

function Modal({ title, children, close, busy }: { title: string; children: ReactNode; close: () => void; busy: boolean }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = ref.current!;
    element.showModal();
    return () => element.close();
  }, []);
  return <dialog ref={ref} className="editor-dialog" aria-labelledby="dialog-title" onCancel={event => { event.preventDefault(); if (!busy) close(); }}>
    <header className="dialog-header"><h2 id="dialog-title">{title}</h2><button type="button" aria-label="Close dialog" disabled={busy} onClick={close}>×</button></header>
    {children}
  </dialog>;
}

export default function EditorDialog({ dialog, project, files, busy, error, save, upload, close }: Props) {
  const dataset = 'id' in dialog ? project.datasets.find(item => item.id === dialog.id) : undefined;
  const layer = 'id' in dialog ? project.network.layers.find(item => item.id === dialog.id) : undefined;
  const feed = 'id' in dialog ? project.feeds.find(item => item.id === dialog.id) : undefined;
  const [id, setId] = useState(dataset?.id ?? layer?.id ?? makeId(dialog.kind === 'source' ? 'data' : 'layer'));
  const [path, setPath] = useState(dataset?.source.path ?? (dialog.kind === 'source' ? dialog.file?.path : '') ?? '');
  const [format, setFormat] = useState<DatasetSpec['source']['type']>(dataset?.source.type ?? inferFormat(path));
  const [delimiter, setDelimiter] = useState(dataset?.source.delimiter ?? ',');
  const [quotechar, setQuotechar] = useState(dataset?.source.quotechar ?? '"');
  const [size, setSize] = useState(layer?.size ?? 16);
  const [activation, setActivation] = useState(layer?.activation ?? 'relu');
  const [target, setTarget] = useState(dialog.kind === 'connection' ? dialog.target ?? '' : '');
  const [field, setField] = useState(feed?.field ?? 'value');
  const [role, setRole] = useState<'input' | 'target'>(feed?.role ?? 'input');
  const [transforms, setTransforms] = useState<TransformSpec[]>(dataset?.transforms ?? []);
  const [localError, setLocalError] = useState('');
  const sourceId = dialog.kind === 'connection' ? dialog.source : dialog.kind === 'layer' ? dialog.parent : undefined;
  const fromDataset = project.datasets.some(item => item.id === sourceId);
  const title = dialog.kind === 'confirm' ? dialog.title : dialog.kind === 'source' ? dataset ? 'Edit data source' : 'Add data source' : dialog.kind === 'layer' ? layer ? 'Edit layer' : 'Add connected layer' : dialog.kind === 'transforms' ? 'Set transforms' : dialog.id ? 'Edit connection' : 'Connect to a layer';

  function addConnection(next: ProjectSpec, source: string, destination: string, connectionId = makeId('connection')) {
    const problem = connectionError(next, source, destination);
    if (problem) throw new Error(problem);
    if (next.datasets.some(item => item.id === source)) {
      next.feeds.push({ id: connectionId, dataset: source, layer: destination, field: field.trim(), role });
    } else next.network.links.push({ id: connectionId, source, target: destination });
  }
  async function submit(event: FormEvent) {
    event.preventDefault();
    setLocalError('');
    try {
      if (dialog.kind === 'confirm') { if (await dialog.confirm()) close(); return; }
      const next = structuredClone(project);
      if (dialog.kind === 'source' || dialog.kind === 'layer') {
        if (!id.trim()) throw new Error('Enter a name.');
        if (!dialog.id && [...next.network.layers, ...next.datasets].some(item => item.id === id.trim())) throw new Error('A node with this name already exists.');
      }
      if (dialog.kind === 'source') {
        if (!path) throw new Error('Choose a stored file or upload one.');
        const item: DatasetSpec = {
          id: id.trim(), pos: dataset?.pos ?? dialog.position ?? nextPosition(project), transforms: dataset?.transforms ?? [],
          source: { ...(dataset?.source.type === format ? dataset.source : {}), type: format, path, ...(format === 'csv' ? { delimiter, quotechar } : {}) },
        };
        next.datasets = dataset ? next.datasets.map(value => value.id === dataset.id ? item : value) : [...next.datasets, item];
      } else if (dialog.kind === 'layer') {
        if (!Number.isSafeInteger(size) || size < 1) throw new Error('Layer size must be a positive whole number.');
        const item = { id: id.trim(), size, activation, pos: layer?.pos ?? nextPosition(project, dialog.parent) };
        next.network.layers = layer ? next.network.layers.map(value => value.id === layer.id ? item : value) : [...next.network.layers, item];
        if (!layer && dialog.parent) addConnection(next, dialog.parent, item.id);
      } else if (dialog.kind === 'connection') {
        next.network.links = next.network.links.filter(item => item.id !== dialog.id);
        next.feeds = next.feeds.filter(item => item.id !== dialog.id);
        addConnection(next, dialog.source, target, dialog.id);
      } else if (dialog.kind === 'transforms') {
        for (const transform of transforms) {
          if (transform.type === 'select' ? !transform.fields.length : !transform.field.trim()) throw new Error('Each transform needs a field.');
          if (transform.type === 'scale' && (!Number.isFinite(transform.divisor) || transform.divisor === 0)) throw new Error('Scale divisor must be a non-zero number.');
        }
        next.datasets = next.datasets.map(item => item.id === dialog.id ? { ...item, transforms } : item);
      }
      if (await save(next)) close();
    } catch (err) { setLocalError(err instanceof Error ? err.message : 'Could not save changes.'); }
  }
  const updateTransform = (index: number, value: TransformSpec) => setTransforms(items => items.map((item, i) => i === index ? value : item));
  return <Modal title={title} close={close} busy={busy}>
    <form onSubmit={submit}>
      <fieldset disabled={busy} className="dialog-body">
        {dialog.kind === 'confirm' ? <p>{dialog.description}</p> : <>
          {(dialog.kind === 'source' || dialog.kind === 'layer') && <label>Name<input autoFocus required value={id} readOnly={!!dialog.id} onChange={event => setId(event.target.value)} /></label>}
          {dialog.kind === 'source' && <>
            <label>Project file<select value={path} onChange={event => { setPath(event.target.value); setFormat(inferFormat(event.target.value)); }} required>
              <option value="">Select a file…</option>
              {path && !files.some(file => file.path === path) && <option value={path}>{basename(path)} (existing source)</option>}
              {files.map(file => <option key={file.id} value={file.path}>{file.name}</option>)}
            </select></label>
            <label className="upload-box">Upload a new file<input type="file" onChange={async event => {
              if (!event.target.files?.length) return;
              const added = await upload(event.target.files);
              if (added[0]) { setPath(added[0].path); setFormat(inferFormat(added[0].name)); }
              event.target.value = '';
            }} /><small>Files are stored on the server and can be reused. Up to 100 MB per file.</small></label>
            <label>File format<select value={format} onChange={event => setFormat(event.target.value as typeof format)}>
              <option value="csv">CSV</option><option value="json">JSON</option><option value="npy">NumPy (.npy)</option><option value="idx">IDX</option>
              {dataset?.source.type === 'image_folder' && <option value="image_folder">Image folder</option>}
            </select></label>
            {format === 'csv' && <div className="form-row"><label>Delimiter<input required maxLength={1} value={delimiter} onChange={event => setDelimiter(event.target.value)} /></label><label>Quote character<input required maxLength={1} value={quotechar} onChange={event => setQuotechar(event.target.value)} /></label></div>}
          </>}
          {dialog.kind === 'layer' && <>
            {dialog.parent && <p className="muted">A connection from <strong>{dialog.parent}</strong> will be added automatically.</p>}
            <div className="form-row"><label>Size<input type="number" min="1" step="1" required value={size} onChange={event => setSize(Number(event.target.value))} /></label>
            <label>Activation<select value={activation} onChange={event => setActivation(event.target.value)}>{Array.from(new Set(['relu', 'identity', 'sigmoid', 'tanh', 'softmax', activation])).map(value => <option key={value}>{value}</option>)}</select></label></div>
          </>}
          {dialog.kind === 'connection' && <>
            <p className="muted">From <strong>{dialog.source}</strong></p>
            <label>Destination layer<select autoFocus required value={target} onChange={event => setTarget(event.target.value)}>
              <option value="">Select a layer…</option>
              {project.network.layers.filter(item => item.id !== dialog.source).map(item => <option key={item.id} value={item.id}>{item.id}</option>)}
            </select></label>
            {!project.network.layers.some(item => item.id !== dialog.source) && <p className="muted">Add a layer from this node first.</p>}
          </>}
          {fromDataset && <div className="connection-fields"><p>Data to pass into the layer</p><div className="form-row"><label>Field<input required value={field} onChange={event => setField(event.target.value)} /></label><label>Use as<select value={role} onChange={event => setRole(event.target.value as typeof role)}><option value="input">Input</option><option value="target">Training target</option></select></label></div><small>Use a CSV column or JSON field name; NumPy and IDX files use “value”.</small></div>}
          {dialog.kind === 'transforms' && <>
            <p className="muted">Transforms run in this order on <strong>{dialog.id}</strong>.</p>
            {!transforms.length && <p>No transforms. Data will be used as loaded.</p>}
            {transforms.map((transform, index) => <div className="transform-row" key={index}>
              <div className="transform-heading"><span>{index + 1}.</span><label>Transform<select value={transform.type} onChange={event => updateTransform(index, defaultTransform(event.target.value as TransformSpec['type']))}>{['flatten', 'cast', 'scale', 'one_hot', 'select'].map(type => <option key={type} value={type}>{type.replace('_', ' ')}</option>)}</select></label>
              <button type="button" aria-label={`Move transform ${index + 1} up`} disabled={index === 0} onClick={() => setTransforms(items => { const copy = [...items]; [copy[index - 1], copy[index]] = [copy[index], copy[index - 1]]; return copy; })}>↑</button>
              <button type="button" className="danger" aria-label={`Remove transform ${index + 1}`} onClick={() => setTransforms(items => items.filter((_, i) => i !== index))}>×</button></div>
              {transform.type === 'select' ? <label>Fields (comma separated)<input required value={transform.fields.join(',')} onChange={event => updateTransform(index, { ...transform, fields: event.target.value.split(',') })} onBlur={() => updateTransform(index, { ...transform, fields: transform.fields.map(value => value.trim()).filter(Boolean) })} /></label> : <label>Field<input required value={transform.field} onChange={event => updateTransform(index, { ...transform, field: event.target.value })} /></label>}
              {transform.type === 'scale' && <label>Divide by<input required type="number" step="any" value={transform.divisor} onChange={event => updateTransform(index, { ...transform, divisor: Number(event.target.value) })} /></label>}
              {transform.type === 'cast' && <label>Data type<select value={transform.dtype} onChange={event => updateTransform(index, { ...transform, dtype: event.target.value })}>{Array.from(new Set(['float32', 'float64', 'int32', 'int64', 'uint8', transform.dtype])).map(type => <option key={type}>{type}</option>)}</select></label>}
              {transform.type === 'one_hot' && <small>{transform.categories ? `Categories: ${JSON.stringify(transform.categories)}` : 'Categories are inferred from the data.'}</small>}
            </div>)}
            <button type="button" onClick={() => setTransforms(items => [...items, defaultTransform('flatten')])}>+ Add transform</button>
          </>}
        </>}
      </fieldset>
      {(localError || error) && <p className="form-error" role="alert">{localError || error}</p>}
      <footer className="dialog-footer"><button type="button" disabled={busy} onClick={close}>Cancel</button><button className={dialog.kind === 'confirm' ? 'danger-button' : 'primary'} disabled={busy} type="submit">{busy ? 'Saving…' : dialog.kind === 'confirm' ? 'Delete' : 'Save'}</button></footer>
    </form>
  </Modal>;
}
function defaultTransform(type: TransformSpec['type']): TransformSpec {
  switch (type) {
    case 'select': return { type, fields: ['value'] };
    case 'cast': return { type, field: 'value', dtype: 'float32' };
    case 'scale': return { type, field: 'value', divisor: 255 };
    default: return { type, field: 'value' };
  }
}
function inferFormat(path: string): DatasetSpec['source']['type'] {
  const name = path.toLowerCase();
  return name.endsWith('.json') ? 'json' : name.endsWith('.npy') ? 'npy' : /idx|ubyte/.test(name) ? 'idx' : 'csv';
}
