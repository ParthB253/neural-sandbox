import { useEffect, useRef, useState, type ReactNode } from 'react';
import { request, type NetworkSummary } from './network';

interface Props {
  title: string;
  disabled: boolean;
  busy: boolean;
  dirty: boolean;
  saved: boolean;
  status: string;
  error: string;
  onRename: (title: string) => void;
  onSave: (copyTitle?: string) => Promise<boolean>;
  onNew: () => Promise<boolean>;
  onOpen: (id: string) => Promise<boolean>;
  onClear: () => void;
  canClear: boolean;
  clearError: () => void;
}
type FileDialog = 'save-as' | 'open' | 'leave' | null;

export default function NetworkHeader(props: Props) {
  const { title, disabled, busy, dirty, saved, status, error, onRename, onSave, onNew, onOpen, onClear, canClear, clearError } = props;
  const [menuOpen, setMenuOpen] = useState(false);
  const [dialog, setDialog] = useState<FileDialog>(null);
  const [copyTitle, setCopyTitle] = useState('');
  const [networks, setNetworks] = useState<NetworkSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [listError, setListError] = useState('');
  const [pending, setPending] = useState<(() => Promise<boolean>) | null>(null);
  const menu = useRef<HTMLDivElement>(null);
  const trigger = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!menuOpen) return;
    menu.current?.querySelector<HTMLButtonElement>('[role="menuitem"]:not(:disabled)')?.focus();
    const outside = (event: PointerEvent) => {
      if (!menu.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    document.addEventListener('pointerdown', outside);
    return () => document.removeEventListener('pointerdown', outside);
  }, [menuOpen]);

  useEffect(() => {
    const shortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
        event.preventDefault();
        if (disabled || dialog || document.querySelector('dialog[open]')) return;
        setMenuOpen(false);
        clearError();
        if (event.shiftKey) { setCopyTitle(`${title} copy`); setDialog('save-as'); }
        else void onSave();
      }
    };
    window.addEventListener('keydown', shortcut);
    return () => window.removeEventListener('keydown', shortcut);
  }, [disabled, dialog, onSave, title, clearError]);

  async function openList() {
    setMenuOpen(false);
    clearError();
    setListError('');
    setLoading(true);
    setDialog('open');
    try { setNetworks(await request<NetworkSummary[]>('/networks')); }
    catch (err) { setListError(err instanceof Error ? err.message : 'Could not load saved networks.'); }
    finally { setLoading(false); }
  }
  async function switchTo(action: () => Promise<boolean>) {
    setMenuOpen(false);
    clearError();
    if (dirty) {
      setPending(() => action);
      setDialog('leave');
    } else if (await action()) setDialog(null);
  }
  function closeDialog() { if (!busy) { setDialog(null); setPending(null); clearError(); } }

  return <header className="app-header">
    <div className="document-heading">
      <span className="brand-mark" aria-label="NeuralSandbox" title="NeuralSandbox">nn</span>
      <div className="document-info">
        <input className="network-title" aria-label="Network title" value={title} maxLength={200} disabled={disabled}
          onChange={event => onRename(event.target.value)}
          onBlur={() => onRename(title.trim() || 'Untitled network')}
          onKeyDown={event => { if (event.key === 'Enter') event.currentTarget.blur(); }} />
        <div className="file-menu" ref={menu} onBlur={event => {
          if (!event.currentTarget.contains(event.relatedTarget)) setMenuOpen(false);
        }} onKeyDown={event => {
          if (event.key === 'Escape') { setMenuOpen(false); trigger.current?.focus(); }
          if (menuOpen && ['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
            event.preventDefault();
            const items = Array.from(menu.current?.querySelectorAll<HTMLButtonElement>('[role="menuitem"]:not(:disabled)') ?? []);
            const index = items.indexOf(document.activeElement as HTMLButtonElement);
            const next = event.key === 'Home' ? 0 : event.key === 'End' ? items.length - 1 : (index + (event.key === 'ArrowDown' ? 1 : -1) + items.length) % items.length;
            items[next]?.focus();
          }
        }}>
          <button className="file-menu-trigger" ref={trigger} disabled={disabled} aria-haspopup="menu" aria-expanded={menuOpen} aria-controls="network-file-menu"
            onClick={() => setMenuOpen(value => !value)} onKeyDown={event => { if (event.key === 'ArrowDown' && !menuOpen) { event.preventDefault(); setMenuOpen(true); } }}>File <span aria-hidden="true">▾</span></button>
          {menuOpen && <div className="file-menu-options" id="network-file-menu" role="menu" aria-label="File">
            <button role="menuitem" onClick={() => void switchTo(onNew)}>New network</button>
            <button role="menuitem" onClick={() => void openList()}>Open saved network…</button>
            <div role="separator" />
            <button role="menuitem" onClick={() => { setMenuOpen(false); clearError(); void onSave(); }}>Save <kbd>Ctrl/⌘ S</kbd></button>
            <button role="menuitem" onClick={() => { setMenuOpen(false); clearError(); setCopyTitle(`${title} copy`); setDialog('save-as'); }}>Save as…</button>
          </div>}
        </div>
      </div>
    </div>
    <div className="header-actions">
      <span className={`save-status${saved && !dirty ? ' save-status--ready' : ''}`} role="status">{busy ? 'Working…' : disabled ? status : dirty ? 'Unsaved changes' : saved ? 'Saved' : 'Not saved'}</span>
      <button disabled={disabled || !canClear} onClick={onClear}>Clear canvas</button>
    </div>
    {dialog && <FileModal title={dialog === 'save-as' ? 'Save network as' : dialog === 'open' ? 'Open saved network' : 'Save changes?'} busy={busy} close={closeDialog}>
      {dialog === 'save-as' ? <form onSubmit={async event => { event.preventDefault(); if (copyTitle.trim() && await onSave(copyTitle.trim())) setDialog(null); }}>
        <fieldset className="dialog-body" disabled={busy}><label>Network title<input data-initial-focus required maxLength={200} value={copyTitle} onChange={event => setCopyTitle(event.target.value)} /></label><p className="muted">Create a separate saved network. Uploaded files remain shared.</p></fieldset>
        {error && <p className="form-error" role="alert">{error}</p>}
        <footer className="dialog-footer"><button type="button" disabled={busy} onClick={closeDialog}>Cancel</button><button className="primary" disabled={busy || !copyTitle.trim()}>Save as</button></footer>
      </form> : dialog === 'open' ? <>
        <div className="dialog-body network-list">
          {loading ? <p role="status">Loading saved networks…</p> : listError ? <><p role="alert">{listError}</p><button onClick={() => void openList()}>Retry</button></> : !networks.length ? <p>No saved networks yet. Use File → Save to keep a network here.</p> : <ul className="item-list">{networks.map(network => <li key={network.id}>
            <button className="saved-network" disabled={busy} onClick={() => void switchTo(() => onOpen(network.id))}>
              <strong>{network.title}</strong><small>{network.layers} layers · {network.data_sources} data sources</small>
              <small>Saved {new Date(network.updated_at).toLocaleString()} · {network.id.slice(0, 8)}</small>
            </button>
          </li>)}</ul>}
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <footer className="dialog-footer"><button disabled={busy} onClick={closeDialog}>Cancel</button></footer>
      </> : <>
        <div className="dialog-body"><p>Your changes to <strong>{title || 'Untitled network'}</strong> haven’t been saved. Save them before switching networks?</p></div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <footer className="dialog-footer"><button disabled={busy} onClick={closeDialog}>Cancel</button>
          <button disabled={busy} onClick={async () => { if (pending && await pending()) setDialog(null); }}>Discard changes</button>
          <button className="primary" disabled={busy} onClick={async () => { if (await onSave() && pending && await pending()) setDialog(null); }}>Save and continue</button>
        </footer>
      </>}
    </FileModal>}
  </header>;
}

function FileModal({ title, busy, close, children }: { title: string; busy: boolean; close: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const element = ref.current!;
    element.showModal();
    element.querySelector<HTMLInputElement>('[data-initial-focus]')?.focus();
    return () => element.close();
  }, []);
  return <dialog ref={ref} className="editor-dialog" aria-labelledby="file-dialog-title" onCancel={event => { event.preventDefault(); if (!busy) close(); }}>
    <header className="dialog-header"><h2 id="file-dialog-title">{title}</h2><button aria-label="Close dialog" disabled={busy} onClick={close}>×</button></header>
    {children}
  </dialog>;
}
