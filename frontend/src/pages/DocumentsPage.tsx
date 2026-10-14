import { useCallback, useEffect, useRef, useState } from 'react';

import { ApiError } from '@/api/client';
import * as documentsApi from '@/api/documents';
import { Spinner } from '@/components/Spinner';
import { StatusBadge } from '@/components/StatusBadge';
import type { DocumentRecord } from '@/types';

type Tab = 'file' | 'url' | 'text';

const POLL_INTERVAL_MS = 1500;

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

export function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentRecord[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [tab, setTab] = useState<Tab>('file');

  const [file, setFile] = useState<File | null>(null);
  const [url, setUrl] = useState('');
  const [title, setTitle] = useState('');
  const [text, setText] = useState('');
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    try {
      const list = await documentsApi.listDocuments();
      setDocuments(list.items);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load documents.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Ingestion is asynchronous, so poll while anything is still in flight and
  // stop as soon as every document has reached a terminal state.
  const hasPending = documents.some(
    (d) => d.status === 'pending' || d.status === 'processing',
  );
  useEffect(() => {
    if (!hasPending) return;
    const timer = setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => clearInterval(timer);
  }, [hasPending, refresh]);

  async function submit(action: () => Promise<DocumentRecord>) {
    setError(null);
    setBusy(true);
    try {
      await action();
      setFile(null);
      setUrl('');
      setTitle('');
      setText('');
      if (fileInput.current) fileInput.current.value = '';
      await refresh();
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Ingestion failed.');
    } finally {
      setBusy(false);
    }
  }

  async function remove(id: string) {
    setError(null);
    try {
      await documentsApi.deleteDocument(id);
      setDocuments((current) => current.filter((d) => d.id !== id));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Delete failed.');
    }
  }

  return (
    <main className="page">
      <div className="page-head">
        <div>
          <h1>Documents</h1>
          <p className="hint">
            Uploads are chunked and embedded in the background. A document becomes
            searchable once it is <strong>Ready</strong>.
          </p>
        </div>
        <button type="button" className="btn btn-sm" onClick={() => void refresh()}>
          Refresh
        </button>
      </div>

      {error && (
        <div className="alert alert-error" role="alert">
          {error}
        </div>
      )}

      <section className="card ingest-panel">
        <div className="tabs" role="tablist" aria-label="Add a document">
          {(['file', 'url', 'text'] as const).map((key) => (
            <button
              key={key}
              type="button"
              role="tab"
              aria-selected={tab === key}
              onClick={() => setTab(key)}
            >
              {key === 'file' ? 'Upload file' : key === 'url' ? 'From URL' : 'Paste text'}
            </button>
          ))}
        </div>

        {tab === 'file' && (
          <div className="field">
            <label htmlFor="file">PDF, TXT, Markdown, HTML or CSV</label>
            <input
              id="file"
              ref={fileInput}
              className="input"
              type="file"
              accept=".pdf,.txt,.md,.markdown,.html,.htm,.csv,.json"
              onChange={(e) => setFile(e.target.files?.[0] ?? null)}
            />
            <button
              type="button"
              className="btn btn-primary"
              disabled={!file || busy}
              onClick={() => file && void submit(() => documentsApi.uploadDocument(file))}
            >
              {busy ? <Spinner label="Uploading" /> : 'Upload'}
            </button>
          </div>
        )}

        {tab === 'url' && (
          <div className="field">
            <label htmlFor="url">Page URL</label>
            <input
              id="url"
              className="input"
              type="url"
              placeholder="https://example.com/article"
              value={url}
              onChange={(e) => setUrl(e.target.value)}
            />
            <button
              type="button"
              className="btn btn-primary"
              disabled={!url || busy}
              onClick={() => void submit(() => documentsApi.ingestUrl(url))}
            >
              {busy ? <Spinner label="Fetching" /> : 'Fetch and index'}
            </button>
          </div>
        )}

        {tab === 'text' && (
          <>
            <div className="field">
              <label htmlFor="title">Title</label>
              <input
                id="title"
                className="input"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
              />
            </div>
            <div className="field">
              <label htmlFor="text">Content</label>
              <textarea
                id="text"
                className="textarea"
                rows={6}
                value={text}
                onChange={(e) => setText(e.target.value)}
              />
              <button
                type="button"
                className="btn btn-primary"
                disabled={!title || !text || busy}
                onClick={() => void submit(() => documentsApi.ingestText(title, text))}
              >
                {busy ? <Spinner label="Indexing" /> : 'Index text'}
              </button>
            </div>
          </>
        )}
      </section>

      {loading ? (
        <div className="empty">
          <Spinner label="Loading documents" />
        </div>
      ) : documents.length === 0 ? (
        <div className="card empty">Nothing indexed yet. Add a document above to begin.</div>
      ) : (
        <div className="doc-grid">
          {documents.map((document) => (
            <article key={document.id} className="card doc-row">
              <div className="grow">
                <p className="title">{document.title}</p>
                <p className="meta">
                  {document.source_type} &middot; {formatSize(document.size_bytes)} &middot;{' '}
                  {document.chunk_count} chunk{document.chunk_count === 1 ? '' : 's'} &middot;{' '}
                  {new Date(document.created_at).toLocaleString()}
                </p>
              </div>
              <StatusBadge status={document.status} />
              {document.status === 'failed' && (
                <button
                  type="button"
                  className="btn btn-sm"
                  onClick={() => void submit(() => documentsApi.reingestDocument(document.id))}
                >
                  Retry
                </button>
              )}
              <button
                type="button"
                className="btn btn-sm btn-danger"
                onClick={() => void remove(document.id)}
                aria-label={`Delete ${document.title}`}
              >
                Delete
              </button>
              {document.error && <p className="err">{document.error}</p>}
            </article>
          ))}
        </div>
      )}
    </main>
  );
}
