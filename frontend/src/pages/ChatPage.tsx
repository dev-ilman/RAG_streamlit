import { useCallback, useEffect, useRef, useState } from 'react';

import * as chatApi from '@/api/chat';
import { ApiError } from '@/api/client';
import { MessageBubble } from '@/components/MessageBubble';
import { Spinner } from '@/components/Spinner';
import type { ChatMessage, ChatSession, Citation } from '@/types';

/** A placeholder message rendered while tokens stream in. */
function draftMessage(sessionId: string): ChatMessage {
  return {
    id: 'streaming-draft',
    session_id: sessionId,
    role: 'assistant',
    content: '',
    citations: [],
    model: null,
    latency_ms: null,
    created_at: new Date().toISOString(),
  };
}

export function ChatPage() {
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [draft, setDraft] = useState<ChatMessage | null>(null);
  const [input, setInput] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);

  const bottom = useRef<HTMLDivElement>(null);
  const abort = useRef<AbortController | null>(null);

  const loadSessions = useCallback(async () => {
    try {
      const list = await chatApi.listSessions();
      setSessions(list);
      setActiveId((current) => current ?? list[0]?.id ?? null);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not load chats.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);

  useEffect(() => {
    if (!activeId) {
      setMessages([]);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const detail = await chatApi.getSession(activeId);
        if (!cancelled) setMessages(detail.messages);
      } catch (caught) {
        if (!cancelled) {
          setError(caught instanceof ApiError ? caught.message : 'Could not load messages.');
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [activeId]);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, draft]);

  // Abandon an in-flight stream if the component unmounts mid-answer.
  useEffect(() => () => abort.current?.abort(), []);

  async function startSession() {
    try {
      const session = await chatApi.createSession();
      setSessions((current) => [session, ...current]);
      setActiveId(session.id);
      setMessages([]);
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not start a chat.');
    }
  }

  async function removeSession(id: string) {
    try {
      await chatApi.deleteSession(id);
      setSessions((current) => current.filter((s) => s.id !== id));
      setActiveId((current) => (current === id ? null : current));
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : 'Could not delete the chat.');
    }
  }

  async function send() {
    const question = input.trim();
    if (!question || sending) return;

    setError(null);
    setSending(true);
    setInput('');

    // Create a session lazily so the sidebar never fills with empty chats.
    let sessionId = activeId;
    if (!sessionId) {
      try {
        const session = await chatApi.createSession();
        setSessions((current) => [session, ...current]);
        setActiveId(session.id);
        sessionId = session.id;
      } catch (caught) {
        setError(caught instanceof ApiError ? caught.message : 'Could not start a chat.');
        setSending(false);
        return;
      }
    }

    const asked: ChatMessage = {
      id: `local-${Date.now()}`,
      session_id: sessionId,
      role: 'user',
      content: question,
      citations: [],
      model: null,
      latency_ms: null,
      created_at: new Date().toISOString(),
    };
    setMessages((current) => [...current, asked]);
    setDraft(draftMessage(sessionId));

    let citations: Citation[] = [];
    let answer = '';
    const controller = new AbortController();
    abort.current = controller;

    await chatApi.askStream(
      sessionId,
      question,
      {
        onMeta: (received) => {
          citations = received;
          setDraft((current) => (current ? { ...current, citations: received } : current));
        },
        onToken: (token) => {
          answer += token;
          setDraft((current) => (current ? { ...current, content: answer } : current));
        },
        onDone: (latencyMs) => {
          setMessages((current) => [
            ...current,
            {
              id: `answer-${Date.now()}`,
              session_id: sessionId,
              role: 'assistant',
              content: answer.trim(),
              citations,
              model: null,
              latency_ms: latencyMs,
              created_at: new Date().toISOString(),
            },
          ]);
          setDraft(null);
          void loadSessions();
        },
        onError: (message) => {
          setError(message);
          setDraft(null);
        },
      },
      { signal: controller.signal },
    );

    abort.current = null;
    setSending(false);
  }

  function handleKeyDown(event: React.KeyboardEvent<HTMLTextAreaElement>) {
    // Enter sends; Shift+Enter inserts a newline.
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void send();
    }
  }

  return (
    <main className="page">
      <div className="chat">
        <aside className="card chat-sidebar">
          <button type="button" className="btn btn-primary btn-sm" onClick={() => void startSession()}>
            + New chat
          </button>
          {loading ? (
            <div className="empty">
              <Spinner label="Loading chats" />
            </div>
          ) : (
            <ul className="session-list">
              {sessions.map((session) => (
                <li key={session.id} className={session.id === activeId ? 'active' : ''}>
                  <button type="button" className="pick" onClick={() => setActiveId(session.id)}>
                    {session.title}
                  </button>
                  <button
                    type="button"
                    className="remove"
                    aria-label={`Delete chat ${session.title}`}
                    onClick={() => void removeSession(session.id)}
                  >
                    ×
                  </button>
                </li>
              ))}
            </ul>
          )}
        </aside>

        <section className="card chat-main">
          {error && (
            <div className="alert alert-error" role="alert" style={{ margin: '0.8rem' }}>
              {error}
            </div>
          )}

          <div className="messages" aria-live="polite">
            {messages.length === 0 && !draft && (
              <div className="empty">
                <h2>Ask about your documents</h2>
                <p>
                  Every answer is generated only from passages retrieved out of the
                  documents you have indexed, and each reply lists its sources.
                </p>
              </div>
            )}
            {messages.map((message) => (
              <MessageBubble key={message.id} message={message} />
            ))}
            {draft && <MessageBubble message={draft} streaming />}
            <div ref={bottom} />
          </div>

          <div className="composer">
            <label htmlFor="composer" className="sr-only">
              Your question
            </label>
            <textarea
              id="composer"
              className="textarea"
              rows={1}
              placeholder="Ask a question about your documents…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
            />
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => void send()}
              disabled={sending || input.trim().length === 0}
            >
              {sending ? <Spinner label="Thinking" /> : 'Send'}
            </button>
          </div>
        </section>
      </div>
    </main>
  );
}
