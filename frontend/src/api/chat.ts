import { API_BASE, api, getAccessToken, refreshAccessToken } from '@/api/client';
import type { ChatMessage, ChatSession, ChatSessionDetail, Citation } from '@/types';

export function listSessions(): Promise<ChatSession[]> {
  return api.get<ChatSession[]>('/chat/sessions');
}

export function createSession(title?: string): Promise<ChatSession> {
  return api.post<ChatSession>('/chat/sessions', { title: title || null });
}

export function getSession(id: string): Promise<ChatSessionDetail> {
  return api.get<ChatSessionDetail>(`/chat/sessions/${id}`);
}

export function renameSession(id: string, title: string): Promise<ChatSession> {
  return api.patch<ChatSession>(`/chat/sessions/${id}`, { title });
}

export function deleteSession(id: string): Promise<void> {
  return api.delete<void>(`/chat/sessions/${id}`);
}

export function ask(
  sessionId: string,
  message: string,
  documentIds?: string[],
): Promise<{ user_message: ChatMessage; assistant_message: ChatMessage }> {
  return api.post(`/chat/sessions/${sessionId}/ask`, {
    message,
    document_ids: documentIds ?? null,
  });
}

export interface StreamHandlers {
  onMeta?: (citations: Citation[]) => void;
  onToken?: (text: string) => void;
  onDone?: (latencyMs: number) => void;
  onError?: (message: string) => void;
}

/**
 * Consume the SSE answer stream.
 *
 * EventSource cannot be used here: it only issues GET requests and cannot
 * carry an Authorization header. Reading the body with fetch + ReadableStream
 * gives both, plus an AbortSignal so the user can stop a long answer.
 */
export async function askStream(
  sessionId: string,
  message: string,
  handlers: StreamHandlers,
  options: { documentIds?: string[]; signal?: AbortSignal } = {},
): Promise<void> {
  const send = (token: string | null) =>
    fetch(`${API_BASE}/chat/sessions/${sessionId}/ask/stream`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      credentials: 'same-origin',
      signal: options.signal,
      body: JSON.stringify({
        message,
        document_ids: options.documentIds ?? null,
      }),
    });

  let response = await send(getAccessToken());
  if (response.status === 401 && (await refreshAccessToken())) {
    response = await send(getAccessToken());
  }
  if (!response.ok || !response.body) {
    handlers.onError?.(`Request failed with status ${response.status}`);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  const dispatch = (frame: string) => {
    let event = 'message';
    const dataLines: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice(6).trim();
      else if (line.startsWith('data:')) dataLines.push(line.slice(5).trim());
    }
    if (dataLines.length === 0) return;

    let payload: Record<string, unknown>;
    try {
      payload = JSON.parse(dataLines.join('\n')) as Record<string, unknown>;
    } catch {
      return;
    }

    switch (event) {
      case 'meta':
        handlers.onMeta?.((payload.citations as Citation[]) ?? []);
        break;
      case 'token':
        handlers.onToken?.(String(payload.text ?? ''));
        break;
      case 'done':
        handlers.onDone?.(Number(payload.latency_ms ?? 0));
        break;
      case 'error':
        handlers.onError?.(String(payload.message ?? 'Unknown streaming error'));
        break;
    }
  };

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      // Frames are separated by a blank line; keep the trailing partial.
      const frames = buffer.split('\n\n');
      buffer = frames.pop() ?? '';
      frames.filter((f) => f.trim()).forEach(dispatch);
    }
    if (buffer.trim()) dispatch(buffer);
  } catch (error) {
    if ((error as Error).name !== 'AbortError') {
      handlers.onError?.((error as Error).message);
    }
  } finally {
    reader.releaseLock();
  }
}
