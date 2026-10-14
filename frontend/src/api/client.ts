/**
 * Thin fetch wrapper.
 *
 * The access token lives in memory only - never localStorage - so an XSS bug
 * cannot walk off with a credential. Durability comes from the httpOnly
 * refresh cookie: on a 401 the client silently calls /auth/refresh once and
 * replays the original request. Concurrent 401s share a single refresh
 * promise, so a page that fires several requests at once does not stampede.
 */

import type { ApiErrorBody } from '@/types';

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly details?: unknown,
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

let accessToken: string | null = null;
let refreshPromise: Promise<boolean> | null = null;
let onUnauthenticated: (() => void) | null = null;

export function setAccessToken(token: string | null): void {
  accessToken = token;
}

export function getAccessToken(): string | null {
  return accessToken;
}

export function setUnauthenticatedHandler(handler: (() => void) | null): void {
  onUnauthenticated = handler;
}

function authHeaders(extra?: HeadersInit): Headers {
  const headers = new Headers(extra);
  if (accessToken) headers.set('Authorization', `Bearer ${accessToken}`);
  return headers;
}

async function toApiError(response: Response): Promise<ApiError> {
  let code = `http_${response.status}`;
  let message = response.statusText || 'Request failed';
  try {
    const body = (await response.json()) as Partial<ApiErrorBody>;
    if (body.error) {
      code = body.error.code ?? code;
      message = body.error.message ?? message;
      return new ApiError(response.status, code, message, body.error.details);
    }
  } catch {
    // Non-JSON error body (a proxy or gateway page); keep the status text.
  }
  return new ApiError(response.status, code, message);
}

/** Exchange the refresh cookie for a new access token. At most one in flight. */
export async function refreshAccessToken(): Promise<boolean> {
  refreshPromise ??= (async () => {
    try {
      const response = await fetch(`${API_BASE}/auth/refresh`, {
        method: 'POST',
        credentials: 'same-origin',
      });
      if (!response.ok) return false;
      const body = (await response.json()) as { access_token: string };
      accessToken = body.access_token;
      return true;
    } catch {
      return false;
    } finally {
      // Clear on the next tick so callers awaiting this promise all see it.
      queueMicrotask(() => {
        refreshPromise = null;
      });
    }
  })();
  return refreshPromise;
}

interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  /** Set internally to stop a refreshed request from looping. */
  _retried?: boolean;
}

export async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, headers, _retried, ...rest } = options;
  const isFormData = body instanceof FormData;

  const requestHeaders = authHeaders(headers);
  if (body !== undefined && !isFormData) {
    requestHeaders.set('Content-Type', 'application/json');
  }

  const response = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: requestHeaders,
    credentials: 'same-origin',
    body: isFormData ? body : body === undefined ? undefined : JSON.stringify(body),
  });

  if (response.status === 401 && !_retried && !path.startsWith('/auth/')) {
    if (await refreshAccessToken()) {
      return request<T>(path, { ...options, _retried: true });
    }
    accessToken = null;
    onUnauthenticated?.();
  }

  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  post: <T>(path: string, body?: unknown) => request<T>(path, { method: 'POST', body }),
  patch: <T>(path: string, body?: unknown) => request<T>(path, { method: 'PATCH', body }),
  delete: <T>(path: string) => request<T>(path, { method: 'DELETE' }),
};
