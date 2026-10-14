import { api, request } from '@/api/client';
import type { DocumentList, DocumentRecord, SearchHit } from '@/types';

export function listDocuments(limit = 50, offset = 0): Promise<DocumentList> {
  return api.get<DocumentList>(`/documents?limit=${limit}&offset=${offset}`);
}

export function getDocument(id: string): Promise<DocumentRecord> {
  return api.get<DocumentRecord>(`/documents/${id}`);
}

export function uploadDocument(file: File, title?: string): Promise<DocumentRecord> {
  const form = new FormData();
  form.append('file', file);
  if (title) form.append('title', title);
  return request<DocumentRecord>('/documents/upload', { method: 'POST', body: form });
}

export function ingestUrl(url: string, title?: string): Promise<DocumentRecord> {
  return api.post<DocumentRecord>('/documents/url', { url, title: title || null });
}

export function ingestText(title: string, content: string): Promise<DocumentRecord> {
  return api.post<DocumentRecord>('/documents/text', { title, content });
}

export function reingestDocument(id: string): Promise<DocumentRecord> {
  return api.post<DocumentRecord>(`/documents/${id}/reingest`);
}

export function deleteDocument(id: string): Promise<void> {
  return api.delete<void>(`/documents/${id}`);
}

export async function search(query: string, topK = 5): Promise<SearchHit[]> {
  const result = await api.post<{ hits: SearchHit[] }>('/documents/search', {
    query,
    top_k: topK,
  });
  return result.hits;
}
