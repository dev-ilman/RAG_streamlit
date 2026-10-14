/** Mirrors the Pydantic schemas served at /openapi.json. */

export type DocumentStatus = 'pending' | 'processing' | 'ready' | 'failed';
export type SourceType = 'upload' | 'url' | 'text';
export type MessageRole = 'user' | 'assistant' | 'system';

export interface User {
  id: string;
  email: string;
  full_name: string | null;
  is_active: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in: number;
  user: User;
}

export interface DocumentRecord {
  id: string;
  title: string;
  status: DocumentStatus;
  source_type: SourceType;
  source_uri: string | null;
  filename: string | null;
  mime_type: string | null;
  size_bytes: number;
  chunk_count: number;
  error: string | null;
  created_at: string;
  updated_at: string;
}

export interface DocumentList {
  items: DocumentRecord[];
  total: number;
  limit: number;
  offset: number;
}

export interface Citation {
  chunk_id: string;
  document_id: string;
  document_title: string;
  chunk_index: number;
  score: number;
  excerpt: string;
}

export interface ChatMessage {
  id: string;
  session_id: string;
  role: MessageRole;
  content: string;
  citations: Citation[];
  model: string | null;
  latency_ms: number | null;
  created_at: string;
}

export interface ChatSession {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  last_message_at: string;
}

export interface ChatSessionDetail extends ChatSession {
  messages: ChatMessage[];
}

export interface SearchHit {
  chunk_id: string;
  document_id: string;
  document_title: string;
  chunk_index: number;
  score: number;
  content: string;
}

export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown };
}
