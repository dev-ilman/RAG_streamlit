import { api, request, setAccessToken } from '@/api/client';
import type { TokenResponse, User } from '@/types';

export async function register(
  email: string,
  password: string,
  fullName?: string,
): Promise<TokenResponse> {
  const result = await api.post<TokenResponse>('/auth/register', {
    email,
    password,
    full_name: fullName || null,
  });
  setAccessToken(result.access_token);
  return result;
}

export async function login(email: string, password: string): Promise<TokenResponse> {
  const result = await api.post<TokenResponse>('/auth/login', { email, password });
  setAccessToken(result.access_token);
  return result;
}

export async function logout(): Promise<void> {
  try {
    await request<void>('/auth/logout', { method: 'POST' });
  } finally {
    setAccessToken(null);
  }
}

export function me(): Promise<User> {
  return api.get<User>('/auth/me');
}
