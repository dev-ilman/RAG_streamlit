import type { ReactNode } from 'react';
import { useCallback, useEffect, useMemo, useState } from 'react';

import * as authApi from '@/api/auth';
import { refreshAccessToken, setAccessToken, setUnauthenticatedHandler } from '@/api/client';
import { AuthContext } from '@/context/authContext';
import type { User } from '@/types';

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [initialising, setInitialising] = useState(true);

  // On a fresh page load there is no access token in memory, but the browser
  // may still hold a valid refresh cookie. Try once to restore the session.
  useEffect(() => {
    let cancelled = false;
    (async () => {
      if (await refreshAccessToken()) {
        try {
          const current = await authApi.me();
          if (!cancelled) setUser(current);
        } catch {
          if (!cancelled) setUser(null);
        }
      }
      if (!cancelled) setInitialising(false);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // When a refresh finally fails, drop the user so the router redirects.
  useEffect(() => {
    setUnauthenticatedHandler(() => setUser(null));
    return () => setUnauthenticatedHandler(null);
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    setUser((await authApi.login(email, password)).user);
  }, []);

  const register = useCallback(
    async (email: string, password: string, fullName?: string) => {
      setUser((await authApi.register(email, password, fullName)).user);
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } finally {
      setAccessToken(null);
      setUser(null);
    }
  }, []);

  const value = useMemo(
    () => ({ user, initialising, login, register, logout }),
    [user, initialising, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}
