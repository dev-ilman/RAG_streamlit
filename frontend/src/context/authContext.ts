import { createContext } from 'react';

import type { User } from '@/types';

export interface AuthState {
  user: User | null;
  /** True until the initial silent refresh settles, so routes do not flash. */
  initialising: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string, fullName?: string) => Promise<void>;
  logout: () => Promise<void>;
}

/** Kept apart from the provider component so that module only exports
 *  components, which is what react-refresh needs for reliable hot reloading. */
export const AuthContext = createContext<AuthState | null>(null);
