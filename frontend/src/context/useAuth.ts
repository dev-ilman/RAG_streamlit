import { useContext } from 'react';

import { AuthContext } from '@/context/authContext';
import type { AuthState } from '@/context/authContext';

export function useAuth(): AuthState {
  const context = useContext(AuthContext);
  if (!context) throw new Error('useAuth must be used inside an AuthProvider');
  return context;
}
