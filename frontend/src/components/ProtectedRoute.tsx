import { Navigate, useLocation } from 'react-router-dom';
import type { ReactNode } from 'react';

import { Spinner } from '@/components/Spinner';
import { useAuth } from '@/context/useAuth';

export function ProtectedRoute({ children }: { children: ReactNode }) {
  const { user, initialising } = useAuth();
  const location = useLocation();

  // Wait for the silent refresh to settle, otherwise a reload would bounce a
  // signed-in user to the login page before the session is restored.
  if (initialising) {
    return (
      <div className="empty">
        <Spinner label="Restoring session" />
      </div>
    );
  }

  if (!user) return <Navigate to="/login" state={{ from: location.pathname }} replace />;
  return <>{children}</>;
}
