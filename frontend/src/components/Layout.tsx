import { NavLink, Outlet } from 'react-router-dom';

import { useAuth } from '@/context/useAuth';

export function Layout() {
  const { user, logout } = useAuth();

  return (
    <div className="layout">
      <header className="topbar">
        <NavLink to="/chat" className="brand">
          📚 RAG Platform
        </NavLink>
        <nav aria-label="Main">
          <NavLink to="/chat">Chat</NavLink>
          <NavLink to="/documents">Documents</NavLink>
        </nav>
        <div className="spacer" />
        <span className="who">{user?.email}</span>
        <button type="button" className="btn btn-sm" onClick={() => void logout()}>
          Sign out
        </button>
      </header>
      <Outlet />
    </div>
  );
}
