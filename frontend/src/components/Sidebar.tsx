import { NavLink } from 'react-router-dom';
import './Sidebar.css';

interface SidebarProps {
  open: boolean;
}

function Sidebar({ open }: SidebarProps) {
  return (
    <aside className={`sidebar ${open ? 'open' : 'closed'}`}>
      <nav className="sidebar-nav">
        <ul className="nav-list">
          <li className="nav-item">
            <NavLink to="/" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">📊</span>
              <span className="nav-text">Dashboard</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/streams" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">📈</span>
              <span className="nav-text">Data Streams</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/alerts" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">🔔</span>
              <span className="nav-text">Anomaly Alerts</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/models" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">🧠</span>
              <span className="nav-text">Detection Models</span>
            </NavLink>
          </li>
          <li className="nav-item">
            <NavLink to="/settings" className={({ isActive }) => isActive ? 'nav-link active' : 'nav-link'}>
              <span className="nav-icon">⚙️</span>
              <span className="nav-text">Settings</span>
            </NavLink>
          </li>
        </ul>
      </nav>
      <div className="sidebar-footer">
        <div className="system-status">
          <div className="status-indicator online"></div>
          <span className="status-text">System Online</span>
        </div>
        <div className="version-info">v0.1.0</div>
      </div>
    </aside>
  );
}

export default Sidebar;
