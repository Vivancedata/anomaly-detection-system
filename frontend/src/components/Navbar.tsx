import { Link } from 'react-router-dom';
import './Navbar.css';

interface NavbarProps {
  toggleSidebar: () => void;
}

function Navbar({ toggleSidebar }: NavbarProps) {
  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-left">
          <button className="menu-button" onClick={toggleSidebar}>
            <span className="menu-icon">☰</span>
          </button>
          <Link to="/" className="navbar-brand">
            <h1 className="navbar-title">VivanceData</h1>
            <span className="navbar-subtitle">Real-Time Anomaly Detection</span>
          </Link>
        </div>
        <div className="navbar-right">
          <button className="nav-button">
            <span className="notification-icon">🔔</span>
            <span className="notification-badge">3</span>
          </button>
          <button className="nav-button">
            <span className="settings-icon">⚙️</span>
          </button>
          <div className="user-profile">
            <img
              src="/profile-placeholder.jpg"
              alt="User"
              className="profile-image"
            />
            <span className="user-name">Admin</span>
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
