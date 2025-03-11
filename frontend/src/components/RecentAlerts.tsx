import React from 'react';
import { AnomalyAlert } from '../types';
import './RecentAlerts.css';

interface RecentAlertsProps {
  alerts: AnomalyAlert[];
}

const RecentAlerts: React.FC<RecentAlertsProps> = ({ alerts }) => {
  // Format time relative to now (e.g., "2 hours ago")
  const formatTimeAgo = (timestamp: Date): string => {
    const now = new Date();
    const diffMs = now.getTime() - timestamp.getTime();
    const diffSec = Math.floor(diffMs / 1000);
    const diffMin = Math.floor(diffSec / 60);
    const diffHour = Math.floor(diffMin / 60);
    const diffDay = Math.floor(diffHour / 24);
    
    if (diffDay > 0) {
      return `${diffDay} day${diffDay > 1 ? 's' : ''} ago`;
    }
    if (diffHour > 0) {
      return `${diffHour} hour${diffHour > 1 ? 's' : ''} ago`;
    }
    if (diffMin > 0) {
      return `${diffMin} minute${diffMin > 1 ? 's' : ''} ago`;
    }
    return 'Just now';
  };

  return (
    <div className="recent-alerts card">
      <div className="card-header">
        <h2 className="card-title">Recent Alerts</h2>
        <button className="view-all-button">View All</button>
      </div>
      <div className="card-body">
        <div className="alerts-list">
          {alerts.length === 0 ? (
            <div className="no-alerts">No recent alerts</div>
          ) : (
            alerts.map(alert => (
              <div key={alert.id} className="alert-item">
                <div className="alert-header">
                  <div className="alert-stream">
                    <span className="alert-icon">🔔</span>
                    {alert.streamName}
                  </div>
                  <div className={`alert-severity ${alert.severity}`}>
                    {alert.severity}
                  </div>
                </div>
                <div className="alert-description">{alert.description}</div>
                <div className="alert-footer">
                  <div className="alert-score">
                    Score: <span className="score-value">{(alert.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="alert-time">{formatTimeAgo(alert.timestamp)}</div>
                </div>
                <div className="alert-status">
                  <span className={`status-indicator ${alert.status}`}></span>
                  {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
};

export default RecentAlerts;
