import React, { useState, useEffect } from 'react';
import { AnomalyAlert } from '../types';
import { mockAlerts } from '../utils/mockData';
import './Alerts.css';

const Alerts: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [alerts, setAlerts] = useState<AnomalyAlert[]>([]);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    // Simulate API loading
    const timer = setTimeout(() => {
      setAlerts(mockAlerts);
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  const filteredAlerts = filter === 'all' 
    ? alerts 
    : alerts.filter(alert => alert.status === filter);

  const getStatusClass = (status: string): string => {
    switch(status) {
      case 'open': return 'status-open';
      case 'acknowledged': return 'status-acknowledged';
      case 'resolved': return 'status-resolved';
      default: return '';
    }
  };

  if (loading) {
    return (
      <div className="alerts-page loading-container">
        <div className="loading-spinner"></div>
        <p>Loading alerts data...</p>
      </div>
    );
  }

  return (
    <div className="alerts-page">
      <h1 className="page-title">Anomaly Alerts</h1>
      
      <div className="alerts-controls">
        <div className="filter-buttons">
          <button 
            className={`filter-btn ${filter === 'all' ? 'active' : ''}`}
            onClick={() => setFilter('all')}
          >
            All Alerts
          </button>
          <button 
            className={`filter-btn ${filter === 'open' ? 'active' : ''}`}
            onClick={() => setFilter('open')}
          >
            Open
          </button>
          <button 
            className={`filter-btn ${filter === 'acknowledged' ? 'active' : ''}`}
            onClick={() => setFilter('acknowledged')}
          >
            Acknowledged
          </button>
          <button 
            className={`filter-btn ${filter === 'resolved' ? 'active' : ''}`}
            onClick={() => setFilter('resolved')}
          >
            Resolved
          </button>
        </div>
      </div>
      
      <div className="alerts-list">
        {filteredAlerts.length === 0 ? (
          <div className="no-alerts">No alerts matching the selected filter</div>
        ) : (
          filteredAlerts.map(alert => (
            <div 
              key={alert.id} 
              className={`alert-card ${getStatusClass(alert.status)}`}
            >
              <div className="alert-header">
                <div className="alert-title">
                  <span className={`severity-indicator ${alert.severity}`}></span>
                  {alert.description}
                </div>
                <div className="alert-stream">{alert.streamName}</div>
              </div>
              
              <div className="alert-details">
                <div className="alert-metrics">
                  <div className="metric">
                    <span className="metric-label">Score:</span>
                    <span className="metric-value">{(alert.score * 100).toFixed(0)}%</span>
                  </div>
                  <div className="metric">
                    <span className="metric-label">Status:</span>
                    <span className={`metric-value status-${alert.status}`}>
                      {alert.status.charAt(0).toUpperCase() + alert.status.slice(1)}
                    </span>
                  </div>
                </div>
                
                <div className="alert-time">
                  <div>Detected at: {new Date(alert.detectedAt).toLocaleString()}</div>
                  <div>Data timestamp: {new Date(alert.timestamp).toLocaleString()}</div>
                </div>
              </div>
              
              <div className="alert-footer">
                <div className="contributing-features">
                  <span className="features-label">Contributing features:</span>
                  {alert.contributingFeatures.map(feature => (
                    <span key={feature} className="feature-tag">{feature}</span>
                  ))}
                </div>
                
                <div className="alert-actions">
                  {alert.status === 'open' && (
                    <>
                      <button className="action-btn acknowledge-btn">Acknowledge</button>
                      <button className="action-btn resolve-btn">Resolve</button>
                    </>
                  )}
                  {alert.status === 'acknowledged' && (
                    <button className="action-btn resolve-btn">Resolve</button>
                  )}
                  <button className="action-btn details-btn">View Details</button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};

export default Alerts;
