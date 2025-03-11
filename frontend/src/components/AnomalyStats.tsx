import React from 'react';
import './AnomalyStats.css';

interface AnomalyStatsProps {
  stats: {
    totalAnomalies: number;
    alertsToday: number;
    falsePositiveRate: number;
    detectionLatency: number;
    streamsMonitored: number;
    modelsActive: number;
  };
}

const AnomalyStats: React.FC<AnomalyStatsProps> = ({ stats }) => {
  return (
    <div className="anomaly-stats">
      <div className="stat-card">
        <div className="stat-value">{stats.totalAnomalies.toLocaleString()}</div>
        <div className="stat-label">Total Anomalies Detected</div>
      </div>
      
      <div className="stat-card">
        <div className="stat-value">{stats.alertsToday}</div>
        <div className="stat-label">Alerts Today</div>
      </div>
      
      <div className="stat-card">
        <div className="stat-value">{(stats.falsePositiveRate * 100).toFixed(1)}%</div>
        <div className="stat-label">False Positive Rate</div>
      </div>
      
      <div className="stat-card">
        <div className="stat-value">{stats.detectionLatency}ms</div>
        <div className="stat-label">Avg. Detection Latency</div>
      </div>
      
      <div className="stat-card">
        <div className="stat-value">{stats.streamsMonitored}</div>
        <div className="stat-label">Streams Monitored</div>
      </div>
      
      <div className="stat-card">
        <div className="stat-value">{stats.modelsActive}</div>
        <div className="stat-label">Active Models</div>
      </div>
    </div>
  );
};

export default AnomalyStats;
