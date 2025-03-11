import { useState, useEffect } from 'react';
import './Dashboard.css';
import AnomalyChart from '../components/AnomalyChart';
import AnomalyStats from '../components/AnomalyStats';
import StreamsOverview from '../components/StreamsOverview';
import RecentAlerts from '../components/RecentAlerts';

// In a real app, this would be fetched from the API
import { mockStats, mockStreams, mockAlerts, mockChartData } from '../utils/mockData';

const Dashboard: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [stats] = useState(mockStats);
  const [streams] = useState(mockStreams);
  const [alerts] = useState(mockAlerts);
  const [chartData] = useState(mockChartData);

  useEffect(() => {
    // Simulate API loading
    const timer = setTimeout(() => {
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="dashboard loading-container">
        <div className="loading-spinner"></div>
        <p>Loading dashboard data...</p>
      </div>
    );
  }

  return (
    <div className="dashboard">
      <h1 className="page-title">Anomaly Detection Dashboard</h1>
      
      <div className="dashboard-stats">
        <AnomalyStats stats={stats} />
      </div>
      
      <div className="dashboard-charts">
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Real-Time Anomaly Detection</h2>
            <div className="chart-controls">
              <select className="chart-select">
                <option>Last 24 Hours</option>
                <option>Last 7 Days</option>
                <option>Last 30 Days</option>
              </select>
            </div>
          </div>
          <div className="card-body">
            <AnomalyChart data={chartData} />
          </div>
        </div>
      </div>
      
      <div className="dashboard-grid">
        <div className="dashboard-col">
          <StreamsOverview streams={streams} />
        </div>
        <div className="dashboard-col">
          <RecentAlerts alerts={alerts} />
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
