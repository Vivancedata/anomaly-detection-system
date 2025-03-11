import React, { useState, useEffect } from 'react';
import { DataStream } from '../types';
import { mockStreams } from '../utils/mockData';
import './Streams.css';

const Streams: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [streams, setStreams] = useState<DataStream[]>([]);

  useEffect(() => {
    // Simulate API loading
    const timer = setTimeout(() => {
      setStreams(mockStreams);
      setLoading(false);
    }, 1000);

    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return (
      <div className="streams-page loading-container">
        <div className="loading-spinner"></div>
        <p>Loading streams data...</p>
      </div>
    );
  }

  return (
    <div className="streams-page">
      <h1 className="page-title">Data Streams</h1>
      
      <div className="toolbar">
        <div className="search-box">
          <input type="text" placeholder="Search streams..." />
        </div>
        <button className="add-button">Add Stream</button>
      </div>
      
      <div className="streams-grid">
        {streams.map(stream => (
          <div key={stream.id} className="stream-card">
            <div className="stream-header">
              <div className="stream-icon">{stream.name.charAt(0)}</div>
              <div className="stream-name">{stream.name}</div>
              <div className="stream-type">{stream.dataType}</div>
            </div>
            <div className="stream-body">
              <p className="stream-description">{stream.description}</p>
              <div className="stream-stats">
                <div className="stat">
                  <div className="stat-label">Dimensions</div>
                  <div className="stat-value">{stream.dimensions}</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Throughput</div>
                  <div className="stat-value">{stream.metrics?.avgThroughput}/s</div>
                </div>
                <div className="stat">
                  <div className="stat-label">Anomalies</div>
                  <div className="stat-value">{stream.metrics?.totalAnomalies}</div>
                </div>
              </div>
            </div>
            <div className="stream-footer">
              <div className="tags">
                {stream.tags.map(tag => (
                  <span key={tag} className="tag">{tag}</span>
                ))}
              </div>
              <div className="stream-actions">
                <button className="action-btn">Configure</button>
                <button className="action-btn">View</button>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Streams;
