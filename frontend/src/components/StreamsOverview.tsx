import React from 'react';
import { DataStream } from '../types';
import './StreamsOverview.css';

interface StreamsOverviewProps {
  streams: DataStream[];
}

const StreamsOverview: React.FC<StreamsOverviewProps> = ({ streams }) => {
  return (
    <div className="streams-overview card">
      <div className="card-header">
        <h2 className="card-title">Data Streams</h2>
        <button className="view-all-button">View All</button>
      </div>
      <div className="card-body">
        <table className="streams-table">
          <thead>
            <tr>
              <th>Stream</th>
              <th>Throughput</th>
              <th>Anomalies</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {streams.map(stream => (
              <tr key={stream.id}>
                <td className="stream-name">
                  <div className="stream-name-container">
                    <div className="stream-icon">{stream.name.charAt(0)}</div>
                    <div className="stream-info">
                      <div className="stream-title">{stream.name}</div>
                      <div className="stream-type">{stream.dataType}</div>
                    </div>
                  </div>
                </td>
                <td>
                  {stream.metrics?.avgThroughput ? 
                    `${stream.metrics.avgThroughput.toLocaleString()}/s` : 
                    '-'}
                </td>
                <td>{stream.metrics?.totalAnomalies?.toLocaleString() || '0'}</td>
                <td>
                  <span className="status-badge active">Active</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default StreamsOverview;
