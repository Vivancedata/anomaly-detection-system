import { StatisticsResponse, AnomalyAlert, DataStream, ModelInfo } from '../types';

// Mock statistics data for the dashboard
export const mockStats = {
  totalAnomalies: 1240,
  alertsToday: 47,
  falsePositiveRate: 0.03,
  detectionLatency: 42, // ms
  streamsMonitored: 24,
  modelsActive: 4
};

// Mock data streams
export const mockStreams: DataStream[] = [
  {
    id: 'financial-transactions',
    name: 'Financial Transactions',
    description: 'Credit card transaction data stream',
    dataType: 'financial',
    dimensions: 8,
    source: 'payment-gateway',
    tags: ['finance', 'transactions', 'fraud'],
    createdAt: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000), // 90 days ago
    updatedAt: new Date(),
    metrics: {
      avgThroughput: 250,
      peakThroughput: 1200,
      totalAnomalies: 1245
    }
  },
  {
    id: 'network-traffic',
    name: 'Network Traffic',
    description: 'Network traffic monitoring',
    dataType: 'network',
    dimensions: 12,
    source: 'firewall-logs',
    tags: ['network', 'security', 'traffic'],
    createdAt: new Date(Date.now() - 120 * 24 * 60 * 60 * 1000), // 120 days ago
    updatedAt: new Date(),
    metrics: {
      avgThroughput: 5000,
      peakThroughput: 25000,
      totalAnomalies: 532
    }
  },
  {
    id: 'manufacturing-sensors',
    name: 'Manufacturing Sensors',
    description: 'Industrial IoT sensor readings',
    dataType: 'iot',
    dimensions: 24,
    source: 'factory-floor',
    tags: ['manufacturing', 'iot', 'sensors'],
    createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000), // 60 days ago
    updatedAt: new Date(),
    metrics: {
      avgThroughput: 500,
      peakThroughput: 800,
      totalAnomalies: 89
    }
  }
];

// Mock alerts for the dashboard
export const mockAlerts: AnomalyAlert[] = [
  {
    id: 'alert-001',
    streamId: 'financial-transactions',
    streamName: 'Financial Transactions',
    timestamp: new Date(Date.now() - 15 * 60 * 1000), // 15 minutes ago
    detectedAt: new Date(Date.now() - 14 * 60 * 1000), // 14 minutes ago
    severity: 'high',
    score: 0.92,
    description: 'Unusual transaction pattern detected',
    values: { amount: 15000, frequency: 12 },
    expectedValues: { amount: 200, frequency: 1 },
    contributingFeatures: ['amount', 'frequency', 'location'],
    status: 'open'
  },
  {
    id: 'alert-002',
    streamId: 'network-traffic',
    streamName: 'Network Traffic',
    timestamp: new Date(Date.now() - 45 * 60 * 1000), // 45 minutes ago
    detectedAt: new Date(Date.now() - 44 * 60 * 1000), // 44 minutes ago
    severity: 'medium',
    score: 0.78,
    description: 'Unusual network traffic pattern',
    values: { bandwidth: 950, packetLoss: 0.15 },
    expectedValues: { bandwidth: 500, packetLoss: 0.02 },
    contributingFeatures: ['bandwidth', 'packetLoss'],
    status: 'acknowledged',
    acknowledgedBy: 'admin',
    acknowledgedAt: new Date(Date.now() - 30 * 60 * 1000) // 30 minutes ago
  },
  {
    id: 'alert-003',
    streamId: 'manufacturing-sensors',
    streamName: 'Manufacturing Sensors',
    timestamp: new Date(Date.now() - 120 * 60 * 1000), // 2 hours ago
    detectedAt: new Date(Date.now() - 119 * 60 * 1000), // 119 minutes ago
    severity: 'low',
    score: 0.65,
    description: 'Slight temperature deviation in production line',
    values: { temperature: 85, pressure: 101 },
    expectedValues: { temperature: 75, pressure: 100 },
    contributingFeatures: ['temperature'],
    status: 'resolved',
    resolvedAt: new Date(Date.now() - 90 * 60 * 1000) // 90 minutes ago
  },
  {
    id: 'alert-004',
    streamId: 'financial-transactions',
    streamName: 'Financial Transactions',
    timestamp: new Date(Date.now() - 180 * 60 * 1000), // 3 hours ago
    detectedAt: new Date(Date.now() - 179 * 60 * 1000), // 179 minutes ago
    severity: 'high',
    score: 0.89,
    description: 'Potential fraudulent transaction pattern',
    values: { amount: 8500, frequency: 7 },
    expectedValues: { amount: 300, frequency: 1 },
    contributingFeatures: ['amount', 'frequency', 'location'],
    status: 'resolved',
    resolvedAt: new Date(Date.now() - 160 * 60 * 1000) // 160 minutes ago
  },
  {
    id: 'alert-005',
    streamId: 'network-traffic',
    streamName: 'Network Traffic',
    timestamp: new Date(Date.now() - 240 * 60 * 1000), // 4 hours ago
    detectedAt: new Date(Date.now() - 239 * 60 * 1000), // 239 minutes ago
    severity: 'medium',
    score: 0.76,
    description: 'Unusual outbound traffic volume',
    values: { outboundTraffic: 800, connections: 120 },
    expectedValues: { outboundTraffic: 400, connections: 50 },
    contributingFeatures: ['outboundTraffic', 'connections'],
    status: 'resolved',
    resolvedAt: new Date(Date.now() - 210 * 60 * 1000) // 210 minutes ago
  }
];

// Mock chart data
export const mockChartData = {
  timestamps: Array.from({ length: 24 }).map((_, i) => {
    const date = new Date();
    date.setHours(date.getHours() - 23 + i);
    return date;
  }),
  normal: Array.from({ length: 24 }).map(() => Math.floor(Math.random() * 200) + 800),
  anomalies: Array.from({ length: 24 }).map(() => Math.floor(Math.random() * 10)),
  threshold: Array.from({ length: 24 }).map(() => 950)
};

// Mock models
export const mockModels: ModelInfo[] = [
  {
    id: 'statistical-model',
    name: 'Statistical Anomaly Detector',
    description: 'Uses statistical methods for anomaly detection',
    type: 'statistical',
    parameters: {
      windowSize: 100,
      threshold: 0.95
    },
    features: ['mean', 'std_dev', 'z_score'],
    performance: {
      precision: 0.92,
      recall: 0.85,
      f1Score: 0.88
    },
    createdAt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000), // 30 days ago
    updatedAt: new Date(Date.now() - 2 * 24 * 60 * 60 * 1000), // 2 days ago
    version: '1.2.0',
    status: 'active'
  },
  {
    id: 'ml-model',
    name: 'Machine Learning Anomaly Detector',
    description: 'Uses isolation forests for anomaly detection',
    type: 'machine_learning',
    parameters: {
      nEstimators: 100,
      contamination: 0.01
    },
    features: ['feature1', 'feature2', 'feature3'],
    performance: {
      precision: 0.94,
      recall: 0.91,
      f1Score: 0.92
    },
    createdAt: new Date(Date.now() - 60 * 24 * 60 * 60 * 1000), // 60 days ago
    updatedAt: new Date(Date.now() - 5 * 24 * 60 * 60 * 1000), // 5 days ago
    version: '2.0.1',
    status: 'active'
  },
  {
    id: 'dl-model',
    name: 'Deep Learning Anomaly Detector',
    description: 'Uses autoencoders for anomaly detection',
    type: 'deep_learning',
    parameters: {
      layers: [32, 16, 8, 16, 32],
      epochs: 100
    },
    features: ['feature1', 'feature2', 'feature3', 'feature4'],
    performance: {
      precision: 0.96,
      recall: 0.93,
      f1Score: 0.94
    },
    createdAt: new Date(Date.now() - 45 * 24 * 60 * 60 * 1000), // 45 days ago
    updatedAt: new Date(Date.now() - 1 * 24 * 60 * 60 * 1000), // 1 day ago
    version: '1.5.0',
    status: 'active'
  },
  {
    id: 'ensemble-model',
    name: 'Ensemble Anomaly Detector',
    description: 'Combines multiple models for better performance',
    type: 'ensemble',
    parameters: {
      models: ['statistical', 'machine_learning', 'deep_learning'],
      voting: 'weighted'
    },
    features: ['all_features'],
    performance: {
      precision: 0.97,
      recall: 0.95,
      f1Score: 0.96
    },
    createdAt: new Date(Date.now() - 20 * 24 * 60 * 60 * 1000), // 20 days ago
    updatedAt: new Date(),
    version: '1.0.0',
    status: 'active'
  }
];

// Mock system statistics
export const mockSystemStats: StatisticsResponse = {
  timePeriod: {
    start: new Date(Date.now() - 24 * 60 * 60 * 1000), // 24 hours ago
    end: new Date()
  },
  totalDataPoints: 1_250_000,
  totalAnomalies: 1_250,
  averageDetectionTimeMs: 45.2,
  falsePositiveRate: 0.03,
  alertCounts: {
    low: 850,
    medium: 350,
    high: 50
  },
  modelPerformance: {
    statistical: {
      precision: 0.92,
      recall: 0.85,
      f1Score: 0.88
    },
    machineLearning: {
      precision: 0.94,
      recall: 0.91,
      f1Score: 0.92
    },
    deepLearning: {
      precision: 0.96,
      recall: 0.93,
      f1Score: 0.94
    },
    ensemble: {
      precision: 0.97,
      recall: 0.95,
      f1Score: 0.96
    }
  },
  streamStatistics: {
    'financial-transactions': {
      dataPoints: 500_000,
      anomalies: 500,
      falsePositives: 15
    },
    'network-traffic': {
      dataPoints: 650_000,
      anomalies: 650,
      falsePositives: 20
    },
    'manufacturing-sensors': {
      dataPoints: 100_000,
      anomalies: 100,
      falsePositives: 3
    }
  },
  systemMetrics: {
    cpuUsage: 35.2,
    memoryUsage: 4.2,
    throughput: 5000,
    latencyP95Ms: 65.3
  }
};
