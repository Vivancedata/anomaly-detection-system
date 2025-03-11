// Data types for the anomaly detection system

// Severity level for anomalies
export type SeverityLevel = 'low' | 'medium' | 'high';

// Status of an anomaly alert
export type AlertStatus = 'open' | 'acknowledged' | 'resolved';

// Types of anomaly detection models
export type ModelType = 'statistical' | 'machine_learning' | 'deep_learning' | 'ensemble';

// Data stream information
export interface DataStream {
  id: string;
  name: string;
  description?: string;
  dataType: string;
  dimensions: number;
  source?: string;
  tags: string[];
  createdAt: Date;
  updatedAt: Date;
  metrics?: {
    avgThroughput?: number;
    peakThroughput?: number;
    totalAnomalies?: number;
    [key: string]: any;
  };
}

// A single data point with timestamp and value
export interface DataPoint {
  timestamp: Date | string;
  value: number | Record<string, number>;
  metadata?: Record<string, any>;
}

// Configuration for anomaly detection
export interface DetectionConfig {
  threshold?: number; // Detection threshold (0.0-1.0)
  sensitivity?: number; // Detection sensitivity (0.0-1.0)
  algorithm?: string; // Algorithm to use
  windowSize?: number; // Window size for detection
  includeMetadata?: boolean; // Include metadata in response
}

// Result of anomaly detection for a single data point
export interface AnomalyResult {
  timestamp: Date;
  value: number | Record<string, number>;
  isAnomaly: boolean;
  score: number; // Anomaly score (0.0-1.0)
  severity?: SeverityLevel;
  explanation?: string;
  contributingFeatures?: string[];
}

// An anomaly alert generated from detection results
export interface AnomalyAlert {
  id: string;
  streamId: string;
  streamName: string;
  timestamp: Date;
  detectedAt: Date;
  severity: SeverityLevel;
  score: number;
  description: string;
  values: Record<string, number>;
  expectedValues?: Record<string, number>;
  contributingFeatures: string[];
  status: AlertStatus;
  acknowledgedBy?: string;
  acknowledgedAt?: Date;
  resolvedAt?: Date;
  feedback?: string;
}

// Information about an anomaly detection model
export interface ModelInfo {
  id: string;
  name: string;
  description?: string;
  type: ModelType;
  parameters: Record<string, any>;
  features: string[];
  performance: {
    precision: number;
    recall: number;
    f1Score: number;
    [key: string]: number;
  };
  createdAt: Date;
  updatedAt: Date;
  version: string;
  status: string;
}

// System statistics and performance metrics
export interface StatisticsResponse {
  timePeriod: {
    start: Date;
    end: Date;
  };
  totalDataPoints: number;
  totalAnomalies: number;
  averageDetectionTimeMs: number;
  falsePositiveRate?: number;
  alertCounts: Record<string, number>;
  modelPerformance: Record<string, Record<string, number>>;
  streamStatistics: Record<string, Record<string, number>>;
  systemMetrics?: Record<string, number>;
}

// Chart data structure
export interface ChartData {
  timestamps: Date[];
  normal: number[];
  anomalies: number[];
  threshold: number[];
}

// Dashboard stats summary
export interface DashboardStats {
  totalAnomalies: number;
  alertsToday: number;
  falsePositiveRate: number;
  detectionLatency: number;
  streamsMonitored: number;
  modelsActive: number;
}
