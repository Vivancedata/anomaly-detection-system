from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Union
from enum import Enum
from datetime import datetime

class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class ModelType(str, Enum):
    STATISTICAL = "statistical"
    MACHINE_LEARNING = "machine_learning"
    DEEP_LEARNING = "deep_learning"
    ENSEMBLE = "ensemble"

class DetectionConfig(BaseModel):
    """Configuration parameters for anomaly detection"""
    threshold: Optional[float] = Field(None, description="Detection threshold (0.0-1.0)")
    sensitivity: Optional[float] = Field(None, description="Detection sensitivity (0.0-1.0)")
    algorithm: Optional[str] = Field(None, description="Algorithm to use")
    window_size: Optional[int] = Field(None, description="Window size for detection")
    include_metadata: bool = Field(True, description="Include metadata in response")

class DataPoint(BaseModel):
    """A single data point with timestamp and value"""
    timestamp: Union[datetime, str] = Field(..., description="Timestamp of the data point")
    value: Union[float, Dict[str, float]] = Field(..., description="Value or dictionary of values")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Optional metadata")
    
    @validator('timestamp', pre=True)
    def parse_timestamp(cls, v):
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace('Z', '+00:00'))
        return v

class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection"""
    data: List[DataPoint] = Field(..., description="Data points to analyze")
    stream_id: Optional[str] = Field(None, description="ID of the data stream")
    config: Optional[DetectionConfig] = Field(None, description="Detection configuration")

class AnomalyResult(BaseModel):
    """Result of anomaly detection for a single data point"""
    timestamp: datetime = Field(..., description="Timestamp of the data point")
    value: Union[float, Dict[str, float]] = Field(..., description="Original value(s)")
    is_anomaly: bool = Field(..., description="Whether this point is an anomaly")
    score: float = Field(..., description="Anomaly score (0.0-1.0)")
    severity: Optional[SeverityLevel] = Field(None, description="Severity level if anomalous")
    explanation: Optional[str] = Field(None, description="Explanation of the anomaly")
    contributing_features: Optional[List[str]] = Field(None, description="Features contributing to anomaly")

class AnomalyDetectionResponse(BaseModel):
    """Response from anomaly detection"""
    results: List[AnomalyResult] = Field(..., description="Anomaly detection results")
    stream_id: Optional[str] = Field(None, description="ID of the data stream")
    model_used: str = Field(..., description="Model used for detection")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    statistics: Optional[Dict[str, Any]] = Field(None, description="Detection statistics")

class DataStream(BaseModel):
    """Information about a data stream"""
    id: str = Field(..., description="Unique identifier of the stream")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Description of the stream")
    data_type: str = Field(..., description="Type of data in this stream")
    dimensions: int = Field(..., description="Number of dimensions in the data")
    source: Optional[str] = Field(None, description="Source of the data")
    tags: List[str] = Field(default_factory=list, description="Tags for this stream")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Stream metrics")

class AnomalyAlert(BaseModel):
    """An anomaly alert"""
    id: str = Field(..., description="Unique identifier of the alert")
    stream_id: str = Field(..., description="ID of the data stream")
    timestamp: datetime = Field(..., description="When the anomaly occurred")
    detected_at: datetime = Field(..., description="When the anomaly was detected")
    severity: SeverityLevel = Field(..., description="Severity level")
    score: float = Field(..., description="Anomaly score")
    description: str = Field(..., description="Description of the anomaly")
    values: Dict[str, float] = Field(..., description="Anomalous values")
    expected_values: Optional[Dict[str, float]] = Field(None, description="Expected values")
    contributing_features: List[str] = Field(default_factory=list, description="Contributing features")
    status: str = Field("open", description="Alert status (open, acknowledged, resolved)")
    acknowledged_by: Optional[str] = Field(None, description="Who acknowledged the alert")
    acknowledged_at: Optional[datetime] = Field(None, description="When the alert was acknowledged")
    resolved_at: Optional[datetime] = Field(None, description="When the alert was resolved")
    feedback: Optional[str] = Field(None, description="User feedback on the alert")

class ModelInfo(BaseModel):
    """Information about an anomaly detection model"""
    id: str = Field(..., description="Unique identifier of the model")
    name: str = Field(..., description="Human-readable name")
    description: Optional[str] = Field(None, description="Description of the model")
    type: ModelType = Field(..., description="Type of model")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Model parameters")
    features: List[str] = Field(default_factory=list, description="Features used by the model")
    performance: Dict[str, float] = Field(default_factory=dict, description="Performance metrics")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")
    version: str = Field(..., description="Model version")
    status: str = Field("active", description="Model status")

class StatisticsResponse(BaseModel):
    """System statistics and performance metrics"""
    time_period: Dict[str, datetime] = Field(..., description="Time period of the statistics")
    total_data_points: int = Field(..., description="Total data points processed")
    total_anomalies: int = Field(..., description="Total anomalies detected")
    average_detection_time_ms: float = Field(..., description="Average detection time")
    false_positive_rate: Optional[float] = Field(None, description="Estimated false positive rate")
    alert_counts: Dict[str, int] = Field(default_factory=dict, description="Alert counts by severity")
    model_performance: Dict[str, Dict[str, float]] = Field(default_factory=dict, description="Performance by model")
    stream_statistics: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Statistics by stream")
    system_metrics: Optional[Dict[str, Any]] = Field(None, description="System performance metrics")
