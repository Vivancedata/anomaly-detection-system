from fastapi import APIRouter, HTTPException, Depends, Query, Path
from typing import List, Dict, Any, Optional
import logging

from api.schemas import (
    AnomalyDetectionRequest, 
    AnomalyDetectionResponse,
    DataStream,
    AnomalyAlert,
    ModelInfo,
    StatisticsResponse
)
from services.detection import AnomalyDetectionService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize the anomaly detection service
detection_service = AnomalyDetectionService()

@router.post("/detect", response_model=AnomalyDetectionResponse)
async def detect_anomalies(request: AnomalyDetectionRequest):
    """
    Analyze data for anomalies
    """
    try:
        result = await detection_service.detect_anomalies(request.data, request.config)
        return result
    except Exception as e:
        logger.error(f"Anomaly detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")

@router.get("/streams", response_model=List[DataStream])
async def list_data_streams():
    """
    List all available data streams
    """
    try:
        streams = await detection_service.list_data_streams()
        return streams
    except Exception as e:
        logger.error(f"Error listing data streams: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing streams: {str(e)}")

@router.get("/streams/{stream_id}", response_model=DataStream)
async def get_data_stream(stream_id: str = Path(..., description="The ID of the data stream")):
    """
    Get details of a specific data stream
    """
    try:
        stream = await detection_service.get_data_stream(stream_id)
        if not stream:
            raise HTTPException(status_code=404, detail=f"Stream with ID {stream_id} not found")
        return stream
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving stream {stream_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving stream: {str(e)}")

@router.get("/alerts", response_model=List[AnomalyAlert])
async def list_alerts(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    severity: Optional[str] = Query(None, description="Filter by severity (low, medium, high)"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)"),
    limit: int = Query(100, description="Maximum number of alerts to return")
):
    """
    List anomaly alerts with optional filtering
    """
    try:
        alerts = await detection_service.list_alerts(
            stream_id=stream_id,
            severity=severity,
            start_time=start_time,
            end_time=end_time,
            limit=limit
        )
        return alerts
    except Exception as e:
        logger.error(f"Error listing alerts: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing alerts: {str(e)}")

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    """
    List all available anomaly detection models
    """
    try:
        models = await detection_service.list_models()
        return models
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing models: {str(e)}")

@router.get("/statistics", response_model=StatisticsResponse)
async def get_statistics(
    stream_id: Optional[str] = Query(None, description="Filter by stream ID"),
    start_time: Optional[str] = Query(None, description="Start time (ISO format)"),
    end_time: Optional[str] = Query(None, description="End time (ISO format)")
):
    """
    Get system statistics and performance metrics
    """
    try:
        stats = await detection_service.get_statistics(
            stream_id=stream_id,
            start_time=start_time,
            end_time=end_time
        )
        return stats
    except Exception as e:
        logger.error(f"Error retrieving statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving statistics: {str(e)}")
