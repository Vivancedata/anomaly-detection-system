import os
from typing import List, Dict, Any, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file if it exists
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API settings
    API_VERSION: str = "v1"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", 8000))
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # Frontend dev server
        "http://localhost:8080",  # Production frontend
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8080",
    ]
    
    # Database settings
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", 5432))
    DB_USER: str = os.getenv("DB_USER", "postgres")
    DB_PASS: str = os.getenv("DB_PASS", "postgres")
    DB_NAME: str = os.getenv("DB_NAME", "anomaly_detection")
    
    # Kafka settings
    KAFKA_BOOTSTRAP_SERVERS: str = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    KAFKA_INPUT_TOPIC: str = os.getenv("KAFKA_INPUT_TOPIC", "data-stream")
    KAFKA_OUTPUT_TOPIC: str = os.getenv("KAFKA_OUTPUT_TOPIC", "anomalies")
    
    # InfluxDB settings (time series database)
    INFLUXDB_URL: str = os.getenv("INFLUXDB_URL", "http://localhost:8086")
    INFLUXDB_TOKEN: str = os.getenv("INFLUXDB_TOKEN", "")
    INFLUXDB_ORG: str = os.getenv("INFLUXDB_ORG", "vivancedata")
    INFLUXDB_BUCKET: str = os.getenv("INFLUXDB_BUCKET", "anomaly_detection")
    
    # Model settings
    MODEL_DIR: str = os.getenv("MODEL_DIR", "data/models")
    MODEL_UPDATE_INTERVAL: int = int(os.getenv("MODEL_UPDATE_INTERVAL", 86400))  # 24 hours
    THRESHOLD_ADAPTIVE: bool = os.getenv("THRESHOLD_ADAPTIVE", "True").lower() in ("true", "1", "t")
    
    # Logging settings
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    class Config:
        case_sensitive = True
        env_file = ".env"

# Create settings instance
settings = Settings()
