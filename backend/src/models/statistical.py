from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
import pickle
import os
from datetime import datetime

from models.base import BaseDetector

logger = logging.getLogger(__name__)

class StatisticalDetector(BaseDetector):
    """
    Statistical methods for anomaly detection:
    - Z-score analysis
    - Moving average decomposition
    - Exponential smoothing
    - Extreme value theory
    """
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="statistical")
        
        # Default configuration
        self.config = {
            "window_size": 100,
            "z_score_threshold": 3.0,
            "ema_alpha": 0.1,
            "use_robust": True,
            "min_samples": 30
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Initialize state
        self._reset_state()
    
    def _reset_state(self) -> None:
        """Reset the detector's state"""
        self.trained = False
        self.n_samples_seen = 0
        self.mean = None
        self.std = None
        self.min_val = None
        self.max_val = None
        self.history = []
        self.ema = None  # Exponential moving average
    
    def detect(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies using statistical methods
        
        Args:
            data: Input data array (n_samples, n_features)
            **kwargs: Additional parameters
                
        Returns:
            Dict with detection results:
                - scores: Anomaly scores (higher = more anomalous)
                - is_anomaly: Boolean mask of anomalies
                - anomaly_indices: Indices of anomalies
                - explanation: Explanation of anomalies
        """
        if not self.trained and self.n_samples_seen < self.config["min_samples"]:
            # Not enough data to reliably detect anomalies
            logger.warning(
                f"Not enough samples seen ({self.n_samples_seen} < {self.config['min_samples']}). "
                "Accumulating data for statistical baseline..."
            )
            
            # Update the stats anyway
            self._update_stats(data)
            
            # Return empty results
            n_samples = data.shape[0]
            return {
                "scores": np.zeros(n_samples),
                "is_anomaly": np.full(n_samples, False),
                "anomaly_indices": np.array([]),
                "explanation": "Insufficient data for detection"
            }
        
        # Detect anomalies using z-score method
        if self.config["use_robust"]:
            # Use median and MAD for robustness
            median = np.median(data, axis=0)
            mad = np.median(np.abs(data - median), axis=0)
            # Avoid division by zero
            mad = np.where(mad < 1e-8, 1e-8, mad)
            z_scores = np.abs((data - median) / (mad * 1.4826))  # 1.4826 to make MAD consistent with std for normal distributions
        else:
            # Use mean and std
            z_scores = np.abs((data - self.mean) / self.std)
        
        # Find points exceeding the threshold
        threshold = kwargs.get("threshold", self.config["z_score_threshold"])
        is_anomaly = np.any(z_scores > threshold, axis=1) if z_scores.ndim > 1 else z_scores > threshold
        anomaly_indices = np.where(is_anomaly)[0]
        
        # Calculate anomaly scores (normalized to 0-1)
        if z_scores.ndim > 1:
            # For multivariate data, take max z-score across features
            max_z_scores = np.max(z_scores, axis=1)
            scores = 1.0 - np.exp(-max_z_scores / threshold)
        else:
            scores = 1.0 - np.exp(-z_scores / threshold)
        
        # Cap scores at 1.0
        scores = np.minimum(scores, 1.0)
        
        # Update stats with new data (excluding anomalies)
        normal_data = data[~is_anomaly] if np.any(~is_anomaly) else None
        if normal_data is not None and len(normal_data) > 0:
            self._update_stats(normal_data)
        
        # Create explanations for anomalies
        explanations = []
        for idx in anomaly_indices:
            if z_scores.ndim > 1:
                # Find which features contributed the most
                feature_contrib = np.where(z_scores[idx] > threshold)[0]
                if len(feature_contrib) > 0:
                    feature_str = ", ".join([f"feature {i}" for i in feature_contrib])
                    explanations.append(f"Anomaly detected due to unusual values in {feature_str}")
                else:
                    explanations.append("Anomaly detected due to unusual combination of values")
            else:
                explanations.append("Anomaly detected due to unusual value")
        
        return {
            "scores": scores,
            "is_anomaly": is_anomaly,
            "anomaly_indices": anomaly_indices,
            "explanation": explanations if explanations else None
        }
    
    def _update_stats(self, data: np.ndarray) -> None:
        """
        Update statistical measures with new data
        
        Args:
            data: New data points (n_samples, n_features)
        """
        n_new = data.shape[0]
        self.n_samples_seen += n_new
        
        # Update mean and std using Welford's online algorithm
        if self.mean is None:
            self.mean = np.mean(data, axis=0)
            self.std = np.std(data, axis=0)
            # Avoid division by zero
            self.std = np.where(self.std < 1e-8, 1e-8, self.std)
            self.min_val = np.min(data, axis=0)
            self.max_val = np.max(data, axis=0)
        else:
            # Compute the new mean
            new_mean = np.mean(data, axis=0)
            delta = new_mean - self.mean
            
            # Update mean
            self.mean = self.mean + delta * (n_new / self.n_samples_seen)
            
            # Update std (approximate - a proper implementation would track the sum of squares)
            new_std = np.std(data, axis=0)
            self.std = np.sqrt(
                (self.std**2 * (self.n_samples_seen - n_new) + new_std**2 * n_new) / self.n_samples_seen
            )
            # Avoid division by zero
            self.std = np.where(self.std < 1e-8, 1e-8, self.std)
            
            # Update min/max
            self.min_val = np.minimum(self.min_val, np.min(data, axis=0))
            self.max_val = np.maximum(self.max_val, np.max(data, axis=0))
        
        # Update EMA if configured
        if self.ema is None:
            self.ema = np.mean(data, axis=0)
        else:
            for i in range(n_new):
                if data.ndim > 1:
                    self.ema = self.ema + self.config["ema_alpha"] * (data[i, :] - self.ema)
                else:
                    self.ema = self.ema + self.config["ema_alpha"] * (data[i] - self.ema)
        
        # Add to history (limited by window size)
        if isinstance(self.history, list):
            self.history.extend(data)
            if len(self.history) > self.config["window_size"]:
                self.history = self.history[-self.config["window_size"]:]
        
        # Mark as trained once we have enough data
        if self.n_samples_seen >= self.config["min_samples"]:
            self.trained = True
    
    def train(self, data: np.ndarray, **kwargs) -> None:
        """
        Train the statistical detector on historical data
        
        Args:
            data: Training data (n_samples, n_features)
            **kwargs: Additional parameters
        """
        self._reset_state()
        self._update_stats(data)
        logger.info(f"Trained statistical detector on {self.n_samples_seen} samples")
    
    def save(self, path: str) -> None:
        """
        Save the detector to disk
        
        Args:
            path: Path to save the detector
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        state = {
            "config": self.config,
            "trained": self.trained,
            "n_samples_seen": self.n_samples_seen,
            "mean": self.mean,
            "std": self.std,
            "min_val": self.min_val,
            "max_val": self.max_val,
            "ema": self.ema,
            "history": np.array(self.history) if isinstance(self.history, list) else self.history
        }
        
        with open(path, "wb") as f:
            pickle.dump(state, f)
        
        logger.info(f"Saved statistical detector to {path}")
    
    def load(self, path: str) -> None:
        """
        Load the detector from disk
        
        Args:
            path: Path to load the detector from
        """
        with open(path, "rb") as f:
            state = pickle.load(f)
        
        self.config = state["config"]
        self.trained = state["trained"]
        self.n_samples_seen = state["n_samples_seen"]
        self.mean = state["mean"]
        self.std = state["std"]
        self.min_val = state["min_val"]
        self.max_val = state["max_val"]
        self.ema = state["ema"]
        self.history = state["history"].tolist() if isinstance(state["history"], np.ndarray) else state["history"]
        
        logger.info(f"Loaded statistical detector from {path}")
