from typing import List, Dict, Any, Optional, Union
import logging
import numpy as np
import pickle
import os
from datetime import datetime

from models.base import BaseDetector

logger = logging.getLogger(__name__)

class EnsembleDetector(BaseDetector):
    """
    Ensemble detector that combines results from multiple anomaly detection models
    to improve overall accuracy and reduce false positives.
    """
    def __init__(self, detectors: List[BaseDetector] = None, config: Optional[Dict[str, Any]] = None):
        super().__init__(name="ensemble")
        
        # Default configuration
        self.config = {
            "fusion_method": "weighted_average",  # weighted_average, majority_vote, max, min
            "weights": None,  # If None, equal weights will be used
            "threshold": 0.7,  # Threshold for the ensemble score
            "require_explanation": True,  # Whether to return explanations
            "min_detectors_agreement": 2  # Minimum number of detectors that must agree for majority vote
        }
        
        # Override defaults with provided config
        if config:
            self.config.update(config)
        
        # Store the individual detectors
        self.detectors = detectors if detectors is not None else []
        
        # Set weights if not provided
        if self.config["weights"] is None and self.detectors:
            self.config["weights"] = [1.0 / len(self.detectors)] * len(self.detectors)
    
    def add_detector(self, detector: BaseDetector, weight: float = None) -> None:
        """
        Add a detector to the ensemble
        
        Args:
            detector: The detector to add
            weight: The weight for this detector (if None, will be set to maintain equal weights)
        """
        self.detectors.append(detector)
        
        # Update weights
        n_detectors = len(self.detectors)
        if weight is None:
            # Equal weights for all detectors
            self.config["weights"] = [1.0 / n_detectors] * n_detectors
        else:
            # Normalize existing weights to sum to (1 - weight)
            if self.config["weights"] is None:
                self.config["weights"] = []
            
            if n_detectors > 1:
                sum_weights = sum(self.config["weights"][:-1])  # Sum without the new detector
                if sum_weights > 0:
                    factor = (1.0 - weight) / sum_weights
                    self.config["weights"] = [w * factor for w in self.config["weights"][:-1]]
            
            # Add the new weight
            self.config["weights"].append(weight)
    
    def detect(self, data: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Detect anomalies using the ensemble of models
        
        Args:
            data: Input data array (n_samples, n_features)
            **kwargs: Additional parameters
                
        Returns:
            Dict with detection results
        """
        if not self.detectors:
            raise ValueError("No detectors in the ensemble")
        
        # Collect results from all detectors
        detector_results = []
        for i, detector in enumerate(self.detectors):
            result = detector.detect(data, **kwargs)
            detector_results.append(result)
        
        n_samples = data.shape[0]
        
        # Combine results using the specified fusion method
        if self.config["fusion_method"] == "weighted_average":
            # Combine scores using weighted average
            combined_scores = np.zeros(n_samples)
            for i, result in enumerate(detector_results):
                weight = self.config["weights"][i]
                combined_scores += weight * result["scores"]
                
            # Determine anomalies using the ensemble threshold
            threshold = kwargs.get("threshold", self.config["threshold"])
            is_anomaly = combined_scores > threshold
            
        elif self.config["fusion_method"] == "majority_vote":
            # Count how many detectors flagged each sample as an anomaly
            anomaly_votes = np.zeros(n_samples)
            for result in detector_results:
                anomaly_votes += result["is_anomaly"].astype(int)
                
            # Determine anomalies based on minimum agreement
            min_agreement = kwargs.get("min_agreement", self.config["min_detectors_agreement"])
            is_anomaly = anomaly_votes >= min_agreement
            
            # Calculate a score based on the proportion of detectors that flagged as anomaly
            combined_scores = anomaly_votes / len(self.detectors)
            
        elif self.config["fusion_method"] == "max":
            # Take the maximum score across all detectors
            combined_scores = np.zeros(n_samples)
            for result in detector_results:
                combined_scores = np.maximum(combined_scores, result["scores"])
                
            # Determine anomalies using the ensemble threshold
            threshold = kwargs.get("threshold", self.config["threshold"])
            is_anomaly = combined_scores > threshold
            
        elif self.config["fusion_method"] == "min":
            # Take the minimum score across all detectors (conservative approach)
            combined_scores = np.ones(n_samples)
            for result in detector_results:
                combined_scores = np.minimum(combined_scores, result["scores"])
                
            # Determine anomalies using the ensemble threshold
            threshold = kwargs.get("threshold", self.config["threshold"])
            is_anomaly = combined_scores > threshold
            
        else:
            raise ValueError(f"Unknown fusion method: {self.config['fusion_method']}")
        
        # Get anomaly indices
        anomaly_indices = np.where(is_anomaly)[0]
        
        # Combine explanations from individual detectors
        explanations = []
        feature_contributions = []
        
        if self.config["require_explanation"] and anomaly_indices.size > 0:
            for idx in anomaly_indices:
                # Collect explanations from detectors that flagged this as an anomaly
                detector_explanations = []
                all_contributing_features = set()
                
                for i, result in enumerate(detector_results):
                    if (idx < len(result["is_anomaly"]) and  # Ensure index is valid
                        result["is_anomaly"][idx] and  # This detector flagged as anomaly
                        "explanation" in result and  # Explanation exists
                        result["explanation"] is not None):
                        
                        # Add detector name to explanation
                        if isinstance(result["explanation"], list):
                            if idx < len(result["explanation"]):
                                detector_explanations.append(
                                    f"{self.detectors[i].name}: {result['explanation'][idx]}"
                                )
                        else:
                            detector_explanations.append(
                                f"{self.detectors[i].name}: {result['explanation']}"
                            )
                        
                        # Collect contributing features
                        if ("feature_contributions" in result and 
                            result["feature_contributions"] is not None and
                            idx < len(result["feature_contributions"])):
                            all_contributing_features.update(result["feature_contributions"][idx])
                
                if detector_explanations:
                    combined_explanation = " | ".join(detector_explanations)
                    explanations.append(combined_explanation)
                else:
                    explanations.append(f"Anomaly detected by ensemble with score {combined_scores[idx]:.2f}")
                
                feature_contributions.append(list(all_contributing_features))
        
        return {
            "scores": combined_scores,
            "is_anomaly": is_anomaly,
            "anomaly_indices": anomaly_indices,
            "explanation": explanations if explanations else None,
            "feature_contributions": feature_contributions if feature_contributions else None,
            "detector_results": detector_results  # Include individual detector results
        }
    
    def train(self, data: np.ndarray, **kwargs) -> None:
        """
        Train all detectors in the ensemble
        
        Args:
            data: Training data (n_samples, n_features)
            **kwargs: Additional parameters
        """
        if not self.detectors:
            raise ValueError("No detectors in the ensemble")
        
        logger.info(f"Training ensemble with {len(self.detectors)} detectors")
        
        # Train each detector
        for i, detector in enumerate(self.detectors):
            logger.info(f"Training detector {i+1}/{len(self.detectors)}: {detector.name}")
            detector.train(data, **kwargs)
    
    def save(self, path: str) -> None:
        """
        Save the ensemble detector to disk
        
        Args:
            path: Path to save the detector
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save ensemble config
        ensemble_path = os.path.join(os.path.dirname(path), "ensemble_config.pkl")
        with open(ensemble_path, "wb") as f:
            pickle.dump(self.config, f)
        
        # Save individual detectors
        for i, detector in enumerate(self.detectors):
            detector_path = os.path.join(os.path.dirname(path), f"detector_{i}_{detector.name}.pkl")
            detector.save(detector_path)
        
        logger.info(f"Saved ensemble detector with {len(self.detectors)} detectors to {os.path.dirname(path)}")
    
    def load(self, path: str) -> None:
        """
        Load the ensemble detector from disk
        
        Args:
            path: Path to load the detector from
        """
        # Load ensemble config
        ensemble_path = os.path.join(os.path.dirname(path), "ensemble_config.pkl")
        with open(ensemble_path, "rb") as f:
            self.config = pickle.load(f)
        
        # Load individual detectors
        detector_files = [f for f in os.listdir(os.path.dirname(path)) if f.startswith("detector_")]
        
        # Sort by index to maintain order
        detector_files.sort(key=lambda x: int(x.split("_")[1]))
        
        self.detectors = []
        
        for detector_file in detector_files:
            # Parse detector type from filename
            detector_type = detector_file.split("_")[2].split(".")[0]
            
            # Create appropriate detector
            if detector_type == "statistical":
                from models.statistical import StatisticalDetector
                detector = StatisticalDetector()
            elif detector_type == "machine_learning":
                from models.machine_learning import MLDetector
                detector = MLDetector()
            elif detector_type == "deep_learning":
                from models.deep_learning import DLDetector
                detector = DLDetector()
            else:
                logger.warning(f"Unknown detector type: {detector_type}, skipping")
                continue
            
            # Load detector
            detector_path = os.path.join(os.path.dirname(path), detector_file)
            detector.load(detector_path)
            
            # Add to ensemble
            self.detectors.append(detector)
        
        logger.info(f"Loaded ensemble detector with {len(self.detectors)} detectors from {os.path.dirname(path)}")
