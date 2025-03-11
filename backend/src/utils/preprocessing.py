from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

def normalize_data(data: np.ndarray, method: str = "z-score") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize data using various methods
    
    Args:
        data: Input data array (n_samples, n_features)
        method: Normalization method ('z-score', 'min-max', 'robust')
        
    Returns:
        Tuple of (normalized_data, normalization_params)
    """
    if method == "z-score":
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        # Avoid division by zero
        std = np.where(std < 1e-8, 1.0, std)
        normalized = (data - mean) / std
        return normalized, {"mean": mean, "std": std}
        
    elif method == "min-max":
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        # Avoid division by zero
        range_val = np.maximum(max_val - min_val, 1e-8)
        normalized = (data - min_val) / range_val
        return normalized, {"min": min_val, "max": max_val}
        
    elif method == "robust":
        median = np.median(data, axis=0)
        q1 = np.percentile(data, 25, axis=0)
        q3 = np.percentile(data, 75, axis=0)
        iqr = q3 - q1
        # Avoid division by zero
        iqr = np.where(iqr < 1e-8, 1.0, iqr)
        normalized = (data - median) / iqr
        return normalized, {"median": median, "iqr": iqr}
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def apply_normalization(data: np.ndarray, params: Dict[str, Any], method: str = "z-score") -> np.ndarray:
    """
    Apply pre-computed normalization to new data
    
    Args:
        data: Input data array (n_samples, n_features)
        params: Normalization parameters from normalize_data
        method: Normalization method ('z-score', 'min-max', 'robust')
        
    Returns:
        Normalized data
    """
    if method == "z-score":
        return (data - params["mean"]) / params["std"]
    elif method == "min-max":
        range_val = np.maximum(params["max"] - params["min"], 1e-8)
        return (data - params["min"]) / range_val
    elif method == "robust":
        return (data - params["median"]) / params["iqr"]
    else:
        raise ValueError(f"Unknown normalization method: {method}")

def create_sequences(data: np.ndarray, sequence_length: int, step: int = 1) -> np.ndarray:
    """
    Create sequences from time series data for sequence models (LSTM, etc.)
    
    Args:
        data: Input data array (n_samples, n_features)
        sequence_length: Length of each sequence
        step: Step size between sequences
        
    Returns:
        Sequences array (n_sequences, sequence_length, n_features)
    """
    n_samples, n_features = data.shape
    
    # Calculate number of sequences
    n_sequences = (n_samples - sequence_length) // step + 1
    
    if n_sequences <= 0:
        raise ValueError(f"Not enough samples ({n_samples}) for sequence length {sequence_length}")
    
    # Create sequences
    sequences = np.zeros((n_sequences, sequence_length, n_features))
    
    for i in range(n_sequences):
        start_idx = i * step
        end_idx = start_idx + sequence_length
        sequences[i] = data[start_idx:end_idx]
    
    return sequences

def impute_missing_values(data: np.ndarray, strategy: str = "mean") -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Impute missing values in data
    
    Args:
        data: Input data array (n_samples, n_features)
        strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'constant')
        
    Returns:
        Tuple of (imputed_data, imputation_params)
    """
    # Make a copy to avoid modifying the original data
    imputed = data.copy()
    
    # Find missing values (NaN)
    mask = np.isnan(data)
    
    if not np.any(mask):
        # No missing values
        return imputed, {}
    
    params = {}
    
    if strategy == "mean":
        # Compute mean ignoring NaN values
        values = np.nanmean(data, axis=0)
        params["values"] = values
        
    elif strategy == "median":
        # Compute median ignoring NaN values
        values = np.nanmedian(data, axis=0)
        params["values"] = values
        
    elif strategy == "most_frequent":
        # Compute most frequent value ignoring NaN values
        values = np.zeros(data.shape[1])
        for j in range(data.shape[1]):
            column = data[:, j]
            unique, counts = np.unique(column[~np.isnan(column)], return_counts=True)
            values[j] = unique[np.argmax(counts)]
        params["values"] = values
        
    elif strategy == "constant":
        # Use a constant value (0 by default)
        values = np.zeros(data.shape[1])
        params["values"] = values
        
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")
    
    # Impute missing values
    for j in range(data.shape[1]):
        imputed[:, j] = np.where(mask[:, j], values[j], data[:, j])
    
    return imputed, params

def resample_time_series(data: np.ndarray, timestamps: List[datetime], target_freq: str = "1H") -> Tuple[np.ndarray, List[datetime]]:
    """
    Resample time series data to a target frequency
    
    Args:
        data: Input data array (n_samples, n_features)
        timestamps: List of timestamps for each sample
        target_freq: Target frequency ('1H' for hourly, '1D' for daily, etc.)
        
    Returns:
        Tuple of (resampled_data, resampled_timestamps)
    """
    import pandas as pd
    
    # Convert to pandas DataFrame with timestamps as index
    df = pd.DataFrame(data, index=timestamps)
    
    # Resample
    resampled = df.resample(target_freq).mean()
    
    # Extract data and timestamps
    resampled_data = resampled.values
    resampled_timestamps = resampled.index.tolist()
    
    return resampled_data, resampled_timestamps

def detect_seasonality(data: np.ndarray, max_lag: int = 100) -> Dict[str, Any]:
    """
    Detect seasonality in time series data using autocorrelation
    
    Args:
        data: Input data array (n_samples,) or (n_samples, 1)
        max_lag: Maximum lag to consider
        
    Returns:
        Dict with seasonality information
    """
    if data.ndim > 1 and data.shape[1] > 1:
        # Use first feature for seasonality detection
        series = data[:, 0]
    else:
        series = data.flatten()
    
    n = len(series)
    max_lag = min(max_lag, n // 2)
    
    # Compute autocorrelation for different lags
    acf = np.zeros(max_lag + 1)
    mean = np.mean(series)
    variance = np.var(series)
    
    if variance < 1e-8:
        # No variance, no seasonality
        return {
            "has_seasonality": False,
            "period": None,
            "strength": 0.0
        }
    
    for lag in range(max_lag + 1):
        if lag == 0:
            acf[lag] = 1.0
        else:
            # Compute autocorrelation for this lag
            product = (series[:(n-lag)] - mean) * (series[lag:] - mean)
            acf[lag] = np.sum(product) / ((n - lag) * variance)
    
    # Find peaks in autocorrelation
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(acf[1:], height=0.2)
    peaks = peaks + 1  # Adjust index for the slice
    
    if len(peaks) > 0:
        # Use the first peak as the seasonal period
        period = peaks[0]
        strength = acf[period]
        
        return {
            "has_seasonality": True,
            "period": int(period),
            "strength": float(strength),
            "acf": acf.tolist()
        }
    else:
        return {
            "has_seasonality": False,
            "period": None,
            "strength": 0.0,
            "acf": acf.tolist()
        }

def detect_trend(data: np.ndarray, method: str = "linear") -> Dict[str, Any]:
    """
    Detect trend in time series data
    
    Args:
        data: Input data array (n_samples,) or (n_samples, 1)
        method: Method for trend detection ('linear', 'polynomial', 'moving_average')
        
    Returns:
        Dict with trend information
    """
    if data.ndim > 1 and data.shape[1] > 1:
        # Use first feature for trend detection
        series = data[:, 0]
    else:
        series = data.flatten()
    
    n = len(series)
    
    if method == "linear":
        # Linear regression
        x = np.arange(n)
        A = np.vstack([x, np.ones(n)]).T
        slope, intercept = np.linalg.lstsq(A, series, rcond=None)[0]
        
        # Compute trend line
        trend = slope * x + intercept
        
        # Compute detrended data
        detrended = series - trend
        
        # Compute trend strength (R^2)
        sse = np.sum((series - trend) ** 2)
        sst = np.sum((series - np.mean(series)) ** 2)
        r_squared = 1 - (sse / (sst + 1e-8))
        
        return {
            "has_trend": abs(slope) > 1e-8,
            "slope": float(slope),
            "intercept": float(intercept),
            "trend_strength": float(r_squared),
            "trend": trend.tolist(),
            "detrended": detrended.tolist()
        }
        
    elif method == "polynomial":
        # Polynomial regression (degree 2)
        x = np.arange(n)
        coeffs = np.polyfit(x, series, 2)
        
        # Compute trend curve
        trend = np.polyval(coeffs, x)
        
        # Compute detrended data
        detrended = series - trend
        
        # Compute trend strength (R^2)
        sse = np.sum((series - trend) ** 2)
        sst = np.sum((series - np.mean(series)) ** 2)
        r_squared = 1 - (sse / (sst + 1e-8))
        
        return {
            "has_trend": abs(coeffs[0]) > 1e-8 or abs(coeffs[1]) > 1e-8,
            "coefficients": coeffs.tolist(),
            "trend_strength": float(r_squared),
            "trend": trend.tolist(),
            "detrended": detrended.tolist()
        }
        
    elif method == "moving_average":
        # Moving average smoothing
        window_size = max(3, n // 10)
        
        # Apply convolution for moving average
        weights = np.ones(window_size) / window_size
        trend = np.convolve(series, weights, mode='valid')
        
        # Pad to match original size
        pad_size = n - len(trend)
        trend = np.pad(trend, (pad_size // 2, pad_size - pad_size // 2), mode='edge')
        
        # Compute detrended data
        detrended = series - trend
        
        # Compute trend strength (R^2)
        sse = np.sum((series - trend) ** 2)
        sst = np.sum((series - np.mean(series)) ** 2)
        r_squared = 1 - (sse / (sst + 1e-8))
        
        return {
            "has_trend": r_squared > 0.1,
            "window_size": window_size,
            "trend_strength": float(r_squared),
            "trend": trend.tolist(),
            "detrended": detrended.tolist()
        }
        
    else:
        raise ValueError(f"Unknown trend detection method: {method}")
