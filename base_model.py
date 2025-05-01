"""
Base Model Module for Energy Consumption Forecasting

This module defines the base model class for energy consumption forecasting.
All specific model implementations should inherit from this base class.

Author: Travis Miragliotta
Date: May 2025
"""

import os
import pickle
import time
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from src.logging.logger import get_logger

# Set up logger
logger = get_logger(__name__)

class BaseModel(ABC):
    """
    Base class for all energy consumption forecast models.
    
    This abstract class defines the common interface and functionality
    for all models in the project.
    """
    
    def __init__(self, name: str, params: Dict = None):
        """
        Initialize the base model.
        
        Args:
            name: Model name
            params: Model parameters
        """
        self.name = name
        self.params = params or {}
        self.model = None
        self.feature_columns = []
        self.target_column = ""
        self.model_path = ""
        self.training_time = 0
        self.trained = False
        self.metrics = {}
    
    @abstractmethod
    def build(self) -> None:
        """
        Build the model with the specified parameters.
        
        This method should be implemented by all child classes to
        create and configure the actual model.
        """
        pass
    
    def _check_data(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> None:
        """
        Check if the data is valid.
        
        Args:
            X: Feature data
            y: Target data (optional)
            
        Raises:
            ValueError: If the data is invalid
        """
        if X is None or X.empty:
            raise ValueError("Feature data is empty")
        
        if y is not None and len(y) != len(X):
            raise ValueError(f"Feature and target data have different lengths: {len(X)} vs {len(y)}")
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before training or prediction.
        
        Args:
            X: Feature data
            
        Returns:
            Preprocessed feature data
        """
        # Store feature columns if not already stored
        if not self.feature_columns:
            self.feature_columns = list(X.columns)
        
        # Basic preprocessing for the base class
        # Child classes can override this method to implement model-specific preprocessing
        
        # Handle missing values
        X_processed = X.copy()
        for col in X_processed.columns:
            if X_processed[col].dtype in [np.float64, np.int64]:
                X_processed[col] = X_processed[col].fillna(X_processed[col].median())
            else:
                X_processed[col] = X_processed[col].fillna(X_processed[col].mode()[0])
        
        # Ensure all features needed by the model are present
        if self.feature_columns:
            missing_cols = set(self.feature_columns) - set(X_processed.columns)
            if missing_cols:
                raise ValueError(f"Missing required feature columns: {missing_cols}")
        
        return X_processed
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Train the model on the given data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of training metrics
        """
        self._check_data(X, y)
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Store target column name
        if isinstance(y, pd.Series):
            self.target_column = y.name
        
        # Build the model if not already built
        if self.model is None:
            self.build()
        
        # Measure training time
        start_time = time.time()
        
        # Train the model (implementation specific to each child class)
        self._train(X_processed, y)
        
        # Record training time
        self.training_time = time.time() - start_time
        
        # Mark as trained
        self.trained = True
        
        # Calculate metrics
        y_pred = self.predict(X)
        self.metrics = self._calculate_metrics(y, y_pred)
        
        # Log training results
        logger.info(f"Model {self.name} trained in {self.training_time:.2f} seconds")
        logger.info(f"Training metrics: {self.metrics}")
        
        return self.metrics
    
    @abstractmethod
    def _train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the model implementation.
        
        This method should be implemented by all child classes to
        train the specific model implementation.
        
        Args:
            X: Preprocessed feature data
            y: Target data
        """
        pass
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Feature data
            
        Returns:
            Array of predictions
            
        Raises:
            ValueError: If the model is not trained
        """
        if not self.trained or self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        
        self._check_data(X)
        
        # Preprocess features
        X_processed = self.preprocess_features(X)
        
        # Make predictions (implementation specific to each child class)
        return self._predict(X_processed)
    
    @abstractmethod
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the specific model implementation.
        
        This method should be implemented by all child classes to
        make predictions with the specific model implementation.
        
        Args:
            X: Preprocessed feature data
            
        Returns:
            Array of predictions
        """
        pass
    
    def predict_country(self, X: pd.DataFrame, country: str) -> np.ndarray:
        """
        Make predictions for a specific country.
        
        Args:
            X: Feature data
            country: Country name
            
        Returns:
            Array of predictions for the specified country
            
        Raises:
            ValueError: If the country is not in the data
        """
        if 'country' not in X.columns:
            raise ValueError("Feature data does not contain 'country' column")
        
        country_data = X[X['country'] == country]
        
        if country_data.empty:
            raise ValueError(f"No data found for country: {country}")
        
        return self.predict(country_data)
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary of metrics
        """
        return {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
    
    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Feature data
            y: Target data
            
        Returns:
            Dictionary of evaluation metrics
            
        Raises:
            ValueError: If the model is not trained
        """
        if not self.trained or self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        
        self._check_data(X, y)
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = self._calculate_metrics(y, y_pred)
        
        # Log evaluation results
        logger.info(f"Model {self.name} evaluation metrics: {metrics}")
        
        return metrics
    
    def save(self, directory: str = "models") -> str:
        """
        Save the trained model to disk.
        
        Args:
            directory: Directory to save the model in
            
        Returns:
            Path to the saved model
            
        Raises:
            ValueError: If the model is not trained
        """
        if not self.trained or self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        
        # Create directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Create a filename
        filename = f"{self.name.lower().replace(' ', '_')}_{int(time.time())}.pkl"
        self.model_path = os.path.join(directory, filename)
        
        # Save the model
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'name': self.name,
                    'params': self.params,
                    'feature_columns': self.feature_columns,
                    'target_column': self.target_column,
                    'metrics': self.metrics,
                    'training_time': self.training_time
                }, f)
            
            logger.info(f"Model saved to {self.model_path}")
            return self.model_path
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    @classmethod
    def load(cls, model_path: str) -> 'BaseModel':
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            Loaded model instance
            
        Raises:
            FileNotFoundError: If the model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Create a new instance of the class
            instance = cls(name=model_data['name'], params=model_data['params'])
            
            # Restore model state
            instance.model = model_data['model']
            instance.feature_columns = model_data.get('feature_columns', [])
            instance.target_column = model_data.get('target_column', "")
            instance.metrics = model_data.get('metrics', {})
            instance.training_time = model_data.get('training_time', 0)
            instance.model_path = model_path
            instance.trained = True
            
            logger.info(f"Model loaded from {model_path}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance if available.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
            
        Raises:
            ValueError: If feature importance is not available
        """
        # This method can be implemented by child classes that support feature importance
        raise NotImplementedError("Feature importance not available for this model type")
    
    def plot_predictions(self, X: pd.DataFrame, y: pd.Series) -> plt.Figure:
        """
        Plot predictions vs actual values.
        
        Args:
            X: Feature data
            y: Actual target values
            
        Returns:
            Matplotlib figure
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot actual vs predicted
        ax.scatter(y, y_pred, alpha=0.5)
        
        # Add reference line
        min_val = min(y.min(), y_pred.min())
        max_val = max(y.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        # Add labels and title
        ax.set_xlabel('Actual Energy Consumption')
        ax.set_ylabel('Predicted Energy Consumption')
        ax.set_title(f'{self.name} - Actual vs Predicted')
        
        # Add metrics as text
        if self.metrics:
            metrics_text = (
                f"RMSE: {self.metrics['rmse']:.4f}\n"
                f"MAE: {self.metrics['mae']:.4f}\n"
                f"R²: {self.metrics['r2']:.4f}"
            )
            ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
                    verticalalignment='top', bbox={'boxstyle': 'round', 'alpha': 0.5})
        
        plt.tight_layout()
        return fig
    
    def plot_residuals(self, X: pd.DataFrame, y: pd.Series) -> plt.Figure:
        """
        Plot residuals.
        
        Args:
            X: Feature data
            y: Actual target values
            
        Returns:
            Matplotlib figure
        """
        # Make predictions
        y_pred = self.predict(X)
        residuals = y - y_pred
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot residuals
        ax.scatter(y_pred, residuals, alpha=0.5)
        ax.axhline(y=0, color='r', linestyle='--')
        
        # Add labels and title
        ax.set_xlabel('Predicted Energy Consumption')
        ax.set_ylabel('Residuals')
        ax.set_title(f'{self.name} - Residuals Plot')
        
        plt.tight_layout()
        return fig
