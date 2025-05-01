"""
Gradient Boosting Model for Energy Consumption Forecasting

This module implements a gradient boosting regression model for energy consumption forecasting.

Author: Travis Miragliotta
Date: May 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder

from src.models.base_model import BaseModel
from src.logging.logger import get_logger

# Set up logger
logger = get_logger(__name__)

class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting Model for energy consumption forecasting.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize the Gradient Boosting model.
        
        Args:
            params: Model parameters
        """
        default_params = {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'subsample': 1.0,
            'max_features': None,
            'random_state': 42
        }
        
        # Update default parameters with provided parameters
        if params:
            default_params.update(params)
        
        super().__init__(name="Gradient Boosting", params=default_params)
        self.encoder = None
        self.numeric_features = []
        self.categorical_features = []
    
    def build(self) -> None:
        """
        Build the Gradient Boosting model.
        """
        try:
            logger.info("Building Gradient Boosting model")
            
            # Create the model with parameters
            self.model = GradientBoostingRegressor(**self.params)
            
            logger.info(f"Gradient Boosting model built with parameters: {self.params}")
            
        except Exception as e:
            logger.error(f"Error building Gradient Boosting model: {str(e)}")
            raise
    
    def preprocess_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features before training or prediction.
        
        Args:
            X: Feature data
            
        Returns:
            Preprocessed feature data
        """
        # Start with basic preprocessing from the parent class
        X_processed = super().preprocess_features(X)
        
        # Identify numeric and categorical features if not already done
        if not self.numeric_features and not self.categorical_features:
            for col in X_processed.columns:
                if col == 'country' or col == 'decade':
                    self.categorical_features.append(col)
                elif X_processed[col].dtype in [np.float64, np.int64]:
                    self.numeric_features.append(col)
        
        # Handle categorical features with one-hot encoding
        if self.categorical_features:
            categorical_data = X_processed[self.categorical_features]
            
            # Initialize and fit encoder if not already done
            if self.encoder is None:
                self.encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
                self.encoder.fit(categorical_data)
            
            # Transform categorical data
            encoded_data = self.encoder.transform(categorical_data)
            
            # Get feature names after encoding
            encoded_feature_names = self.encoder.get_feature_names_out(self.categorical_features)
            
            # Create DataFrame with encoded data
            encoded_df = pd.DataFrame(
                encoded_data, 
                columns=encoded_feature_names, 
                index=X_processed.index
            )
            
            # Combine with numeric features
            X_processed = pd.concat([
                X_processed[self.numeric_features].reset_index(drop=True),
                encoded_df.reset_index(drop=True)
            ], axis=1)
        
        return X_processed
    
    def _train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Gradient Boosting model.
        
        Args:
            X: Preprocessed feature data
            y: Target data
        """
        try:
            logger.info(f"Training Gradient Boosting model on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Fit the model
            self.model.fit(X, y)
            
            logger.info("Gradient Boosting model training completed")
            
        except Exception as e:
            logger.error(f"Error training Gradient Boosting model: {str(e)}")
            raise
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the Gradient Boosting model.
        
        Args:
            X: Preprocessed feature data
            
        Returns:
            Array of predictions
        """
        try:
            logger.info(f"Making predictions with Gradient Boosting model on {X.shape[0]} samples")
            
            # Predict using the model
            predictions = self.model.predict(X)
            
            logger.info("Predictions completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with Gradient Boosting model: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance for the Gradient Boosting model.
        
        Args:
            top_n: Number of top features to show
            
        Returns:
            Matplotlib figure
            
        Raises:
            ValueError: If the model is not trained
        """
        if not self.trained or self.model is None:
            raise ValueError("Model is not trained. Call fit() first.")
        
        try:
            # Get feature names
            feature_names = self.feature_columns
            
            # Get feature importances
            importances = self.model.feature_importances_
            
            # Create a dataframe of features and importances
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
            
            # Sort by importance value
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
            # Take top N features
            importance_df = importance_df.head(top_n)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot horizontal bar chart
            ax.barh(importance_df['Feature'], importance_df['Importance'], color='goldenrod')
            
            # Add labels and title
            ax.set_xlabel('Importance')
            ax.set_ylabel('Feature')
            ax.set_title(f'Top {top_n} Feature Importances - Gradient Boosting')
            
            # Add gridlines
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            raise
