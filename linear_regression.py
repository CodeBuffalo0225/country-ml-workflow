"""
Linear Regression Model for Energy Consumption Forecasting

This module implements a linear regression model for energy consumption forecasting.

Author: Travis Miragliotta
Date: May 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.models.base_model import BaseModel
from src.logging.logger import get_logger

# Set up logger
logger = get_logger(__name__)

class LinearRegressionModel(BaseModel):
    """
    Linear Regression Model for energy consumption forecasting.
    """
    
    def __init__(self, params: Dict = None):
        """
        Initialize the Linear Regression model.
        
        Args:
            params: Model parameters
        """
        super().__init__(name="Linear Regression", params=params or {})
        self.scaler = None
        self.numeric_features = []
        self.categorical_features = []
    
    def build(self) -> None:
        """
        Build the Linear Regression model.
        """
        try:
            logger.info("Building Linear Regression model")
            
            # Set up model parameters
            model_params = {
                'fit_intercept': self.params.get('fit_intercept', True),
                'normalize': self.params.get('normalize', False),
                'copy_X': self.params.get('copy_X', True),
                'n_jobs': self.params.get('n_jobs', None),
                'positive': self.params.get('positive', False)
            }
            
            # Create the model
            self.model = LinearRegression(**model_params)
            
            logger.info(f"Linear Regression model built with parameters: {model_params}")
            
        except Exception as e:
            logger.error(f"Error building Linear Regression model: {str(e)}")
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
        
        # Drop unnecessary columns
        cols_to_keep = self.numeric_features + self.categorical_features
        X_processed = X_processed[cols_to_keep]
        
        # Handle categorical features
        for col in self.categorical_features:
            if col in X_processed.columns:
                # Convert to one-hot encoding
                X_processed = pd.get_dummies(X_processed, columns=[col], drop_first=True)
        
        # Scale numeric features if not already scaled
        if self.scaler is None and self.numeric_features:
            self.scaler = StandardScaler()
            numeric_data = X_processed[self.numeric_features]
            self.scaler.fit(numeric_data)
        
        if self.scaler is not None and self.numeric_features:
            # Scale only numeric features
            numeric_data = X_processed[self.numeric_features]
            X_processed[self.numeric_features] = self.scaler.transform(numeric_data)
        
        return X_processed
    
    def _train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the Linear Regression model.
        
        Args:
            X: Preprocessed feature data
            y: Target data
        """
        try:
            logger.info(f"Training Linear Regression model on {X.shape[0]} samples with {X.shape[1]} features")
            
            # Fit the model
            self.model.fit(X, y)
            
            logger.info("Linear Regression model training completed")
            
        except Exception as e:
            logger.error(f"Error training Linear Regression model: {str(e)}")
            raise
    
    def _predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the Linear Regression model.
        
        Args:
            X: Preprocessed feature data
            
        Returns:
            Array of predictions
        """
        try:
            logger.info(f"Making predictions with Linear Regression model on {X.shape[0]} samples")
            
            # Predict using the model
            predictions = self.model.predict(X)
            
            logger.info("Predictions completed")
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions with Linear Regression model: {str(e)}")
            raise
    
    def plot_feature_importance(self, top_n: int = 10) -> plt.Figure:
        """
        Plot feature importance (coefficients) for the Linear Regression model.
        
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
            
            # If we used categorical features with one-hot encoding, the feature names might have changed
            if hasattr(self.model, 'feature_names_in_'):
                feature_names = self.model.feature_names_in_
            
            # Get coefficients
            coefficients = self.model.coef_
            
            # Create a dataframe of features and coefficients
            coef_df = pd.DataFrame({
                'Feature': feature_names,
                'Coefficient': coefficients
            })
            
            # Sort by absolute coefficient value
            coef_df['Abs_Coefficient'] = coef_df['Coefficient'].abs()
            coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
            
            # Take top N features
            coef_df = coef_df.head(top_n)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot horizontal bar chart
            ax.barh(coef_df['Feature'], coef_df['Coefficient'], color='skyblue')
            
            # Add labels and title
            ax.set_xlabel('Coefficient Value')
            ax.set_ylabel('Feature')
            ax.set_title(f'Top {top_n} Feature Coefficients - Linear Regression')
            
            # Add gridlines
            ax.grid(axis='x', linestyle='--', alpha=0.7)
            
            # Add intercept value as text
            intercept_text = f"Intercept: {self.model.intercept_:.4f}"
            ax.text(0.02, 0.95, intercept_text, transform=ax.transAxes, 
                   bbox={'boxstyle': 'round', 'alpha': 0.5})
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
