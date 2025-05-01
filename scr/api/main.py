"""
FastAPI Application for Energy Consumption Forecasting

This module implements a REST API using FastAPI for energy consumption forecasting.

Author: Travis Miragliotta
Date: May 2025
"""

import os
import sys
import json
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Depends, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.linear_regression import LinearRegressionModel
from src.models.random_forest import RandomForestModel
from src.models.gradient_boosting import GradientBoostingModel
from src.logging.logger import get_logger, log_prediction
from src.monitoring.performance import log_api_performance, get_model_performance_metrics
from src.data_ingestion import load_data

# Set up logger
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Global Energy Demand Forecasting API",
    description="API for predicting energy consumption across different countries",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models")

# Available models with their file paths (these will be populated at startup)
available_models = {}

# Pydantic models for API requests and responses
class PredictionInput(BaseModel):
    """Input data structure for energy consumption prediction."""
    country: str = Field(..., description="Country name")
    year: int = Field(..., description="Year for prediction")
    population: Optional[float] = Field(None, description="Population")
    gdp: Optional[float] = Field(None, description="Gross Domestic Product")
    
    class Config:
        schema_extra = {
            "example": {
                "country": "United States",
                "year": 2025,
                "population": 332915073,
                "gdp": 25460000000000
            }
        }

class PredictionResponse(
