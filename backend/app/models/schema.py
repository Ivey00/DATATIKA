from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
from datetime import datetime
import json
import base64

# Custom JSON encoders for non-standard types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient='records')
        if isinstance(obj, pd.Series):
            return obj.to_dict()
        return super().default(obj)

# Helper function to convert numpy/pandas types to Python native types
def convert_to_native_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    else:
        return obj

# Base model with custom model_dump method to handle numpy/pandas types
class CustomBaseModel(BaseModel):
    def model_dump(self, **kwargs):
        # Get the standard model_dump output
        data = super().model_dump(**kwargs)
        # Convert numpy/pandas types to Python native types
        return convert_to_native_types(data)

# Request Models
class DataImportRequest(CustomBaseModel):  
    file_content: str = Field(..., description="Base64 encoded file content")
    file_name: str = Field(..., description="Original file name with extension (.csv or .xlsx)")

class FeatureDefinitionRequest(CustomBaseModel):
    features: List[str] = Field(..., description="List of column names to use as features")
    target: str = Field(..., description="Name of the target column")
    categorical_features: Optional[List[str]] = Field(default=None, description="List of categorical features")
    numerical_features: Optional[List[str]] = Field(default=None, description="List of numerical features")
    datetime_column: Optional[str] = Field(default=None, description="Name of the datetime column")
    item_id_column: Optional[str] = Field(default=None, description="Column name for item/machine ID if applicable")

class ItemFilterRequest(CustomBaseModel):
    item_id_value: Union[str, int] = Field(..., description="The value of the item ID to filter by")

class AlgorithmSelectionRequest(CustomBaseModel):
    algorithm_name: str = Field(..., description="Name of the algorithm to use")

class HyperparameterConfigRequest(CustomBaseModel):
    hyperparameters: Dict[str, Any] = Field(..., description="Dictionary of hyperparameter names and values")

class GridSearchRequest(CustomBaseModel):
    param_grid: Optional[Dict[str, List[Any]]] = Field(None, description="Dictionary with parameters names as keys and lists of parameter values")

class PredictionRequest(CustomBaseModel):
    file_content: str = Field(..., description="Base64 encoded file content")
    file_name: str = Field(..., description="Original file name with extension (.csv or .xlsx)")

class ComparisonRequest(CustomBaseModel):
    file_content: str = Field(..., description="Base64 encoded file content")
    file_name: str = Field(..., description="Original file name with extension (.csv or .xlsx)")

class ModelSaveRequest(BaseModel):
    model_name: str
    save_directory: str

# Response Models
class DataImportResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the data import was successful")
    message: str = Field(..., description="Status message")
    data_shape: Optional[List[int]] = Field(None, description="Shape of the imported data [rows, columns]")
    data_sample: Optional[List[Dict[str, Any]]] = Field(None, description="Sample of the imported data (first 5 rows)")
    data_types: Optional[Dict[str, str]] = Field(None, description="Data types of each column")
    missing_values: Optional[Dict[str, int]] = Field(None, description="Missing values count for each column")
    summary_statistics: Optional[Dict[str, Dict[str, float]]] = Field(None, description="Summary statistics for numerical columns")

class ColumnInfoResponse(CustomBaseModel):
    column_info: Dict[str, Dict[str, Any]] = Field(..., description="Information about columns in the dataset")

class FeatureDefinitionResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the feature definition was successful")
    message: str = Field(..., description="Status message")
    features: List[str] = Field(..., description="Selected features")
    target: str = Field(..., description="Selected target")
    categorical_features: List[str] = Field(..., description="Categorical features")
    numerical_features: List[str] = Field(..., description="Numerical features")
    datetime_column: Optional[str] = Field(default=None, description="Name of the datetime column")
    item_id_column: Optional[str] = Field(default=None, description="Item ID column if specified")

class ItemFilterResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the filtering was successful")
    message: str = Field(..., description="Status message")
    item_id_column: str = Field(..., description="Item ID column")
    item_id_value: Union[str, int] = Field(..., description="Item ID value used for filtering")
    filtered_shape: List[int] = Field(..., description="Shape of the filtered data [rows, columns]")

class VisualizationResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the visualization was successful")
    message: str = Field(..., description="Status message")
    visualizations: Dict[str, str] = Field(..., description="Base64 encoded visualization images")

class FeatureImportanceResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the feature importance analysis was successful")
    message: str = Field(..., description="Status message")
    importance_data: List[Dict[str, Any]] = Field(..., description="Feature importance data")
    visualization: str = Field(..., description="Base64 encoded visualization image")

class AlgorithmSelectionResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the algorithm selection was successful")
    message: str = Field(..., description="Status message")
    algorithm_name: str = Field(..., description="Selected algorithm name")
    hyperparameters: Dict[str, Dict[str, Any]] = Field(..., description="Available hyperparameters with descriptions and suggested values")
    algorithm_description: str = Field(..., description="Description of the selected algorithm")

class HyperparameterConfigResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the hyperparameter configuration was successful")
    message: str = Field(..., description="Status message")
    hyperparameters: Dict[str, Any] = Field(..., description="Configured hyperparameters")

class PreprocessResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the preprocessing was successful")
    message: str = Field(..., description="Status message")
    train_shape: List[int] = Field(..., description="Shape of the training data [rows, columns]")
    test_shape: List[int] = Field(..., description="Shape of the test data [rows, columns]")

class TrainingResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the training was successful")
    message: str = Field(..., description="Status message")
    training_time: float = Field(..., description="Training time in seconds")
    cross_validation: Optional[Dict[str, Any]] = Field(None, description="Cross-validation results if performed")

class EvaluationResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the evaluation was successful")
    message: str = Field(..., description="Status message")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    classification_report: Dict[str, Dict[str, float]] = Field(..., description="Classification report")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    visualizations: Dict[str, str] = Field(..., description="Base64 encoded visualization images")

class PredictionResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the prediction was successful")
    message: str = Field(..., description="Status message")
    predictions: List[Dict[str, Any]] = Field(..., description="Prediction results")
    prediction_count: int = Field(..., description="Number of predictions made")

class ComparisonResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the comparison was successful")
    message: str = Field(..., description="Status message")
    metrics: Dict[str, float] = Field(..., description="Evaluation metrics")
    classification_report: Dict[str, Dict[str, float]] = Field(..., description="Classification report")
    confusion_matrix: List[List[int]] = Field(..., description="Confusion matrix")
    comparison_sample: List[Dict[str, Any]] = Field(..., description="Sample of actual vs predicted values")
    class_accuracy: Dict[str, float] = Field(..., description="Accuracy by class")
    visualizations: Dict[str, str] = Field(..., description="Base64 encoded visualization images")

class GridSearchResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the grid search was successful")
    message: str = Field(..., description="Status message")
    best_params: Dict[str, Any] = Field(..., description="Best parameters found")
    best_score: float = Field(..., description="Best cross-validation score")
    search_time: float = Field(..., description="Grid search time in seconds")

class ModelSaveResponse(CustomBaseModel):
    success: bool = Field(..., description="Whether the model was saved successfully")
    message: str = Field(..., description="Status message")
    model_path: str = Field(..., description="Path to the saved model")
    timestamp: str = Field(..., description="Timestamp when the model was saved")

class ErrorResponse(CustomBaseModel):
    success: bool = Field(False, description="Operation failed")
    message: str = Field(..., description="Error message")
    error_type: Optional[str] = Field(None, description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
