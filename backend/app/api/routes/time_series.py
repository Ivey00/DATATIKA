import base64
from datetime import datetime
import io
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List, Optional
import json
from io import BytesIO
import seaborn as sns

from app.core.time_series import TimeSeriesForecaster
from app.models.schema import (
    DataImportRequest, DataImportResponse,
    ColumnInfoResponse,
    FeatureDefinitionRequest, FeatureDefinitionResponse,
    ItemFilterRequest, ItemFilterResponse,
    VisualizationResponse,
    FeatureImportanceResponse,
    AlgorithmSelectionRequest, AlgorithmSelectionResponse,
    HyperparameterConfigRequest, HyperparameterConfigResponse,
    PreprocessResponse,
    TrainingResponse,
    EvaluationResponse,
    PredictionRequest, PredictionResponse,
    ComparisonRequest, ComparisonResponse,
    ModelSaveResponse,
    ErrorResponse,
    convert_to_native_types
)

router = APIRouter()

# Global instance of the TimeSeriesForecaster class
# In a production environment, this would be handled differently
# with a database to store session state
time_series_forecaster = TimeSeriesForecaster()

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

@router.post("/import-data", response_model=DataImportResponse)
async def import_data(request: DataImportRequest):
    """
    Import data from a CSV or Excel file.
    """
    try:
        # Decode base64 file content
        file_content = base64.b64decode(request.file_content)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_name)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Import the data
            data = time_series_forecaster.load_data(temp_file_path)
            
            if data is None:
                return DataImportResponse(
                    success=False,
                    message="Failed to import data. Please check the file format."
                )
            
            # Prepare response
            response = DataImportResponse(
                success=True,
                message="Data imported successfully",
                data_shape=list(data.shape),
                data_sample=convert_to_native_types(data.head().to_dict(orient='records')),
                data_types={col: str(dtype) for col, dtype in data.dtypes.items()},
                missing_values={col: int(count) for col, count in data.isnull().sum().items() if count > 0}
            )
            
            # Add summary statistics if available
            try:
                stats = data.describe().to_dict()
                response.summary_statistics = convert_to_native_types(stats)
            except Exception:
                # Some columns might not support describe()
                pass
                
            return response
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        return DataImportResponse(
            success=False,
            message=f"Error importing data: {str(e)}"
        )

@router.get("/column-info", response_model=ColumnInfoResponse)
async def get_column_info():
    """
    Get information about columns in the dataset.
    """
    try:
        if time_series_forecaster.data is None:
            raise HTTPException(
                status_code=400,
                detail="No data available. Please import data first."
            )
        
        # Analyze column information
        column_info = {}
        for column in time_series_forecaster.data.columns:
            dtype = str(time_series_forecaster.data[column].dtype)
            unique_values = time_series_forecaster.data[column].nunique()
            missing_values = time_series_forecaster.data[column].isnull().sum()
            
            # Suggest column type based on data
            if pd.api.types.is_datetime64_any_dtype(time_series_forecaster.data[column]):
                suggested_type = "datetime"
            elif pd.api.types.is_numeric_dtype(time_series_forecaster.data[column]):
                suggested_type = "numerical"
            else:
                suggested_type = "categorical"
            
            column_info[column] = {
                "dtype": dtype,
                "unique_values": unique_values,
                "missing_values": int(missing_values),
                "suggested_type": suggested_type
            }
        
        return ColumnInfoResponse(
            column_info=convert_to_native_types(column_info)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting column information: {str(e)}"
        )

@router.post("/define-features", response_model=FeatureDefinitionResponse)
async def define_features(request: FeatureDefinitionRequest):
    """
    Define the features and target for the model.
    """
    try:
        # Define column types
        columns_info = time_series_forecaster.define_columns(
            datetime_col=request.datetime_column,
            target_col=request.target,
            item_id_col=request.item_id_column,
            categorical_cols=request.categorical_features,
            numerical_cols=request.numerical_features
        )
        
        if columns_info is None:
            return FeatureDefinitionResponse(
                success=False,
                message="Failed to define features and target. Please import data first.",
                features=request.features,
                target=request.target,
                categorical_features=request.categorical_features or [],
                numerical_features=request.numerical_features or [],
                datetime_column=request.datetime_column
            )
        
        return FeatureDefinitionResponse(
            success=True,
            message="Features and target defined successfully",
            features=time_series_forecaster.features,
            target=time_series_forecaster.target,
            categorical_features=time_series_forecaster.categorical_cols,
            numerical_features=time_series_forecaster.numerical_cols,
            item_id_column=time_series_forecaster.item_id_col,
            datetime_column=time_series_forecaster.datetime_col
        )
        
    except Exception as e:
        return FeatureDefinitionResponse(
            success=False,
            message=f"Error defining features and target: {str(e)}",
            features=request.features,
            target=request.target,
            categorical_features=request.categorical_features or [],
            numerical_features=request.numerical_features or [],
            datetime_column=request.datetime_column
        )

@router.post("/define-time-unit", response_model=Dict[str, Any])
async def define_time_unit(time_unit: str, forecast_horizon: int):
    """
    Define the time unit and forecast horizon for time series forecasting.
    """
    try:
        result = time_series_forecaster.define_time_unit(time_unit, forecast_horizon)
        
        if result is None:
            return {
                "success": False,
                "message": f"Failed to define time unit. Valid time units are: 'hour', 'day', 'month'.",
                "time_unit": time_unit,
                "forecast_horizon": forecast_horizon
            }
        
        return {
            "success": True,
            "message": f"Time unit defined successfully: {time_unit}, forecast horizon: {forecast_horizon}",
            "time_unit": time_series_forecaster.time_unit,
            "forecast_horizon": time_series_forecaster.forecast_horizon
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Error defining time unit: {str(e)}",
            "time_unit": time_unit,
            "forecast_horizon": forecast_horizon
        }

@router.post("/filter-by-item", response_model=ItemFilterResponse)
async def filter_by_item(request: ItemFilterRequest):
    """
    Filter the dataset to only include rows with the specified item ID.
    """
    try:
        if time_series_forecaster.data is None or time_series_forecaster.item_id_col is None:
            return ItemFilterResponse(
                success=False,
                message="Failed to filter data. Please define features and target first, and ensure the item ID column is specified.",
                item_id_column="",
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        # Create a copy of the data with the item filter
        filtered_data = time_series_forecaster.data[time_series_forecaster.data[time_series_forecaster.item_id_col] == request.item_id_value].copy()
        
        if filtered_data.empty:
            return ItemFilterResponse(
                success=False,
                message=f"No data found for {time_series_forecaster.item_id_col} = {request.item_id_value}",
                item_id_column=time_series_forecaster.item_id_col,
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        # Store specific item ID for later use in preprocessing
        time_series_forecaster.specific_item_id = request.item_id_value
        
        return ItemFilterResponse(
            success=True,
            message=f"Data filtered successfully for {time_series_forecaster.item_id_col} = {request.item_id_value}",
            item_id_column=time_series_forecaster.item_id_col,
            item_id_value=request.item_id_value,
            filtered_shape=list(filtered_data.shape)
        )
        
    except Exception as e:
        return ItemFilterResponse(
            success=False,
            message=f"Error filtering data: {str(e)}",
            item_id_column=time_series_forecaster.item_id_col or "",
            item_id_value=request.item_id_value,
            filtered_shape=[0, 0]
        )

@router.get("/visualize-data", response_model=VisualizationResponse)
async def visualize_data(visualization_type: str = "all", n_samples: int = 1000, specific_cols: Optional[List[str]] = None):
    """
    Visualize the data based on the specified type.
    """
    try:
        # Check if data and columns are defined
        if time_series_forecaster.data is None or time_series_forecaster.datetime_col is None or time_series_forecaster.target is None:
            return VisualizationResponse(
                success=False,
                message="Please import data and define features first.",
                visualizations={}
            )
        
        # Create a directory for figures if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Visualize data and capture figures
        time_series_forecaster.visualize_data(
            visualization_type=visualization_type,
            n_samples=n_samples,
            specific_cols=specific_cols
        )
        
        # Collect all generated figures
        visualizations = {}
        
        # Look for generated files in the figures directory
        for filename in os.listdir('figures'):
            if filename.endswith('.png'):
                filepath = os.path.join('figures', filename)
                with open(filepath, 'rb') as img_file:
                    img_data = base64.b64encode(img_file.read()).decode('utf-8')
                    # Use the filename (without extension) as the key
                    key = os.path.splitext(filename)[0]
                    visualizations[key] = img_data
        
        if not visualizations:
            return VisualizationResponse(
                success=False,
                message="No visualizations were generated. Please check your data and parameters.",
                visualizations={}
            )
        
        return VisualizationResponse(
            success=True,
            message=f"Generated {len(visualizations)} visualizations.",
            visualizations=visualizations
        )
        
    except Exception as e:
        return VisualizationResponse(
            success=False,
            message=f"Error visualizing data: {str(e)}",
            visualizations={}
        )

@router.get("/analyze-feature-importance", response_model=FeatureImportanceResponse)
async def analyze_feature_importance(algorithm: str = "random_forest", n_estimators: int = 100, max_depth: int = 10):
    """
    Analyze feature importance using the specified algorithm.
    """
    try:
        # Check if data and features are defined
        if time_series_forecaster.data is None or time_series_forecaster.features is None or time_series_forecaster.target is None:
            return FeatureImportanceResponse(
                success=False,
                message="Please import data and define features first.",
                importance_data=[],
                visualization=""
            )
        
        # Analyze feature importance
        importance_df = time_series_forecaster.analyze_feature_importance(
            algorithm=algorithm,
            n_estimators=n_estimators,
            max_depth=max_depth
        )
        
        if importance_df is None:
            return FeatureImportanceResponse(
                success=False,
                message="Failed to analyze feature importance.",
                importance_data=[],
                visualization=""
            )
        
        # Read the generated feature importance image
        try:
            with open('figures/feature_importance.png', 'rb') as img_file:
                visualization = base64.b64encode(img_file.read()).decode('utf-8')
        except:
            # If the file doesn't exist, create a simple visualization
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
            plt.title(f'Top 20 Feature Importance using {algorithm}')
            plt.tight_layout()
            visualization = fig_to_base64(plt.gcf())
        
        return FeatureImportanceResponse(
            success=True,
            message="Feature importance analyzed successfully.",
            importance_data=convert_to_native_types(importance_df.to_dict(orient='records')),
            visualization=visualization
        )
        
    except Exception as e:
        return FeatureImportanceResponse(
            success=False,
            message=f"Error analyzing feature importance: {str(e)}",
            importance_data=[],
            visualization=""
        )

@router.post("/select-algorithm", response_model=AlgorithmSelectionResponse)
async def select_algorithm(request: AlgorithmSelectionRequest):
    """
    Select the time series forecasting algorithm.
    """
    try:
        # Check if the algorithm is supported
        if request.algorithm_name not in time_series_forecaster.available_algorithms:
            return AlgorithmSelectionResponse(
                success=False,
                message=f"Algorithm '{request.algorithm_name}' not supported. Supported algorithms: {list(time_series_forecaster.available_algorithms.keys())}",
                algorithm_name=request.algorithm_name,
                hyperparameters={},
                algorithm_description=""
            )
        
        # Set the algorithm
        time_series_forecaster.algorithm = request.algorithm_name
        
        # Get algorithm information
        algorithm_info = time_series_forecaster.available_algorithms[request.algorithm_name]
        
        # Get algorithm description
        algorithm_description = f"{request.algorithm_name} is a time series forecasting algorithm for predicting future values."
        
        # Prepare hyperparameters with descriptions
        hyperparameters = {}
        for param, value in algorithm_info['default_params'].items():
            description = algorithm_info['description'].get(param, "No description available.")
            
            # Create suggested values based on the parameter type
            suggested_values = [value]  # Start with the default value
            
            if param == "n_estimators":
                suggested_values = [10, 50, 100, 200, 500]
            elif param == "max_depth":
                suggested_values = [None, 3, 5, 10, 15]
            elif param == "learning_rate":
                suggested_values = [0.01, 0.05, 0.1, 0.2, 0.5]
            elif param == "subsample":
                suggested_values = [0.6, 0.7, 0.8, 0.9, 1.0]
            elif param == "colsample_bytree":
                suggested_values = [0.6, 0.7, 0.8, 0.9, 1.0]
            elif param == "min_samples_split":
                suggested_values = [2, 5, 10, 20]
            elif param == "min_samples_leaf":
                suggested_values = [1, 2, 4, 8]
            elif param == "seasonality_mode":
                suggested_values = ["additive", "multiplicative"]
            elif param == "changepoint_prior_scale":
                suggested_values = [0.01, 0.05, 0.1, 0.5]
            elif isinstance(value, bool):
                suggested_values = [True, False]
            elif param.endswith("seasonality"):
                suggested_values = ["auto", True, False]
            
            hyperparameters[param] = {
                "description": description,
                "suggested_values": convert_to_native_types(suggested_values)
            }
        
        # Return the response
        return AlgorithmSelectionResponse(
            success=True,
            message=f"Algorithm '{request.algorithm_name}' selected successfully.",
            algorithm_name=request.algorithm_name,
            hyperparameters=hyperparameters,
            algorithm_description=algorithm_description
        )
        
    except Exception as e:
        return AlgorithmSelectionResponse(
            success=False,
            message=f"Error selecting algorithm: {str(e)}",
            algorithm_name=request.algorithm_name,
            hyperparameters={},
            algorithm_description=""
        )

@router.post("/configure-hyperparameters", response_model=HyperparameterConfigResponse)
async def configure_hyperparameters(request: HyperparameterConfigRequest):
    """
    Configure hyperparameters for the selected algorithm.
    """
    try:
        # Check if an algorithm is selected
        if time_series_forecaster.algorithm is None:
            return HyperparameterConfigResponse(
                success=False,
                message="No algorithm selected. Please select an algorithm first.",
                hyperparameters={}
            )
        
        # Set the hyperparameters
        time_series_forecaster.hyperparameters = request.hyperparameters
        
        return HyperparameterConfigResponse(
            success=True,
            message="Hyperparameters configured successfully.",
            hyperparameters=convert_to_native_types(request.hyperparameters)
        )
        
    except Exception as e:
        return HyperparameterConfigResponse(
            success=False,
            message=f"Error configuring hyperparameters: {str(e)}",
            hyperparameters={}
        )

@router.get("/preprocess-data", response_model=PreprocessResponse)
async def preprocess_data(test_size: float = 0.2):
    """
    Preprocess the data for time series forecasting.
    """
    try:
        # Check if data and features are defined
        if time_series_forecaster.data is None or time_series_forecaster.features is None or time_series_forecaster.target is None:
            return PreprocessResponse(
                success=False,
                message="Please import data and define features first.",
                train_shape=[0, 0],
                test_shape=[0, 0]
            )
        
        # Get specific item ID if it's set
        specific_item_id = getattr(time_series_forecaster, 'specific_item_id', None)
        
        # Preprocess the data
        result = time_series_forecaster.preprocess_data(
            test_size=test_size,
            specific_item_id=specific_item_id
        )
        
        if result is None:
            return PreprocessResponse(
                success=False,
                message="Failed to preprocess data.",
                train_shape=[0, 0],
                test_shape=[0, 0]
            )
        
        return PreprocessResponse(
            success=True,
            message="Data preprocessed successfully.",
            train_shape=[time_series_forecaster.X_train.shape[0], len(time_series_forecaster.features)],
            test_shape=[time_series_forecaster.X_test.shape[0], len(time_series_forecaster.features)]
        )
        
    except Exception as e:
        return PreprocessResponse(
            success=False,
            message=f"Error preprocessing data: {str(e)}",
            train_shape=[0, 0],
            test_shape=[0, 0]
        )

@router.get("/train-model", response_model=TrainingResponse)
async def train_model():
    """
    Train the time series forecasting model.
    """
    try:
        # Check if an algorithm is selected and data is preprocessed
        if time_series_forecaster.algorithm is None or time_series_forecaster.X_train is None:
            return TrainingResponse(
                success=False,
                message="No algorithm selected or data not preprocessed. Please select an algorithm and preprocess the data first.",
                training_time=0.0
            )
        
        # Train the model
        import time
        start_time = time.time()
        
        model = time_series_forecaster.train_model()
        
        training_time = time.time() - start_time
        
        if model is None:
            return TrainingResponse(
                success=False,
                message="Failed to train model.",
                training_time=0.0
            )
        
        return TrainingResponse(
            success=True,
            message="Model trained successfully.",
            training_time=training_time
        )
        
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Error training model: {str(e)}",
            training_time=0.0
        )

@router.get("/evaluate-model", response_model=EvaluationResponse)
async def evaluate_model():
    """
    Evaluate the trained model on the test set.
    """
    try:
        # Check if a model is trained
        if time_series_forecaster.model is None:
            return EvaluationResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                metrics={},
                classification_report={},
                confusion_matrix=[],
                visualizations={}
            )
        
        # Evaluate the model
        metrics = time_series_forecaster.evaluate_model()
        
        if metrics is None:
            return EvaluationResponse(
                success=False,
                message="Failed to evaluate model.",
                metrics={},
                classification_report={},
                confusion_matrix=[],
                visualizations={}
            )
        
        # Get visualizations from files
        visualizations = {}
        try:
            with open('figures/actual_vs_predicted.png', 'rb') as img_file:
                visualizations['actual_vs_predicted'] = base64.b64encode(img_file.read()).decode('utf-8')
        except:
            pass
        
        # Create dummy classification report and confusion matrix for compatibility
        dummy_classification_report = {
            "regression": {
                "mse": metrics['MSE'],
                "rmse": metrics['RMSE'],
                "mae": metrics['MAE'],
                "r2": metrics['R2']
            }
        }
        
        dummy_confusion_matrix = [[0]]
        
        return EvaluationResponse(
            success=True,
            message="Model evaluated successfully.",
            metrics=convert_to_native_types(metrics),
            classification_report=convert_to_native_types(dummy_classification_report),
            confusion_matrix=convert_to_native_types(dummy_confusion_matrix),
            visualizations=visualizations
        )
        
    except Exception as e:
        return EvaluationResponse(
            success=False,
            message=f"Error evaluating model: {str(e)}",
            metrics={},
            classification_report={},
            confusion_matrix=[],
            visualizations={}
        )

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Make predictions on new data.
    """
    try:
        # Check if a model is trained
        if time_series_forecaster.model is None:
            return PredictionResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                predictions=[],
                prediction_count=0
            )
        
        # Decode base64 file content
        file_content = base64.b64decode(request.file_content)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_name)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Load the prediction data
            if temp_file_path.endswith('.csv'):
                prediction_data = pd.read_csv(temp_file_path)
            elif temp_file_path.endswith(('.xlsx', '.xls')):
                prediction_data = pd.read_excel(temp_file_path)
            else:
                return PredictionResponse(
                    success=False,
                    message="Unsupported file format. Please use CSV or Excel files.",
                    predictions=[],
                    prediction_count=0
                )
            
            # Check if the required features are present
            missing_features = [f for f in time_series_forecaster.features if f not in prediction_data.columns]
            if missing_features:
                return PredictionResponse(
                    success=False,
                    message=f"Missing features in prediction data: {', '.join(missing_features)}",
                    predictions=[],
                    prediction_count=0
                )
            
            # Make predictions using future_data parameter
            result = time_series_forecaster.predict_future(future_data=prediction_data)
            
            if result is None:
                return PredictionResponse(
                    success=False,
                    message="Failed to make predictions.",
                    predictions=[],
                    prediction_count=0
                )
            
            # Convert to records
            predictions = convert_to_native_types(result.to_dict(orient='records'))
            
            return PredictionResponse(
                success=True,
                message=f"Made {len(predictions)} predictions successfully.",
                predictions=predictions,
                prediction_count=len(predictions)
            )
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"Error making predictions: {str(e)}",
            predictions=[],
            prediction_count=0
        )

@router.post("/predict-future", response_model=PredictionResponse)
async def predict_future(future_periods: int):
    """
    Predict future time periods based on the trained model.
    """
    try:
        # Check if a model is trained
        if time_series_forecaster.model is None:
            return PredictionResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                predictions=[],
                prediction_count=0
            )
        
        # Check if time unit is defined
        if time_series_forecaster.time_unit is None or time_series_forecaster.forecast_horizon is None:
            return PredictionResponse(
                success=False,
                message="Time unit not defined. Please define time unit first.",
                predictions=[],
                prediction_count=0
            )
        
        # Make predictions for future periods
        result = time_series_forecaster.predict_future(future_periods=future_periods)
        
        if result is None:
            return PredictionResponse(
                success=False,
                message="Failed to make future predictions.",
                predictions=[],
                prediction_count=0
            )
        
        # Convert to records
        predictions = convert_to_native_types(result.to_dict(orient='records'))
        
        return PredictionResponse(
            success=True,
            message=f"Made {len(predictions)} future predictions successfully.",
            predictions=predictions,
            prediction_count=len(predictions)
        )
        
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"Error making future predictions: {str(e)}",
            predictions=[],
            prediction_count=0
        )

@router.post("/compare-predictions", response_model=ComparisonResponse)
async def compare_predictions(request: ComparisonRequest):
    """
    Compare predictions with actual values.
    """
    try:
        # Check if a model is trained
        if time_series_forecaster.model is None:
            return ComparisonResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                metrics={},
                classification_report={},
                confusion_matrix=[],
                comparison_sample=[],
                class_accuracy={},
                visualizations={}
            )
        
        # Decode base64 file content
        file_content = base64.b64decode(request.file_content)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_name)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Load the comparison data
            if temp_file_path.endswith('.csv'):
                comparison_data = pd.read_csv(temp_file_path)
            elif temp_file_path.endswith(('.xlsx', '.xls')):
                comparison_data = pd.read_excel(temp_file_path)
            else:
                return ComparisonResponse(
                    success=False,
                    message="Unsupported file format. Please use CSV or Excel files.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    comparison_sample=[],
                    class_accuracy={},
                    visualizations={}
                )
            
            # Check if the required target column is present
            if time_series_forecaster.target not in comparison_data.columns:
                return ComparisonResponse(
                    success=False,
                    message=f"Missing target column '{time_series_forecaster.target}' in comparison data.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    comparison_sample=[],
                    class_accuracy={},
                    visualizations={}
                )
            
            # Check if required datetime column is present
            if time_series_forecaster.datetime_col not in comparison_data.columns:
                return ComparisonResponse(
                    success=False,
                    message=f"Missing datetime column '{time_series_forecaster.datetime_col}' in comparison data.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    comparison_sample=[],
                    class_accuracy={},
                    visualizations={}
                )
            
            # Check if required features are present
            features = [f for f in time_series_forecaster.features if f != time_series_forecaster.target]
            missing_features = [f for f in features if f not in comparison_data.columns]
            if missing_features:
                return ComparisonResponse(
                    success=False,
                    message=f"Missing features in comparison data: {', '.join(missing_features)}",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    comparison_sample=[],
                    class_accuracy={},
                    visualizations={}
                )
            
            # Make predictions on the comparison data
            result = time_series_forecaster.predict_future(future_data=comparison_data)
            
            if result is None:
                return ComparisonResponse(
                    success=False,
                    message="Failed to make predictions for comparison.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    comparison_sample=[],
                    class_accuracy={},
                    visualizations={}
                )
            
            # Combine actual and predicted values
            y_actual = comparison_data[time_series_forecaster.target]
            y_pred = result['Prediction']
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_actual, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_actual, y_pred)
            r2 = r2_score(y_actual, y_pred)
            
            metrics = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            }
            
            # Create visualizations for comparison
            visualizations = {}
            
            # Actual vs Predicted plot
            plt.figure(figsize=(12, 6))
            plt.plot(comparison_data[time_series_forecaster.datetime_col], y_actual, label='Actual', color='blue')
            plt.plot(comparison_data[time_series_forecaster.datetime_col], y_pred, label='Predicted', color='red')
            plt.title('Actual vs Predicted Values')
            plt.xlabel(time_series_forecaster.datetime_col)
            plt.ylabel(time_series_forecaster.target)
            plt.legend()
            plt.tight_layout()
            visualizations['actual_vs_predicted'] = fig_to_base64(plt.gcf())
            
            # Create comparison sample
            comparison_df = pd.DataFrame({
                'Datetime': comparison_data[time_series_forecaster.datetime_col],
                'Actual': y_actual,
                'Predicted': y_pred,
                'Error': y_actual - y_pred,
                'Abs_Error': np.abs(y_actual - y_pred)
            })
            
            comparison_sample = convert_to_native_types(comparison_df.head(20).to_dict(orient='records'))
            
            # Create dummy values for classification report, confusion matrix, and class accuracy
            dummy_classification_report = {
                "regression": {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            }
            
            dummy_confusion_matrix = [[0]]
            
            dummy_class_accuracy = {
                "regression": r2
            }
            
            return ComparisonResponse(
                success=True,
                message="Comparison completed successfully.",
                metrics=convert_to_native_types(metrics),
                classification_report=convert_to_native_types(dummy_classification_report),
                confusion_matrix=convert_to_native_types(dummy_confusion_matrix),
                comparison_sample=comparison_sample,
                class_accuracy=convert_to_native_types(dummy_class_accuracy),
                visualizations=visualizations
            )
            
        finally:
            # Clean up the temporary file
            os.unlink(temp_file_path)
            
    except Exception as e:
        return ComparisonResponse(
            success=False,
            message=f"Error comparing predictions: {str(e)}",
            metrics={},
            classification_report={},
            confusion_matrix=[],
            comparison_sample=[],
            class_accuracy={},
            visualizations={}
        )

@router.get("/save-model", response_model=ModelSaveResponse)
async def save_model(directory: str = "models"):
    """
    Save the trained model to disk.
    """
    try:
        # Check if a model is trained
        if time_series_forecaster.model is None:
            return ModelSaveResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                model_path="",
                timestamp=""
            )
        
        # Create the models directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate a filename based on algorithm and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"time_series_{time_series_forecaster.algorithm}_{timestamp}.joblib"
        filepath = os.path.join(directory, filename)
        
        # Save the model
        success = time_series_forecaster.save_model(filepath)
        
        if not success:
            return ModelSaveResponse(
                success=False,
                message="Failed to save model.",
                model_path="",
                timestamp=""
            )
        
        return ModelSaveResponse(
            success=True,
            message=f"Model saved successfully to {filepath}.",
            model_path=filepath,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return ModelSaveResponse(
            success=False,
            message=f"Error saving model: {str(e)}",
            model_path="",
            timestamp=""
        )

@router.get("/available-algorithms")
async def get_available_algorithms():
    """
    Get a list of available time series forecasting algorithms.
    """
    try:
        algorithms = list(time_series_forecaster.available_algorithms.keys())
        return {
            "success": True,
            "algorithms": algorithms
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting available algorithms: {str(e)}",
            "algorithms": []
        }