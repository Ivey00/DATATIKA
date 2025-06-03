import base64
from datetime import datetime
import io
import os
import tempfile
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import Dict, Any, List, Optional
import json
from io import BytesIO
import seaborn as sns
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from app.core.time_series import TimeSeriesForecaster
from app.models.schema import (
    DataImportRequest, DataImportResponse,
    ColumnInfoResponse,
    TimeSeriesFeatureDefinitionRequest, TimeSeriesFeatureDefinitionResponse,
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

# Progress tracker for long-running tasks
class ProgressTracker:
    def __init__(self):
        self.progress = 0
        self.message = ""
        self.error = None
        self.is_running = False
        self.start_time = None

    def update(self, progress: int, message: str = ""):
        self.progress = progress
        self.message = message

    def set_error(self, error_msg: str):
        self.error = error_msg
        self.is_running = False

    def start_task(self):
        self.progress = 0
        self.message = "Starting task..."
        self.error = None
        self.is_running = True
        self.start_time = time.time()

    def reset(self):
        self.__init__()

# Global progress tracker
progress_tracker = ProgressTracker()

# Create a thread pool for background tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

async def run_in_threadpool(func, *args, **kwargs):
    """Run a CPU-bound function in a thread pool."""
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool, 
        partial(func, *args, **kwargs)
    )

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

@router.post("/define-features", response_model=TimeSeriesFeatureDefinitionResponse)
async def define_features(request: TimeSeriesFeatureDefinitionRequest):
    """
    Define the features for time series forecasting.
    """
    try:
        # Define column types
        columns_info = time_series_forecaster.define_columns(
            datetime_col=request.datetime_column,
            target_col=request.target,
            item_id_col=request.item_id_column,
            categorical_cols=request.categorical_features,
            numerical_cols=request.numerical_features,
            additional_features=request.additional_features
        )
        
        if columns_info is None:
            return TimeSeriesFeatureDefinitionResponse(
                success=False,
                message="Failed to define features. Please import data first.",
                datetime_column=request.datetime_column,
                target=request.target,
                features=[],
                additional_features=request.additional_features,
                categorical_features=request.categorical_features,
                numerical_features=request.numerical_features,
                auto_generated_features=[]
            )
        
        return TimeSeriesFeatureDefinitionResponse(
            success=True,
            message="Features defined successfully",
            datetime_column=time_series_forecaster.datetime_col,
            target=time_series_forecaster.target,
            features=time_series_forecaster.features,
            additional_features=request.additional_features,
            categorical_features=time_series_forecaster.categorical_cols,
            numerical_features=time_series_forecaster.numerical_cols,
            item_id_column=time_series_forecaster.item_id_col,
            auto_generated_features=time_series_forecaster.auto_generated_features
        )
        
    except Exception as e:
        return TimeSeriesFeatureDefinitionResponse(
            success=False,
            message=f"Error defining features: {str(e)}",
            datetime_column=request.datetime_column,
            target=request.target,
            features=[],
            additional_features=request.additional_features,
            categorical_features=request.categorical_features,
            numerical_features=request.numerical_features,
            auto_generated_features=[]
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

@router.get("/check-progress")
async def check_progress():
    """Check the progress of the current task."""
    return {
        "progress": progress_tracker.progress,
        "message": progress_tracker.message,
        "error": progress_tracker.error,
        "is_running": progress_tracker.is_running
    }

@router.get("/train-model", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
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
        
        # Check if training is already in progress
        if progress_tracker.is_running:
            return TrainingResponse(
                success=False,
                message="Training is already in progress. Please wait for it to complete.",
                training_time=0.0
            )

        # Reset progress tracker
        progress_tracker.reset()
        progress_tracker.start_task()

        async def train_with_timeout():
            timeout = 30 * 60  # 30 minutes timeout

            async def training_task():
                try:
                    # Train the model in a separate thread to avoid blocking
                    progress_tracker.update(20, "Preprocessing data...")
                    await asyncio.sleep(0.1)

                    # Run the CPU-intensive training in a thread pool
                    progress_tracker.update(40, "Training model...")
                    model = await run_in_threadpool(time_series_forecaster.train_model)
                    
                    if model is None:
                        progress_tracker.set_error("Failed to train model.")
                        return

                    progress_tracker.update(80, "Finalizing training...")
                    await asyncio.sleep(0.1)

                    progress_tracker.update(100, "Training completed successfully.")
                    progress_tracker.is_running = False

                except Exception as e:
                    progress_tracker.set_error(f"Error during training: {str(e)}")
                    raise

            try:
                # Run with timeout
                await asyncio.wait_for(training_task(), timeout=timeout)
            except asyncio.TimeoutError:
                progress_tracker.set_error(f"Training timed out after {timeout} seconds")
                raise HTTPException(
                    status_code=408,
                    detail="Training timed out"
                )
            except Exception as e:
                progress_tracker.set_error(f"Error during training: {str(e)}")
                raise

        # Start training in background
        background_tasks.add_task(train_with_timeout)
        
        return TrainingResponse(
            success=True,
            message="Model training started in background.",
            training_time=0.0
        )
        
    except Exception as e:
        progress_tracker.set_error(f"Error starting training: {str(e)}")
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
            
            # Check if datetime column exists
            if time_series_forecaster.datetime_col not in prediction_data.columns:
                return PredictionResponse(
                    success=False,
                    message=f"Missing datetime column: {time_series_forecaster.datetime_col}",
                    predictions=[],
                    prediction_count=0
                )
            
            # Convert datetime column
            prediction_data[time_series_forecaster.datetime_col] = pd.to_datetime(prediction_data[time_series_forecaster.datetime_col])
            
            # Sort by datetime
            prediction_data = prediction_data.sort_values(by=time_series_forecaster.datetime_col)
            
            # Create time-based features
            prediction_data['hour'] = prediction_data[time_series_forecaster.datetime_col].dt.hour
            prediction_data['day'] = prediction_data[time_series_forecaster.datetime_col].dt.day
            prediction_data['month'] = prediction_data[time_series_forecaster.datetime_col].dt.month
            prediction_data['year'] = prediction_data[time_series_forecaster.datetime_col].dt.year
            prediction_data['dayofweek'] = prediction_data[time_series_forecaster.datetime_col].dt.dayofweek

            # Create lag features if target column exists in prediction data
            if time_series_forecaster.target in prediction_data.columns:
                for lag in [1, 2, 3, 5, 7]:
                    lag_col = f'{time_series_forecaster.target}_lag_{lag}'
                    prediction_data[lag_col] = prediction_data[time_series_forecaster.target].shift(lag)

                # Create rolling window features
                for window in [3, 5, 7]:
                    # Rolling mean
                    mean_col = f'{time_series_forecaster.target}_rolling_mean_{window}'
                    prediction_data[mean_col] = prediction_data[time_series_forecaster.target].rolling(window=window).mean()
                    
                    # Rolling std
                    std_col = f'{time_series_forecaster.target}_rolling_std_{window}'
                    prediction_data[std_col] = prediction_data[time_series_forecaster.target].rolling(window=window).std()
            else:
                # If target column doesn't exist, use historical values from training data
                last_values = time_series_forecaster.data[time_series_forecaster.target].tail(10).values
                
                # Add lag features using historical values
                for lag in [1, 2, 3, 5, 7]:
                    lag_col = f'{time_series_forecaster.target}_lag_{lag}'
                    if lag < len(last_values):
                        prediction_data[lag_col] = [last_values[-lag]] * len(prediction_data)
                    else:
                        prediction_data[lag_col] = [last_values[0]] * len(prediction_data)
                
                # Add rolling features using historical values
                for window in [3, 5, 7]:
                    # Rolling mean
                    mean_col = f'{time_series_forecaster.target}_rolling_mean_{window}'
                    if window < len(last_values):
                        prediction_data[mean_col] = [np.mean(last_values[-window:])] * len(prediction_data)
                    else:
                        prediction_data[mean_col] = [np.mean(last_values)] * len(prediction_data)
                    
                    # Rolling std
                    std_col = f'{time_series_forecaster.target}_rolling_std_{window}'
                    if window < len(last_values):
                        prediction_data[std_col] = [np.std(last_values[-window:])] * len(prediction_data)
                    else:
                        prediction_data[std_col] = [np.std(last_values)] * len(prediction_data)

            # Check for missing numerical features and impute with training mean
            for col in time_series_forecaster.numerical_cols:
                if col not in prediction_data.columns and col not in ['hour', 'day', 'month', 'year', 'dayofweek'] and not col.startswith(f'{time_series_forecaster.target}_'):
                    if col in time_series_forecaster.data.columns:
                        prediction_data[col] = time_series_forecaster.data[col].mean()
                    else:
                        prediction_data[col] = 0  # Fallback value

            # Check for missing categorical features and impute with mode
            for col in time_series_forecaster.categorical_cols:
                if col not in prediction_data.columns and col != time_series_forecaster.item_id_col:
                    if col in time_series_forecaster.data.columns:
                        prediction_data[col] = time_series_forecaster.data[col].mode()[0]
                    else:
                        prediction_data[col] = 'unknown'  # Fallback value

            # Prepare features for prediction
            X_pred = prediction_data[time_series_forecaster.features].copy()
            
            # Transform features using the same preprocessor
            X_pred_processed = time_series_forecaster.preprocessor.transform(X_pred)
            
            # Make predictions
            predictions = time_series_forecaster.model.predict(X_pred_processed)
            
            # Create results DataFrame
            results_df = pd.DataFrame({
                time_series_forecaster.datetime_col: prediction_data[time_series_forecaster.datetime_col],
                'Prediction': predictions
            })
            
            # Add actual values if available
            if time_series_forecaster.target in prediction_data.columns:
                results_df['Actual'] = prediction_data[time_series_forecaster.target]
            
            # Add item ID if available
            if time_series_forecaster.item_id_col and time_series_forecaster.item_id_col in prediction_data.columns:
                results_df[time_series_forecaster.item_id_col] = prediction_data[time_series_forecaster.item_id_col]
            
            # Convert to records
            predictions_list = convert_to_native_types(results_df.to_dict(orient='records'))
            
            return PredictionResponse(
                success=True,
                message=f"Made {len(predictions_list)} predictions successfully.",
                predictions=predictions_list,
                prediction_count=len(predictions_list)
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