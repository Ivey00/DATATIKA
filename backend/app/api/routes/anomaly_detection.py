import base64
from datetime import datetime
import io
import os
import tempfile
import asyncio
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import Dict, Any, List, Optional
import json
from io import BytesIO
import seaborn as sns

from app.core.unsupervised_learning import UnsupervisedModelTrainer
from app.models.schema import (
    DataImportRequest, DataImportResponse,
    ColumnInfoResponse,
    FeatureDefinitionRequest, FeatureDefinitionResponse,
    ItemFilterRequest, ItemFilterResponse,
    VisualizationResponse,
    AlgorithmSelectionRequest, AlgorithmSelectionResponse,
    HyperparameterConfigRequest, HyperparameterConfigResponse,
    PreprocessResponse,
    TrainingResponse,
    EvaluationResponse,
    PredictionRequest, PredictionResponse,
    GridSearchRequest, GridSearchResponse,
    ModelSaveResponse,
    ErrorResponse,
    convert_to_native_types
)

router = APIRouter()

# Global instance of the UnsupervisedModelTrainer class
# In a production environment, this would be handled differently
# with a database to store session state
anomaly_detection_system = UnsupervisedModelTrainer()

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

class ProgressReporter:
    def __init__(self):
        self.progress = 0
        self.message = ""

    def update(self, progress, message=""):
        self.progress = progress
        self.message = message

progress_reporter = ProgressReporter()

async def progress_stream():
    """Generate SSE events for progress reporting."""
    event_id = 1
    last_progress = None
    
    # Initial connection confirmation
    print("SSE: New client connected to progress stream")
    
    # Send initial event to establish connection
    yield "retry: 1000\n"
    yield f"id: {event_id}\n"
    yield f"data: {json.dumps({'progress': progress_reporter.progress, 'message': progress_reporter.message})}\n\n"
    event_id += 1
    
    try:
        while True:
            try:
                current_progress = progress_reporter.progress
                current_message = progress_reporter.message
                current_data = {'progress': current_progress, 'message': current_message}
                current_json = json.dumps(current_data)
                
                # Send update when progress changes or for 100% completion
                if current_progress != last_progress or current_progress == 100:
                    print(f"SSE: Sending progress event: {current_progress}% - {current_message}")
                    
                    # Format properly according to SSE spec with ID and retry directive
                    yield f"id: {event_id}\n"
                    yield "retry: 1000\n"
                    yield f"data: {current_json}\n\n"
                    
                    event_id += 1
                    last_progress = current_progress
                    
                    # For critical progress points (70%, 90%, 100%), send multiple confirmations
                    # to increase chances of reaching client despite connection issues
                    if current_progress in [70, 90, 100]:
                        print(f"SSE: Sending critical {current_progress}% confirmation")
                        await asyncio.sleep(0.3)
                        
                        # Send extra confirmations for critical updates (more for 100%)
                        repeats = 5 if current_progress == 100 else 2
                        for i in range(repeats):
                            try:
                                yield f"id: {event_id}\n"
                                yield "retry: 1000\n"
                                yield f"data: {current_json}\n\n"
                                event_id += 1
                                await asyncio.sleep(0.3)
                            except Exception as e:
                                print(f"SSE: Error sending confirmation {i+1}/{repeats} for {current_progress}%: {str(e)}")
                                # Continue with next attempt even if this one fails
                                continue
                else:
                    # Send keep-alive comment every few seconds to maintain connection
                    yield f": keepalive {event_id}\n\n"
                    
                await asyncio.sleep(0.5)
                
            except ConnectionResetError as e:
                # Handle connection reset specifically but try to continue
                print(f"SSE: Connection reset during progress stream: {str(e)}")
                
                # If we were at critical progress point, try to recover
                if current_progress in [70, 90, 100]:
                    print(f"SSE: Connection reset at critical progress {current_progress}%, attempting recovery")
                    
                    # Short sleep to allow client reconnection
                    await asyncio.sleep(1.0)
                    
                    # Return to the main loop to continue sending updates
                    continue
                else:
                    # For non-critical points, re-raise to exit the stream
                    raise
    except Exception as e:
        print(f"SSE: Error in progress stream: {str(e)}")
        try:
            # Send error event
            yield f"id: {event_id}\n"
            yield f"event: error\n"
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        except Exception:
            # Ignore errors when trying to send the error message
            pass
    finally:
        print("SSE: Client disconnected from progress stream")
        
        # If disconnection happens at 70% or above, make sure to update to 100%
        # This ensures the backend state is consistent even if client connection was lost
        if progress_reporter.progress >= 70:
            current_progress = progress_reporter.progress
            print(f"SSE: Client disconnected at {current_progress}%, ensuring final state is set")
            
            # Use background task to finalize progress to 100% to ensure backend state consistency
            asyncio.create_task(finalize_progress_after_disconnect())

async def finalize_progress_after_disconnect():
    """Ensure progress reaches 100% after a client disconnect during evaluation phase."""
    try:
        await asyncio.sleep(1.0)  # Small delay to let any running processes complete
        
        # Only update if we were in the evaluation or finalization phase
        if progress_reporter.progress >= 70 and progress_reporter.progress < 100:
            print("SSE: Setting final progress to 100% after disconnect")
            progress_reporter.update(100, "Model training and evaluation completed successfully")
    except Exception as e:
        print(f"SSE: Error in finalize_progress_after_disconnect: {str(e)}")

@router.get("/progress")
async def get_progress():
    """Return SSE stream for progress reporting."""
    headers = {
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"  # Disable buffering for Nginx
    }
    return StreamingResponse(
        progress_stream(),
        media_type="text/event-stream",
        headers=headers
    )

@router.post("/import-data", response_model=DataImportResponse)
async def import_data(request: DataImportRequest, background_tasks: BackgroundTasks):
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
            # Update progress
            progress_reporter.update(10, "Importing data...")
            
            # Import the data
            data = anomaly_detection_system.load_data(temp_file_path)
            
            progress_reporter.update(100, "Data imported successfully")
            
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
    Get information about columns in the dataset and auto-detect column types.
    """
    try:
        if anomaly_detection_system.data is None:
            raise HTTPException(
                status_code=400,
                detail="No data available. Please import data first."
            )
        
        # Auto-detect column types
        datetime_cols, numerical_cols, categorical_cols = anomaly_detection_system.detect_column_types()
        
        # Prepare column info
        column_info = {}
        for column in anomaly_detection_system.data.columns:
            dtype = str(anomaly_detection_system.data[column].dtype)
            unique_count = len(anomaly_detection_system.data[column].unique())
            
            column_type = "Unknown"
            if column in datetime_cols:
                column_type = "Datetime"
            elif column in numerical_cols:
                column_type = "Numerical"
            elif column in categorical_cols:
                column_type = "Categorical"
                
            column_info[column] = {
                "dtype": dtype,
                "unique_values": unique_count,
                "column_type": column_type,
                "missing_values": int(anomaly_detection_system.data[column].isnull().sum())
            }
            
            # Include sample values
            if column_type == "Categorical" and unique_count <= 20:  # Only for categorical with few unique values
                column_info[column]["unique_values_list"] = convert_to_native_types(
                    anomaly_detection_system.data[column].dropna().unique().tolist()
                )
                
            # For numerical columns, include min, max, mean
            if column_type == "Numerical":
                column_info[column]["min"] = float(anomaly_detection_system.data[column].min())
                column_info[column]["max"] = float(anomaly_detection_system.data[column].max())
                column_info[column]["mean"] = float(anomaly_detection_system.data[column].mean())
        
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
    Define the features and column types for anomaly detection.
    """
    try:
        if anomaly_detection_system.data is None:
            return FeatureDefinitionResponse(
                success=False,
                message="No data available. Please import data first.",
                features=[],
                target="",
                categorical_features=[],
                numerical_features=[],
                datetime_column="",
                item_id_column=None
            )
        
        # Set column types
        anomaly_detection_system.set_column_types(
            datetime_col=request.datetime_column,
            item_id_col=request.item_id_column,
            numerical_cols=request.numerical_features,
            categorical_cols=request.categorical_features
        )
        
        return FeatureDefinitionResponse(
            success=True,
            message="Features defined successfully",
            features=request.features,
            target="anomaly",  # For anomaly detection, target is always "anomaly"
            categorical_features=anomaly_detection_system.categorical_cols,
            numerical_features=anomaly_detection_system.numerical_cols,
            datetime_column=anomaly_detection_system.datetime_col,
            item_id_column=anomaly_detection_system.item_id_col
        )
        
    except Exception as e:
        return FeatureDefinitionResponse(
            success=False,
            message=f"Error defining features: {str(e)}",
            features=[],
            target="",
            categorical_features=[],
            numerical_features=[],
            datetime_column="",
            item_id_column=None
        )

@router.post("/filter-by-item", response_model=ItemFilterResponse)
async def filter_by_item(request: ItemFilterRequest):
    """
    Filter the dataset to only include rows with the specified item ID.
    """
    try:
        if anomaly_detection_system.data is None or not anomaly_detection_system.item_id_col:
            return ItemFilterResponse(
                success=False,
                message="No data available or item ID column not defined. Please import data and define features first.",
                item_id_column="",
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        # Filter data by category (item ID)
        filtered_data = anomaly_detection_system.filter_by_category(
            category_col=anomaly_detection_system.item_id_col,
            category_value=request.item_id_value
        )
        
        if filtered_data is None:
            return ItemFilterResponse(
                success=False,
                message=f"No data found for {anomaly_detection_system.item_id_col} = {request.item_id_value}",
                item_id_column=anomaly_detection_system.item_id_col,
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        return ItemFilterResponse(
            success=True,
            message=f"Data filtered successfully for {anomaly_detection_system.item_id_col} = {request.item_id_value}",
            item_id_column=anomaly_detection_system.item_id_col,
            item_id_value=request.item_id_value,
            filtered_shape=list(filtered_data.shape)
        )
        
    except Exception as e:
        return ItemFilterResponse(
            success=False,
            message=f"Error filtering data: {str(e)}",
            item_id_column=anomaly_detection_system.item_id_col or "",
            item_id_value=request.item_id_value,
            filtered_shape=[0, 0]
        )

@router.get("/visualize-data", response_model=VisualizationResponse)
async def visualize_data(background_tasks: BackgroundTasks, max_features: Optional[int] = 10):
    """
    Visualize the data with appropriate plots for each feature type using SSE for progress updates.
    """
    try:
        if anomaly_detection_system.data is None or not anomaly_detection_system.features:
            return VisualizationResponse(
                success=False,
                message="No data available or features not defined. Please import data and define features first.",
                visualizations={}
            )
        
        # Prepare a shared dict to store visualizations that can be accessed by the background task
        visualizations = {}
        
        # Define the background task to generate visualizations
        def generate_visualizations_task():
            try:
                # Update progress at the start
                progress_reporter.update(10, "Starting data visualization...")
                
                # Save the original backend
                original_backend = plt.get_backend()
                plt.switch_backend('Agg')
                
                try:
                    # Limit the number of features to visualize
                    features_to_viz = anomaly_detection_system.features[:max_features] if len(anomaly_detection_system.features) > max_features else anomaly_detection_system.features
                    
                    # Numerical features histograms
                    progress_reporter.update(20, "Generating numerical feature distributions...")
                    num_features = [f for f in features_to_viz if f in anomaly_detection_system.numerical_cols]
                    if num_features:
                        n_cols = min(3, len(num_features))
                        n_rows = (len(num_features) + n_cols - 1) // n_cols
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
                        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                        
                        for i, feature in enumerate(num_features):
                            sns.histplot(anomaly_detection_system.data[feature], kde=True, ax=axes[i])
                            axes[i].set_title(f"Distribution of {feature}")
                        
                        # Hide unused subplots
                        for j in range(len(num_features), len(axes)):
                            fig.delaxes(axes[j])
                        
                        plt.tight_layout()
                        visualizations['numerical_distributions'] = fig_to_base64(fig)
                    
                    # Categorical features bar plots
                    progress_reporter.update(40, "Generating categorical feature distributions...")
                    cat_features = [f for f in features_to_viz if f in anomaly_detection_system.categorical_cols]
                    if cat_features:
                        n_cols = min(2, len(cat_features))
                        n_rows = (len(cat_features) + n_cols - 1) // n_cols
                        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4))
                        axes = axes.flatten() if n_rows * n_cols > 1 else [axes]
                        
                        for i, feature in enumerate(cat_features):
                            counts = anomaly_detection_system.data[feature].value_counts()
                            counts = counts.iloc[:10] if len(counts) > 10 else counts  # Limit to top 10 categories
                            counts.plot(kind='bar', ax=axes[i])
                            axes[i].set_title(f"Distribution of {feature}")
                            axes[i].set_ylabel('Count')
                            axes[i].tick_params(axis='x', rotation=45)
                        
                        # Hide unused subplots
                        for j in range(len(cat_features), len(axes)):
                            fig.delaxes(axes[j])
                        
                        plt.tight_layout()
                        visualizations['categorical_distributions'] = fig_to_base64(fig)
                    
                    # Correlation matrix for numerical features
                    progress_reporter.update(60, "Generating correlation matrix...")
                    if len(anomaly_detection_system.numerical_cols) > 1:
                        corr_features = anomaly_detection_system.numerical_cols[:10]  # Limit to 10 features for readability
                        correlation_matrix = anomaly_detection_system.data[corr_features].corr()
                        
                        fig = plt.figure(figsize=(10, 8))
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                        plt.title('Correlation Matrix')
                        plt.tight_layout()
                        visualizations['correlation_matrix'] = fig_to_base64(fig)
                    
                    # Time series plots if datetime column exists
                    progress_reporter.update(80, "Generating time series visualizations...")
                    if anomaly_detection_system.datetime_col and anomaly_detection_system.numerical_cols:
                        time_series_features = anomaly_detection_system.numerical_cols[:3]  # Limit to 3 features
                        fig, axes = plt.subplots(len(time_series_features), 1, figsize=(12, 4 * len(time_series_features)))
                        axes = [axes] if len(time_series_features) == 1 else axes
                        
                        for i, feature in enumerate(time_series_features):
                            # Convert to datetime and sort
                            temp_df = anomaly_detection_system.data.copy()
                            temp_df[anomaly_detection_system.datetime_col] = pd.to_datetime(temp_df[anomaly_detection_system.datetime_col])
                            temp_df = temp_df.sort_values(anomaly_detection_system.datetime_col)
                            
                            # Plot time series
                            axes[i].plot(temp_df[anomaly_detection_system.datetime_col], temp_df[feature])
                            axes[i].set_title(f"{feature} Over Time")
                            axes[i].set_ylabel(feature)
                            axes[i].tick_params(axis='x', rotation=45)
                        
                        plt.tight_layout()
                        visualizations['time_series'] = fig_to_base64(fig)
                    
                    # Update progress to complete
                    progress_reporter.update(100, "Data visualization completed successfully")
                    
                finally:
                    # Restore the original backend
                    plt.switch_backend(original_backend)
                    plt.close('all')  # Clean up all figures
                    
            except Exception as e:
                progress_reporter.update(0, f"Error in visualization: {str(e)}")
        
        # Start the visualization task in the background
        background_tasks.add_task(generate_visualizations_task)
        
        # Return an initial response indicating the task has started
        return VisualizationResponse(
            success=True,
            message="Data visualization started. Check progress endpoint for updates.",
            visualizations={}
        )
            
    except Exception as e:
        return VisualizationResponse(
            success=False,
            message=f"Error starting visualization: {str(e)}",
            visualizations={}
        )

@router.post("/select-algorithm", response_model=AlgorithmSelectionResponse)
async def select_algorithm(request: AlgorithmSelectionRequest):
    """
    Select and configure the anomaly detection algorithm.
    """
    try:
        success = anomaly_detection_system.select_algorithm(request.algorithm_name)
        
        if not success:
            return AlgorithmSelectionResponse(
                success=False,
                message=f"Algorithm '{request.algorithm_name}' not supported.",
                algorithm_name=request.algorithm_name,
                hyperparameters={},
                algorithm_description=""
            )
        
        # Get parameters for the selected algorithm
        hyperparameters = {}
        for param, details in anomaly_detection_system.algorithm_params[request.algorithm_name].items():
            hyperparameters[param] = {
                'description': details['description'],
                'default': details['default'],
                'suggested_values': details['suggested_values']
            }
        
        # Create algorithm description
        algorithm_descriptions = {
            'Isolation Forest': "Isolates anomalies by randomly selecting features and split values. Anomalies require fewer splits to isolate.",
            'Local Outlier Factor': "Identifies anomalies by measuring local deviation of a point with respect to its neighbors.",
            'One-Class SVM': "Learns a boundary around normal data points. Points outside this boundary are considered anomalies."
        }
        
        algorithm_description = algorithm_descriptions.get(request.algorithm_name, "No description available.")
        
        return AlgorithmSelectionResponse(
            success=True,
            message=f"Algorithm '{request.algorithm_name}' selected successfully",
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
        success = anomaly_detection_system.set_hyperparameters(**request.hyperparameters)
        
        if not success:
            return HyperparameterConfigResponse(
                success=False,
                message="Failed to configure hyperparameters. Please select an algorithm first or check parameter values.",
                hyperparameters={}
            )
        
        return HyperparameterConfigResponse(
            success=True,
            message="Hyperparameters configured successfully",
            hyperparameters=convert_to_native_types(request.hyperparameters)
        )
        
    except Exception as e:
        return HyperparameterConfigResponse(
            success=False,
            message=f"Error configuring hyperparameters: {str(e)}",
            hyperparameters={}
        )

@router.get("/preprocess-data", response_model=PreprocessResponse)
async def preprocess_data():
    """
    Preprocess the data for model training.
    """
    try:
        X_transformed = anomaly_detection_system.preprocess_data()
        
        if X_transformed is None:
            return PreprocessResponse(
                success=False,
                message="Failed to preprocess data. Please import data and define features first.",
                train_shape=[0, 0],
                test_shape=[0, 0]  # No test set for anomaly detection
            )
        
        return PreprocessResponse(
            success=True,
            message="Data preprocessed successfully",
            train_shape=list(X_transformed.shape),
            test_shape=[0, 0]  # No test set for anomaly detection
        )
        
    except Exception as e:
        return PreprocessResponse(
            success=False,
            message=f"Error preprocessing data: {str(e)}",
            train_shape=[0, 0],
            test_shape=[0, 0]
        )

@router.get("/train-model", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the model with the selected algorithm and configured hyperparameters.
    Uses SSE for progress updates.
    """
    try:
        # Force reset the progress reporter to ensure clean state
        progress_reporter.progress = 0
        progress_reporter.message = ""
        print("\n--- NEW TRAINING SESSION STARTED ---")
        print("Resetting progress reporter to initial state")
        
        # Set initial progress
        progress_reporter.update(10, "Starting model training...")
        print("Progress set to 10% - Starting model training...")
        
        async def async_train_model_task():
            try:
                print("Beginning async training task")
                # Set progress to 20%
                progress_reporter.update(20, "Initializing model training...")
                print("Progress set to 20% - Initializing model training...")
                await asyncio.sleep(0.3)
                
                # Preprocessing
                if anomaly_detection_system.X_transformed is None:
                    progress_reporter.update(30, "Preprocessing data...")
                    print("Progress set to 30% - Preprocessing data...")
                    await asyncio.sleep(0.3)
                    anomaly_detection_system.preprocess_data()
                
                # Train the model
                progress_reporter.update(40, "Training model with selected algorithm...")
                print("Progress set to 40% - Training model with selected algorithm...")
                await asyncio.sleep(0.3)
                
                # Execute the actual model training
                print("Executing model.train_model() function...")
                success = anomaly_detection_system.train_model()
                print(f"Model training completed with success={success}")
                
                if success:
                    # Update progress to indicate evaluation is starting
                    progress_reporter.update(70, "Model trained, now evaluating...")
                    print("Progress set to 70% - Model trained, now evaluating...")
                    await asyncio.sleep(0.5)
                    
                    # Perform model evaluation as part of the training process
                    print("Executing model evaluation as part of training process...")
                    anomaly_detection_system.evaluate_model()
                    print("Model evaluation completed")
                    
                    # Update to 90%
                    progress_reporter.update(90, "Finalizing model and evaluation...")
                    print("Progress set to 90% - Finalizing model and evaluation...")
                    await asyncio.sleep(1.0)
                    
                    # Force multiple updates to 100% to ensure client receives it
                    print("CRITICAL: Setting progress to 100% with multiple attempts")
                    for i in range(5):  # Try 5 times to ensure delivery
                        progress_reporter.update(100, "Model training and evaluation completed successfully")
                        print(f"Progress 100% update attempt {i+1}/5")
                        await asyncio.sleep(0.7)  # Longer delay between attempts
                else:
                    progress_reporter.update(0, "Model training failed")
                    print("Training failed - progress reset to 0%")
            except Exception as e:
                print(f"ERROR in training task: {str(e)}")
                progress_reporter.update(0, f"Error in model training: {str(e)}")
        
        # Start training in background
        background_tasks.add_task(async_train_model_task)
        
        return TrainingResponse(
            success=True,
            message="Model training started. Check progress endpoint for updates.",
            training_time=0.0
        )
        
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Error starting model training: {str(e)}",
            training_time=0.0
        )

@router.get("/evaluate-model", response_model=EvaluationResponse)
async def evaluate_model():
    """
    Retrieve the evaluation results from the trained model.
    The evaluation is performed during model training, this endpoint just returns the results.
    """
    try:
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            # Get previously computed evaluation results instead of recomputing
            evaluation_results = anomaly_detection_system.get_evaluation_results()
            
            if not evaluation_results["success"]:
                return EvaluationResponse(
                    success=False,
                    message=evaluation_results["message"],
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    visualizations={}
                )
            
            # Use the metrics from the pre-computed results
            metrics = evaluation_results["metrics"]
            classification_report = {}
            confusion_matrix = []
            visualizations = {}
            
            # For clustering algorithms
            if anomaly_detection_system.algorithm_name in ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture']:
                # Count points in each cluster
                cluster_counts = anomaly_detection_system.anomaly_results['cluster'].value_counts().to_dict()
                
                # Create cluster distribution visualization
                fig = plt.figure(figsize=(10, 6))
                plt.bar(
                    [f"Cluster {k}" for k in sorted(cluster_counts.keys())],
                    [v for k, v in sorted(cluster_counts.items())]
                )
                plt.title('Cluster Distribution')
                plt.ylabel('Number of Points')
                plt.xticks(rotation=45)
                plt.tight_layout()
                visualizations['cluster_distribution'] = fig_to_base64(fig)
                
                # If we have numerical features, create scatter plot of first two components
                if len(anomaly_detection_system.numerical_cols) >= 2 and anomaly_detection_system.X_transformed is not None:
                    # Use PCA for high-dimensional data
                    if anomaly_detection_system.X_transformed.shape[1] > 2:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        X_2d = pca.fit_transform(anomaly_detection_system.X_transformed)
                        x_label, y_label = 'Principal Component 1', 'Principal Component 2'
                    else:
                        X_2d = anomaly_detection_system.X_transformed
                        x_label = anomaly_detection_system.numerical_cols[0]
                        y_label = anomaly_detection_system.numerical_cols[1]
                    
                    # Create scatter plot colored by cluster
                    fig = plt.figure(figsize=(10, 8))
                    clusters = anomaly_detection_system.anomaly_results['cluster'].unique()
                    for cluster in sorted(clusters):
                        mask = anomaly_detection_system.anomaly_results['cluster'] == cluster
                        plt.scatter(
                            X_2d[mask, 0], X_2d[mask, 1],
                            label=f"Cluster {cluster}",
                            alpha=0.7
                        )
                    plt.title('Cluster Visualization')
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.legend()
                    plt.tight_layout()
                    visualizations['cluster_visualization'] = fig_to_base64(fig)
            
            # For anomaly detection algorithms
            if anomaly_detection_system.algorithm_name in ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']:
                # Create visualization of anomalies
                if anomaly_detection_system.X_transformed is not None:
                    # Use PCA for high-dimensional data
                    if anomaly_detection_system.X_transformed.shape[1] > 2:
                        from sklearn.decomposition import PCA
                        pca = PCA(n_components=2)
                        X_2d = pca.fit_transform(anomaly_detection_system.X_transformed)
                        x_label, y_label = 'Principal Component 1', 'Principal Component 2'
                    else:
                        X_2d = anomaly_detection_system.X_transformed
                        x_label = anomaly_detection_system.numerical_cols[0]
                        y_label = anomaly_detection_system.numerical_cols[1]
                    
                    # Create scatter plot showing anomalies
                    fig = plt.figure(figsize=(10, 8))
                    normal_mask = anomaly_detection_system.anomaly_results['anomaly'] == 'No'
                    anomaly_mask = anomaly_detection_system.anomaly_results['anomaly'] == 'Yes'
                    
                    plt.scatter(
                        X_2d[normal_mask, 0], X_2d[normal_mask, 1],
                        label="Normal", alpha=0.5, c='blue'
                    )
                    plt.scatter(
                        X_2d[anomaly_mask, 0], X_2d[anomaly_mask, 1],
                        label="Anomaly", alpha=0.7, c='red', edgecolors='k'
                    )
                    plt.title('Anomaly Detection Results')
                    plt.xlabel(x_label)
                    plt.ylabel(y_label)
                    plt.legend()
                    plt.tight_layout()
                    visualizations['anomaly_visualization'] = fig_to_base64(fig)
            
            # For time series visualization with anomalies
            if anomaly_detection_system.datetime_col and anomaly_detection_system.numerical_cols:
                # Select up to 3 numerical features for time series plots
                time_series_features = anomaly_detection_system.numerical_cols[:3]
                
                # Create time series plot with anomalies highlighted
                fig, axes = plt.subplots(len(time_series_features), 1, figsize=(12, 4 * len(time_series_features)))
                axes = [axes] if len(time_series_features) == 1 else axes
                
                # Sort by datetime
                temp_df = anomaly_detection_system.anomaly_results.copy()
                temp_df[anomaly_detection_system.datetime_col] = pd.to_datetime(temp_df[anomaly_detection_system.datetime_col])
                temp_df = temp_df.sort_values(anomaly_detection_system.datetime_col)
                
                # Plot each feature
                for i, feature in enumerate(time_series_features):
                    # Plot normal points
                    normal_mask = temp_df['anomaly'] == 'No'
                    axes[i].plot(
                        temp_df[normal_mask][anomaly_detection_system.datetime_col],
                        temp_df[normal_mask][feature],
                        'b-', alpha=0.7, label='Normal'
                    )
                    
                    # Plot anomalies
                    anomaly_mask = temp_df['anomaly'] == 'Yes'
                    axes[i].scatter(
                        temp_df[anomaly_mask][anomaly_detection_system.datetime_col],
                        temp_df[anomaly_mask][feature],
                        c='red', s=50, label='Anomaly', zorder=5
                    )
                    
                    axes[i].set_title(f"{feature} Over Time with Anomalies")
                    axes[i].set_ylabel(feature)
                    axes[i].tick_params(axis='x', rotation=45)
                    axes[i].legend()
                
                plt.tight_layout()
                visualizations['time_series_anomalies'] = fig_to_base64(fig)
            
            return EvaluationResponse(
                success=True,
                message="Model evaluated successfully",
                metrics=metrics,
                classification_report=classification_report,
                confusion_matrix=confusion_matrix,
                visualizations=visualizations
            )
            
        finally:
            # Clean up
            plt.close('all')
            
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
        # Decode base64 file content
        file_content = base64.b64decode(request.file_content)
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_name)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Load the new data
            if temp_file_path.endswith('.csv'):
                new_data = pd.read_csv(temp_file_path)
            elif temp_file_path.endswith(('.xls', '.xlsx')):
                new_data = pd.read_excel(temp_file_path)
            else:
                return PredictionResponse(
                    success=False,
                    message="Unsupported file format. Please provide a CSV or Excel file.",
                    predictions=[],
                    prediction_count=0
                )
            
            # Make predictions
            results = anomaly_detection_system.predict_new_data(new_data)
            
            if results is None:
                return PredictionResponse(
                    success=False,
                    message="No trained model available or error in prediction. Please train a model first.",
                    predictions=[],
                    prediction_count=0
                )
            
            # Extract anomaly predictions
            predictions = results.copy()
            
            return PredictionResponse(
                success=True,
                message="Predictions made successfully",
                predictions=convert_to_native_types(predictions.to_dict(orient='records')),
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

@router.get("/save-results", response_model=ModelSaveResponse)
async def save_results(directory: str = "results"):
    """
    Save anomaly detection results to a CSV file.
    """
    try:
        if anomaly_detection_system.anomaly_results is None:
            return ModelSaveResponse(
                success=False,
                message="No results available. Please train a model first.",
                model_path="",
                timestamp=""
            )
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{directory}/anomaly_results_{timestamp}.csv"
        
        anomaly_detection_system.anomaly_results.to_csv(output_path, index=False)
        
        return ModelSaveResponse(
            success=True,
            message="Results saved successfully",
            model_path=output_path,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return ModelSaveResponse(
            success=False,
            message=f"Error saving results: {str(e)}",
            model_path="",
            timestamp=""
        )

@router.get("/save-model", response_model=ModelSaveResponse)
async def save_model(directory: str = "models"):
    """
    Save the trained model and preprocessor to disk.
    """
    try:
        if anomaly_detection_system.model is None:
            return ModelSaveResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                model_path="",
                timestamp=""
            )
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{directory}/anomaly_model_{anomaly_detection_system.algorithm_name}_{timestamp}"
        
        model_path = anomaly_detection_system.save_model(output_path)
        
        if model_path is None:
            return ModelSaveResponse(
                success=False,
                message="Error saving model.",
                model_path="",
                timestamp=""
            )
        
        return ModelSaveResponse(
            success=True,
            message="Model saved successfully",
            model_path=str(model_path),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        return ModelSaveResponse(
            success=False,
            message=f"Error saving model: {str(e)}",
            model_path="",
            timestamp=""
        )

@router.get("/algorithm-descriptions")
async def get_algorithm_descriptions():
    """
    Get descriptions of all supported anomaly detection algorithms.
    """
    try:
        algorithm_descriptions = {
            'Isolation Forest': "Isolates anomalies by randomly selecting features and split values. Anomalies require fewer splits to isolate.",
            'Local Outlier Factor': "Identifies anomalies by measuring local deviation of a point with respect to its neighbors.",
            'One-Class SVM': "Learns a boundary around normal data points. Points outside this boundary are considered anomalies."
        }
        
        return {"descriptions": algorithm_descriptions}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting algorithm descriptions: {str(e)}"
        )

@router.get("/available-algorithms")
async def get_available_algorithms():
    """
    Get list of available anomaly detection algorithms.
    """
    try:
        algorithms = ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']
        return {"algorithms": algorithms}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting available algorithms: {str(e)}"
        )