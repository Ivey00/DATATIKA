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
from sklearn.metrics import classification_report, confusion_matrix
import asyncio

from app.core.classification import Classification
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
    ComparisonRequest, ComparisonResponse,
    GridSearchRequest, GridSearchResponse,
    ModelSaveResponse,
    ErrorResponse,
    convert_to_native_types,
    ModelSaveRequest
)

router = APIRouter()

# Global instance of the Classification class
# In a production environment, this would be handled differently
# with a database to store session state
classification_system = Classification()

# Add training state variables
training_progress = 0
training_message = "Not started"
training_complete = False
training_error = None
is_training = False

def reset_training_state():
    global training_progress, training_message, training_complete, training_error, is_training
    training_progress = 0
    training_message = "Not started"
    training_complete = False
    training_error = None
    is_training = False

async def train_model_task():
    """Background task for model training"""
    global training_progress, training_message, training_complete, training_error, is_training
    
    try:
        is_training = True
        training_message = "Preprocessing data..."
        training_progress = 10
        
        # Preprocess data
        X_train, X_test, y_train, y_test = classification_system.preprocess_data()
        if X_train is None:
            raise Exception("Failed to preprocess data")
        
        await asyncio.sleep(1)  # Simulate some work
        training_progress = 40
        training_message = "Training model..."
        
        # Train model
        model = classification_system.train_model()
        if model is None:
            raise Exception("Failed to train model")
        
        await asyncio.sleep(1)  # Simulate some work
        training_progress = 70
        training_message = "Evaluating model..."
        
        # Evaluate model
        metrics = classification_system.evaluate_model()
        if metrics is None:
            raise Exception("Failed to evaluate model")
        
        training_progress = 100
        training_message = "Training complete!"
        training_complete = True
        
    except Exception as e:
        training_error = str(e)
        training_message = f"Error: {str(e)}"
    finally:
        is_training = False

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
            data = classification_system.import_data(temp_file_path)
            
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
        column_info = classification_system.get_column_info()
        
        if column_info is None:
            raise HTTPException(
                status_code=400,
                detail="No data available. Please import data first."
            )
        
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
        X, y = classification_system.define_features_target(
            features=request.features,
            target=request.target,
            categorical_features=request.categorical_features,
            numerical_features=request.numerical_features,
            item_id_column=request.item_id_column
        )
        
        # Store datetime column for time-based visualizations if provided
        classification_system.datetime_column = request.datetime_column
        
        if X is None or y is None:
            return FeatureDefinitionResponse(
                success=False,
                message="Failed to define features and target",
                features=[],
                target="",
                categorical_features=[],
                numerical_features=[],
                datetime_column=None,
                item_id_column=None
            )
        
        return FeatureDefinitionResponse(
            success=True,
            message="Features and target defined successfully",
            features=request.features,
            target=request.target,
            categorical_features=request.categorical_features or [],
            numerical_features=request.numerical_features or [],
            datetime_column=request.datetime_column,
            item_id_column=request.item_id_column
        )
        
    except Exception as e:
        return FeatureDefinitionResponse(
            success=False,
            message=f"Error defining features and target: {str(e)}",
            features=[],
            target="",
            categorical_features=[],
            numerical_features=[],
            datetime_column=None,
            item_id_column=None
        )

@router.post("/filter-by-item", response_model=ItemFilterResponse)
async def filter_by_item(request: ItemFilterRequest):
    """
    Filter the dataset to only include rows with the specified item ID.
    """
    try:
        X_filtered, y_filtered = classification_system.filter_by_item_id(request.item_id_value)
        
        if X_filtered is None or y_filtered is None:
            return ItemFilterResponse(
                success=False,
                message="Failed to filter data. Please define features and target first, and ensure the item ID column is specified.",
                item_id_column=classification_system.item_id_column or "",
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        return ItemFilterResponse(
            success=True,
            message=f"Data filtered successfully for {classification_system.item_id_column} = {request.item_id_value}",
            item_id_column=classification_system.item_id_column,
            item_id_value=request.item_id_value,
            filtered_shape=list(X_filtered.shape)
        )
        
    except Exception as e:
        return ItemFilterResponse(
            success=False,
            message=f"Error filtering data: {str(e)}",
            item_id_column=classification_system.item_id_column or "",
            item_id_value=request.item_id_value,
            filtered_shape=[0, 0]
        )

@router.get("/visualize-data", response_model=VisualizationResponse)
async def visualize_data(max_features: Optional[int] = 10):
    """
    Visualize the data with appropriate plots for each feature type.
    """
    try:
        # Redirect matplotlib output to capture visualizations
        visualizations = {}
        
        # Save the original backend
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')
        
        try:
            # Feature distributions
            if classification_system.X is not None and classification_system.y is not None:
                # Limit the number of features to visualize
                features_to_viz = classification_system.features[:max_features] if len(classification_system.features) > max_features else classification_system.features
                
                # Set up the plotting area
                num_features = len(features_to_viz)
                if num_features <= 3:
                    fig_height = 5
                    n_rows = num_features
                else:
                    fig_height = min(20, num_features * 2)  # Limit the figure height
                    n_rows = (num_features + 1) // 2  # Two columns if more than 3 features
                
                fig, axes = plt.subplots(n_rows, 2 if num_features > 3 else 1, figsize=(12, fig_height))
                
                # Make axes iterable even for a single subplot
                if num_features == 1:
                    axes = [axes]
                elif num_features <= 3:
                    axes = axes.flatten()
                else:
                    axes = axes.flatten()
                    if num_features % 2 == 1:  # Odd number of features
                        fig.delaxes(axes[-1])  # Remove the last subplot
                        axes = axes[:-1]
                
                for i, feature in enumerate(features_to_viz):
                    ax = axes[i]
                    
                    # Check if feature is categorical or numerical
                    if feature in classification_system.categorical_features:
                        # For categorical features
                        counts = pd.crosstab(classification_system.X[feature], classification_system.y)
                        counts.plot(kind='bar', stacked=True, ax=ax)
                        ax.set_title(f'Distribution of {feature} by Target')
                        ax.set_ylabel('Count')
                        ax.set_xlabel(feature)
                        ax.legend(title=classification_system.target)
                        
                    elif feature in classification_system.numerical_features:
                        # For numerical features
                        if classification_system.y.nunique() <= 5:  # If target has few unique values, use box plot
                            ax.boxplot([classification_system.X[classification_system.y == c][feature].dropna() for c in classification_system.y.unique()],
                                    labels=classification_system.y.unique())
                            ax.set_title(f'Distribution of {feature} by Target')
                            ax.set_ylabel(feature)
                            ax.set_xlabel(classification_system.target)
                        else:
                            # For many target values or continuous targets, use scatter plot
                            ax.scatter(classification_system.X[feature], classification_system.y, alpha=0.5)
                            ax.set_title(f'{feature} vs {classification_system.target}')
                            ax.set_xlabel(feature)
                            ax.set_ylabel(classification_system.target)
                
                plt.tight_layout()
                visualizations['feature_distributions'] = fig_to_base64(fig)
                
                # Correlation matrix for numerical features
                if len(classification_system.numerical_features) > 1:
                    correlation_matrix = classification_system.X[classification_system.numerical_features].corr()
                    fig = plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
                    plt.title('Correlation Matrix')
                    plt.tight_layout()
                    visualizations['correlation_matrix'] = fig_to_base64(fig)
                
                # Target distribution
                fig = plt.figure(figsize=(8, 6))
                classification_system.y.value_counts().plot(kind='bar')
                plt.title(f'Distribution of Target: {classification_system.target}')
                plt.ylabel('Count')
                plt.xlabel(classification_system.target)
                plt.tight_layout()
                visualizations['target_distribution'] = fig_to_base64(fig)
                
                return VisualizationResponse(
                    success=True,
                    message="Data visualized successfully",
                    visualizations=visualizations
                )
            else:
                return VisualizationResponse(
                    success=False,
                    message="Features and target not defined. Please define features and target first.",
                    visualizations={}
                )
                
        finally:
            # Restore the original backend
            plt.switch_backend(original_backend)
            
    except Exception as e:
        return VisualizationResponse(
            success=False,
            message=f"Error visualizing data: {str(e)}",
            visualizations={}
        )

@router.post("/select-algorithm", response_model=AlgorithmSelectionResponse)
async def select_algorithm(request: AlgorithmSelectionRequest):
    """
    Select and configure the classification algorithm.
    """
    try:
        model_info = classification_system.select_algorithm(request.algorithm_name)
        
        if model_info is None:
            return AlgorithmSelectionResponse(
                success=False,
                message=f"Algorithm '{request.algorithm_name}' not supported.",
                algorithm_name=request.algorithm_name,
                hyperparameters={},
                algorithm_description=""
            )
        
        # Get algorithm description
        algorithm_descriptions = classification_system.get_algorithm_descriptions()
        algorithm_description = algorithm_descriptions.get(request.algorithm_name, "No description available.")
        
        # Prepare hyperparameters with descriptions
        hyperparameters = {}
        for param, values in model_info['params'].items():
            hyperparameters[param] = {
                'description': model_info['description'].get(param, "No description available."),
                'suggested_values': convert_to_native_types(values)
            }
        
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
        model = classification_system.configure_hyperparameters(request.hyperparameters)
        
        if model is None:
            return HyperparameterConfigResponse(
                success=False,
                message="No algorithm selected. Please select an algorithm first.",
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
        X_train, X_test, y_train, y_test = classification_system.preprocess_data()
        
        if X_train is None or X_test is None or y_train is None or y_test is None:
            return PreprocessResponse(
                success=False,
                message="Features and target not defined. Please define features and target first.",
                train_shape=[0, 0],
                test_shape=[0, 0]
            )
        
        return PreprocessResponse(
            success=True,
            message="Data preprocessed successfully",
            train_shape=list(X_train.shape),
            test_shape=list(X_test.shape)
        )
        
    except Exception as e:
        return PreprocessResponse(
            success=False,
            message=f"Error preprocessing data: {str(e)}",
            train_shape=[0, 0],
            test_shape=[0, 0]
        )

@router.post("/train-model", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
    """
    Train the model asynchronously.
    """
    try:
        # Reset training state
        reset_training_state()
        
        # Start training in background
        background_tasks.add_task(train_model_task)
        
        return TrainingResponse(
            success=True,
            message="Model training started",
            training_time=0,
            cross_validation=None
        )
        
    except Exception as e:
        return TrainingResponse(
            success=False,
            message=f"Error starting model training: {str(e)}",
            training_time=0,
            cross_validation=None
        )

@router.get("/check-progress")
async def check_progress():
    """
    Check the progress of model training.
    """
    return {
        "success": True,
        "progress": training_progress,
        "message": training_message,
        "is_complete": training_complete,
        "task_running": is_training,
        "error": training_error
    }

@router.get("/evaluate-model", response_model=EvaluationResponse)
async def evaluate_model():
    """
    Evaluate the trained model on the test set.
    """
    try:
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            metrics = classification_system.evaluate_model()
            
            if metrics is None:
                return EvaluationResponse(
                    success=False,
                    message="No trained model available or test data not available. Please train a model first.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    visualizations={}
                )
            
            # Get confusion matrix
            y_pred = classification_system.trained_model.predict(classification_system.X_test)
            cm = confusion_matrix(classification_system.y_test, y_pred)
            
            # Create confusion matrix visualization
            fig = plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_viz = fig_to_base64(fig)
            
            # Create ROC curve visualization for binary classification
            roc_viz = ""
            if len(np.unique(classification_system.y_test)) == 2 and hasattr(classification_system.trained_model, 'predict_proba'):
                y_prob = classification_system.trained_model.predict_proba(classification_system.X_test)[:, 1]
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(classification_system.y_test, y_prob)
                roc_auc = auc(fpr, tpr)
                
                fig = plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], 'k--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('Receiver Operating Characteristic (ROC) Curve')
                plt.legend(loc="lower right")
                plt.tight_layout()
                roc_viz = fig_to_base64(fig)
            
            # Get classification report
            report = classification_report(classification_system.y_test, y_pred, output_dict=True)
            
            # Convert accuracy from float to dict to match expected structure
            if 'accuracy' in report and isinstance(report['accuracy'], float):
                accuracy_value = report['accuracy']
                report['accuracy'] = {
                    'precision': accuracy_value,
                    'recall': accuracy_value,
                    'f1_score': accuracy_value,
                    'support': len(classification_system.y_test)
                }
            
            return EvaluationResponse(
                success=True,
                message="Model evaluated successfully",
                metrics=convert_to_native_types(metrics),
                classification_report=convert_to_native_types(report),
                confusion_matrix=convert_to_native_types(cm.tolist() if isinstance(cm, np.ndarray) else cm),
                visualizations={
                    'confusion_matrix': cm_viz,
                    'roc_curve': roc_viz
                }
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
            # Make predictions
            results = classification_system.predict_new_data(temp_file_path)
            
            if results is None:
                return PredictionResponse(
                    success=False,
                    message="No trained model available or error in prediction. Please train a model first.",
                    predictions=[],
                    prediction_count=0
                )
            
            return PredictionResponse(
                success=True,
                message="Predictions made successfully",
                predictions=convert_to_native_types(results.to_dict(orient='records')),
                prediction_count=len(results)
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

@router.post("/compare-predictions", response_model=ComparisonResponse)
async def compare_predictions(request: ComparisonRequest):
    """
    Compare predictions with actual values for validation.
    """
    try:
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            # Decode base64 file content
            file_content = base64.b64decode(request.file_content)
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(request.file_name)[1]) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Compare predictions with actual values
                metrics = classification_system.compare_predictions_with_actual(temp_file_path)
                
                if metrics is None:
                    return ComparisonResponse(
                        success=False,
                        message="No trained model available or error in comparison. Please train a model first.",
                        metrics={},
                        classification_report={},
                        confusion_matrix=[],
                        comparison_sample=[],
                        class_accuracy={},
                        visualizations={}
                    )
                
                # Get comparison data
                new_data = pd.read_csv(temp_file_path) if temp_file_path.lower().endswith('.csv') else pd.read_excel(temp_file_path)
                results = classification_system.predict_new_data(new_data)
                
                # Extract actual and predicted values
                y_actual = new_data[classification_system.target]
                y_pred = results[f'predicted_{classification_system.target}']
                
                # Create confusion matrix visualization
                from sklearn.metrics import confusion_matrix
                cm = confusion_matrix(y_actual, y_pred)
                fig = plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title('Confusion Matrix (Validation)')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_viz = fig_to_base64(fig)
                
                # Get classification report
                report = classification_report(y_actual, y_pred, output_dict=True)
                
                # Calculate accuracy by class
                class_accuracy = {}
                for cls in y_actual.unique():
                    mask = y_actual == cls
                    from sklearn.metrics import accuracy_score
                    class_accuracy[str(cls)] = float(accuracy_score(y_actual[mask], y_pred[mask]))
                
                # Prepare comparison sample
                comparison = pd.DataFrame({
                    'Actual': y_actual,
                    'Predicted': y_pred,
                    'Correct': y_actual == y_pred
                })
                
                return ComparisonResponse(
                    success=True,
                    message="Comparison completed successfully",
                    metrics=convert_to_native_types(metrics),
                    classification_report=convert_to_native_types(report),
                    confusion_matrix=convert_to_native_types(cm.tolist()),
                    comparison_sample=convert_to_native_types(comparison.head(20).to_dict(orient='records')),
                    class_accuracy=convert_to_native_types(class_accuracy),
                    visualizations={
                        'confusion_matrix': cm_viz
                    }
                )
                
            finally:
                # Clean up the temporary file
                os.unlink(temp_file_path)
                
        finally:
            # Clean up
            plt.close('all')
            
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

@router.post("/grid-search", response_model=GridSearchResponse)
async def perform_grid_search(request: GridSearchRequest):
    """
    Perform grid search for hyperparameter tuning.
    """
    try:
        best_params = classification_system.perform_grid_search(request.param_grid)
        
        if best_params is None:
            return GridSearchResponse(
                success=False,
                message="No algorithm selected or data not preprocessed. Please select an algorithm and preprocess the data first.",
                best_params={},
                best_score=0.0,
                search_time=0.0
            )
        
        return GridSearchResponse(
            success=True,
            message="Grid search completed successfully",
            best_params=convert_to_native_types(best_params),
            best_score=0.0,  # We don't have the actual best score here
            search_time=0.0  # We don't have the actual search time here
        )
        
    except Exception as e:
        return GridSearchResponse(
            success=False,
            message=f"Error performing grid search: {str(e)}",
            best_params={},
            best_score=0.0,
            search_time=0.0
        )

@router.post("/save-model", response_model=ModelSaveResponse)
async def save_model(request: ModelSaveRequest):
    """
    Save the trained model to disk.
    """
    try:
        if not classification_system.trained_model:
            return ModelSaveResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                model_path="",
                timestamp=""
            )
        
        # Create directory if it doesn't exist
        os.makedirs(request.save_directory, exist_ok=True)
        
        # Save the model
        model_path = classification_system.save_model(request.save_directory)
        
        if not model_path:
            return ModelSaveResponse(
                success=False,
                message="Failed to save model",
                model_path="",
                timestamp=""
            )
        
        return ModelSaveResponse(
            success=True,
            message="Model saved successfully",
            model_path=model_path,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    Get descriptions of all supported algorithms.
    """
    try:
        descriptions = classification_system.get_algorithm_descriptions()
        return {"descriptions": descriptions}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting algorithm descriptions: {str(e)}"
        )
