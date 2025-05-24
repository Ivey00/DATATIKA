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

from app.core.regression import Regression
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
    GridSearchRequest, GridSearchResponse,
    ModelSaveResponse,
    ErrorResponse,
    convert_to_native_types
)

router = APIRouter()

# Global instance of the Regression class
# In a production environment, this would be handled differently
# with a database to store session state
regression_system = Regression()

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
            data = regression_system.load_data(temp_file_path)
            
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
        if regression_system.data is None:
            raise HTTPException(
                status_code=400,
                detail="No data available. Please import data first."
            )
        
        # Analyze column information
        column_info = {}
        for column in regression_system.data.columns:
            dtype = str(regression_system.data[column].dtype)
            unique_values = regression_system.data[column].nunique()
            missing_values = regression_system.data[column].isnull().sum()
            
            # Suggest column type based on data
            if pd.api.types.is_numeric_dtype(regression_system.data[column]):
                suggested_type = "numerical"
            elif pd.api.types.is_datetime64_any_dtype(regression_system.data[column]):
                suggested_type = "datetime"
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
        regression_system.set_features(
            target_column=request.target,
            feature_columns=request.features,
            numerical_features=request.numerical_features,
            categorical_features=request.categorical_features,
            datetime_features=[],  # Add datetime features if needed
            machine_id_column=request.item_id_column
        )
        
        if regression_system.X is None or regression_system.y is None:
            return FeatureDefinitionResponse(
                success=False,
                message="Failed to define features and target. Please import data first.",
                features=request.features,
                target=request.target,
                categorical_features=request.categorical_features or [],
                numerical_features=request.numerical_features or []
            )
        
        return FeatureDefinitionResponse(
            success=True,
            message="Features and target defined successfully",
            features=regression_system.feature_names,
            target=regression_system.target_name,
            categorical_features=regression_system.categorical_features,
            numerical_features=regression_system.numerical_features,
            item_id_column=regression_system.machine_id_column
        )
        
    except Exception as e:
        return FeatureDefinitionResponse(
            success=False,
            message=f"Error defining features and target: {str(e)}",
            features=request.features,
            target=request.target,
            categorical_features=request.categorical_features or [],
            numerical_features=request.numerical_features or []
        )

@router.post("/filter-by-item", response_model=ItemFilterResponse)
async def filter_by_item(request: ItemFilterRequest):
    """
    Filter the dataset to only include rows with the specified item ID.
    """
    try:
        if regression_system.data is None or regression_system.machine_id_column is None:
            return ItemFilterResponse(
                success=False,
                message="Failed to filter data. Please define features and target first, and ensure the item ID column is specified.",
                item_id_column="",
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        # Filter data
        filtered_data = regression_system.data[regression_system.data[regression_system.machine_id_column] == request.item_id_value]
        
        if filtered_data.empty:
            return ItemFilterResponse(
                success=False,
                message=f"No data found for {regression_system.machine_id_column} = {request.item_id_value}",
                item_id_column=regression_system.machine_id_column,
                item_id_value=request.item_id_value,
                filtered_shape=[0, 0]
            )
        
        # Update the regression system with filtered data
        regression_system.data = filtered_data
        
        # Reapply feature definition if already defined
        if regression_system.feature_names and regression_system.target_name:
            regression_system.set_features(
                target_column=regression_system.target_name,
                feature_columns=regression_system.feature_names,
                numerical_features=regression_system.numerical_features,
                categorical_features=regression_system.categorical_features,
                datetime_features=[],  # Add datetime features if needed
                machine_id_column=regression_system.machine_id_column
            )
        
        return ItemFilterResponse(
            success=True,
            message=f"Data filtered successfully for {regression_system.machine_id_column} = {request.item_id_value}",
            item_id_column=regression_system.machine_id_column,
            item_id_value=request.item_id_value,
            filtered_shape=list(filtered_data.shape)
        )
        
    except Exception as e:
        return ItemFilterResponse(
            success=False,
            message=f"Error filtering data: {str(e)}",
            item_id_column=regression_system.machine_id_column or "",
            item_id_value=request.item_id_value,
            filtered_shape=[0, 0]
        )

@router.get("/visualize-data", response_model=VisualizationResponse)
async def visualize_data(plot_type: str = "numerical", max_features: Optional[int] = 10):
    """
    Visualize the data with appropriate plots for each feature type.
    """
    try:
        # Add a timeout for the visualization process
        from starlette.background import BackgroundTask
        
        # Redirect matplotlib output to capture visualizations
        visualizations = {}
        
        # Save the original backend
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')
        
        try:
            if regression_system.X is None or regression_system.y is None:
                return VisualizationResponse(
                    success=False,
                    message="Features and target not defined. Please define features and target first.",
                    visualizations={}
                )
            
            # Determine which features to visualize based on plot_type
            # Allow both "numerical" and "numeric" for backward compatibility
            if plot_type.lower() in ["numerical", "numeric"]:
                # Limit the data size for visualization to prevent timeouts
                max_rows = min(10000, len(regression_system.data))
                data_sample = regression_system.data.sample(n=max_rows) if len(regression_system.data) > max_rows else regression_system.data
                
                features_to_viz = regression_system.numerical_features[:max_features] if len(regression_system.numerical_features) > max_features else regression_system.numerical_features
                
                # Create visualizations for numerical features with lower resolution
                for feature in features_to_viz:
                    # Distribution plot with reduced complexity
                    fig = plt.figure(figsize=(8, 5), dpi=80)  # Lower DPI
                    sns.histplot(data=data_sample, x=feature, bins=min(50, int(max_rows/100)))
                    plt.title(f'Distribution of {feature}')
                    plt.tight_layout()
                    visualizations[f'distribution_{feature}'] = fig_to_base64(fig)
                    
                    # Release memory
                    plt.close(fig)
                    
                    # Scatter plot with reduced number of points
                    fig = plt.figure(figsize=(8, 5), dpi=80)  # Lower DPI
                    sns.scatterplot(data=data_sample.sample(n=min(1000, len(data_sample))), x=feature, y=regression_system.target_name, alpha=0.5)
                    plt.title(f'{feature} vs {regression_system.target_name}')
                    plt.tight_layout()
                    visualizations[f'scatter_{feature}'] = fig_to_base64(fig)
                    
                    # Release memory
                    plt.close(fig)
                
                # Return the visualizations
                return VisualizationResponse(
                    success=True,
                    message="Numerical visualizations generated successfully",
                    visualizations=visualizations
                )
            
            elif plot_type == "categorical":
                # Limit the data size for visualization to prevent timeouts
                max_rows = min(10000, len(regression_system.data))
                data_sample = regression_system.data.sample(n=max_rows) if len(regression_system.data) > max_rows else regression_system.data
                
                features_to_viz = regression_system.categorical_features[:max_features] if len(regression_system.categorical_features) > max_features else regression_system.categorical_features
                
                if not features_to_viz:
                    return VisualizationResponse(
                        success=False,
                        message="No categorical features defined. Please define categorical features first.",
                        visualizations={}
                    )
                
                # Create visualizations for categorical features
                for feature in features_to_viz:
                    try:
                        # Count plot
                        fig = plt.figure(figsize=(10, 6), dpi=80)
                        sns.countplot(data=data_sample, x=feature)
                        plt.title(f'Distribution of {feature}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        visualizations[f'countplot_{feature}'] = fig_to_base64(fig)
                        
                        # Release memory
                        plt.close(fig)
                        
                        # Box plot with target
                        fig = plt.figure(figsize=(10, 6), dpi=80)
                        sns.boxplot(data=data_sample, x=feature, y=regression_system.target_name)
                        plt.title(f'{regression_system.target_name} by {feature}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        visualizations[f'boxplot_{feature}'] = fig_to_base64(fig)
                        
                        # Release memory
                        plt.close(fig)
                    except Exception as e:
                        # Log the error but continue with other features
                        print(f"Error generating plot for feature {feature}: {str(e)}")
                        continue
                
                # Return the visualizations if any were generated
                if visualizations:
                    return VisualizationResponse(
                        success=True,
                        message="Categorical visualizations generated successfully",
                        visualizations=visualizations
                    )
                else:
                    return VisualizationResponse(
                        success=False,
                        message="Failed to generate categorical visualizations. Please check your categorical features.",
                        visualizations={}
                    )
            
            elif plot_type == "time":
                # Check if there are any datetime features defined
                if not regression_system.datetime_features or len(regression_system.datetime_features) == 0:
                    # Try to identify potential datetime columns
                    potential_datetime_cols = []
                    for col in regression_system.data.columns:
                        if col in regression_system.feature_names:
                            try:
                                # Try to convert to datetime
                                pd.to_datetime(regression_system.data[col])
                                potential_datetime_cols.append(col)
                            except:
                                pass
                    
                    if not potential_datetime_cols:
                        return VisualizationResponse(
                            success=False,
                            message="No datetime features defined or detected. Please define datetime features first.",
                            visualizations={}
                        )
                    
                    # Use the detected datetime columns
                    datetime_features = potential_datetime_cols
                else:
                    datetime_features = regression_system.datetime_features
                
                # Limit the data size for visualization to prevent timeouts
                max_rows = min(10000, len(regression_system.data))
                data_sample = regression_system.data.sample(n=max_rows) if len(regression_system.data) > max_rows else regression_system.data
                
                # Create time series visualizations
                for feature in datetime_features[:max_features]:
                    # Convert to datetime if not already
                    try:
                        # Create a copy to avoid modifying the original data
                        temp_data = data_sample.copy()
                        temp_data[feature] = pd.to_datetime(temp_data[feature])
                        
                        # Time series plot
                        fig = plt.figure(figsize=(10, 6), dpi=80)
                        temp_data.sort_values(by=feature, inplace=True)
                        plt.plot(temp_data[feature], temp_data[regression_system.target_name])
                        plt.title(f'{regression_system.target_name} over {feature}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        visualizations[f'timeseries_{feature}'] = fig_to_base64(fig)
                        
                        # Release memory
                        plt.close(fig)
                    except Exception as e:
                        # Skip this feature if there's an error
                        continue
                
                # Return the visualizations
                return VisualizationResponse(
                    success=True,
                    message="Time series visualizations generated successfully",
                    visualizations=visualizations
                )
            
            # If we get here, the plot_type wasn't handled
            # Normalize plot_type for better error messages
            supported_types = "'numerical' (or 'numeric'), 'categorical', and 'time'"
            return VisualizationResponse(
                success=False,
                message=f"Unsupported plot type: {plot_type}. Supported types are {supported_types}.",
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

# @router.get("/analyze-feature-importance", response_model=FeatureImportanceResponse)
# async def analyze_feature_importance(method: str = "shap"):
#     """
#     Analyze the importance of features using the specified method.
#     """
#     try:
#         # Redirect matplotlib output to capture visualizations
#         plt.switch_backend('Agg')
        
#         try:
#             if regression_system.X is None or regression_system.y is None:
#                 return FeatureImportanceResponse(
#                     success=False,
#                     message="Features and target not defined. Please define features and target first.",
#                     importance_data=[],
#                     visualization=""
#                 )
            
#             # Ensure preprocessor is created
#             if regression_system.preprocessor is None:
#                 regression_system.preprocess_data()
            
#             try:
#                 # Create a RandomForestRegressor model
#                 from sklearn.ensemble import RandomForestRegressor
#                 from sklearn.pipeline import Pipeline
                
#                 # Create a pipeline with preprocessor and model
#                 model = RandomForestRegressor(n_estimators=100, random_state=42)
#                 pipeline = Pipeline([
#                     ('preprocessor', regression_system.preprocessor),
#                     ('model', model)
#                 ])
                
#                 # Fit the pipeline
#                 pipeline.fit(regression_system.X, regression_system.y)
                
#                 # Get feature names after preprocessing
#                 feature_names = []
                
#                 # Add numerical feature names
#                 if regression_system.numerical_features:
#                     feature_names.extend(regression_system.numerical_features)
                
#                 # Add one-hot encoded categorical feature names
#                 if regression_system.categorical_features:
#                     try:
#                         # Get the categorical transformer
#                         cat_transformer = regression_system.preprocessor.named_transformers_['cat']
#                         # Get the one-hot encoder
#                         one_hot = cat_transformer.named_steps['onehot']
#                         # Get the categories
#                         for i, feature in enumerate(regression_system.categorical_features):
#                             categories = one_hot.categories_[i]
#                             for category in categories:
#                                 feature_names.append(f"{feature}_{category}")
#                     except Exception as e:
#                         # If there's an error getting categorical feature names, log it but continue
#                         print(f"Error getting categorical feature names: {str(e)}")
                
#                 # Calculate feature importances from the model
#                 importances = model.feature_importances_
                
#                 # Create a DataFrame with feature importance
#                 importance_df = pd.DataFrame({
#                     'Feature': feature_names[:len(importances)],  # Ensure lengths match
#                     'Importance': importances
#                 }).sort_values('Importance', ascending=False)
                
#                 # Create visualization
#                 fig = plt.figure(figsize=(10, 8))
#                 sns.barplot(x='Importance', y='Feature', data=importance_df)
#                 plt.title(f'Feature Importance using Random Forest')
#                 plt.tight_layout()
                
#                 visualization = fig_to_base64(fig)
                
#                 return FeatureImportanceResponse(
#                     success=True,
#                     message="Feature importance analyzed successfully",
#                     importance_data=convert_to_native_types(importance_df.to_dict(orient='records')),
#                     visualization=visualization
#                 )
#             except Exception as e:
#                 return FeatureImportanceResponse(
#                     success=False,
#                     message=f"Error parsing feature importance: {str(e)}",
#                     importance_data=[],
#                     visualization=""
#                 )
            
#         finally:
#             # Clean up
#             plt.close('all')
            
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
    Select the regression algorithm.
    """
    try:
        if regression_system.available_models is None or request.algorithm_name not in regression_system.available_models:
            return AlgorithmSelectionResponse(
                success=False,
                message=f"Algorithm '{request.algorithm_name}' not supported.",
                algorithm_name=request.algorithm_name,
                hyperparameters={},
                algorithm_description=""
            )
        
        # Get algorithm description
        algorithm_description = f"{request.algorithm_name} is a regression algorithm for predicting continuous values."
        
        # Get hyperparameter descriptions
        hyperparameters = {}
        if request.algorithm_name in regression_system.hyperparameter_descriptions:
            for param, description in regression_system.hyperparameter_descriptions[request.algorithm_name].items():
                # Define suggested values based on parameter type
                suggested_values = []
                
                if param == "n_estimators":
                    suggested_values = [10, 50, 100, 200, 500]
                elif param == "max_depth":
                    suggested_values = [None, 3, 5, 10, 15, 20]
                elif param == "learning_rate":
                    suggested_values = [0.01, 0.05, 0.1, 0.2, 0.5]
                elif param == "alpha":
                    suggested_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
                elif param == "l1_ratio":
                    suggested_values = [0.0, 0.25, 0.5, 0.75, 1.0]
                elif param == "min_samples_split":
                    suggested_values = [2, 5, 10, 20]
                elif param == "min_samples_leaf":
                    suggested_values = [1, 2, 4, 8]
                elif param == "n_neighbors":
                    suggested_values = [3, 5, 7, 9, 11]
                elif param == "weights":
                    suggested_values = ["uniform", "distance"]
                elif param == "algorithm":
                    suggested_values = ["auto", "ball_tree", "kd_tree", "brute"]
                elif param == "solver":
                    suggested_values = ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]
                elif param == "fit_intercept":
                    suggested_values = [True, False]
                elif param == "normalize":
                    suggested_values = [True, False]
                elif param == "n_jobs":
                    suggested_values = [None, 1, -1]
                elif param == "criterion":
                    suggested_values = ["mse", "friedman_mse", "mae"]
                elif param == "splitter":
                    suggested_values = ["best", "random"]
                elif param == "loss":
                    suggested_values = ["linear", "square", "huber", "epsilon_insensitive"]
                elif param == "num_leaves":
                    suggested_values = [31, 50, 100, 200]
                else:
                    suggested_values = ["Please specify a value"]
                
                hyperparameters[param] = {
                    "description": description,
                    "suggested_values": suggested_values
                }
        
        # Set the selected algorithm
        regression_system.model_name = request.algorithm_name
        
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
        if regression_system.model_name is None:
            return HyperparameterConfigResponse(
                success=False,
                message="No algorithm selected. Please select an algorithm first.",
                hyperparameters={}
            )
        
        # Store hyperparameters
        regression_system.hyperparameters = request.hyperparameters
        
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
        if regression_system.X is None or regression_system.y is None:
            return PreprocessResponse(
                success=False,
                message="Features and target not defined. Please define features and target first.",
                train_shape=[0, 0],
                test_shape=[0, 0]
            )
        
        # Preprocess data
        preprocessor = regression_system.preprocess_data()
        
        if preprocessor is None:
            return PreprocessResponse(
                success=False,
                message="Failed to preprocess data.",
                train_shape=[0, 0],
                test_shape=[0, 0]
            )
        
        # For simplicity, we'll use a 80/20 train/test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            regression_system.X, regression_system.y, test_size=0.2, random_state=42
        )
        
        return PreprocessResponse(
            success=True,
            message="Data preprocessed successfully",
            train_shape=[X_train.shape[0], len(regression_system.feature_names)],
            test_shape=[X_test.shape[0], len(regression_system.feature_names)]
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
    Train the model with the selected algorithm and configured hyperparameters.
    """
    try:
        if regression_system.model_name is None or regression_system.X is None or regression_system.y is None:
            return TrainingResponse(
                success=False,
                message="No algorithm selected or features not defined. Please select an algorithm and define features first.",
                training_time=0.0
            )
        
        # Train the model
        import time
        start_time = time.time()
        
        metrics = regression_system.train_model(
            model_name=regression_system.model_name,
            hyperparameters=regression_system.hyperparameters or {}
        )
        
        training_time = time.time() - start_time
        
        if metrics is None:
            return TrainingResponse(
                success=False,
                message="Failed to train model.",
                training_time=0.0
            )
        
        return TrainingResponse(
            success=True,
            message="Model trained successfully",
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
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            if regression_system.trained_model is None:
                return EvaluationResponse(
                    success=False,
                    message="No trained model available. Please train a model first.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    visualizations={}
                )
            
            # Split data for evaluation
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                regression_system.X, regression_system.y, test_size=0.2, random_state=42
            )
            
            # Make predictions
            y_pred = regression_system.trained_model.predict(X_test)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            metrics = {
                "mse": mse,
                "rmse": rmse,
                "mae": mae,
                "r2": r2
            }
            
            # Create visualizations
            visualizations = {}
            
            # Actual vs Predicted plot
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(y_test, y_pred, alpha=0.5)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title('Actual vs Predicted Values')
            plt.tight_layout()
            visualizations['actual_vs_predicted'] = fig_to_base64(fig)
            
            # Residuals plot
            residuals = y_test - y_pred
            fig = plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.5)
            plt.hlines(y=0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--', lw=2)
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title('Residuals Plot')
            plt.tight_layout()
            visualizations['residuals'] = fig_to_base64(fig)
            
            # Residuals distribution
            fig = plt.figure(figsize=(10, 6))
            sns.histplot(residuals, kde=True)
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title('Residuals Distribution')
            plt.tight_layout()
            visualizations['residuals_distribution'] = fig_to_base64(fig)
            
            # Create a dummy classification report and confusion matrix for compatibility
            # with the existing EvaluationResponse model
            dummy_classification_report = {
                "regression": {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
            }
            
            dummy_confusion_matrix = [[0]]
            
            return EvaluationResponse(
                success=True,
                message="Model evaluated successfully",
                metrics=convert_to_native_types(metrics),
                classification_report=convert_to_native_types(dummy_classification_report),
                confusion_matrix=convert_to_native_types(dummy_confusion_matrix),
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
        if regression_system.trained_model is None:
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
            # Load the new data
            if temp_file_path.endswith('.csv'):
                new_data = pd.read_csv(temp_file_path)
            elif temp_file_path.endswith(('.xlsx', '.xls')):
                new_data = pd.read_excel(temp_file_path)
            else:
                return PredictionResponse(
                    success=False,
                    message="Unsupported file format. Please use .csv or .xlsx files.",
                    predictions=[],
                    prediction_count=0
                )
            
            # Check if all required features are present
            missing_features = [f for f in regression_system.feature_names if f not in new_data.columns]
            if missing_features:
                return PredictionResponse(
                    success=False,
                    message=f"Missing features in the prediction data: {', '.join(missing_features)}",
                    predictions=[],
                    prediction_count=0
                )
            
            # Make predictions
            predictions = regression_system.predict(new_data[regression_system.feature_names])
            
            # Add predictions to the original data
            new_data[f'predicted_{regression_system.target_name}'] = predictions
            
            return PredictionResponse(
                success=True,
                message="Predictions made successfully",
                predictions=convert_to_native_types(new_data.to_dict(orient='records')),
                prediction_count=len(new_data)
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
            if regression_system.trained_model is None:
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
                # Load the new data
                if temp_file_path.endswith('.csv'):
                    new_data = pd.read_csv(temp_file_path)
                elif temp_file_path.endswith(('.xlsx', '.xls')):
                    new_data = pd.read_excel(temp_file_path)
                else:
                    return ComparisonResponse(
                        success=False,
                        message="Unsupported file format. Please use .csv or .xlsx files.",
                        metrics={},
                        classification_report={},
                        confusion_matrix=[],
                        comparison_sample=[],
                        class_accuracy={},
                        visualizations={}
                    )
                
                # Check if all required features and target are present
                missing_features = [f for f in regression_system.feature_names if f not in new_data.columns]
                if missing_features:
                    return ComparisonResponse(
                        success=False,
                        message=f"Missing features in the comparison data: {', '.join(missing_features)}",
                        metrics={},
                        classification_report={},
                        confusion_matrix=[],
                        comparison_sample=[],
                        class_accuracy={},
                        visualizations={}
                    )
                
                if regression_system.target_name not in new_data.columns:
                    return ComparisonResponse(
                        success=False,
                        message=f"Missing target column '{regression_system.target_name}' in the comparison data",
                        metrics={},
                        classification_report={},
                        confusion_matrix=[],
                        comparison_sample=[],
                        class_accuracy={},
                        visualizations={}
                    )
                
                # Make predictions
                predictions = regression_system.predict(new_data[regression_system.feature_names])
                
                # Calculate metrics
                from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                mse = mean_squared_error(new_data[regression_system.target_name], predictions)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(new_data[regression_system.target_name], predictions)
                r2 = r2_score(new_data[regression_system.target_name], predictions)
                
                metrics = {
                    "mse": mse,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2
                }
                
                # Create comparison DataFrame
                comparison_df = pd.DataFrame({
                    'Actual': new_data[regression_system.target_name],
                    'Predicted': predictions,
                    'Absolute_Error': np.abs(new_data[regression_system.target_name] - predictions)
                })
                
                # Create visualizations
                visualizations = {}
                
                # Actual vs Predicted plot
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(new_data[regression_system.target_name], predictions, alpha=0.5)
                plt.plot([new_data[regression_system.target_name].min(), new_data[regression_system.target_name].max()], 
                        [new_data[regression_system.target_name].min(), new_data[regression_system.target_name].max()], 
                        'r--', lw=2)
                plt.xlabel('Actual Values')
                plt.ylabel('Predicted Values')
                plt.title('Actual vs Predicted Values')
                plt.tight_layout()
                visualizations['actual_vs_predicted'] = fig_to_base64(fig)
                
                # Residuals plot
                residuals = new_data[regression_system.target_name] - predictions
                fig = plt.figure(figsize=(10, 6))
                plt.scatter(predictions, residuals, alpha=0.5)
                plt.hlines(y=0, xmin=predictions.min(), xmax=predictions.max(), colors='r', linestyles='--', lw=2)
                plt.xlabel('Predicted Values')
                plt.ylabel('Residuals')
                plt.title('Residuals Plot')
                plt.tight_layout()
                visualizations['residuals'] = fig_to_base64(fig)
                
                # Create a dummy classification report and confusion matrix for compatibility
                # with the existing ComparisonResponse model
                dummy_classification_report = {
                    "regression": {
                        "mse": mse,
                        "rmse": rmse,
                        "mae": mae,
                        "r2": r2
                    }
                }
                
                dummy_confusion_matrix = [[0]]
                
                # Create a dummy class_accuracy for compatibility
                dummy_class_accuracy = {
                    "regression": r2
                }
                
                return ComparisonResponse(
                    success=True,
                    message="Comparison completed successfully",
                    metrics=convert_to_native_types(metrics),
                    classification_report=convert_to_native_types(dummy_classification_report),
                    confusion_matrix=convert_to_native_types(dummy_confusion_matrix),
                    comparison_sample=convert_to_native_types(comparison_df.head(20).to_dict(orient='records')),
                    class_accuracy=convert_to_native_types(dummy_class_accuracy),
                    visualizations=visualizations
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
        if regression_system.model_name is None or regression_system.X is None or regression_system.y is None:
            return GridSearchResponse(
                success=False,
                message="No algorithm selected or features not defined. Please select an algorithm and define features first.",
                best_params={},
                best_score=0.0,
                search_time=0.0
            )
        
        # Perform grid search
        import time
        start_time = time.time()
        
        from sklearn.model_selection import GridSearchCV
        from sklearn.pipeline import Pipeline
        
        # Create model instance
        model = regression_system.available_models[regression_system.model_name]()
        
        # Create pipeline with preprocessor
        pipeline = Pipeline([
            ('preprocessor', regression_system.preprocessor),
            ('model', model)
        ])
        
        # Perform grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid={'model__' + key: value for key, value in request.param_grid.items()},
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(regression_system.X, regression_system.y)
        
        search_time = time.time() - start_time
        
        # Extract best parameters (remove 'model__' prefix)
        best_params = {key.replace('model__', ''): value for key, value in grid_search.best_params_.items()}
        
        return GridSearchResponse(
            success=True,
            message="Grid search completed successfully",
            best_params=convert_to_native_types(best_params),
            best_score=-grid_search.best_score_,  # Convert negative MSE back to positive
            search_time=search_time
        )
        
    except Exception as e:
        return GridSearchResponse(
            success=False,
            message=f"Error performing grid search: {str(e)}",
            best_params={},
            best_score=0.0,
            search_time=0.0
        )

@router.get("/save-model", response_model=ModelSaveResponse)
async def save_model(directory: str = "models"):
    """
    Save the trained model and preprocessor to disk.
    """
    try:
        if regression_system.trained_model is None:
            return ModelSaveResponse(
                success=False,
                message="No trained model available. Please train a model first.",
                model_path="",
                timestamp=""
            )
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{regression_system.model_name}_{timestamp}.joblib"
        filepath = os.path.join(directory, filename)
        
        # Save model
        regression_system.save_model(filepath)
        
        return ModelSaveResponse(
            success=True,
            message="Model saved successfully",
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

@router.get("/algorithm-descriptions")
async def get_algorithm_descriptions():
    """
    Get descriptions of all supported algorithms.
    """
    try:
        descriptions = {}
        for algo_name in regression_system.available_models.keys():
            descriptions[algo_name] = f"{algo_name} is a regression algorithm for predicting continuous values."
        
        return {"descriptions": descriptions}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting algorithm descriptions: {str(e)}"
        )