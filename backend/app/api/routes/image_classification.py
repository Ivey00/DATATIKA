import base64
import io
import os
import tempfile
import zipfile
import shutil
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks, Form
from fastapi.responses import FileResponse
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

from app.core.image_classification import IndustrialVisionApp, DataManager, ModelManager
from app.models.schema import (
    CustomBaseModel,
    VisualizationResponse,
    AlgorithmSelectionRequest, AlgorithmSelectionResponse,
    HyperparameterConfigRequest, HyperparameterConfigResponse,
    TrainingResponse,
    EvaluationResponse,
    ErrorResponse,
    convert_to_native_types
)

router = APIRouter()

# Global instance of the IndustrialVisionApp
# In a production environment, this would be handled differently
# with a database to store session state
vision_app = IndustrialVisionApp()

# Define Pydantic models for request/response
class ImageDimensionsRequest(CustomBaseModel):
    width: int
    height: int
    channels: int = 3

class ImageDimensionsResponse(CustomBaseModel):
    success: bool
    message: str
    width: int
    height: int
    channels: int

class DatasetUploadResponse(CustomBaseModel):
    success: bool
    message: str
    extracted_path: Optional[str] = None
    classes: Optional[List[str]] = None
    image_counts: Optional[Dict[str, int]] = None
    total_images: Optional[int] = None

class SampleDatasetRequest(CustomBaseModel):
    source_directories: List[str]
    target_directories: List[str]
    samples_per_class: int = 1000

class SampleDatasetResponse(CustomBaseModel):
    success: bool
    message: str
    sampled_counts: Optional[Dict[str, int]] = None
    total_sampled: Optional[int] = None

class ModelResponse(CustomBaseModel):
    success: bool
    message: str
    available_models: List[str]

class PredictionResponse(CustomBaseModel):
    success: bool
    message: str
    prediction: Optional[str] = None
    confidence: Optional[float] = None
    
def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return img_str

def extract_zip_file(file_path, extract_to):
    """
    Extract a zip file to the specified directory
    """
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    return extract_to

def count_images_in_directory(directory):
    """
    Count images in each subdirectory that contains images (actual classes)
    """
    counts = {}
    total = 0
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # First, check if there's a main directory inside the extracted path
    # (e.g., "dataset" directory containing class directories)
    main_dirs = [d for d in os.listdir(directory) 
                if os.path.isdir(os.path.join(directory, d))]
    
    # If there's only one directory and it contains subdirectories, it's likely the main directory
    main_dir_path = directory
    if len(main_dirs) == 1:
        potential_main_dir = os.path.join(directory, main_dirs[0])
        sub_dirs = [d for d in os.listdir(potential_main_dir) 
                   if os.path.isdir(os.path.join(potential_main_dir, d))]
        if sub_dirs:  # If it contains subdirectories, use it as the main directory
            main_dir_path = potential_main_dir
    
    # Get all directories within the main directory
    all_dirs = []
    for item in os.listdir(main_dir_path):
        item_path = os.path.join(main_dir_path, item)
        if os.path.isdir(item_path):
            all_dirs.append(item_path)
    
    # Check each directory to see if it contains images - only directories with images are classes
    for dir_path in all_dirs:
        dir_name = os.path.basename(dir_path)
        image_files = [f for f in os.listdir(dir_path) 
                      if os.path.isfile(os.path.join(dir_path, f)) and 
                      any(f.lower().endswith(ext) for ext in image_extensions)]
        
        image_count = len(image_files)
        if image_count > 0:  # Only include directories that contain images
            counts[dir_name] = image_count
            total += image_count
    
    return counts, total

@router.post("/set-dimensions", response_model=ImageDimensionsResponse)
async def set_image_dimensions(request: ImageDimensionsRequest):
    """
    Set the dimensions for image processing
    """
    try:
        # Create a new IndustrialVisionApp with the specified dimensions
        global vision_app
        vision_app = IndustrialVisionApp(
            img_width=request.width,
            img_height=request.height,
            channels=request.channels
        )
        
        return ImageDimensionsResponse(
            success=True,
            message=f"Image dimensions set to {request.width}x{request.height}x{request.channels}",
            width=request.width,
            height=request.height,
            channels=request.channels
        )
    except Exception as e:
        return ImageDimensionsResponse(
            success=False,
            message=f"Error setting image dimensions: {str(e)}",
            width=0,
            height=0,
            channels=0
        )

@router.post("/upload-dataset", response_model=DatasetUploadResponse)
async def upload_dataset(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    """
    Upload and extract a zip file containing image dataset
    """
    try:
        # Check if the file is a zip file
        if not file.filename.endswith('.zip'):
            return DatasetUploadResponse(
                success=False,
                message="Uploaded file must be a zip file"
            )
        
        # Create a temporary directory for extraction
        temp_dir = tempfile.mkdtemp()
        temp_zip_path = os.path.join(temp_dir, file.filename)
        
        # Save the uploaded file
        with open(temp_zip_path, "wb") as buffer:
            contents = await file.read()
            buffer.write(contents)
        
        # Extract the zip file
        extract_path = os.path.join(temp_dir, "extracted")
        os.makedirs(extract_path, exist_ok=True)
        extract_zip_file(temp_zip_path, extract_path)
        
        # Check if there's a main directory (e.g. "dataset") inside the extracted path
        main_dirs = [d for d in os.listdir(extract_path) 
                    if os.path.isdir(os.path.join(extract_path, d))]
        
        main_dir_path = extract_path
        if len(main_dirs) == 1:
            potential_main_dir = os.path.join(extract_path, main_dirs[0])
            sub_dirs = [d for d in os.listdir(potential_main_dir) 
                      if os.path.isdir(os.path.join(potential_main_dir, d))]
            if sub_dirs:  # If it contains subdirectories, use it as the main directory
                main_dir_path = potential_main_dir
        
        # Count images in each class (only directories with images)
        image_counts, total_images = count_images_in_directory(extract_path)
        
        # Get only directories that contain images (actual classes)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        class_dirs = []
        
        if main_dir_path == extract_path:
            for dir_name in main_dirs:
                dir_path = os.path.join(main_dir_path, dir_name)
                has_images = any(any(f.lower().endswith(ext) for ext in image_extensions) 
                                for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
                if has_images:
                    class_dirs.append(dir_name)
        else:
            for dir_name in os.listdir(main_dir_path):
                dir_path = os.path.join(main_dir_path, dir_name)
                if os.path.isdir(dir_path):
                    has_images = any(any(f.lower().endswith(ext) for ext in image_extensions) 
                                    for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)))
                    if has_images:
                        class_dirs.append(dir_name)
        
        return DatasetUploadResponse(
            success=True,
            message=f"Dataset uploaded and extracted successfully. Found {len(class_dirs)} classes with {total_images} total images.",
            extracted_path=main_dir_path,  # Use the main directory path
            classes=class_dirs,
            image_counts=image_counts,
            total_images=total_images
        )
        
    except Exception as e:
        return DatasetUploadResponse(
            success=False,
            message=f"Error uploading dataset: {str(e)}"
        )

@router.post("/sample-dataset", response_model=SampleDatasetResponse)
async def sample_dataset(request: SampleDatasetRequest):
    """
    Sample a subset of images from source directories
    """
    try:
        sampled_counts = {}
        total_sampled = 0
        
        # First check if source directories contain images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        valid_source_dirs = []
        valid_target_dirs = []
        
        for i in range(len(request.source_directories)):
            source_dir = request.source_directories[i]
            target_dir = request.target_directories[i]
            
            # Check if the directory contains images
            has_images = False
            for file in os.listdir(source_dir):
                file_path = os.path.join(source_dir, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                    has_images = True
                    break
            
            if has_images:
                valid_source_dirs.append(source_dir)
                valid_target_dirs.append(target_dir)
        
        # Process only valid directories that contain images
        for i in range(len(valid_source_dirs)):
            source_dir = valid_source_dirs[i]
            target_dir = valid_target_dirs[i]
            
            count = vision_app.sample_dataset(
                source_dir,
                target_dir,
                request.samples_per_class
            )
            
            dir_name = os.path.basename(source_dir)
            sampled_counts[dir_name] = count
            total_sampled += count
        
        return SampleDatasetResponse(
            success=True,
            message=f"Sampled {total_sampled} images across {len(valid_source_dirs)} classes",
            sampled_counts=sampled_counts,
            total_sampled=total_sampled
        )
        
    except Exception as e:
        return SampleDatasetResponse(
            success=False,
            message=f"Error sampling dataset: {str(e)}"
        )

@router.post("/load-dataset", response_model=DatasetUploadResponse)
async def load_dataset(data_dir: str = Form(...), categories: List[str] = Form(...)):
    """
    Load the dataset from specified directory and categories
    """
    try:
        # Check if the data_dir is the extracted root or the main directory
        # We want to make sure we're using the directory that contains the class folders
        main_dir_path = data_dir
        
        # Try to detect if there's a nested structure we need to navigate
        main_dirs = [d for d in os.listdir(data_dir) 
                    if os.path.isdir(os.path.join(data_dir, d))]
                    
        # If categories aren't found directly in data_dir, check if they're in a subdirectory
        categories_found = all(c in main_dirs for c in categories)
        
        if not categories_found and len(main_dirs) == 1:
            potential_main_dir = os.path.join(data_dir, main_dirs[0])
            sub_dirs = [d for d in os.listdir(potential_main_dir) 
                       if os.path.isdir(os.path.join(potential_main_dir, d))]
            if all(c in sub_dirs for c in categories):
                main_dir_path = potential_main_dir
        
        # Filter categories to only include directories that contain images
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        valid_categories = []
        
        for category in categories:
            category_path = os.path.join(main_dir_path, category)
            if os.path.exists(category_path) and os.path.isdir(category_path):
                has_images = any(any(f.lower().endswith(ext) for ext in image_extensions) 
                                for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f)))
                if has_images:
                    valid_categories.append(category)
        
        # Now load the dataset from the correct directory with valid categories
        vision_app.load_dataset(main_dir_path, valid_categories)
        
        image_counts = {}
        for category in valid_categories:
            category_path = os.path.join(main_dir_path, category)
            if os.path.exists(category_path):
                image_counts[category] = len([f for f in os.listdir(category_path) 
                                           if os.path.isfile(os.path.join(category_path, f)) and
                                           any(f.lower().endswith(ext) for ext in image_extensions)])
        
        total_images = sum(image_counts.values())
        
        return DatasetUploadResponse(
            success=True,
            message=f"Dataset loaded successfully. Loaded {total_images} images from {len(valid_categories)} classes.",
            extracted_path=main_dir_path,
            classes=valid_categories,
            image_counts=image_counts,
            total_images=total_images
        )
        
    except Exception as e:
        return DatasetUploadResponse(
            success=False,
            message=f"Error loading dataset: {str(e)}"
        )

@router.get("/visualize-data", response_model=VisualizationResponse)
async def visualize_data():
    """
    Visualize the dataset using PCA
    """
    try:
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            # Create a temporary file to save the visualization
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                save_path = temp_file.name
            
            # Generate the visualization
            vision_app.visualize_data(save_path=save_path)
            
            # Read the saved image
            with open(save_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Clean up
            os.unlink(save_path)
            
            return VisualizationResponse(
                success=True,
                message="Data visualized successfully",
                visualizations={"pca_visualization": img_data}
            )
            
        finally:
            # Restore the original backend
            plt.close('all')
            
    except Exception as e:
        return VisualizationResponse(
            success=False,
            message=f"Error visualizing data: {str(e)}",
            visualizations={}
        )

@router.get("/get-models", response_model=ModelResponse)
async def get_available_models():
    """
    Get the list of available models
    """
    try:
        models = vision_app.get_available_models()
        
        return ModelResponse(
            success=True,
            message="Retrieved available models successfully",
            available_models=models
        )
        
    except Exception as e:
        return ModelResponse(
            success=False,
            message=f"Error retrieving models: {str(e)}",
            available_models=[]
        )

@router.post("/select-model", response_model=AlgorithmSelectionResponse)
async def select_model(request: AlgorithmSelectionRequest):
    """
    Select a model for training
    """
    try:
        model_params = vision_app.get_model_params(request.algorithm_name)
        
        if model_params is None:
            return AlgorithmSelectionResponse(
                success=False,
                message=f"Model '{request.algorithm_name}' not found",
                algorithm_name=request.algorithm_name,
                hyperparameters={},
                algorithm_description=""
            )
        
        # Get model descriptions
        model_descriptions = {
            'KNN': 'K-Nearest Neighbors classifier works by finding the K closest training examples in the feature space and voting for the most common class among them.',
            'SVM': 'Support Vector Machine finds the hyperplane that best separates the classes with the maximum margin.',
            'RandomForest': 'Random Forest builds multiple decision trees and merges their predictions to improve accuracy and control over-fitting.',
            'DecisionTree': 'Decision Tree creates a model that predicts the target value by learning simple decision rules inferred from the data features.',
            'GradientBoosting': 'Gradient Boosting builds an ensemble of sequential trees, each correcting the errors of its predecessors.'
        }
        
        # Format hyperparameters
        hyperparameters = {}
        for param, value in model_params.items():
            hyperparameters[param] = {
                'description': f"Parameter: {param}",
                'suggested_values': [value]  # Use current value as suggestion
            }
        
        return AlgorithmSelectionResponse(
            success=True,
            message=f"Model '{request.algorithm_name}' selected successfully",
            algorithm_name=request.algorithm_name,
            hyperparameters=hyperparameters,
            algorithm_description=model_descriptions.get(request.algorithm_name, "No description available")
        )
        
    except Exception as e:
        return AlgorithmSelectionResponse(
            success=False,
            message=f"Error selecting model: {str(e)}",
            algorithm_name=request.algorithm_name,
            hyperparameters={},
            algorithm_description=""
        )

@router.post("/configure-model", response_model=HyperparameterConfigResponse)
async def configure_model(request: HyperparameterConfigRequest):
    """
    Configure the selected model's hyperparameters
    """
    try:
        vision_app.model_manager.set_model_params(
            vision_app.model_manager.model_name,
            request.hyperparameters
        )
        
        return HyperparameterConfigResponse(
            success=True,
            message="Model configured successfully",
            hyperparameters=convert_to_native_types(request.hyperparameters)
        )
        
    except Exception as e:
        return HyperparameterConfigResponse(
            success=False,
            message=f"Error configuring model: {str(e)}",
            hyperparameters={}
        )

@router.post("/train-model", response_model=TrainingResponse)
async def train_model(model_name: str = Form(...)):
    """
    Train the selected model
    """
    try:
        start_time = datetime.now()
        
        results = vision_app.train_model(model_name)
        
        if results is None:
            return TrainingResponse(
                success=False,
                message=f"Error training model '{model_name}'",
                training_time=0.0
            )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        return TrainingResponse(
            success=True,
            message=f"Model '{model_name}' trained successfully",
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
    Evaluate the trained model
    """
    try:
        # Redirect matplotlib output to capture visualizations
        plt.switch_backend('Agg')
        
        try:
            if not vision_app.model_manager.trained_models:
                return EvaluationResponse(
                    success=False,
                    message="No trained model available. Please train a model first.",
                    metrics={},
                    classification_report={},
                    confusion_matrix=[],
                    visualizations={}
                )
            
            # Get metrics from the last training result
            model_name = list(vision_app.model_manager.trained_models.keys())[0]
            model = vision_app.model_manager.trained_models[model_name]
            
            # Get predictions for test data
            y_pred = model.predict(vision_app.data_manager.X_test)
            
            # Calculate metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
            accuracy = accuracy_score(vision_app.data_manager.y_test, y_pred)
            
            # Handle binary and multiclass scenarios
            if len(np.unique(vision_app.data_manager.y_test)) == 2:
                precision = precision_score(vision_app.data_manager.y_test, y_pred, average='binary')
                recall = recall_score(vision_app.data_manager.y_test, y_pred, average='binary')
                f1 = f1_score(vision_app.data_manager.y_test, y_pred, average='binary')
            else:
                precision = precision_score(vision_app.data_manager.y_test, y_pred, average='weighted')
                recall = recall_score(vision_app.data_manager.y_test, y_pred, average='weighted')
                f1 = f1_score(vision_app.data_manager.y_test, y_pred, average='weighted')
                
            metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Generate confusion matrix
            cm = confusion_matrix(vision_app.data_manager.y_test, y_pred)
            
            # Create confusion matrix visualization
            fig = plt.figure(figsize=(8, 6))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            
            # Add class labels
            classes = np.unique(vision_app.data_manager.y_test)
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            
            # Add text annotations
            thresh = cm.max() / 2.
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, format(cm[i, j], 'd'),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black")
            
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            
            cm_viz = fig_to_base64(fig)
            
            # Get classification report
            report = classification_report(vision_app.data_manager.y_test, y_pred, output_dict=True)
            
            return EvaluationResponse(
                success=True,
                message="Model evaluated successfully",
                metrics=convert_to_native_types(metrics),
                classification_report=convert_to_native_types(report),
                confusion_matrix=convert_to_native_types(cm.tolist()),
                visualizations={'confusion_matrix': cm_viz}
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

@router.post("/predict-image", response_model=PredictionResponse)
async def predict_image(file: UploadFile = File(...), model_name: str = Form(...)):
    """
    Make a prediction on a single image
    """
    try:
        # Save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            contents = await file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name
        
        try:
            # Make prediction
            prediction = vision_app.predict_image(model_name, temp_file_path)
            
            if prediction is None:
                return PredictionResponse(
                    success=False,
                    message="Failed to make prediction. Model may not be trained."
                )
            
            return PredictionResponse(
                success=True,
                message="Prediction made successfully",
                prediction=prediction,
                confidence=0.0  # We don't have confidence scores in the current implementation
            )
            
        finally:
            # Clean up
            os.unlink(temp_file_path)
            
    except Exception as e:
        return PredictionResponse(
            success=False,
            message=f"Error making prediction: {str(e)}"
        )