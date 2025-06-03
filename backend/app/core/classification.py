import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import time
from datetime import datetime
import pickle
import os


class Classification:
    """
    A comprehensive system for training classification models on industrial data.
    
    This system provides functionality for:
    - Data import (.csv and .excel)
    - Feature selection and visualization
    - Algorithm selection and hyperparameter tuning
    - Data preprocessing
    - Model training
    - Model evaluation
    - Prediction on new data
    """
    
    def __init__(self):
        """Initialize the SupervisedLearningSystem."""
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.trained_model = None
        self.preprocessor = None
        self.features = None
        self.target = None
        self.categorical_features = []
        self.numerical_features = []
        self.item_id_column = None
        self.item_filter = None
        self.datetime_column = None  # Initialize datetime_column as None
        self.supported_algorithms = {
            'logistic_regression': {
                'model': LogisticRegression(),
                'params': {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2', 'elasticnet', None],
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'max_iter': [100, 500, 1000]
                },
                'description': {
                    'C': 'Inverse of regularization strength; smaller values specify stronger regularization',
                    'penalty': 'Norm used in the penalization (l1, l2, elasticnet, or None)',
                    'solver': 'Algorithm for optimization problem',
                    'max_iter': 'Maximum number of iterations for solver convergence'
                }
            },
            'decision_tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 5, 10, 15, 20],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                'description': {
                    'criterion': 'Function to measure the quality of a split (gini for Gini impurity, entropy for information gain)',
                    'max_depth': 'Maximum depth of the tree (None means unlimited)',
                    'min_samples_split': 'Minimum number of samples required to split an internal node',
                    'min_samples_leaf': 'Minimum number of samples required to be at a leaf node',
                    'max_features': 'Number of features to consider when looking for the best split'
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'criterion': ['gini', 'entropy'],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': [None, 'sqrt', 'log2']
                },
                'description': {
                    'n_estimators': 'Number of trees in the forest',
                    'criterion': 'Function to measure the quality of a split',
                    'max_depth': 'Maximum depth of the trees (None means unlimited)',
                    'min_samples_split': 'Minimum number of samples required to split an internal node',
                    'min_samples_leaf': 'Minimum number of samples required to be at a leaf node',
                    'max_features': 'Number of features to consider for the best split'
                }
            },
            'svm': {
                'model': SVC(probability=True),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'gamma': ['scale', 'auto', 0.1, 1],
                    'degree': [2, 3, 4]
                },
                'description': {
                    'C': 'Regularization parameter. The strength of the regularization is inversely proportional to C',
                    'kernel': 'Specifies the kernel type to be used in the algorithm',
                    'gamma': 'Kernel coefficient for RBF, poly and sigmoid kernels',
                    'degree': 'Degree of the polynomial kernel function'
                }
            },
            'naive_bayes': {
                'model': GaussianNB(),
                'params': {
                    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]
                },
                'description': {
                    'var_smoothing': 'Portion of the largest variance of all features that is added to variances for calculation stability'
                }
            },
            'knn': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                    'leaf_size': [10, 20, 30, 40, 50],
                    'p': [1, 2]
                },
                'description': {
                    'n_neighbors': 'Number of neighbors to consider',
                    'weights': 'Weight function used in prediction (uniform: all points in neighborhood are weighted equally, distance: closer neighbors have greater influence)',
                    'algorithm': 'Algorithm used to compute the nearest neighbors',
                    'leaf_size': 'Leaf size passed to BallTree or KDTree',
                    'p': 'Power parameter for the Minkowski metric (p=1 is Manhattan distance, p=2 is Euclidean distance)'
                }
            }
        }
        
    def import_data(self, file_path):
        """
        Import data from a CSV or Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file.
            
        Returns:
        --------
        pandas.DataFrame
            The imported data.
        """
        try:
            if file_path.lower().endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
                
            print(f"Data imported successfully. Shape: {self.data.shape}")
            print("\nFirst 5 rows of data:")
            print(self.data.head())
            print("\nData types:")
            print(self.data.dtypes)
            print("\nSummary statistics:")
            print(self.data.describe())
            
            # Check for missing values
            missing_values = self.data.isnull().sum()
            print("\nMissing values:")
            print(missing_values[missing_values > 0] if any(missing_values > 0) else "No missing values found")
            
            return self.data
            
        except Exception as e:
            print(f"Error importing data: {e}")
            return None
    
    def get_column_info(self):
        """
        Get information about columns in the dataset.
        
        Returns:
        --------
        dict
            Dictionary with column names as keys and their data types as values.
        """
        if self.data is None:
            print("No data available. Please import data first.")
            return None
        
        column_info = {}
        for column in self.data.columns:
            dtype = self.data[column].dtype
            unique_count = self.data[column].nunique()
            missing = self.data[column].isnull().sum()
            
            # Infer column type
            if pd.api.types.is_numeric_dtype(dtype):
                if unique_count <= 10:  # Arbitrary threshold
                    col_type = "Categorical (numeric)"
                else:
                    col_type = "Numerical"
            elif pd.api.types.is_datetime64_dtype(dtype):
                col_type = "DateTime"
            else:
                if unique_count <= 10:
                    col_type = "Categorical"
                else:
                    col_type = "Text"
            
            column_info[column] = {
                "dtype": str(dtype),
                "unique_values": unique_count,
                "missing_values": missing,
                "suggested_type": col_type
            }
        
        # Print column information
        print("\nColumn Information:")
        for col, info in column_info.items():
            print(f"- {col}:")
            print(f"  Data type: {info['dtype']}")
            print(f"  Unique values: {info['unique_values']}")
            print(f"  Missing values: {info['missing_values']}")
            print(f"  Suggested type: {info['suggested_type']}")
            print()
        
        return column_info
    
    def define_features_target(self, features, target, categorical_features=None, 
                               numerical_features=None, item_id_column=None):
        """
        Define the features and target for the model.
        
        Parameters:
        -----------
        features : list
            List of column names to use as features.
        target : str
            Name of the target column.
        categorical_features : list, optional
            List of categorical features. If None, they will be inferred.
        numerical_features : list, optional
            List of numerical features. If None, they will be inferred.
        item_id_column : str, optional
            Column name for item/machine ID if applicable.
            
        Returns:
        --------
        tuple
            (X, y) - feature matrix and target vector
        """
        if self.data is None:
            print("No data available. Please import data first.")
            return None, None
        
        self.features = features
        self.target = target
        self.item_id_column = item_id_column
        
        # Select features and target
        X = self.data[features]
        y = self.data[target]
        
        # Determine categorical and numerical features if not specified
        if categorical_features is None and numerical_features is None:
            self.categorical_features = []
            self.numerical_features = []
            
            for column in features:
                if pd.api.types.is_numeric_dtype(X[column]):
                    if X[column].nunique() <= 10:  # Arbitrary threshold for categorical
                        self.categorical_features.append(column)
                    else:
                        self.numerical_features.append(column)
                else:
                    self.categorical_features.append(column)
        else:
            self.categorical_features = categorical_features or []
            self.numerical_features = numerical_features or []
        
        print("\nFeatures and Target defined:")
        print(f"Features: {', '.join(features)}")
        print(f"Target: {target}")
        print(f"Categorical features: {', '.join(self.categorical_features)}")
        print(f"Numerical features: {', '.join(self.numerical_features)}")
        if item_id_column:
            print(f"Item ID column: {item_id_column}")
        
        self.X = X
        self.y = y
        
        return X, y
    
    def filter_by_item_id(self, item_id_value):
        """
        Filter the dataset to only include rows with the specified item ID.
        
        Parameters:
        -----------
        item_id_value : str or int
            The value of the item ID to filter by.
            
        Returns:
        --------
        tuple
            (X_filtered, y_filtered) - filtered feature matrix and target vector
        """
        if self.item_id_column is None:
            print("No item ID column specified. Please define features and target first.")
            return None, None
        
        if item_id_value not in self.data[self.item_id_column].unique():
            print(f"Item ID {item_id_value} not found in the dataset.")
            return None, None
        
        # Filter the dataset
        self.item_filter = item_id_value
        filtered_data = self.data[self.data[self.item_id_column] == item_id_value]
        
        X_filtered = filtered_data[self.features]
        y_filtered = filtered_data[self.target]
        
        print(f"\nDataset filtered for {self.item_id_column} = {item_id_value}")
        print(f"Filtered dataset shape: {X_filtered.shape}")
        
        self.X = X_filtered
        self.y = y_filtered
        
        return X_filtered, y_filtered
    
    def visualize_data(self, max_features=10):
        """
        Visualize the data with appropriate plots for each feature type.
        
        Parameters:
        -----------
        max_features : int, optional
            Maximum number of features to visualize.
        """
        if self.X is None or self.y is None:
            print("Features and target not defined. Please define features and target first.")
            return
        
        print("\nGenerating visualizations...")
        
        # Limit the number of features to visualize
        features_to_viz = self.features[:max_features] if len(self.features) > max_features else self.features
        
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
            if feature in self.categorical_features:
                # For categorical features
                counts = pd.crosstab(self.X[feature], self.y)
                counts.plot(kind='bar', stacked=True, ax=ax)
                ax.set_title(f'Distribution of {feature} by Target')
                ax.set_ylabel('Count')
                ax.set_xlabel(feature)
                ax.legend(title=self.target)
                
            elif feature in self.numerical_features:
                # For numerical features
                if self.y.nunique() <= 5:  # If target has few unique values, use box plot
                    ax.boxplot([self.X[self.y == c][feature].dropna() for c in self.y.unique()],
                              labels=self.y.unique())
                    ax.set_title(f'Distribution of {feature} by Target')
                    ax.set_ylabel(feature)
                    ax.set_xlabel(self.target)
                else:
                    # For many target values or continuous targets, use scatter plot
                    ax.scatter(self.X[feature], self.y, alpha=0.5)
                    ax.set_title(f'{feature} vs {self.target}')
                    ax.set_xlabel(feature)
                    ax.set_ylabel(self.target)
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualizations
        
        # Correlation matrix for numerical features
        if len(self.numerical_features) > 1:
            correlation_matrix = self.X[self.numerical_features].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.show()
        
        # Target distribution
        plt.figure(figsize=(8, 6))
        self.y.value_counts().plot(kind='bar')
        plt.title(f'Distribution of Target: {self.target}')
        plt.ylabel('Count')
        plt.xlabel(self.target)
        plt.tight_layout()
        plt.show()
    
    def select_algorithm(self, algorithm_name):
        """
        Select and configure the classification algorithm.
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to use. Must be in supported_algorithms.
            
        Returns:
        --------
        dict
            Information about the selected algorithm.
        """
        if algorithm_name not in self.supported_algorithms:
            print(f"Algorithm '{algorithm_name}' not supported. Available algorithms:")
            for name in self.supported_algorithms.keys():
                print(f"- {name}")
            return None
        
        self.model = self.supported_algorithms[algorithm_name]
        
        print(f"\nSelected algorithm: {algorithm_name}")
        print("Available hyperparameters:")
        for param, desc in self.model['description'].items():
            print(f"- {param}: {desc}")
            print(f"  Suggested values: {self.model['params'][param]}")
        
        return self.model
    
    def configure_hyperparameters(self, hyperparameters):
        """
        Configure hyperparameters for the selected algorithm.
        
        Parameters:
        -----------
        hyperparameters : dict
            Dictionary of hyperparameter names and values.
            
        Returns:
        --------
        object
            Configured model.
        """
        if self.model is None:
            print("No algorithm selected. Please select an algorithm first.")
            return None
        
        # Create a new instance of the model with the specified hyperparameters
        model_instance = self.model['model'].__class__(**hyperparameters)
        
        print("\nHyperparameters configured:")
        for param, value in hyperparameters.items():
            print(f"- {param}: {value}")
        
        # Update the model
        self.model['model'] = model_instance
        
        return model_instance
    
    def preprocess_data(self):
        """
        Preprocess the data for model training.
        
        Returns:
        --------
        tuple
            (X_train, X_test, y_train, y_test) - preprocessed train and test sets
        """
        if self.X is None or self.y is None:
            print("Features and target not defined. Please define features and target first.")
            return None, None, None, None
        
        print("\nPreprocessing data...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Create preprocessing pipelines
        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        # Create preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_transformer, self.numerical_features),
                ('cat', cat_transformer, self.categorical_features)
            ]
        )
        
        # Preprocess training data
        X_train_processed = self.preprocessor.fit_transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)
        
        print(f"Preprocessed training data shape: {X_train_processed.shape}")
        print(f"Preprocessed test data shape: {X_test_processed.shape}")
        
        self.X_train = X_train_processed
        self.X_test = X_test_processed
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train_processed, X_test_processed, y_train, y_test
    
    def train_model(self, cross_validate=True, cv=5):
        """
        Train the model with the selected algorithm and configured hyperparameters.
        
        Parameters:
        -----------
        cross_validate : bool, optional
            Whether to perform cross-validation.
        cv : int, optional
            Number of folds for cross-validation.
            
        Returns:
        --------
        object
            Trained model.
        """
        if self.model is None:
            print("No algorithm selected. Please select an algorithm first.")
            return None
        
        if self.X_train is None:
            print("Data not preprocessed. Please preprocess the data first.")
            return None
        
        print("\nTraining model...")
        start_time = time.time()
        
        # Get the model
        model = self.model['model']
        
        # Train the model
        model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        print(f"Model trained in {training_time:.2f} seconds")
        
        # Cross-validation if requested
        if cross_validate:
            print("\nPerforming cross-validation...")
            if hasattr(self.preprocessor, 'transform'):
                # Create a pipeline for cross-validation
                cv_pipeline = Pipeline([
                    ('preprocessor', self.preprocessor),
                    ('classifier', model.__class__(**model.get_params()))
                ])
                
                # Use the original data for cross-validation
                cv_scores = cross_val_score(cv_pipeline, self.X, self.y, cv=cv, scoring='accuracy')
            else:
                # If the preprocessor is already fitted, just use the processed data
                cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=cv, scoring='accuracy')
            
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean CV score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
        
        self.trained_model = model
        
        return model
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
        --------
        dict
            Dictionary with evaluation metrics.
        """
        if self.trained_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        if self.X_test is None or self.y_test is None:
            print("Test data not available. Please preprocess the data first.")
            return None
        
        print("\nEvaluating model on test set...")
        
        # Make predictions
        y_pred = self.trained_model.predict(self.X_test)
        y_prob = self.trained_model.predict_proba(self.X_test)[:, 1] if hasattr(self.trained_model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        # Print metrics
        print("\nModel Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # ROC curve for binary classification
        if len(np.unique(self.y_test)) == 2 and y_prob is not None:
            fpr, tpr, _ = roc_curve(self.y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.show()
        
        return metrics
    
    def predict_new_data(self, new_data):
        """
        Make predictions on new data.
        
        Parameters:
        -----------
        new_data : pandas.DataFrame or str
            New data to predict on. Can be a DataFrame or a path to a CSV/Excel file.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with the original data and prediction results.
        """
        if self.trained_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        if self.preprocessor is None:
            print("No preprocessor available. Please preprocess the data first.")
            return None
        
        print("\nPredicting on new data...")
        
        # Check if new_data is a file path
        if isinstance(new_data, str):
            try:
                if new_data.lower().endswith('.csv'):
                    new_data = pd.read_csv(new_data)
                elif new_data.lower().endswith(('.xls', '.xlsx')):
                    new_data = pd.read_excel(new_data)
                else:
                    print("Unsupported file format. Please provide a CSV or Excel file.")
                    return None
                print(f"New data imported successfully. Shape: {new_data.shape}")
            except Exception as e:
                print(f"Error importing new data: {e}")
                return None
        
        # Check if required features are in the new data
        missing_features = [feature for feature in self.features if feature not in new_data.columns]
        if missing_features:
            print(f"New data is missing the following features: {missing_features}")
            return None
        
        # Extract features from new data
        X_new = new_data[self.features]
        
        # Preprocess new data
        X_new_processed = self.preprocessor.transform(X_new)
        
        # Make predictions
        predictions = self.trained_model.predict(X_new_processed)
        probabilities = None
        if hasattr(self.trained_model, 'predict_proba'):
            probabilities = self.trained_model.predict_proba(X_new_processed)
        
        # Create results DataFrame
        results = new_data.copy()
        results[f'predicted_{self.target}'] = predictions
        
        # Add probabilities for binary classification
        if probabilities is not None and probabilities.shape[1] == 2:
            results[f'probability_{self.target}'] = probabilities[:, 1]
        
        print("\nPrediction results:")
        print(results[[self.features[0], f'predicted_{self.target}']].head())
        print(f"\nShape of results: {results.shape}")
        
        return results
    
    def compare_predictions_with_actual(self, new_data_with_target):
        """
        Compare predictions with actual values for validation.
        
        Parameters:
        -----------
        new_data_with_target : pandas.DataFrame or str
            New data with target column to validate predictions. Can be a DataFrame or a path to a CSV/Excel file.
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics.
        """
        if self.trained_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        # Check if new_data is a file path
        if isinstance(new_data_with_target, str):
            try:
                if new_data_with_target.lower().endswith('.csv'):
                    new_data_with_target = pd.read_csv(new_data_with_target)
                elif new_data_with_target.lower().endswith(('.xls', '.xlsx')):
                    new_data_with_target = pd.read_excel(new_data_with_target)
                else:
                    print("Unsupported file format. Please provide a CSV or Excel file.")
                    return None
                print(f"Validation data imported successfully. Shape: {new_data_with_target.shape}")
            except Exception as e:
                print(f"Error importing validation data: {e}")
                return None
        
        # Check if target column exists
        if self.target not in new_data_with_target.columns:
            print(f"Target column '{self.target}' not found in the validation data.")
            return None
        
        # Get predictions
        results = self.predict_new_data(new_data_with_target)
        if results is None:
            return None
        
        # Extract actual and predicted values
        y_actual = new_data_with_target[self.target]
        y_pred = results[f'predicted_{self.target}']
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_actual, y_pred),
            'precision': precision_score(y_actual, y_pred, average='weighted'),
            'recall': recall_score(y_actual, y_pred, average='weighted'),
            'f1': f1_score(y_actual, y_pred, average='weighted')
        }
        
        # Print metrics
        print("\nValidation Performance Metrics:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_actual, y_pred))
        
        # Confusion matrix
        cm = confusion_matrix(y_actual, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Validation)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        
        # Show comparison
        comparison = pd.DataFrame({
            'Actual': y_actual,
            'Predicted': y_pred,
            'Correct': y_actual == y_pred
        })
        print("\nActual vs Predicted (first 20 rows):")
        print(comparison.head(20))
        
        # Calculate accuracy by class
        class_accuracy = {}
        for cls in y_actual.unique():
            mask = y_actual == cls
            class_accuracy[cls] = accuracy_score(y_actual[mask], y_pred[mask])
        
        print("\nAccuracy by class:")
        for cls, acc in class_accuracy.items():
            print(f"Class {cls}: {acc:.4f}")
        
        return metrics
    
    def save_model(self, directory="models"):
        """
        Save the trained model and preprocessor to disk.
        
        Parameters:
        -----------
        directory : str
            Directory to save the model in.
            
        Returns:
        --------
        str
            Path to the saved model.
        """
        if self.trained_model is None:
            print("No trained model available. Please train a model first.")
            return None
        
        if self.preprocessor is None:
            print("No preprocessor available. Please preprocess the data first.")
            return None
        
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Generate a timestamp for the model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.model['model'].__class__.__name__}_{timestamp}"
        
        # Save model
        model_path = os.path.join(directory, f"{model_name}_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.trained_model, f)
        
        # Save preprocessor
        preprocessor_path = os.path.join(directory, f"{model_name}_preprocessor.pkl")
        with open(preprocessor_path, 'wb') as f:
            pickle.dump(self.preprocessor, f)
        
        # Save metadata
        metadata = {
            'model_class': self.model['model'].__class__.__name__,
            'features': self.features,
            'target': self.target,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'item_id_column': self.item_id_column,
            'item_filter': self.item_filter,
            'hyperparameters': self.trained_model.get_params()
        }
        
        metadata_path = os.path.join(directory, f"{model_name}_metadata.pkl")
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\nModel saved successfully to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        print(f"Metadata saved to {metadata_path}")
        
        return model_path
    
    def load_model(self, model_path, preprocessor_path, metadata_path):
        """
        Load a previously saved model from disk.
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file.
        preprocessor_path : str
            Path to the saved preprocessor file.
        metadata_path : str
            Path to the saved metadata file.
            
        Returns:
        --------
        bool
            True if the model was loaded successfully, False otherwise.
        """
        try:
            # Load model
            with open(model_path, 'rb') as f:
                self.trained_model = pickle.load(f)
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = pickle.load(f)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            # Update attributes
            self.features = metadata['features']
            self.target = metadata['target']
            self.numerical_features = metadata['numerical_features']
            self.categorical_features = metadata['categorical_features']
            self.item_id_column = metadata['item_id_column']
            self.item_filter = metadata['item_filter']
            
            # Find the corresponding model in supported_algorithms
            for algo_name, algo_info in self.supported_algorithms.items():
                if algo_info['model'].__class__.__name__ == metadata['model_class']:
                    self.model = algo_info
                    break
            
            print(f"\nModel loaded successfully from {model_path}")
            print(f"Model type: {metadata['model_class']}")
            print(f"Features: {self.features}")
            print(f"Target: {self.target}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def perform_grid_search(self, param_grid=None):
        """
        Perform grid search for hyperparameter tuning.
        
        Parameters:
        -----------
        param_grid : dict, optional
            Dictionary with parameters names as keys and lists of parameter values.
            If None, use the default parameter grid from the selected algorithm.
            
        Returns:
        --------
        dict
            Best parameters found.
        """
        if self.model is None:
            print("No algorithm selected. Please select an algorithm first.")
            return None
        
        if self.X_train is None or self.y_train is None:
            print("Data not preprocessed. Please preprocess the data first.")
            return None
        
        print("\nPerforming grid search for hyperparameter tuning...")
        
        # Get parameter grid
        if param_grid is None:
            param_grid = self.model['params']
        
        # Print parameter grid
        print("Parameter grid:")
        for param, values in param_grid.items():
            print(f"- {param}: {values}")
        
        # Create a base model
        base_model = self.model['model'].__class__()
        
        # Create grid search
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            scoring='accuracy',
            cv=5,
            n_jobs=-1,
            verbose=1
        )
        
        # Perform grid search
        start_time = time.time()
        grid_search.fit(self.X_train, self.y_train)
        grid_search_time = time.time() - start_time
        
        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        print(f"\nGrid search completed in {grid_search_time:.2f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best cross-validation score: {best_score:.4f}")
        
        # Configure the model with best parameters
        self.configure_hyperparameters(best_params)
        
        return best_params
    
    def get_algorithm_descriptions(self):
        """
        Get descriptions of all supported algorithms.
        
        Returns:
        --------
        dict
            Dictionary with algorithm descriptions.
        """
        descriptions = {
            'logistic_regression': """
                Logistic Regression
                
                A linear model that uses a logistic function to model a binary dependent variable. It's particularly good for binary classification problems. Logistic regression is simple to implement, interpret, and efficient to train.
                
                Strengths:
                - Simple to understand and interpret
                - Less prone to overfitting when the dataset is small
                - Works well for linearly separable data
                - Can provide probabilities for outcomes
                
                Weaknesses:
                - May underperform with complex non-linear relationships
                - Assumes features are independent
                - Not ideal for high-dimensional data with many features
            """,
            'decision_tree': """
                  Decision Tree  
                
                A non-parametric supervised learning algorithm that creates a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
                
                Strengths:
                - Easy to understand and interpret
                - Handles both numerical and categorical data
                - Requires little data preprocessing
                - Can handle multi-output problems
                
                Weaknesses:
                - Can create overly complex trees that overfit
                - Can be unstable (small variations in data can result in different trees)
                - Biased towards features with more levels
                - Not very good at extrapolating
            """,
            'random_forest': """
                Random Forest
                
                An ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes of the individual trees.
                
                  Strengths:  
                - Reduces overfitting compared to decision trees
                - Handles high-dimensional data well
                - Good performance on many problems
                - Provides feature importance metrics
                
                  Weaknesses:  
                - More complex and harder to interpret than a single decision tree
                - More computationally intensive
                - Slower to train than decision trees
                - Can still overfit on noisy datasets
            """,
            'svm': """
                  Support Vector Machine (SVM)  
                
                A supervised learning algorithm that can be used for both classification and regression. SVM finds the hyperplane that best separates the classes.
                
                  Strengths:  
                - Effective in high-dimensional spaces
                - Memory efficient
                - Versatile through different kernel functions
                - Works well when classes are separable
                
                  Weaknesses:  
                - Not suitable for large datasets due to training time
                - Sensitive to the choice of kernel and regularization
                - Doesn't directly provide probability estimates
                - Can be hard to interpret
            """,
            'naive_bayes': """
                  Naive Bayes  
                
                A family of probabilistic algorithms based on applying Bayes' theorem with the "naive" assumption of conditional independence between features.
                
                  Strengths:  
                - Fast training and prediction
                - Works well with high-dimensional data
                - Simple and easy to implement
                - Good for text classification problems
                
                  Weaknesses:  
                - Assumes features are independent (often not true in real data)
                - Limited by the "zero frequency" problem
                - Can be outperformed by more sophisticated models
                - Not ideal for regression problems
            """,
            'knn': """
                  K-Nearest Neighbors (KNN)  
                
                A non-parametric method used for classification and regression. The output is determined by the majority vote of the k nearest neighbors.
                
                  Strengths:  
                - Simple to implement
                - Adapts easily to new data
                - No assumptions about data distribution
                - Effective for many classification tasks
                
                  Weaknesses:  
                - Computationally intensive for large datasets
                - Sensitive to irrelevant features
                - Requires feature scaling
                - Can be affected by imbalanced datasets
            """
        }
        
        return descriptions