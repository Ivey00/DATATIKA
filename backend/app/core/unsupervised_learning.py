import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import warnings
import os
from datetime import datetime

class UnsupervisedModelTrainer:
    """
    A class to automate unsupervised model training for industrial users
    """
    
    def __init__(self):
        self.data = None
        self.features = None
        self.datetime_col = None
        self.item_id_col = None
        self.numerical_cols = None
        self.categorical_cols = None
        self.model = None
        self.preprocessor = None
        self.scaler = None
        self.X_transformed = None
        self.anomaly_results = None
        self.algorithm_name = None
        self.category_col = None  # For filtering by specific machine/item
        self.category_value = None  # The specific value to filter by
        
        # Available algorithms
        self.algorithms = {
            'Isolation Forest': IsolationForest,
            'Local Outlier Factor': LocalOutlierFactor,
            'One-Class SVM': OneClassSVM,
        }
        
        self.algorithm_params = {
            'Isolation Forest': {
                'n_estimators': {
                    'default': 100,
                    'description': 'The number of base estimators in the ensemble',
                    'suggested_values': [50, 100, 200, 500]
                },
                'contamination': {
                    'default': 'auto',
                    'description': 'The proportion of outliers in the data set',
                    'suggested_values': ['auto', 0.1, 0.2, 0.3]
                },
                'max_samples': {
                    'default': 'auto',
                    'description': 'The number of samples to draw from X to train each base estimator',
                    'suggested_values': ['auto', 100, 256, 512, 'all']
                },
                'random_state': {
                    'default': 42,
                    'description': 'Determines random number generation for dataset shuffling and tree building',
                    'suggested_values': [None, 42, 0, 1, 123]
                },
                'max_features': {
                    'default': 1.0,
                    'description': 'The number of features to draw from X to train each base estimator',
                    'suggested_values': [0.5, 0.8, 1.0, 'auto', 'sqrt', 'log2']
                }
            },
            'Local Outlier Factor': {
                'n_neighbors': {
                    'default': 20,
                    'description': 'Number of neighbors to use by default for kneighbors queries',
                    'suggested_values': [5, 10, 20, 30, 50]
                },
                'algorithm': {
                    'default': 'auto',
                    'description': 'Algorithm used to compute the nearest neighbors',
                    'suggested_values': ['auto', 'ball_tree', 'kd_tree', 'brute']
                },
                'leaf_size': {
                    'default': 30,
                    'description': 'Leaf size passed to BallTree or KDTree',
                    'suggested_values': [10, 20, 30, 40, 50]
                },
                'metric': {
                    'default': 'minkowski',
                    'description': 'Metric used for the distance computation',
                    'suggested_values': ['minkowski', 'euclidean', 'manhattan', 'chebyshev']
                },
                'contamination': {
                    'default': 'auto',
                    'description': 'The proportion of outliers in the data set',
                    'suggested_values': ['auto', 0.1, 0.2, 0.3]
                },
                'p': {
                    'default': 2,
                    'description': 'Parameter for the Minkowski metric',
                    'suggested_values': [1, 2, 3, 'inf']
                }
            },
            'One-Class SVM': {
                'kernel': {
                    'default': 'rbf',
                    'description': 'Specifies the kernel type to be used in the algorithm',
                    'suggested_values': ['linear', 'poly', 'rbf', 'sigmoid']
                },
                'nu': {
                    'default': 0.5,
                    'description': 'An upper bound on the fraction of training errors',
                    'suggested_values': [0.1, 0.2, 0.3, 0.4, 0.5]
                },
                'gamma': {
                    'default': 'scale',
                    'description': 'Kernel coefficient',
                    'suggested_values': ['scale', 'auto', 0.1, 0.01, 0.001]
                },
                'coef0': {
                    'default': 0.0,
                    'description': 'Independent term in kernel function. Only significant in poly and sigmoid',
                    'suggested_values': [0.0, 0.1, 0.5, 1.0]
                },
                'degree': {
                    'default': 3,
                    'description': 'Degree of the polynomial kernel function',
                    'suggested_values': [2, 3, 4, 5]
                },
                'shrinking': {
                    'default': True,
                    'description': 'Whether to use the shrinking heuristic',
                    'suggested_values': [True, False]
                },
                'tol': {
                    'default': 0.001,
                    'description': 'Tolerance for stopping criterion',
                    'suggested_values': [0.1, 0.01, 0.001, 0.0001]
                }
            }
        }
    def load_data(self, file_path):
        """
        Load data from CSV or Excel file
        """
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
                
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            print("\nFirst 5 rows of the dataset:")
            print(self.data.head())
            print("\nData Info:")
            self.data.info()
            print("\nSummary statistics:")
            print(self.data.describe())
            
            return self.data
        except Exception as e:
            print(f"Error loading data: {e}")
            return None

    def detect_column_types(self):
        """
        Automatically detect numerical, categorical, and datetime columns
        """
        if self.data is None:
            print("Please load data first.")
            return
            
        # Auto-detect datetime columns
        datetime_cols = []
        for col in self.data.columns:
            if self.data[col].dtype == 'object':
                try:
                    pd.to_datetime(self.data[col])
                    datetime_cols.append(col)
                except:
                    pass
        
        # Auto-detect numerical and categorical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = self.data.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_cols = [col for col in categorical_cols if col not in datetime_cols]
        
        print(f"Detected datetime columns: {datetime_cols}")
        print(f"Detected numerical columns: {numerical_cols}")
        print(f"Detected categorical columns: {categorical_cols}")
        
        return datetime_cols, numerical_cols, categorical_cols

    def set_column_types(self, datetime_col=None, item_id_col=None, numerical_cols=None, categorical_cols=None):
        """
        Manually set column types
        """
        self.datetime_col = datetime_col
        self.item_id_col = item_id_col
        self.numerical_cols = numerical_cols if numerical_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        
        # Set features as all numerical and categorical columns
        self.features = self.numerical_cols + self.categorical_cols
        
        print(f"Datetime column: {self.datetime_col}")
        print(f"Item ID column: {self.item_id_col}")
        print(f"Numerical columns: {self.numerical_cols}")
        print(f"Categorical columns: {self.categorical_cols}")
        print(f"Features selected: {self.features}")

    def filter_by_category(self, category_col, category_value):
        """
        Filter data by a specific category (e.g., specific machine_id)
        """
        if category_col not in self.data.columns:
            print(f"Column {category_col} not found in data.")
            return
            
        self.category_col = category_col
        self.category_value = category_value
        
        filtered_data = self.data[self.data[category_col] == category_value]
        if len(filtered_data) == 0:
            print(f"No data found for {category_col} = {category_value}")
            return
            
        print(f"Data filtered for {category_col} = {category_value}")
        print(f"Filtered data shape: {filtered_data.shape}")
        
        self.data = filtered_data
        return self.data

    def visualize_features(self, max_features=10):
        """
        Visualize features distributions and correlations
        """
        if self.data is None or not self.features:
            print("Please load data and set features first.")
            return
            
        # Limit the number of plots to avoid too many visualizations
        plot_features = self.features[:min(len(self.features), max_features)]
        
        # Create a figure with subplots
        plt.figure(figsize=(15, 10))
        
        # Plot numerical features
        num_features = [f for f in plot_features if f in self.numerical_cols]
        n_num = len(num_features)
        if n_num > 0:
            ncols = 3
            nrows = int(np.ceil(n_num / ncols))
            fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4 * nrows))
            axes = axes.flatten() if n_num > 1 else [axes]
            for i, feature in enumerate(num_features):
                sns.histplot(self.data[feature], kde=True, ax=axes[i])
                axes[i].set_title(f"Distribution of {feature}")
            # Hide any unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])
            plt.tight_layout()
            plt.show()
        
        # Plot categorical features
        cat_features = [f for f in plot_features if f in self.categorical_cols]
        for i, feature in enumerate(cat_features):
            plt.subplot(3, min(len(cat_features), 3), i+1)
            self.data[feature].value_counts().plot(kind='bar')
            plt.title(f"Count of {feature}")
            plt.tight_layout()
            
        # Plot correlation matrix for numerical features
        if len(num_features) > 1:
            plt.figure(figsize=(12, 10))
            correlation = self.data[num_features].corr()
            sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title("Correlation Matrix")
            plt.tight_layout()
            
        # Plot time series if datetime column exists
        if self.datetime_col and len(num_features) > 0:
            plt.figure(figsize=(15, 8))
            for feature in num_features[:3]:  # Plot first 3 numerical features only
                temp_df = self.data.copy()
                temp_df[self.datetime_col] = pd.to_datetime(temp_df[self.datetime_col])
                temp_df.set_index(self.datetime_col, inplace=True)
                temp_df[feature].plot()
                plt.title(f"Time Series of {feature}")
                plt.tight_layout()
                
        plt.show()
    
    def select_algorithm(self, algorithm_name):
        """
        Select an unsupervised learning algorithm
        """
        if algorithm_name not in self.algorithms:
            print(f"Algorithm {algorithm_name} not supported. Available algorithms: {list(self.algorithms.keys())}")
            return False
            
        self.algorithm_name = algorithm_name
        print(f"Selected algorithm: {algorithm_name}")
        
        # Display parameter information
        print("\nParameters for this algorithm:")
        for param, details in self.algorithm_params[algorithm_name].items():
            print(f"- {param} (default: {details['default']}): {details['description']}")
            
        return True

    def set_hyperparameters(self, **kwargs):
        """
        Set hyperparameters for the selected algorithm
        """
        if not self.algorithm_name:
            print("Please select an algorithm first.")
            return False
            
        # Check if provided parameters are valid for the selected algorithm
        invalid_params = [param for param in kwargs if param not in self.algorithm_params[self.algorithm_name]]
        if invalid_params:
            print(f"Invalid parameters for {self.algorithm_name}: {invalid_params}")
            print(f"Available parameters: {list(self.algorithm_params[self.algorithm_name].keys())}")
            return False
            
        # Create model with specified parameters
        try:
            self.model = self.algorithms[self.algorithm_name](**kwargs)
            print(f"Hyperparameters set for {self.algorithm_name}:")
            for param, value in kwargs.items():
                print(f"- {param}: {value}")
            return True
        except Exception as e:
            print(f"Error setting hyperparameters: {e}")
            return False

    def preprocess_data(self):
        """
        Preprocess data for model training
        """
        if self.data is None or not self.features:
            print("Please load data and set features first.")
            return None
            
        # Create preprocessing pipeline
        transformers = []
        
        # Numerical features
        if self.numerical_cols:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_transformer, self.numerical_cols))
            
        # Categorical features
        if self.categorical_cols:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])
            transformers.append(('cat', cat_transformer, self.categorical_cols))
            
        # Check if we have any transformers
        if not transformers:
            print("No features to preprocess.")
            return None
            
        # Create preprocessor
        self.preprocessor = ColumnTransformer(transformers=transformers)
        
        # Apply preprocessing
        X = self.data[self.features]
        try:
            self.X_transformed = self.preprocessor.fit_transform(X)
            print("Data preprocessing completed.")
            print(f"Transformed data shape: {self.X_transformed.shape}")
            return self.X_transformed
        except Exception as e:
            print(f"Error preprocessing data: {e}")
            return None

    def train_model(self):
        """
        Train the selected unsupervised model
        """
        if self.model is None:
            print("Please select an algorithm and set hyperparameters first.")
            return False
            
        if self.X_transformed is None:
            print("Please preprocess data first.")
            return False
            
        try:
            # For clustering algorithms
            if self.algorithm_name in ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture']:
                self.model.fit(self.X_transformed)
                if hasattr(self.model, 'labels_'):
                    labels = self.model.labels_
                elif hasattr(self.model, 'predict'):
                    labels = self.model.predict(self.X_transformed)
                
                # Store results in a dataframe
                result_df = self.data.copy()
                result_df['cluster'] = labels
                
                # Count elements in each cluster
                cluster_counts = result_df['cluster'].value_counts().to_dict()
                print("\nCluster distribution:")
                for cluster, count in sorted(cluster_counts.items()):
                    print(f"Cluster {cluster}: {count} samples ({count/len(result_df)*100:.2f}%)")
                
                # Identify anomalous clusters (smallest clusters)
                if self.algorithm_name != 'DBSCAN':
                    total_samples = len(result_df)
                    # Mark clusters as anomalous if they are less than 10% of the total samples
                    small_cluster_labels = [cluster for cluster, count in cluster_counts.items() if count / total_samples < 0.1]
                    result_df['anomaly'] = result_df['cluster'].apply(
                        lambda x: 'Yes' if x in small_cluster_labels else 'No')
                else:
                    # For DBSCAN, -1 indicates noise points (anomalies)
                    result_df['anomaly'] = result_df['cluster'].apply(
                        lambda x: 'Yes' if x == -1 else 'No')
                    
            # For anomaly detection algorithms
            elif self.algorithm_name in ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']:
                # For these algorithms, we get anomaly scores or predictions directly
                if hasattr(self.model, 'fit_predict'):
                    # Return -1 for outliers and 1 for inliers
                    predictions = self.model.fit_predict(self.X_transformed)
                elif hasattr(self.model, 'predict'):
                    self.model.fit(self.X_transformed)
                    predictions = self.model.predict(self.X_transformed)
                
                # Convert predictions to meaningful labels (anomaly/normal)
                result_df = self.data.copy()
                result_df['anomaly_score'] = predictions
                result_df['anomaly'] = result_df['anomaly_score'].apply(
                    lambda x: 'Yes' if x == -1 else 'No')
                
                # Count anomalies
                anomaly_count = (result_df['anomaly'] == 'Yes').sum()
                print(f"\nDetected {anomaly_count} potential anomalies out of {len(result_df)} samples "
                     f"({anomaly_count/len(result_df)*100:.2f}%)")
                
            # Save the results
            self.anomaly_results = result_df
            
            print("\nModel training completed.")
            return True
        except Exception as e:
            print(f"Error training model: {e}")
            return False

    def evaluate_model(self):
        """
        Evaluate the trained model
        """
        if self.model is None or self.X_transformed is None:
            print("Please train a model first.")
            return False
            
        # Initialize metrics storage if not already present
        if not hasattr(self, 'evaluation_metrics'):
            self.evaluation_metrics = {}
            
        # Different metrics based on algorithm type
        if self.algorithm_name in ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture']:
            # For clustering algorithms
            try:
                if hasattr(self.model, 'labels_'):
                    labels = self.model.labels_
                elif hasattr(self.model, 'predict'):
                    labels = self.model.predict(self.X_transformed)
                
                # Skip evaluation if all points are in the same cluster
                if len(set(labels)) <= 1:
                    print("All points assigned to the same cluster. Cannot calculate clustering metrics.")
                    return False
                
                # Filter out noise points for DBSCAN (-1 label)
                if self.algorithm_name == 'DBSCAN':
                    mask = labels != -1
                    if sum(mask) <= 1:
                        print("Not enough non-noise points to calculate clustering metrics.")
                        return False
                    X_valid = self.X_transformed[mask]
                    labels_valid = labels[mask]
                else:
                    X_valid = self.X_transformed
                    labels_valid = labels
                
                if len(set(labels_valid)) > 1:  # Need at least 2 clusters for these metrics
                    # Calculate clustering metrics
                    try:
                        sil_score = silhouette_score(X_valid, labels_valid)
                        print(f"Silhouette Score: {sil_score:.4f} (higher is better, range: -1 to 1)")
                        print("Interpretation: Measures how similar an object is to its own cluster compared to other clusters.")
                        print("Values near 1 indicate well-separated clusters.")
                        self.evaluation_metrics['silhouette_score'] = float(sil_score)
                    except Exception as e:
                        print(f"Could not calculate Silhouette Score: {str(e)}")
                        
                    try:
                        ch_score = calinski_harabasz_score(X_valid, labels_valid)
                        print(f"Calinski-Harabasz Index: {ch_score:.4f} (higher is better)")
                        print("Interpretation: Ratio of between-cluster variance to within-cluster variance.")
                        print("Higher values indicate better-defined clusters.")
                        self.evaluation_metrics['calinski_harabasz_score'] = float(ch_score)
                    except Exception as e:
                        print(f"Could not calculate Calinski-Harabasz Index: {str(e)}")
                        
                    try:
                        db_score = davies_bouldin_score(X_valid, labels_valid)
                        print(f"Davies-Bouldin Index: {db_score:.4f} (lower is better)")
                        print("Interpretation: Average similarity between clusters.")
                        print("Lower values indicate better separation between clusters.")
                        self.evaluation_metrics['davies_bouldin_score'] = float(db_score)
                    except Exception as e:
                        print(f"Could not calculate Davies-Bouldin Index: {str(e)}")
            except Exception as e:
                print(f"Error evaluating clustering model: {e}")
                return False
        elif self.algorithm_name in ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']:
            # For anomaly detection algorithms
            anomaly_count = (self.anomaly_results['anomaly'] == 'Yes').sum()
            normal_count = (self.anomaly_results['anomaly'] == 'No').sum()
            
            print(f"Anomaly Detection Results:")
            print(f"- Normal instances: {normal_count} ({normal_count/len(self.anomaly_results)*100:.2f}%)")
            print(f"- Anomalous instances: {anomaly_count} ({anomaly_count/len(self.anomaly_results)*100:.2f}%)")
            
            # Store these metrics
            self.evaluation_metrics['anomaly_count'] = int(anomaly_count)
            self.evaluation_metrics['normal_count'] = int(normal_count)
            self.evaluation_metrics['anomaly_percentage'] = float(anomaly_count/len(self.anomaly_results)*100)
            
            # Visualize anomalies if we have numerical features
            if len(self.numerical_cols) >= 2:
                plt.figure(figsize=(12, 8))
                
                # PCA for visualization if we have many features
                if self.X_transformed.shape[1] > 2:
                    pca = PCA(n_components=2)
                    X_2d = pca.fit_transform(self.X_transformed)
                    x_col, y_col = 0, 1  # PCA components
                    plt.title('Anomaly Detection Results (PCA projection)')
                    plt.xlabel('Principal Component 1')
                    plt.ylabel('Principal Component 2')
                else:
                    X_2d = self.X_transformed
                    x_col, y_col = 0, 1  # First two columns
                    plt.title('Anomaly Detection Results')
                    plt.xlabel(self.numerical_cols[0])
                    plt.ylabel(self.numerical_cols[1] if len(self.numerical_cols) > 1 else 'Feature 2')
                
                # Plot normal points
                normal_mask = self.anomaly_results['anomaly'] == 'No'
                plt.scatter(X_2d[normal_mask, x_col], X_2d[normal_mask, y_col], 
                           c='blue', label='Normal', alpha=0.5)
                
                # Plot anomalies
                anomaly_mask = self.anomaly_results['anomaly'] == 'Yes'
                plt.scatter(X_2d[anomaly_mask, x_col], X_2d[anomaly_mask, y_col], 
                           c='red', label='Anomaly', alpha=0.5)
                
                plt.legend()
                plt.tight_layout()
                plt.show()
        # Mark evaluation as completed
        self.evaluation_completed = True
        return True
        
    def get_evaluation_results(self):
        """
        Get the evaluation results without re-computing them.
        Returns the metrics and success status that can be directly returned by the API.
        """
        # Check if we have trained a model and performed evaluation
        if not hasattr(self, 'model') or self.model is None or not hasattr(self, 'evaluation_completed'):
            return {
                "success": False,
                "message": "No evaluation results available. Please train the model first.",
                "metrics": {},
                "classification_report": {},
                "confusion_matrix": [],
                "visualizations": {}
            }
            
        # Prepare the metrics
        metrics = {}
        if hasattr(self, 'evaluation_metrics'):
            metrics.update(self.evaluation_metrics)
        else:
            # Calculate basic metrics if not already done
            if hasattr(self, 'anomaly_results') and self.anomaly_results is not None:
                anomaly_count = (self.anomaly_results['anomaly'] == 'Yes').sum()
                normal_count = (self.anomaly_results['anomaly'] == 'No').sum()
                metrics['anomaly_count'] = int(anomaly_count)
                metrics['normal_count'] = int(normal_count)
                metrics['anomaly_percentage'] = float(anomaly_count/len(self.anomaly_results)*100)
        
        # Return the results in the format expected by the API
        return {
            "success": True,
            "message": "Evaluation results retrieved successfully",
            "metrics": metrics,
            "classification_report": {},  # The visualization endpoint will generate these
            "confusion_matrix": [],
            "visualizations": {}  # The visualization endpoint will generate these
        }
    def view_anomalies(self):
        """
        View detected anomalies
        """
        if self.anomaly_results is None:
            print("Please train a model first.")
            return
            
        anomalies = self.anomaly_results[self.anomaly_results['anomaly'] == 'Yes']
        
        if len(anomalies) == 0:
            print("No anomalies detected.")
            return
            
        print(f"Found {len(anomalies)} anomalies ({len(anomalies)/len(self.anomaly_results)*100:.2f}% of data).")
        
        # Display anomalies with datetime and item_id if available
        display_cols = []
        if self.datetime_col:
            display_cols.append(self.datetime_col)
        if self.item_id_col:
            display_cols.append(self.item_id_col)
            
        # Add some feature columns
        display_cols.extend(self.features[5:])  # Show last 5 features
        display_cols.append('anomaly')
        
        # Make sure we only include columns that exist
        display_cols = [col for col in display_cols if col in anomalies.columns]
        
        print("\nSample of detected anomalies:")
        print(anomalies[display_cols].head(10))
        
        return anomalies

    def predict_new_data(self, new_data):
        """
        Predict anomalies on new data
        """
        if self.model is None or self.preprocessor is None:
            print("Please train a model first.")
            return None
            
        try:
            # Verify that new_data has all required columns
            missing_cols = [col for col in self.features if col not in new_data.columns]
            if missing_cols:
                print(f"New data is missing these columns: {missing_cols}")
                return None
            
            # Apply preprocessing 
            X_new = new_data[self.features]
            X_new_transformed = self.preprocessor.transform(X_new)
            
            # For clustering algorithms
            if self.algorithm_name in ['K-Means', 'DBSCAN', 'Agglomerative Clustering', 'Gaussian Mixture']:
                if hasattr(self.model, 'predict'):
                    labels = self.model.predict(X_new_transformed)
                else:
                    print(f"The {self.algorithm_name} model doesn't support predictions on new data.")
                    return None
                
                # Create result dataframe
                result_df = new_data.copy()
                result_df['cluster'] = labels
                
                # Determine which clusters represent anomalies (use the same logic as in training)
                if self.algorithm_name == 'DBSCAN':
                    result_df['anomaly'] = result_df['cluster'].apply(
                        lambda x: 'Yes' if x == -1 else 'No')
                else:
                    # For other clustering algorithms, use the smallest clusters identified during training
                    original_clusters = self.anomaly_results['cluster'].value_counts().to_dict()
                    total_samples = sum(original_clusters.values())
                    small_cluster_labels = [cluster for cluster, count in original_clusters.items() if count / total_samples < 0.1]
                    result_df['anomaly'] = result_df['cluster'].apply(
                        lambda x: 'Yes' if x in small_cluster_labels else 'No')
                
            # For anomaly detection algorithms
            elif self.algorithm_name in ['Isolation Forest', 'Local Outlier Factor', 'One-Class SVM']:
                predictions = self.model.predict(X_new_transformed)
                
                # Create result dataframe
                result_df = new_data.copy()
                result_df['anomaly_score'] = predictions
                result_df['anomaly'] = result_df['anomaly_score'].apply(
                    lambda x: 'Yes' if x == -1 else 'No')
            
            print(f"Predictions completed. {(result_df['anomaly'] == 'Yes').sum()} potential anomalies detected.")
            return result_df
        except Exception as e:
            print(f"Error predicting new data: {e}")
            return None

    def visualize_anomalies(self):
        """
        Visualize detected anomalies
        """
        if self.anomaly_results is None:
            print("Please train a model first.")
            return
            
        # Check if we have numerical features
        if len(self.numerical_cols) < 2:
            print("Not enough numerical features to visualize anomalies.")
            return
            
        plt.figure(figsize=(12, 8))
        
        # PCA for visualization if we have many features
        if self.X_transformed.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(self.X_transformed)
            x_col, y_col = 0, 1
            plt.title('Anomaly Detection Results (PCA projection)')
            plt.xlabel('Principal Component 1')
            plt.ylabel('Principal Component 2')
        else:
            X_2d = self.X_transformed
            x_col, y_col = 0, 1
            plt.title('Anomaly Detection Results')
            plt.xlabel(self.numerical_cols[0])
            plt.ylabel(self.numerical_cols[1] if len(self.numerical_cols) > 1 else 'Feature 2')
        # Plot normal points
        normal_mask = self.anomaly_results['anomaly'] == 'No'
        plt.scatter(X_2d[normal_mask, x_col], X_2d[normal_mask, y_col],
                     c='blue', label='Normal', alpha=0.5)
        # Plot anomalies
        anomaly_mask = self.anomaly_results['anomaly'] == 'Yes'
        plt.scatter(X_2d[anomaly_mask, x_col], X_2d[anomaly_mask, y_col],
                    c='red', label='Anomaly', alpha=0.8, edgecolors='k')
        
    def save_results(self, output_path=None):
        """
        Save anomaly detection results
        """
        if self.anomaly_results is None:
            print("No results to save. Please train a model first.")
            return
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"anomaly_results_{timestamp}.csv"
            
        self.anomaly_results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        
        return output_path

    def save_model(self, output_path=None):
        """
        Save trained model and preprocessor
        """
        if self.model is None or self.preprocessor is None:
            print("No model to save. Please train a model first.")
            return
            
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"model_{self.algorithm_name}_{timestamp}.pkl"
            preproc_path = f"preprocessor_{self.algorithm_name}_{timestamp}.pkl"
        else:
            model_path = f"{output_path}_model.pkl"
            preproc_path = f"{output_path}_preprocessor.pkl"
            
        try:
            import pickle
            
            # Save model
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
                
            # Save preprocessor
            with open(preproc_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
                
            print(f"Model saved to {model_path}")
            print(f"Preprocessor saved to {preproc_path}")
            
            return model_path, preproc_path
        except Exception as e:
            print(f"Error saving model: {e}")
            return None