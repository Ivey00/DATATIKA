import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import joblib
from datetime import datetime
from sklearn.impute import SimpleImputer
import warnings
import os
warnings.filterwarnings('ignore')

# Import all regressors
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet, LinearRegression, Ridge, BayesianRidge, HuberRegressor
from sklearn.linear_model import PassiveAggressiveRegressor, Lars, LassoLars, Lasso
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import OrthogonalMatchingPursuit
import lightgbm as lgb

class Regression:
    def __init__(self):
        self.data = None
        self.X = None
        self.y = None
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.target_name = None
        self.numerical_features = None
        self.categorical_features = None
        self.datetime_features = None
        self.machine_id_column = None
        self.trained_model = None
        self.model_name = None
        self.hyperparameters = None
        
        # Available models dictionary
        self.available_models = {
            'KNeighborsRegressor': KNeighborsRegressor,
            'ElasticNet': ElasticNet,
            'LinearRegression': LinearRegression,
            'Ridge': Ridge,
            'BayesianRidge': BayesianRidge,
            'HuberRegressor': HuberRegressor,
            'PassiveAggressiveRegressor': PassiveAggressiveRegressor,
            'Lars': Lars,
            'LassoLars': LassoLars,
            'Lasso': Lasso,
            'ExtraTreesRegressor': ExtraTreesRegressor,
            'DecisionTreeRegressor': DecisionTreeRegressor,
            'RandomForestRegressor': RandomForestRegressor,
            'DummyRegressor': DummyRegressor,
            'LGBMRegressor': lgb.LGBMRegressor,
            'GradientBoostingRegressor': GradientBoostingRegressor,
            'AdaBoostRegressor': AdaBoostRegressor,
            'OrthogonalMatchingPursuit': OrthogonalMatchingPursuit
        }
        
        # Model hyperparameter descriptions
        self.hyperparameter_descriptions = {
            'KNeighborsRegressor': {
                'n_neighbors': 'Number of neighbors to use for prediction',
                'weights': 'Weight function used in prediction (uniform or distance)',
                'algorithm': 'Algorithm used to compute the nearest neighbors'
            },
            'ElasticNet': {
                'alpha': 'Constant that multiplies the penalty terms',
                'l1_ratio': 'Mixing parameter between L1 and L2 penalties'
            },
            'RandomForestRegressor': {
                'n_estimators': 'Number of trees in the forest',
                'max_depth': 'Maximum depth of the trees',
                'min_samples_split': 'Minimum number of samples required to split a node',
                'min_samples_leaf': 'Minimum number of samples required to be at a leaf node'
            },
            'GradientBoostingRegressor': {
                'n_estimators': 'Number of boosting stages',
                'learning_rate': 'Learning rate shrinks the contribution of each tree',
                'max_depth': 'Maximum depth of the individual regression estimators',
                'min_samples_split': 'Minimum number of samples required to split a node'
            },
            'LGBMRegressor': {
                'n_estimators': 'Number of boosting iterations',
                'learning_rate': 'Boosting learning rate',
                'max_depth': 'Maximum tree depth',
                'num_leaves': 'Maximum number of leaves in one tree'
            },
            'LinearRegression': {
                'fit_intercept': 'Whether to calculate the intercept for this model',
                'normalize': 'Whether to normalize the input variables',
                'n_jobs': 'Number of jobs to use for the computation'
            },
            'Ridge': {
                'alpha': 'Regularization strength',
                'solver': 'Algorithm to use in the optimization problem',
                'fit_intercept': 'Whether to calculate the intercept for this model'
            },
            'BayesianRidge': {
                'n_iter': 'Maximum number of iterations',
                'alpha_1': 'Shape parameter for the Gamma distribution',
                'alpha_2': 'Inverse scale parameter for the Gamma distribution',
                'lambda_1': 'Shape parameter for the Gamma distribution of the precision',
                'lambda_2': 'Inverse scale parameter for the Gamma distribution of the precision'
            },
            'HuberRegressor': {
                'epsilon': 'The parameter that controls the number of outliers',
                'alpha': 'Regularization strength',
                'max_iter': 'Maximum number of iterations'
            },
            'PassiveAggressiveRegressor': {
                'C': 'Regularization parameter',
                'max_iter': 'Maximum number of passes over the training data',
                'early_stopping': 'Whether to use early stopping to terminate training'
            },
            'Lars': {
                'n_nonzero_coefs': 'Target number of non-zero coefficients',
                'fit_intercept': 'Whether to calculate the intercept for this model'
            },
            'LassoLars': {
                'alpha': 'Constant that multiplies the L1 term',
                'fit_intercept': 'Whether to calculate the intercept for this model'
            },
            'Lasso': {
                'alpha': 'Constant that multiplies the L1 term',
                'max_iter': 'Maximum number of iterations',
                'tol': 'Tolerance for optimization'
            },
            'ExtraTreesRegressor': {
                'n_estimators': 'Number of trees in the forest',
                'max_depth': 'Maximum depth of the trees',
                'min_samples_split': 'Minimum number of samples required to split a node',
                'min_samples_leaf': 'Minimum number of samples required to be at a leaf node'
            },
            'DecisionTreeRegressor': {
                'criterion': 'Function to measure the quality of a split',
                'splitter': 'Strategy used to choose the split at each node',
                'max_depth': 'Maximum depth of the tree'
            },
            'DummyRegressor': {
                'strategy': 'Strategy used to generate predictions',
                'constant': 'Constant value to predict if strategy is constant',
                'quantile': 'Quantile value to predict if strategy is quantile'
            },
            'AdaBoostRegressor': {
                'n_estimators': 'Number of boosting stages',
                'learning_rate': 'Learning rate shrinks the contribution of each tree',
                'loss': 'Loss function to use when updating weights'
            },
            'OrthogonalMatchingPursuit': {
                'n_nonzero_coefs': 'Target number of non-zero coefficients',
                'fit_intercept': 'Whether to calculate the intercept for this model'
            }
        }

    def load_data(self, file_path):
        """Load data from CSV or Excel file"""
        if file_path.endswith('.csv'):
            self.data = pd.read_csv(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            self.data = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please use .csv or .xlsx files.")
        return self.data

    def set_features(self, target_column, feature_columns, numerical_features=None, 
                    categorical_features=None, datetime_features=None, machine_id_column=None):
        """Set target and feature columns, along with their types"""
        self.target_name = target_column
        self.feature_names = feature_columns
        self.numerical_features = numerical_features or []
        self.categorical_features = categorical_features or []
        self.datetime_features = datetime_features or []
        self.machine_id_column = machine_id_column

        # Prepare features and target
        self.X = self.data[feature_columns]
        self.y = self.data[target_column]

    def visualize_data(self, plot_type='numerical'):
        """Visualize data based on feature types"""
        if plot_type == 'numerical':
            # Numerical features visualization
            for feature in self.numerical_features:
                plt.figure(figsize=(10, 6))
                sns.histplot(data=self.data, x=feature)
                plt.title(f'Distribution of {feature}')
                plt.show()
                
                # Scatter plot with target
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=self.data, x=feature, y=self.target_name)
                plt.title(f'{feature} vs {self.target_name}')
                plt.show()
        
        elif plot_type == 'categorical':
            # Categorical features visualization
            for feature in self.categorical_features:
                plt.figure(figsize=(10, 6))
                sns.countplot(data=self.data, x=feature)
                plt.title(f'Distribution of {feature}')
                plt.xticks(rotation=45)
                plt.show()
        
        elif plot_type == 'time':
            # Time-based features visualization
            for feature in self.datetime_features:
                plt.figure(figsize=(12, 6))
                self.data[feature] = pd.to_datetime(self.data[feature])
                self.data.set_index(feature)[self.target_name].plot()
                plt.title(f'{self.target_name} over time')
                plt.show()

    def preprocess_data(self):
        """Preprocess the data based on feature types"""
        # Create preprocessing pipelines
        numerical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Add numerical imputer
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Add categorical imputer
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  # Set sparse_output=False for better compatibility
        ])
        
        # Combine transformers
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ],
            remainder='drop'  # Drop columns that are not specified in the transformers
        )
        
        self.preprocessor = preprocessor
        return preprocessor

    def train_model(self, model_name, hyperparameters=None):
        """Train the selected model with specified hyperparameters"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not supported")
        
        self.model_name = model_name
        self.hyperparameters = hyperparameters or {}
        
        # Create model instance
        model = self.available_models[model_name](**self.hyperparameters)
        
        # Create full pipeline
        pipeline = Pipeline([
            ('preprocessor', self.preprocessor),
            ('model', model)
        ])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Train model
        pipeline.fit(X_train, y_train)
        self.trained_model = pipeline
        
        # Evaluate model
        y_pred = pipeline.predict(X_test)
        metrics = {
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': mean_absolute_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred)
        }
        
        return metrics

    def predict(self, new_data):
        """Make predictions on new data"""
        if self.trained_model is None:
            raise ValueError("Model needs to be trained first")
        
        predictions = self.trained_model.predict(new_data)
        return predictions

    def compare_predictions(self, actual_data, predicted_values):
        """Compare actual vs predicted values"""
        comparison_df = pd.DataFrame({
            'Actual': actual_data,
            'Predicted': predicted_values
        })
        
        # Plot comparison
        plt.figure(figsize=(10, 6))
        plt.scatter(actual_data, predicted_values, alpha=0.5)
        plt.plot([actual_data.min(), actual_data.max()], 
                [actual_data.min(), actual_data.max()], 
                'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        plt.show()
        
        return comparison_df

    def save_model(self, filepath):
        """
        Save the trained model, preprocessor, and metadata to disk.
        
        Parameters:
        -----------
        filepath : str
            Base filepath to save the model components.
            
        Returns:
        --------
        tuple
            Tuple containing (model_path, preprocessor_path, metadata_path).
        """
        if self.trained_model is None:
            raise ValueError("No trained model to save")
            
        if self.preprocessor is None:
            raise ValueError("No preprocessor available. Please preprocess the data first.")

        # Extract directory and base filename from filepath
        directory = os.path.dirname(filepath)
        base_name = os.path.splitext(os.path.basename(filepath))[0]

        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Generate paths for all components
        model_path = os.path.join(directory, f"{base_name}_model.pkl")
        preprocessor_path = os.path.join(directory, f"{base_name}_preprocessor.pkl")
        metadata_path = os.path.join(directory, f"{base_name}_metadata.pkl")

        # Save model
        with open(model_path, 'wb') as f:
            joblib.dump(self.trained_model, f)

        # Save preprocessor
        with open(preprocessor_path, 'wb') as f:
            joblib.dump(self.preprocessor, f)

        # Get evaluation metrics if available
        metrics = getattr(self, 'last_evaluation', None)

        # Save metadata
        metadata = {
            'model_class': self.model_name,
            'model_type': 'regression',  # For database schema
            'features': self.feature_names,
            'target': self.target_name,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'datetime_features': self.datetime_features,
            'machine_id_column': self.machine_id_column,
            'hyperparameters': self.hyperparameters,
            'metrics': metrics,
            'created_at': datetime.now().strftime("%Y%m%d_%H%M%S")
        }

        with open(metadata_path, 'wb') as f:
            joblib.dump(metadata, f)

        print(f"\nModel saved successfully to {model_path}")
        print(f"Preprocessor saved to {preprocessor_path}")
        print(f"Metadata saved to {metadata_path}")

        return model_path, preprocessor_path, metadata_path

    def load_model(self, filepath):
        """
        Load a previously saved model, preprocessor, and metadata from disk.
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model file. The preprocessor and metadata files
            should be in the same directory with _preprocessor.pkl and _metadata.pkl suffixes.
            
        Returns:
        --------
        bool
            True if the model was loaded successfully, False otherwise.
        """
        try:
            # Extract directory and base filename
            directory = os.path.dirname(filepath)
            base_name = os.path.splitext(os.path.basename(filepath))[0]
            
            # Remove any existing suffix like _model
            base_name = base_name.replace('_model', '')
            
            # Generate paths for all components
            model_path = os.path.join(directory, f"{base_name}_model.pkl")
            preprocessor_path = os.path.join(directory, f"{base_name}_preprocessor.pkl")
            metadata_path = os.path.join(directory, f"{base_name}_metadata.pkl")
            
            # Load model
            with open(model_path, 'rb') as f:
                self.trained_model = joblib.load(f)
            
            # Load preprocessor
            with open(preprocessor_path, 'rb') as f:
                self.preprocessor = joblib.load(f)
            
            # Load metadata
            with open(metadata_path, 'rb') as f:
                metadata = joblib.load(f)
            
            # Update attributes from metadata
            self.model_name = metadata['model_class']
            self.feature_names = metadata['features']
            self.target_name = metadata['target']
            self.numerical_features = metadata['numerical_features']
            self.categorical_features = metadata['categorical_features']
            self.datetime_features = metadata['datetime_features']
            self.machine_id_column = metadata['machine_id_column']
            self.hyperparameters = metadata['hyperparameters']
            
            print(f"\nModel loaded successfully from {model_path}")
            print(f"Model type: {metadata['model_class']}")
            print(f"Features: {self.feature_names}")
            print(f"Target: {self.target_name}")
            
            return True
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False