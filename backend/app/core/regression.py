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
# import shap
import joblib
from datetime import datetime
from sklearn.impute import SimpleImputer
import warnings
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

    # def analyze_feature_importance(self, method='shap'):
    #     """Analyze feature importance using SHAP values or other methods"""
    #     if method == 'shap':
    #         # Create a simple model for SHAP analysis
    #         model = RandomForestRegressor(n_estimators=100, random_state=42)
    #         model.fit(self.X, self.y)
            
    #         # Calculate SHAP values
    #         explainer = shap.TreeExplainer(model)
    #         shap_values = explainer.shap_values(self.X)
            
    #         # Plot SHAP summary
    #         plt.figure(figsize=(10, 6))
    #         shap.summary_plot(shap_values, self.X)
    #         plt.title('Feature Importance (SHAP Values)')
    #         plt.show()
            
    #         return shap_values
    #     else:
    #         raise ValueError("Currently only 'shap' method is supported")

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
        """Save the trained model"""
        if self.trained_model is None:
            raise ValueError("No trained model to save")
        joblib.dump(self.trained_model, filepath)

    def load_model(self, filepath):
        """Load a saved model"""
        self.trained_model = joblib.load(filepath)