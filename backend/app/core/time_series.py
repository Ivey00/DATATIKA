import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgbm
from prophet import Prophet
import warnings
import os
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

class TimeSeriesForecaster:
    """
    A class for time series forecasting in industrial applications.
    This class provides functionality for data loading, preprocessing, 
    feature importance analysis, model training, evaluation, and prediction.
    """
    
    def __init__(self):
        """Initialize the TimeSeriesForecaster with default values."""
        self.data = None
        self.features = None
        self.target = None
        self.datetime_col = None
        self.item_id_col = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.model = None
        self.preprocessor = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.train_dates = None
        self.test_dates = None
        self.item_ids_test = None
        self.time_unit = None
        self.forecast_horizon = None
        self.algorithm = None
        self.hyperparameters = None
        self.evaluation_metrics = None
        self.feature_importance = None
        
        # Define available algorithms
        self.available_algorithms = {
            'xgboost': {
                'model': xgb.XGBRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'reg:squarederror',
                    'random_state': 42
                },
                'description': {
                    'n_estimators': 'Number of boosting rounds (trees) to build. Higher values may improve performance but increase training time.',
                    'learning_rate': 'Step size shrinkage used to prevent overfitting. Lower values make the model more robust but require more trees.',
                    'max_depth': 'Maximum depth of a tree. Higher values allow the model to learn more complex patterns but may lead to overfitting.',
                    'subsample': 'Fraction of samples used for fitting the trees. Values < 1.0 help prevent overfitting.',
                    'colsample_bytree': 'Fraction of features used for building each tree. Values < 1.0 help prevent overfitting.',
                    'objective': 'The loss function to be optimized. For regression, "reg:squarederror" is commonly used.',
                    'random_state': 'Seed for random number generation to ensure reproducibility.'
                }
            },
            'lightgbm': {
                'model': lgbm.LGBMRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 5,
                    'num_leaves': 31,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42
                },
                'description': {
                    'n_estimators': 'Number of boosting rounds (trees) to build. Higher values may improve performance but increase training time.',
                    'learning_rate': 'Step size shrinkage used to prevent overfitting. Lower values make the model more robust but require more trees.',
                    'max_depth': 'Maximum depth of a tree. Higher values allow the model to learn more complex patterns but may lead to overfitting.',
                    'num_leaves': 'Maximum number of leaves in one tree. Higher values may increase accuracy but may lead to overfitting.',
                    'subsample': 'Fraction of samples used for fitting the trees. Values < 1.0 help prevent overfitting.',
                    'colsample_bytree': 'Fraction of features used for building each tree. Values < 1.0 help prevent overfitting.',
                    'random_state': 'Seed for random number generation to ensure reproducibility.'
                }
            },
            'random_forest': {
                'model': RandomForestRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'random_state': 42
                },
                'description': {
                    'n_estimators': 'Number of trees in the forest. Higher values generally improve performance but increase training time.',
                    'max_depth': 'Maximum depth of the trees. Higher values allow the model to learn more complex patterns but may lead to overfitting.',
                    'min_samples_split': 'Minimum number of samples required to split an internal node. Higher values prevent overfitting.',
                    'min_samples_leaf': 'Minimum number of samples required to be at a leaf node. Higher values prevent overfitting.',
                    'random_state': 'Seed for random number generation to ensure reproducibility.'
                }
            },
            'gradient_boosting': {
                'model': GradientBoostingRegressor,
                'default_params': {
                    'n_estimators': 100,
                    'learning_rate': 0.1,
                    'max_depth': 3,
                    'subsample': 0.8,
                    'random_state': 42
                },
                'description': {
                    'n_estimators': 'Number of boosting stages (trees) to build. Higher values may improve performance but increase training time.',
                    'learning_rate': 'Step size shrinkage used to prevent overfitting. Lower values make the model more robust but require more trees.',
                    'max_depth': 'Maximum depth of the trees. Higher values allow the model to learn more complex patterns but may lead to overfitting.',
                    'subsample': 'Fraction of samples used for fitting the trees. Values < 1.0 help prevent overfitting.',
                    'random_state': 'Seed for random number generation to ensure reproducibility.'
                }
            },
            'prophet': {
                'model': Prophet,
                'default_params': {
                    'seasonality_mode': 'additive',
                    'yearly_seasonality': 'auto',
                    'weekly_seasonality': 'auto',
                    'daily_seasonality': 'auto',
                    'changepoint_prior_scale': 0.05
                },
                'description': {
                    'seasonality_mode': 'Type of seasonality, either "additive" or "multiplicative".',
                    'yearly_seasonality': 'Fit yearly seasonality. Can be "auto", True, False, or a number of Fourier terms.',
                    'weekly_seasonality': 'Fit weekly seasonality. Can be "auto", True, False, or a number of Fourier terms.',
                    'daily_seasonality': 'Fit daily seasonality. Can be "auto", True, False, or a number of Fourier terms.',
                    'changepoint_prior_scale': 'Controls flexibility of the trend. Higher values allow more flexibility.'
                }
            }
        }
    
    def load_data(self, file_path):
        """
        Load data from a CSV or Excel file.
        
        Parameters:
        -----------
        file_path : str
            Path to the data file (CSV or Excel)
            
        Returns:
        --------
        pandas.DataFrame
            The loaded data
        """
        try:
            if file_path.endswith('.csv'):
                # Try different delimiters
                try:
                    self.data = pd.read_csv(file_path)
                except:
                    try:
                        self.data = pd.read_csv(file_path, sep=';')
                    except:
                        self.data = pd.read_csv(file_path, sep='\t')
            elif file_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(file_path)
            else:
                raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
            
            print(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            return self.data
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return None
    
    def define_columns(self, datetime_col, target_col, item_id_col=None, 
                      categorical_cols=None, numerical_cols=None):
        """
        Define the column types in the dataset.
        
        Parameters:
        -----------
        datetime_col : str
            Name of the datetime column
        target_col : str
            Name of the target column to predict
        item_id_col : str, optional
            Name of the item/machine ID column
        categorical_cols : list, optional
            List of categorical column names
        numerical_cols : list, optional
            List of numerical column names
            
        Returns:
        --------
        dict
            Dictionary containing the defined columns
        """
        if self.data is None:
            print("Please load data first using load_data() method.")
            return None
        
        self.datetime_col = datetime_col
        self.target = target_col
        self.item_id_col = item_id_col
        
        # Convert datetime column to datetime type
        try:
            self.data[datetime_col] = pd.to_datetime(self.data[datetime_col])
            print(f"Converted {datetime_col} to datetime format.")
        except Exception as e:
            print(f"Error converting {datetime_col} to datetime: {str(e)}")
            return None
        
        # Auto-detect column types if not provided
        if categorical_cols is None and numerical_cols is None:
            self.categorical_cols = []
            self.numerical_cols = []
            
            for col in self.data.columns:
                if col in [datetime_col, target_col]:
                    continue
                    
                if item_id_col is not None and col == item_id_col:
                    self.categorical_cols.append(col)
                    continue
                
                if self.data[col].dtype == 'object' or self.data[col].nunique() < 10:
                    self.categorical_cols.append(col)
                else:
                    self.numerical_cols.append(col)
        else:
            self.categorical_cols = categorical_cols if categorical_cols is not None else []
            self.numerical_cols = numerical_cols if numerical_cols is not None else []
        
        # Define features
        self.features = self.numerical_cols + self.categorical_cols
        
        print(f"Defined columns:")
        print(f"  - Datetime column: {self.datetime_col}")
        print(f"  - Target column: {self.target}")
        print(f"  - Item ID column: {self.item_id_col if self.item_id_col else 'None'}")
        print(f"  - Categorical columns: {self.categorical_cols}")
        print(f"  - Numerical columns: {self.numerical_cols}")
        
        return {
            'datetime_col': self.datetime_col,
            'target_col': self.target,
            'item_id_col': self.item_id_col,
            'categorical_cols': self.categorical_cols,
            'numerical_cols': self.numerical_cols,
            'features': self.features
        }
    
    def visualize_data(self, visualization_type='all', n_samples=1000, specific_cols=None):
        """
        Visualize the data based on the specified type.
        
        Parameters:
        -----------
        visualization_type : str
            Type of visualization ('numerical', 'categorical', 'time', or 'all')
        n_samples : int
            Number of samples to use for visualization
        specific_cols : list, optional
            List of specific columns to visualize
            
        Returns:
        --------
        None
        """
        if self.data is None:
            print("Please load data first using load_data() method.")
            return
        
        if self.datetime_col is None or self.target is None:
            print("Please define columns first using define_columns() method.")
            return
        
        # Sample data for visualization if it's too large
        if len(self.data) > n_samples:
            data_sample = self.data.sample(n_samples, random_state=42)
        else:
            data_sample = self.data
        
        # Create a figure directory if it doesn't exist
        os.makedirs('figures', exist_ok=True)
        
        # Filter columns if specific_cols is provided
        num_cols = specific_cols if specific_cols else self.numerical_cols
        cat_cols = specific_cols if specific_cols else self.categorical_cols
        
        # Visualize numerical features
        if visualization_type in ['numerical', 'all'] and num_cols:
            print("Visualizing numerical features...")
            
            # Distribution of numerical features
            n_cols = min(3, len(num_cols))
            n_rows = (len(num_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, n_rows * 5))
            for i, col in enumerate(num_cols):
                if col in self.data.columns:
                    plt.subplot(n_rows, n_cols, i + 1)
                    sns.histplot(data_sample[col].dropna(), kde=True)
                    plt.title(f'Distribution of {col}')
                    plt.tight_layout()
            
            plt.savefig('figures/numerical_distributions.png')
            plt.close()
            
            # Correlation heatmap
            plt.figure(figsize=(12, 10))
            correlation_cols = [col for col in num_cols if col in self.data.columns]
            if self.target in self.data.columns:
                correlation_cols.append(self.target)
            
            if correlation_cols:
                corr = data_sample[correlation_cols].corr()
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', 
                           square=True, linewidths=.5)
                plt.title('Correlation Heatmap')
                plt.tight_layout()
                plt.savefig('figures/correlation_heatmap.png')
                plt.close()
            
            # Scatter plots against target
            if self.target in self.data.columns:
                n_cols = min(2, len(num_cols))
                n_rows = (len(num_cols) + n_cols - 1) // n_cols
                
                plt.figure(figsize=(15, n_rows * 5))
                for i, col in enumerate(num_cols):
                    if col in self.data.columns and col != self.target:
                        plt.subplot(n_rows, n_cols, i + 1)
                        sns.scatterplot(x=col, y=self.target, data=data_sample)
                        plt.title(f'{col} vs {self.target}')
                        plt.tight_layout()
                
                plt.savefig('figures/numerical_scatter_plots.png')
                plt.close()
        
        # Visualize categorical features
        if visualization_type in ['categorical', 'all'] and cat_cols:
            print("Visualizing categorical features...")
            
            n_cols = min(2, len(cat_cols))
            n_rows = (len(cat_cols) + n_cols - 1) // n_cols
            
            plt.figure(figsize=(15, n_rows * 5))
            for i, col in enumerate(cat_cols):
                if col in self.data.columns:
                    plt.subplot(n_rows, n_cols, i + 1)
                    value_counts = data_sample[col].value_counts().sort_values(ascending=False).head(10)
                    sns.barplot(x=value_counts.index, y=value_counts.values)
                    plt.title(f'Top 10 values for {col}')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
            
            plt.savefig('figures/categorical_distributions.png')
            plt.close()
            
            # Box plots for categorical vs target
            if self.target in self.data.columns:
                n_cols = min(2, len(cat_cols))
                n_rows = (len(cat_cols) + n_cols - 1) // n_cols
                
                plt.figure(figsize=(15, n_rows * 5))
                for i, col in enumerate(cat_cols):
                    if col in self.data.columns and col != self.target:
                        plt.subplot(n_rows, n_cols, i + 1)
                        # Limit to top 10 categories if there are too many
                        top_cats = data_sample[col].value_counts().nlargest(10).index
                        sns.boxplot(x=col, y=self.target, data=data_sample[data_sample[col].isin(top_cats)])
                        plt.title(f'{col} vs {self.target}')
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                
                plt.savefig('figures/categorical_boxplots.png')
                plt.close()
        
        # Visualize time-based features
        if visualization_type in ['time', 'all'] and self.datetime_col in self.data.columns:
            print("Visualizing time-based features...")
            
            # Time series plot of target
            if self.target in self.data.columns:
                plt.figure(figsize=(15, 6))
                
                # If item_id_col is defined, plot for a few items
                if self.item_id_col and self.item_id_col in self.data.columns:
                    # Get a few unique item IDs
                    unique_items = data_sample[self.item_id_col].unique()[:5]
                    for item in unique_items:
                        item_data = data_sample[data_sample[self.item_id_col] == item]
                        item_data = item_data.sort_values(by=self.datetime_col)
                        plt.plot(item_data[self.datetime_col], item_data[self.target], 
                                label=f'{self.item_id_col}={item}')
                    
                    plt.legend()
                else:
                    # Sort by datetime
                    time_data = data_sample.sort_values(by=self.datetime_col)
                    plt.plot(time_data[self.datetime_col], time_data[self.target])
                
                plt.title(f'{self.target} over time')
                plt.xlabel(self.datetime_col)
                plt.ylabel(self.target)
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig('figures/time_series_plot.png')
                plt.close()
                
                # Seasonal decomposition if enough data
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    
                    # Sort by datetime
                    time_data = data_sample.sort_values(by=self.datetime_col)
                    
                    # Check if data has a regular frequency
                    time_diff = time_data[self.datetime_col].diff().dropna()
                    
                    if len(time_diff.unique()) <= 5:  # Assuming somewhat regular frequency
                        # Set datetime as index
                        time_data = time_data.set_index(self.datetime_col)
                        
                        # Try to infer frequency
                        try:
                            freq = pd.infer_freq(time_data.index)
                            if freq:
                                # Perform seasonal decomposition
                                result = seasonal_decompose(time_data[self.target], model='additive')
                                
                                # Plot decomposition
                                plt.figure(figsize=(15, 12))
                                plt.subplot(411)
                                plt.plot(result.observed)
                                plt.title('Observed')
                                plt.subplot(412)
                                plt.plot(result.trend)
                                plt.title('Trend')
                                plt.subplot(413)
                                plt.plot(result.seasonal)
                                plt.title('Seasonal')
                                plt.subplot(414)
                                plt.plot(result.resid)
                                plt.title('Residual')
                                plt.tight_layout()
                                plt.savefig('figures/seasonal_decomposition.png')
                                plt.close()
                        except Exception as e:
                            print(f"Could not perform seasonal decomposition: {str(e)}")
                except Exception as e:
                    print(f"Could not perform seasonal decomposition: {str(e)}")
        
        print("Visualizations saved to 'figures' directory.")
    
    def analyze_feature_importance(self, algorithm='random_forest', n_estimators=100, max_depth=10):
        """
        Analyze feature importance using the specified algorithm.
        
        Parameters:
        -----------
        algorithm : str
            Algorithm to use for feature importance ('random_forest', 'xgboost', 'permutation')
        n_estimators : int
            Number of estimators for tree-based models
        max_depth : int
            Maximum depth for tree-based models
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing feature importance scores
        """
        if self.data is None:
            print("Please load data first using load_data() method.")
            return None
        
        if self.features is None or self.target is None:
            print("Please define columns first using define_columns() method.")
            return None
        
        print(f"Analyzing feature importance using {algorithm}...")
        
        # Prepare data
        X = self.data[self.features].copy()
        y = self.data[self.target].copy()
        
        # Handle categorical features
        if self.categorical_cols:
            for col in self.categorical_cols:
                if col in X.columns:
                    X[col] = X[col].astype('category')
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        # Fit preprocessor
        X_processed = preprocessor.fit_transform(X)
        
        # Get feature names after preprocessing
        feature_names = []
        
        # Add numerical feature names
        feature_names.extend(self.numerical_cols)
        
        # Add categorical feature names (with one-hot encoding)
        if self.categorical_cols:
            ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = ohe.get_feature_names_out(self.categorical_cols)
            feature_names.extend(cat_feature_names)
        
        # Calculate feature importance
        importance_df = None
        
        if algorithm == 'random_forest':
            # Use Random Forest for feature importance
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_processed, y)
            
            # Get feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
        elif algorithm == 'xgboost':
            # Use XGBoost for feature importance
            model = xgb.XGBRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_processed, y)
            
            # Get feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            
        elif algorithm == 'permutation':
            # Use permutation importance
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_processed, y)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(model, X_processed, y, n_repeats=10, random_state=42)
            
            # Get feature importance
            importance = perm_importance.importances_mean
            importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Visualize feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
        plt.title(f'Top 20 Feature Importance using {algorithm}')
        plt.tight_layout()
        plt.savefig('figures/feature_importance.png')
        plt.close()
        
        self.feature_importance = importance_df
        print("Feature importance analysis completed.")
        return importance_df
    
    def define_algorithm(self, algorithm_name, hyperparameters=None):
        """
        Define the algorithm and hyperparameters for time series forecasting.
        
        Parameters:
        -----------
        algorithm_name : str
            Name of the algorithm to use
        hyperparameters : dict, optional
            Dictionary of hyperparameters for the algorithm
            
        Returns:
        --------
        dict
            Dictionary containing the algorithm and hyperparameters
        """
        if algorithm_name not in self.available_algorithms:
            print(f"Algorithm {algorithm_name} not available. Available algorithms: {list(self.available_algorithms.keys())}")
            return None
        
        self.algorithm = algorithm_name
        
        # Use default hyperparameters if not provided
        if hyperparameters is None:
            self.hyperparameters = self.available_algorithms[algorithm_name]['default_params']
        else:
            # Merge provided hyperparameters with defaults
            self.hyperparameters = {**self.available_algorithms[algorithm_name]['default_params'], **hyperparameters}
        
        print(f"Algorithm defined: {algorithm_name}")
        print("Hyperparameters:")
        for param, value in self.hyperparameters.items():
            description = self.available_algorithms[algorithm_name]['description'].get(param, "No description available")
            print(f"  - {param}: {value}")
            print(f"    Description: {description}")
        
        return {
            'algorithm': self.algorithm,
            'hyperparameters': self.hyperparameters
        }
    
    def define_time_unit(self, time_unit, forecast_horizon):
        """
        Define the time unit and forecast horizon for time series forecasting.
        
        Parameters:
        -----------
        time_unit : str
            Time unit for forecasting ('hour', 'day', 'month')
        forecast_horizon : int
            Number of time units to forecast
            
        Returns:
        --------
        dict
            Dictionary containing the time unit and forecast horizon
        """
        if time_unit not in ['hour', 'day', 'month']:
            print("Time unit must be one of: 'hour', 'day', 'month'")
            return None
        
        self.time_unit = time_unit
        self.forecast_horizon = forecast_horizon
        
        print(f"Time unit defined: {time_unit}")
        print(f"Forecast horizon: {forecast_horizon} {time_unit}s")
        
        return {
            'time_unit': self.time_unit,
            'forecast_horizon': self.forecast_horizon
        }
    
    def preprocess_data(self, test_size=0.2, specific_item_id=None):
        """
        Preprocess the data for time series forecasting.
        
        Parameters:
        -----------
        test_size : float
            Proportion of data to use for testing
        specific_item_id : any, optional
            Specific item ID to train on
            
        Returns:
        --------
        dict
            Dictionary containing the preprocessed data
        """
        if self.data is None:
            print("Please load data first using load_data() method.")
            return None
        
        if self.features is None or self.target is None or self.datetime_col is None:
            print("Please define columns first using define_columns() method.")
            return None
        
        print("Preprocessing data...")
        
        # Filter data for specific item if provided
        if specific_item_id is not None and self.item_id_col is not None:
            if specific_item_id in self.data[self.item_id_col].unique():
                self.data = self.data[self.data[self.item_id_col] == specific_item_id].copy()
                print(f"Filtered data for {self.item_id_col}={specific_item_id}")
            else:
                print(f"Item ID {specific_item_id} not found in data. Using all data.")
        
        # Sort data by datetime
        self.data = self.data.sort_values(by=self.datetime_col)
        
        # Create time-based features
        self.data['hour'] = self.data[self.datetime_col].dt.hour
        self.data['day'] = self.data[self.datetime_col].dt.day
        self.data['month'] = self.data[self.datetime_col].dt.month
        self.data['year'] = self.data[self.datetime_col].dt.year
        self.data['dayofweek'] = self.data[self.datetime_col].dt.dayofweek
        
        # Add these new features to numerical columns
        time_features = ['hour', 'day', 'month', 'year', 'dayofweek']
        self.numerical_cols.extend(time_features)
        self.features = self.numerical_cols + self.categorical_cols
        
        # Create lag features for the target
        for lag in [1, 2, 3, 5, 7]:
            lag_col = f'{self.target}_lag_{lag}'
            self.data[lag_col] = self.data[self.target].shift(lag)
            self.numerical_cols.append(lag_col)
        
            # Create rolling window features
        for window in [3, 5, 7]:
            # Rolling mean
            mean_col = f'{self.target}_rolling_mean_{window}'
            self.data[mean_col] = self.data[self.target].rolling(window=window).mean()
            self.numerical_cols.append(mean_col)
            
            # Rolling std
            std_col = f'{self.target}_rolling_std_{window}'
            self.data[std_col] = self.data[self.target].rolling(window=window).std()
            self.numerical_cols.append(std_col)
        
        # Update features list ensuring uniqueness
        # Using set to remove any duplicates from numerical_cols
        unique_numerical_cols = list(dict.fromkeys(self.numerical_cols))
        self.numerical_cols = unique_numerical_cols
        self.features = self.numerical_cols + self.categorical_cols
        
        # Drop rows with NaN values (from lag and rolling features)
        self.data = self.data.dropna()
        
        # Split data into train and test sets
        # For time series, we use the last part of the data for testing
        train_size = int(len(self.data) * (1 - test_size))
        train_data = self.data.iloc[:train_size]
        test_data = self.data.iloc[train_size:]
        
        # Store train and test dates and item IDs for later use
        self.train_dates = train_data[self.datetime_col].values
        self.test_dates = test_data[self.datetime_col].values
        
        if self.item_id_col and self.item_id_col in self.data.columns:
            self.item_ids_test = test_data[self.item_id_col].values
        
        # Prepare X and y for training and testing
        X_train = train_data[self.features].copy()
        y_train = train_data[self.target].copy()
        X_test = test_data[self.features].copy()
        y_test = test_data[self.target].copy()
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_cols),
                ('cat', categorical_transformer, self.categorical_cols)
            ])
        
        # Fit preprocessor on training data
        self.preprocessor = preprocessor.fit(X_train)
        
        # Transform data
        self.X_train = self.preprocessor.transform(X_train)
        self.X_test = self.preprocessor.transform(X_test)
        self.y_train = y_train
        self.y_test = y_test
        
        print("Data preprocessing completed.")
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        
        return {
            'X_train': self.X_train,
            'y_train': self.y_train,
            'X_test': self.X_test,
            'y_test': self.y_test,
            'train_dates': self.train_dates,
            'test_dates': self.test_dates
        }
    
    def train_model(self):
        """
        Train the time series forecasting model.
        
        Returns:
        --------
        object
            The trained model
        """
        if self.X_train is None or self.y_train is None:
            print("Please preprocess data first using preprocess_data() method.")
            return None
        
        if self.algorithm is None or self.hyperparameters is None:
            print("Please define algorithm first using define_algorithm() method.")
            return None
        
        print(f"Training {self.algorithm} model...")
        
        # Get model class and hyperparameters
        model_class = self.available_algorithms[self.algorithm]['model']
        
        # Special case for Prophet
        if self.algorithm == 'prophet':
            # Prophet requires a specific data format
            prophet_data = pd.DataFrame({
                'ds': self.train_dates,
                'y': self.y_train
            })
            
            # Create and fit model
            self.model = model_class(**self.hyperparameters)
            self.model.fit(prophet_data)
        else:
            # Create and fit model
            self.model = model_class(**self.hyperparameters)
            self.model.fit(self.X_train, self.y_train)
        
        print("Model training completed.")
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model on the test set.
        
        Returns:
        --------
        dict
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            print("Please train model first using train_model() method.")
            return None
        
        if self.X_test is None or self.y_test is None:
            print("Please preprocess data first using preprocess_data() method.")
            return None
        
        print("Evaluating model...")
        
        # Make predictions
        if self.algorithm == 'prophet':
            # Prophet requires a specific data format
            future = pd.DataFrame({'ds': self.test_dates})
            forecast = self.model.predict(future)
            y_pred = forecast['yhat'].values
        else:
            y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        # Filter out zero values to prevent division by zero
        mask = self.y_test != 0
        if np.any(mask):
            mape = np.mean(np.abs((self.y_test[mask] - y_pred[mask]) / self.y_test[mask])) * 100
        else:
            mape = np.nan  # Use NaN if all values are zero
        
        # Store metrics
        self.evaluation_metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape
        }
        
        # Sanitize metrics to prevent JSON serialization issues with infinite values
        for key, value in self.evaluation_metrics.items():
            if np.isnan(value) or np.isinf(value):
                self.evaluation_metrics[key] = None
        
        print("Evaluation metrics:")
        print(f"  - Mean Squared Error (MSE): {mse:.4f}")
        print(f"  - Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  - Mean Absolute Error (MAE): {mae:.4f}")
        print(f"  - R-squared (R2): {r2:.4f}")
        print(f"  - Mean Absolute Percentage Error (MAPE): {mape:.4f}%")
        
        # Visualize predictions vs actual
        plt.figure(figsize=(15, 6))
        plt.plot(self.test_dates, self.y_test, label='Actual')
        plt.plot(self.test_dates, y_pred, label='Predicted')
        plt.title('Actual vs Predicted Values')
        plt.xlabel(self.datetime_col)
        plt.ylabel(self.target)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('figures/actual_vs_predicted.png')
        plt.close()
        
        # Create a DataFrame with actual and predicted values
        results_df = pd.DataFrame({
            self.datetime_col: self.test_dates,
            'Actual': self.y_test,
            'Predicted': y_pred
        })
        
        # Add item ID if available
        if hasattr(self, 'item_ids_test') and self.item_ids_test is not None:
            results_df[self.item_id_col] = self.item_ids_test
        
        # Save results to CSV
        results_df.to_csv('results/prediction_results.csv', index=False)
        
        return self.evaluation_metrics
    
    def predict_future(self, future_periods=None, future_data=None):
        """
        Predict future values.
        
        Parameters:
        -----------
        future_periods : int, optional
            Number of future periods to predict
        future_data : pandas.DataFrame, optional
            Future data with features for prediction
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing future predictions
        """
        if self.model is None:
            print("Please train model first using train_model() method.")
            return None
        
        if future_periods is None and future_data is None:
            print("Please provide either future_periods or future_data.")
            return None
        
        print("Predicting future values...")
        
        # Case 1: Predict using future_periods
        if future_periods is not None:
            if self.time_unit is None or self.forecast_horizon is None:
                print("Please define time unit first using define_time_unit() method.")
                return None
            
            # Get the last date in the data
            last_date = self.data[self.datetime_col].max()
            
            # Generate future dates
            if self.time_unit == 'hour':
                future_dates = [last_date + timedelta(hours=i+1) for i in range(future_periods)]
            elif self.time_unit == 'day':
                future_dates = [last_date + timedelta(days=i+1) for i in range(future_periods)]
            elif self.time_unit == 'month':
                future_dates = [last_date + pd.DateOffset(months=i+1) for i in range(future_periods)]
            
            # Special case for Prophet
            if self.algorithm == 'prophet':
                # Prophet requires a specific data format
                future = pd.DataFrame({'ds': future_dates})
                forecast = self.model.predict(future)
                future_predictions = forecast['yhat'].values
                
                # Create a DataFrame with predictions
                predictions_df = pd.DataFrame({
                    self.datetime_col: future_dates,
                    'Prediction': future_predictions
                })
                
                return predictions_df
            
            # For other algorithms, we need to generate features for future dates
            future_data = pd.DataFrame({self.datetime_col: future_dates})
            
            # Add time-based features
            future_data['hour'] = future_data[self.datetime_col].dt.hour
            future_data['day'] = future_data[self.datetime_col].dt.day
            future_data['month'] = future_data[self.datetime_col].dt.month
            future_data['year'] = future_data[self.datetime_col].dt.year
            future_data['dayofweek'] = future_data[self.datetime_col].dt.dayofweek
            
            # For lag and rolling features, we need historical data
            # Use the last values from the training data
            last_values = self.data[self.target].tail(10).values
            
            # Add lag features
            for lag in [1, 2, 3, 5, 7]:
                lag_col = f'{self.target}_lag_{lag}'
                if lag < len(last_values):
                    future_data[lag_col] = [last_values[-lag]] * len(future_data)
                else:
                    future_data[lag_col] = [last_values[0]] * len(future_data)
            
            # Add rolling features
            for window in [3, 5, 7]:
                # Rolling mean
                mean_col = f'{self.target}_rolling_mean_{window}'
                if window < len(last_values):
                    future_data[mean_col] = [np.mean(last_values[-window:])] * len(future_data)
                else:
                    future_data[mean_col] = [np.mean(last_values)] * len(future_data)
                
                # Rolling std
                std_col = f'{self.target}_rolling_std_{window}'
                if window < len(last_values):
                    future_data[std_col] = [np.std(last_values[-window:])] * len(future_data)
                else:
                    future_data[std_col] = [np.std(last_values)] * len(future_data)
            
            # Add original numerical features with imputed values (mean from training data)
            for col in self.numerical_cols:
                if col not in future_data.columns and col not in ['hour', 'day', 'month', 'year', 'dayofweek'] and not col.startswith(f'{self.target}_'):
                    if col in self.data.columns:
                        future_data[col] = self.data[col].mean()
                    else:
                        future_data[col] = 0  # Fallback if column not in training data
            
            # Add categorical features with imputed values (mode from training data)
            if self.categorical_cols:
                for col in self.categorical_cols:
                    if col not in future_data.columns and col != self.item_id_col:
                        if col in self.data.columns:
                            most_common = self.data[col].mode()[0]
                            future_data[col] = most_common
                        else:
                            future_data[col] = 'unknown'  # Fallback for unknown categories
            
            # Add item ID if needed
            if self.item_id_col and self.item_id_col not in future_data.columns:
                # Use the last item ID from the training data
                last_item_id = self.data[self.item_id_col].iloc[-1]
                future_data[self.item_id_col] = last_item_id
        
        # Case 2: Use provided future_data
        else:
            # Check if future_data has all required columns
            missing_cols = [col for col in self.features if col not in future_data.columns]
            if missing_cols:
                print(f"Future data is missing the following columns: {missing_cols}")
                # Impute missing columns
                for col in missing_cols:
                    if col in self.numerical_cols:
                        future_data[col] = self.data[col].mean() if col in self.data.columns else 0
                    elif col in self.categorical_cols:
                        future_data[col] = self.data[col].mode()[0] if col in self.data.columns else 'unknown'
        
        # Prepare future data for prediction
        X_future = future_data[self.features].copy()
        
        # Transform future data
        X_future_processed = self.preprocessor.transform(X_future)
        
        # Make predictions
        future_predictions = self.model.predict(X_future_processed)
        
        # Create a DataFrame with predictions
        predictions_df = pd.DataFrame({
            self.datetime_col: future_data[self.datetime_col],
            'Prediction': future_predictions
        })
        
        # Add item ID if available
        if self.item_id_col and self.item_id_col in future_data.columns:
            predictions_df[self.item_id_col] = future_data[self.item_id_col]
        
        # Save predictions to CSV
        os.makedirs('results', exist_ok=True)
        predictions_df.to_csv('results/future_predictions.csv', index=False)
        
        print("Future predictions completed.")
        return predictions_df
    
    def visualize_predictions(self, predictions_df, actual_df=None):
        """
        Visualize predictions.
        
        Parameters:
        -----------
        predictions_df : pandas.DataFrame
            DataFrame containing predictions
        actual_df : pandas.DataFrame, optional
            DataFrame containing actual values
            
        Returns:
        --------
        None
        """
        if predictions_df is None:
            print("Please provide predictions DataFrame.")
            return
        
        print("Visualizing predictions...")
        
        plt.figure(figsize=(15, 6))
        
        # Plot predictions
        plt.plot(predictions_df[self.datetime_col], predictions_df['Prediction'], label='Prediction', color='blue')
        
        # Plot actual values if available
        if actual_df is not None:
            plt.plot(actual_df[self.datetime_col], actual_df[self.target], label='Actual', color='red')
        
        plt.title('Predictions vs Actual Values')
        plt.xlabel(self.datetime_col)
        plt.ylabel(self.target)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save figure
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/predictions_visualization.png')
        plt.close()
        
        print("Predictions visualization saved to 'figures/predictions_visualization.png'.")
    
    def save_model(self, file_path):
        """
        Save the trained model to a file.
        
        Parameters:
        -----------
        file_path : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.model is None:
            print("Please train model first using train_model() method.")
            return False
        
        try:
            import joblib
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save model
            joblib.dump(self.model, file_path)
            
            print(f"Model saved to {file_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {str(e)}")
            return False
    
    def load_model(self, file_path):
        """
        Load a trained model from a file.
        
        Parameters:
        -----------
        file_path : str
            Path to the saved model
            
        Returns:
        --------
        object
            The loaded model
        """
        try:
            import joblib
            
            # Load model
            self.model = joblib.load(file_path)
            
            print(f"Model loaded from {file_path}")
            return self.model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None
