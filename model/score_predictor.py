import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class ScorePredictionModel:
    def __init__(self, dataset: pd.DataFrame, kfold_splits: int = 20):
        self.dataset = self.prepare_data(dataset)
        self.scaler = StandardScaler()
        self.best_model = None
        self.best_score = None
        self.best_params = None
        self.models_performance = {}
        self.kfold_splits = kfold_splits

    def prepare_data(self, dataset: pd.DataFrame) -> dict:
        data = {
            "X": [],
            "y": []
        }

        features = [feature for feature in dataset.columns if feature != "Exam_Score"]
        target = ["Exam_Score"]

        data["X"] = dataset[features].values
        data["y"] = dataset[target].values.ravel()  # ravel() to flatten for sklearn

        return data
    
    def train(self):
        X = self.dataset["X"]
        y = self.dataset["y"]

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Set up cross-validation with configurable splits
        kfold = KFold(n_splits=self.kfold_splits, shuffle=True, random_state=42)
        
        # Define models and their parameter grids
        models = {
            'Linear Regression': {
                'model': LinearRegression(),
                'params': {}
            },
            'Ridge Regression': {
                'model': Ridge(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Lasso Regression': {
                'model': Lasso(),
                'params': {
                    'alpha': [0.1, 1.0, 10.0, 100.0]
                }
            },
            'Random Forest': {
                'model': RandomForestRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [5, 10, None],
                    'min_samples_split': [2, 5]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingRegressor(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            },
            'K-Nearest Neighbors': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': [3, 5, 7, 10],
                    'weights': ['uniform', 'distance']
                }
            }
        }
        
        print(f"Training and evaluating models with {self.kfold_splits}-fold cross-validation...")
        print("=" * 60)
        
        best_cv_score = -np.inf
        
        for name, model_info in models.items():
            print(f"\nTraining {name}...")
            
            if model_info['params']:
                # Hyperparameter tuning with GridSearchCV
                grid_search = GridSearchCV(
                    model_info['model'], 
                    model_info['params'], 
                    cv=kfold, 
                    scoring='r2',
                    n_jobs=-1
                )
                grid_search.fit(X_train_scaled, y_train)
                
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                cv_score = grid_search.best_score_
            else:
                # No hyperparameters to tune
                model = model_info['model']
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=kfold, scoring='r2')
                cv_score = cv_scores.mean()
                model.fit(X_train_scaled, y_train)
                best_model = model
                best_params = {}
            
            # Evaluate on test set
            y_pred = best_model.predict(X_test_scaled)
            test_r2 = r2_score(y_test, y_pred)
            test_mse = mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_rmse = np.sqrt(test_mse)
            
            # Store performance metrics
            self.models_performance[name] = {
                'cv_r2_mean': cv_score,
                'test_r2': test_r2,
                'test_mse': test_mse,
                'test_mae': test_mae,
                'test_rmse': test_rmse,
                'best_params': best_params,
                'model': best_model
            }
            
            print(f"  Cross-validation R² (mean): {cv_score:.4f}")
            print(f"  Test R²: {test_r2:.4f}")
            print(f"  Test RMSE: {test_rmse:.4f}")
            print(f"  Test MAE: {test_mae:.4f}")
            if best_params:
                print(f"  Best parameters: {best_params}")
            
            # Track the best model
            if cv_score > best_cv_score:
                best_cv_score = cv_score
                self.best_model = best_model
                self.best_score = cv_score
                self.best_params = best_params
        
        print("\n" + "=" * 60)
        print("TRAINING SUMMARY")
        print("=" * 60)
        print(f"Best model: {self._get_best_model_name()}")
        print(f"Best cross-validation R² ({self.kfold_splits}-fold): {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")
        
        # Display performance comparison
        self._display_model_comparison()
        
        return self.best_model
    
    def _get_best_model_name(self):
        """Get the name of the best performing model"""
        for name, performance in self.models_performance.items():
            if performance['model'] == self.best_model:
                return name
        return "Unknown"
    
    def _display_model_comparison(self):
        """Display a comparison table of all models"""
        print(f"\nMODEL PERFORMANCE COMPARISON ({self.kfold_splits}-Fold CV)")
        print("-" * 80)
        print(f"{'Model':<20} {'CV R²':<10} {'Test R²':<10} {'Test RMSE':<12} {'Test MAE':<10}")
        print("-" * 80)
        
        # Sort by CV R² score
        sorted_models = sorted(
            self.models_performance.items(), 
            key=lambda x: x[1]['cv_r2_mean'], 
            reverse=True
        )
        
        for name, performance in sorted_models:
            print(f"{name:<20} {performance['cv_r2_mean']:<10.4f} {performance['test_r2']:<10.4f} "
                  f"{performance['test_rmse']:<12.4f} {performance['test_mae']:<10.4f}")
    
    def get_performance_data(self):
        """Get performance data for chart generation"""
        if not self.models_performance:
            raise ValueError("No performance data available. Please train the model first.")
        
        data = {
            'Model': [],
            'CV_R2': [],
            'Test_R2': [],
            'Test_RMSE': [],
            'Test_MAE': [],
            'kfold_splits': self.kfold_splits
        }
        
        for name, performance in self.models_performance.items():
            data['Model'].append(name)
            data['CV_R2'].append(performance['cv_r2_mean'])
            data['Test_R2'].append(performance['test_r2'])
            data['Test_RMSE'].append(performance['test_rmse'])
            data['Test_MAE'].append(performance['test_mae'])
        
        return data
    
    def predict(self, X):
        """Make predictions using the best model"""
        if self.best_model is None:
            raise ValueError("Model not trained yet. Please call train() first.")
        
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)
    
    def save(self, filename=None):
        """Save the trained model and scaler"""
        if self.best_model is None:
            raise ValueError("No model to save. Please train the model first.")
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"score_predictor_{self.kfold_splits}fold_{timestamp}"
        
        # Create saved_models directory if it doesn't exist
        os.makedirs("saved_models", exist_ok=True)
        
        # Save the model and scaler
        model_path = f"saved_models/{filename}_model.joblib"
        scaler_path = f"saved_models/{filename}_scaler.joblib"
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save model metadata
        metadata = {
            'best_model_name': self._get_best_model_name(),
            'best_score': self.best_score,
            'best_params': self.best_params,
            'models_performance': self.models_performance,
            'kfold_splits': self.kfold_splits
        }
        
        metadata_path = f"saved_models/{filename}_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        print(f"Model saved successfully!")
        print(f"  Model: {model_path}")
        print(f"  Scaler: {scaler_path}")
        print(f"  Metadata: {metadata_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }
    
    def load(self, filename):
        """Load a saved model and scaler"""
        model_path = f"saved_models/{filename}_model.joblib"
        scaler_path = f"saved_models/{filename}_scaler.joblib"
        metadata_path = f"saved_models/{filename}_metadata.joblib"
        
        self.best_model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        if os.path.exists(metadata_path):
            metadata = joblib.load(metadata_path)
            self.best_score = metadata.get('best_score')
            self.best_params = metadata.get('best_params')
            self.models_performance = metadata.get('models_performance', {})
            self.kfold_splits = metadata.get('kfold_splits', 20)
        
        print(f"Model loaded successfully from {model_path}")
        print(f"KFold splits: {self.kfold_splits}")