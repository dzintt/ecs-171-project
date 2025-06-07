import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def plot_training_curves(model, X_train, y_train, X_test, y_test, model_name, kfold_splits):
    """
    Plot MSE training curves for models that support iterative training
    """
    # Only create training curves for models that support it
    if hasattr(model, 'partial_fit') or isinstance(model, (SGDRegressor, RandomForestRegressor, GradientBoostingRegressor)):
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # For iterative models like SGD
        if isinstance(model, SGDRegressor) or hasattr(model, 'partial_fit'):
            train_errors = []
            test_errors = []
            epochs = range(1, 101)  # 100 epochs
            
            # Clone the model for iterative training
            from sklearn.base import clone
            iterative_model = clone(model)
            iterative_model.set_params(max_iter=1, warm_start=True)
            
            for epoch in epochs:
                iterative_model.fit(X_train, y_train)
                
                # Calculate MSE for train and test
                train_pred = iterative_model.predict(X_train)
                test_pred = iterative_model.predict(X_test)
                
                train_mse = mean_squared_error(y_train, train_pred)
                test_mse = mean_squared_error(y_test, test_pred)
                
                train_errors.append(train_mse)
                test_errors.append(test_mse)
                
                # Increase max_iter for next iteration
                iterative_model.set_params(max_iter=epoch + 1)
        
        # For ensemble models, use validation curve approach
        elif isinstance(model, (RandomForestRegressor, GradientBoostingRegressor)):
            from sklearn.model_selection import validation_curve
            
            if isinstance(model, RandomForestRegressor):
                param_name = 'n_estimators'
                param_range = range(10, 201, 10)  # 10 to 200 trees
                xlabel = 'Number of Trees'
            else:  # GradientBoostingRegressor
                param_name = 'n_estimators'
                param_range = range(10, 201, 10)  # 10 to 200 estimators
                xlabel = 'Number of Estimators'
            
            train_scores, test_scores = validation_curve(
                model, X_train, y_train, param_name=param_name,
                param_range=param_range, cv=3, scoring='neg_mean_squared_error'
            )
            
            # Convert to positive MSE
            train_errors = -train_scores.mean(axis=1)
            test_errors = -test_scores.mean(axis=1)
            epochs = param_range
            
        # Plot the curves
        ax.plot(epochs, train_errors, 'b-', label='Train MSE', linewidth=2)
        ax.plot(epochs, test_errors, 'r-', label='Test MSE', linewidth=2)
        
        ax.set_xlabel(xlabel if 'xlabel' in locals() else 'Epoch', fontsize=16)
        ax.set_ylabel('Mean Squared Error', fontsize=16)
        ax.set_title(f'{model_name} - MSE Train vs Test over {xlabel if "xlabel" in locals() else "Epochs"}', 
                    fontsize=18, fontweight='bold', pad=20)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        plt.savefig(f'plots/training_curves_{model_name.lower().replace(" ", "_")}_{kfold_splits}fold.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        return True
    else:
        print(f"Training curves not available for {model_name} (no iterative training support)")
        return False


def plot_actual_vs_predicted(model, X, y, scaler, model_name, kfold_splits, cv_folds):
    """
    Create actual vs predicted scatter plot using cross-validation predictions
    """
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Get cross-validation predictions
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv_folds)
    
    # Calculate metrics
    r2 = r2_score(y, y_pred_cv)
    mse = mean_squared_error(y, y_pred_cv)
    rmse = np.sqrt(mse)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Scatter plot
    ax.scatter(y, y_pred_cv, alpha=0.6, s=50, color='blue', edgecolors='navy', linewidth=0.5)
    
    # Perfect prediction line (diagonal)
    min_val = min(y.min(), y_pred_cv.min())
    max_val = max(y.max(), y_pred_cv.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Trend line
    z = np.polyfit(y, y_pred_cv, 1)
    p = np.poly1d(z)
    ax.plot(y, p(y), "r-", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Labels and title
    ax.set_xlabel('Actual Values', fontsize=16)
    ax.set_ylabel('Predicted Values', fontsize=16)
    ax.set_title(f'{model_name} - Actual vs Predicted Values\n'
                f'({kfold_splits}-fold cross validation)', 
                fontsize=18, fontweight='bold', pad=20)
    
    # Add metrics text
    textstr = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMSE = {mse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Equal aspect ratio for better visualization
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(f'plots/actual_vs_predicted_{model_name.lower().replace(" ", "_")}_{kfold_splits}fold.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    return r2, rmse, mse


def create_comprehensive_model_diagnostics(model, X, y, scaler, model_name, kfold_splits, cv_folds, X_train=None, y_train=None, X_test=None, y_test=None):
    """
    Create comprehensive diagnostic plots for a trained model
    """
    print(f"\n📊 Creating diagnostic plots for {model_name}...")
    
    # Plot 1: Training curves (if available)
    if X_train is not None and y_train is not None and X_test is not None and y_test is not None:
        X_train_scaled = scaler.transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        plot_training_curves(model, X_train_scaled, y_train, X_test_scaled, y_test, model_name, kfold_splits)
    
    # Plot 2: Actual vs Predicted
    r2, rmse, mse = plot_actual_vs_predicted(model, X, y, scaler, model_name, kfold_splits, cv_folds)
    
    print(f"✅ Diagnostic plots created for {model_name}")
    print(f"   Cross-validation R²: {r2:.4f}")
    print(f"   Cross-validation RMSE: {rmse:.4f}")
    
    return {
        'r2': r2,
        'rmse': rmse,
        'mse': mse
    }


def plot_residuals(model, X, y, scaler, model_name, kfold_splits, cv_folds):
    """
    Create residual plots to check for patterns in prediction errors
    """
    # Scale the features
    X_scaled = scaler.transform(X)
    
    # Get cross-validation predictions
    y_pred_cv = cross_val_predict(model, X_scaled, y, cv=cv_folds)
    
    # Calculate residuals
    residuals = y - y_pred_cv
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Residual Analysis ({kfold_splits}-fold CV)', 
                 fontsize=20, fontweight='bold')
    
    # Plot 1: Residuals vs Predicted
    axes[0, 0].scatter(y_pred_cv, residuals, alpha=0.6, color='blue')
    axes[0, 0].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Predicted Values', fontsize=12)
    axes[0, 0].set_ylabel('Residuals', fontsize=12)
    axes[0, 0].set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Residuals vs Actual
    axes[0, 1].scatter(y, residuals, alpha=0.6, color='green')
    axes[0, 1].axhline(y=0, color='red', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Actual Values', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residuals vs Actual', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Histogram of residuals
    axes[1, 0].hist(residuals, bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Q-Q plot (normal probability plot)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Normal Distribution)', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'plots/residual_analysis_{model_name.lower().replace(" ", "_")}_{kfold_splits}fold.png', 
               dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print residual statistics
    print(f"\n📈 Residual Analysis for {model_name}:")
    print(f"   Mean residual: {np.mean(residuals):.6f}")
    print(f"   Std residual: {np.std(residuals):.4f}")
    print(f"   Min residual: {np.min(residuals):.4f}")
    print(f"   Max residual: {np.max(residuals):.4f}")
    
    return residuals 