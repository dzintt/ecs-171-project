from data.dataset import load_data, clean_null_values, filter_features
from data.eda import run_full_eda
from model.score_predictor import ScorePredictionModel
from plots.model_comparison_charts import create_model_performance_charts
from plots.model_diagnostics import create_comprehensive_model_diagnostics, plot_residuals
import os


def load_and_clean_data():
    print("Loading and cleaning dataset...")
    dataset = load_data()
    cleaned_dataset = clean_null_values(dataset)
    cleaned_dataset = filter_features(cleaned_dataset, features_to_keep=[
        "Attendance",
        "Hours_Studied",
        "Sleep_Hours",
        "Tutoring_Sessions",
        "Access_to_Resources",
        "Extracurricular_Activities",
        "Teacher_Quality",
        "Exam_Score"
    ])
    print(f"Dataset loaded successfully! Shape: {cleaned_dataset.shape}")
    return cleaned_dataset


def get_kfold_input():
    while True:
        try:
            print("\nCross-Validation Configuration")
            print("Common values: 5 (fast), 10 (standard), 20 (thorough), 30 (extensive)")
            kfold = input("Enter number of KFold splits (5-30, default=10): ").strip()
            
            if kfold == "":
                return 10
            
            kfold = int(kfold)
            
            if 2 <= kfold <= 30:
                return kfold
            else:
                print("Please enter a value between 2 and 30.")
        except ValueError:
            print("Please enter a valid number.")


def train_and_save_model(dataset, kfold_splits=10):
    print("\n" + "="*60)
    print(f"STARTING MODEL TRAINING ({kfold_splits}-Fold CV)")
    print("="*60)
    
    model = ScorePredictionModel(dataset, kfold_splits=kfold_splits)
    model.train()
    
    print("\n" + "="*60)
    print("SAVING MODEL")
    print("="*60)
    saved_paths = model.save()
    
    return model, saved_paths


def generate_charts(model):
    print("\n" + "="*60)
    print("GENERATING PERFORMANCE CHARTS")
    print("="*60)
    try:
        performance_data = model.get_performance_data()
        create_model_performance_charts(performance_data)
        kfold_splits = performance_data['kfold_splits']
        print(f"Performance charts saved to plots/ directory:")
        print(f"  model_performance_comparison_{kfold_splits}fold.png")
        print(f"  model_performance_summary_{kfold_splits}fold.png")
    except Exception as e:
        print(f"Could not generate charts: {e}")


def generate_diagnostic_plots(model, dataset):
    print("\n" + "="*60)
    print("GENERATING DIAGNOSTIC PLOTS")
    print("="*60)
    
    try:
        X = model.dataset["X"]
        y = model.dataset["y"]
        best_model = model.best_model
        scaler = model.scaler
        kfold_splits = model.kfold_splits
        best_model_name = model._get_best_model_name()
        
        from sklearn.model_selection import KFold
        cv_folds = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)
        
        metrics = create_comprehensive_model_diagnostics(
            best_model, X, y, scaler, best_model_name, kfold_splits, cv_folds
        )
        
        residuals = plot_residuals(
            best_model, X, y, scaler, best_model_name, kfold_splits, cv_folds
        )
        
        print(f"\nDiagnostic plots created for {best_model_name}:")
        print(f"  Training curves (if applicable)")
        print(f"  Actual vs Predicted scatter plot")  
        print(f"  Residual analysis (4 plots)")
        print(f"  All saved to plots/ directory")
        
        return metrics
        
    except Exception as e:
        print(f"Could not generate diagnostic plots: {e}")
        return None


def train_single_model_with_diagnostics(dataset, model_name, kfold_splits=10):
    print(f"\nTraining {model_name} with full diagnostic analysis...")
    
    temp_model = ScorePredictionModel(dataset, kfold_splits=kfold_splits)
    
    models_dict = {
        'Linear Regression': temp_model._get_single_model('Linear Regression'),
        'Ridge Regression': temp_model._get_single_model('Ridge Regression'),
        'Random Forest': temp_model._get_single_model('Random Forest'),
        'Gradient Boosting': temp_model._get_single_model('Gradient Boosting')
    }
    
    if model_name not in models_dict:
        print(f"Model {model_name} not available")
        return None
    
    single_model = ScorePredictionModel(dataset, kfold_splits=kfold_splits)
    single_model._train_single_model(model_name)
    
    return generate_diagnostic_plots(single_model, dataset)


def demonstrate_prediction(model, dataset):
    print("\n" + "="*60)
    print("PREDICTION DEMONSTRATION")
    print("="*60)
    
    features = [col for col in dataset.columns if col != "Exam_Score"]
    sample_data = dataset[features].iloc[:5]
    actual_scores = dataset["Exam_Score"].iloc[:5]
    
    predictions = model.predict(sample_data.values)
    
    print("Sample Predictions:")
    print("-" * 40)
    for i, (pred, actual) in enumerate(zip(predictions, actual_scores)):
        print(f"Student {i+1}: Predicted = {pred:.2f}, Actual = {actual:.2f}, Difference = {abs(pred-actual):.2f}")


def compare_kfold_experiments(dataset):
    print("\n" + "="*60)
    print("KFOLD COMPARISON EXPERIMENT")
    print("="*60)
    
    kfold_values = [5, 10, 20]
    print("Testing KFold values:", kfold_values)
    
    results = {}
    
    for kfold in kfold_values:
        print(f"\nTraining with {kfold}-fold cross-validation...")
        model = ScorePredictionModel(dataset, kfold_splits=kfold)
        model.train()
        
        generate_charts(model)
        
        results[kfold] = {
            'best_model': model._get_best_model_name(),
            'best_score': model.best_score,
            'performance_data': model.get_performance_data()
        }
    
    print("\n" + "="*80)
    print("KFOLD COMPARISON SUMMARY")
    print("="*80)
    print(f"{'KFold':<8} {'Best Model':<20} {'Best CV R²':<12}")
    print("-" * 80)
    
    for kfold, result in results.items():
        print(f"{kfold:<8} {result['best_model']:<20} {result['best_score']:<12.4f}")
    
    return results


def main():
    dataset = load_and_clean_data()
    
    while True:
        print("\n" + "="*60)
        print("MAIN MENU")
        print("="*60)
        print("1. Run Exploratory Data Analysis (EDA)")
        print("2. Train Model with Custom KFold")
        print("3. Train Model + Generate Charts")
        print("4. Train Model + Full Diagnostics (Charts + Plots)")
        print("5. Compare Different KFold Values (5, 10, 20)")
        print("6. Quick Train (Default Settings)")
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                print("\nRunning EDA...")
                run_full_eda(dataset)
                
            elif choice == "2":
                kfold_splits = get_kfold_input()
                model, saved_paths = train_and_save_model(dataset, kfold_splits)
                demonstrate_prediction(model, dataset)
                
            elif choice == "3":
                kfold_splits = get_kfold_input()
                model, saved_paths = train_and_save_model(dataset, kfold_splits)
                generate_charts(model)
                demonstrate_prediction(model, dataset)
                print("\nComplete pipeline executed successfully!")
                print(f"Model saved at: {saved_paths['model_path']}")
                print(f"Charts generated with {kfold_splits}-fold CV")
                
            elif choice == "4":
                kfold_splits = get_kfold_input()
                model, saved_paths = train_and_save_model(dataset, kfold_splits)
                generate_charts(model)
                generate_diagnostic_plots(model, dataset)
                demonstrate_prediction(model, dataset)
                print("\nComplete pipeline with diagnostics executed!")
                print(f"Model saved at: {saved_paths['model_path']}")
                print(f"All charts and diagnostic plots generated!")
                
            elif choice == "5":
                results = compare_kfold_experiments(dataset)
                print("\nKFold comparison completed!")
                print("Charts generated for each KFold value")
                
            elif choice == "6":
                print("\nUsing default settings (10-fold CV)...")
                model, saved_paths = train_and_save_model(dataset, 10)
                generate_charts(model)
                demonstrate_prediction(model, dataset)
                print("\nQuick training completed!")
                
            else:
                print("Invalid choice. Please enter 1-6.")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")


if __name__ == "__main__":
    main()


