import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from dataset import load_data, clean_null_values, filter_features

# Set up plotting style for publication-quality figures
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False
})

# Create report plots directory if it doesn't exist
REPORT_PLOTS_DIR = "plots/report"
if not os.path.exists(REPORT_PLOTS_DIR):
    os.makedirs(REPORT_PLOTS_DIR, exist_ok=True)
    print(f"📁 Created report plots directory: {REPORT_PLOTS_DIR}")

# Features used in the analysis
FEATURES = [
    "Attendance",
    "Hours_Studied", 
    "Sleep_Hours",
    "Tutoring_Sessions",
    "Access_to_Resources",
    "Extracurricular_Activities",
    "Teacher_Quality"
]

def load_filtered_dataset():
    """Load and filter the dataset with the selected features."""
    dataset = load_data()
    cleaned_dataset = clean_null_values(dataset)
    filtered_dataset = filter_features(cleaned_dataset, features_to_keep=FEATURES)
    print(f"📊 Loaded dataset with {len(filtered_dataset)} samples and {len(FEATURES)} features")
    return filtered_dataset

def generate_feature_distributions(dataset: pd.DataFrame):
    """Generate individual feature distribution plots for the report."""
    print("\n📈 Generating feature distribution plots...")
    
    # Create a figure with subplots for each feature
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Distribution of Student Performance Factors', fontsize=16, y=0.98)
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        
        # Create histogram with statistics
        n, bins, patches = ax.hist(dataset[feature], bins=30, alpha=0.7, 
                                 color=sns.color_palette("husl", len(FEATURES))[i],
                                 edgecolor='black', linewidth=0.5)
        
        # Add mean and median lines
        mean_val = dataset[feature].mean()
        median_val = dataset[feature].median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, 
                  label=f'Median: {median_val:.2f}')
        
        # Formatting
        ax.set_title(f'{feature.replace("_", " ")}', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add basic statistics as text
        std_val = dataset[feature].std()
        ax.text(0.02, 0.98, f'σ: {std_val:.2f}', transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", 
                facecolor="white", alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(FEATURES), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    save_path = os.path.join(REPORT_PLOTS_DIR, "feature_distributions.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Feature distributions saved to: {save_path}")

def generate_kernel_density_plots(dataset: pd.DataFrame):
    """Generate kernel density estimation plots for all features."""
    print("\n📈 Generating kernel density plots...")
    
    # Create a comprehensive KDE plot
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Kernel Density Estimation of Student Performance Factors', fontsize=16, y=0.98)
    
    axes = axes.flatten()
    
    for i, feature in enumerate(FEATURES):
        ax = axes[i]
        
        # Create KDE plot
        sns.kdeplot(data=dataset, x=feature, ax=ax, fill=True, alpha=0.6,
                   color=sns.color_palette("husl", len(FEATURES))[i])
        
        # Add distribution statistics
        mean_val = dataset[feature].mean()
        std_val = dataset[feature].std()
        
        # Add mean line
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                  label=f'Mean: {mean_val:.2f}')
        
        # Add ±1 std deviation shading
        ax.axvspan(mean_val - std_val, mean_val + std_val, alpha=0.2, 
                  color='gray', label='±1 SD')
        
        # Formatting
        ax.set_title(f'{feature.replace("_", " ")} - KDE', fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add skewness and kurtosis
        skewness = stats.skew(dataset[feature])
        kurtosis = stats.kurtosis(dataset[feature])
        # ax.text(0.02, 0.98, f'Skew: {skewness:.2f}\nKurt: {kurtosis:.2f}', 
        #         transform=ax.transAxes, verticalalignment='top',
        #         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Remove empty subplots
    for i in range(len(FEATURES), len(axes)):
        axes[i].remove()
    
    plt.tight_layout()
    save_path = os.path.join(REPORT_PLOTS_DIR, "kernel_density_plots.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Kernel density plots saved to: {save_path}")

def analyze_feature_cardinality(dataset: pd.DataFrame):
    """Analyze and visualize high and low cardinality features."""
    print("\n📈 Analyzing feature cardinality...")
    
    # Calculate cardinality for each feature
    cardinality_data = []
    for feature in FEATURES:
        unique_count = dataset[feature].nunique()
        total_count = len(dataset)
        cardinality_ratio = unique_count / total_count
        
        cardinality_data.append({
            'Feature': feature.replace("_", " "),
            'Unique_Values': unique_count,
            'Total_Values': total_count,
            'Cardinality_Ratio': cardinality_ratio,
            'Category': 'High Cardinality' if cardinality_ratio > 0.1 else 'Low Cardinality'
        })
    
    cardinality_df = pd.DataFrame(cardinality_data)
    
    # Create cardinality visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Feature Cardinality Analysis', fontsize=16)
    
    # Plot 1: Unique values count
    colors = ['#ff7f0e' if cat == 'High Cardinality' else '#1f77b4' 
              for cat in cardinality_df['Category']]
    
    bars1 = ax1.bar(range(len(cardinality_df)), cardinality_df['Unique_Values'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax1.set_title('Number of Unique Values per Feature', fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Unique Value Count')
    ax1.set_xticks(range(len(cardinality_df)))
    ax1.set_xticklabels(cardinality_df['Feature'], rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars1, cardinality_df['Unique_Values'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Cardinality ratio
    bars2 = ax2.bar(range(len(cardinality_df)), cardinality_df['Cardinality_Ratio'], 
                   color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_title('Cardinality Ratio (Unique/Total)', fontweight='bold')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Cardinality Ratio')
    ax2.set_xticks(range(len(cardinality_df)))
    ax2.set_xticklabels(cardinality_df['Feature'], rotation=45, ha='right')
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=2, 
               label='High/Low Threshold (0.1)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars2, cardinality_df['Cardinality_Ratio'])):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    save_path = os.path.join(REPORT_PLOTS_DIR, "feature_cardinality.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Feature cardinality analysis saved to: {save_path}")
    
    # Print cardinality summary
    print("\n📊 CARDINALITY ANALYSIS SUMMARY")
    print("=" * 50)
    high_card = cardinality_df[cardinality_df['Category'] == 'High Cardinality']
    low_card = cardinality_df[cardinality_df['Category'] == 'Low Cardinality']
    
    print(f"\nHigh Cardinality Features ({len(high_card)}):")
    for _, row in high_card.iterrows():
        print(f"  • {row['Feature']}: {row['Unique_Values']} unique values "
              f"(ratio: {row['Cardinality_Ratio']:.3f})")
    
    print(f"\nLow Cardinality Features ({len(low_card)}):")
    for _, row in low_card.iterrows():
        print(f"  • {row['Feature']}: {row['Unique_Values']} unique values "
              f"(ratio: {row['Cardinality_Ratio']:.3f})")
    
    return cardinality_df

def generate_comparative_analysis(dataset: pd.DataFrame):
    """Generate comparative analysis plots for the report."""
    print("\n📈 Generating comparative analysis plots...")
    
    # Create normalized comparison plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Comparative Feature Analysis for Student Performance Factors', fontsize=16)
    
    # Plot 1: Box plots for all features (normalized)
    # Normalize features for comparison
    normalized_data = dataset[FEATURES].copy()
    for feature in FEATURES:
        normalized_data[feature] = (dataset[feature] - dataset[feature].min()) / \
                                  (dataset[feature].max() - dataset[feature].min())
    
    # Melt data for seaborn
    melted_data = normalized_data.melt(var_name='Feature', value_name='Normalized_Value')
    melted_data['Feature'] = melted_data['Feature'].str.replace('_', ' ')
    
    sns.boxplot(data=melted_data, x='Feature', y='Normalized_Value', ax=ax1)
    ax1.set_title('Normalized Feature Distributions (Box Plots)', fontweight='bold')
    ax1.set_xlabel('Features')
    ax1.set_ylabel('Normalized Value (0-1)')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Violin plots for distribution shapes
    sns.violinplot(data=melted_data, x='Feature', y='Normalized_Value', ax=ax2)
    ax2.set_title('Feature Distribution Shapes (Violin Plots)', fontweight='bold')
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Normalized Value (0-1)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(REPORT_PLOTS_DIR, "comparative_analysis.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Comparative analysis saved to: {save_path}")

def generate_summary_statistics_table(dataset: pd.DataFrame):
    """Generate a comprehensive summary statistics table."""
    print("\n📊 Generating summary statistics...")
    
    # Calculate comprehensive statistics
    stats_data = []
    for feature in FEATURES:
        data = dataset[feature]
        stats_data.append({
            'Feature': feature.replace('_', ' '),
            'Count': len(data),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std': data.std(),
            'Min': data.min(),
            'Max': data.max(),
            'Q25': data.quantile(0.25),
            'Q75': data.quantile(0.75),
            'Skewness': stats.skew(data),
            'Kurtosis': stats.kurtosis(data),
            'Unique_Values': data.nunique()
        })
    
    stats_df = pd.DataFrame(stats_data)
    
    # Save as CSV for reference
    csv_path = os.path.join(REPORT_PLOTS_DIR, "summary_statistics.csv")
    stats_df.to_csv(csv_path, index=False)
    print(f"✅ Summary statistics saved to: {csv_path}")
    
    # Print formatted table
    print("\n📊 SUMMARY STATISTICS TABLE")
    print("=" * 100)
    print(stats_df.round(3).to_string(index=False))
    
    return stats_df

def generate_all_report_charts():
    """Generate all charts needed for the project report."""
    print("🎯 GENERATING REPORT CHARTS FOR STUDENT PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    # Load the filtered dataset
    dataset = load_filtered_dataset()
    
    # Generate all required charts
    generate_feature_distributions(dataset)
    generate_kernel_density_plots(dataset) 
    cardinality_df = analyze_feature_cardinality(dataset)
    generate_comparative_analysis(dataset)
    stats_df = generate_summary_statistics_table(dataset)
    
    print(f"\n🎉 ALL REPORT CHARTS GENERATED SUCCESSFULLY!")
    print("=" * 70)
    print(f"📂 Charts saved in: {os.path.abspath(REPORT_PLOTS_DIR)}")
    print("\nGenerated files:")
    print("  📊 feature_distributions.png - Individual feature histograms")
    print("  📊 kernel_density_plots.png - KDE plots for all features")
    print("  📊 feature_cardinality.png - High/low cardinality analysis")
    print("  📊 comparative_analysis.png - Normalized comparative plots")
    print("  📄 summary_statistics.csv - Comprehensive statistics table")
    
    return dataset, cardinality_df, stats_df

if __name__ == "__main__":
    generate_all_report_charts()
