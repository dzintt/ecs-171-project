import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def create_model_performance_charts(performance_data=None):

    data = performance_data
    
    df = pd.DataFrame({k: v for k, v in data.items() if k != 'kfold_splits'})
    kfold_splits = data['kfold_splits']
    
    plt.style.use('seaborn-v0_8')
    plt.rcParams.update({'font.size': 18})
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 20))
    fig.suptitle(f'Model Performance Comparison ({kfold_splits}-Fold Cross-Validation)', 
                 fontsize=32, fontweight='bold', y=0.98)
    
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Model'], df['CV_R2'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax1.set_title('Cross-Validation R² Score', fontsize=30, fontweight='bold', pad=30)
    ax1.set_ylabel('R² Score', fontsize=22)
    ax1.set_ylim(0, max(df['CV_R2']) * 1.1)
    ax1.grid(axis='y', alpha=0.3)
    ax1.tick_params(axis='both', which='major', labelsize=24)
    
    for bar, value in zip(bars1, df['CV_R2']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=30)
    
    ax1.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=20)
    
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['Model'], df['Test_R2'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax2.set_title('Test R² Score', fontsize=30, fontweight='bold', pad=30)
    ax2.set_ylabel('R² Score', fontsize=22)
    ax2.set_ylim(0, max(df['Test_R2']) * 1.1)
    ax2.grid(axis='y', alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=24)
    
    for bar, value in zip(bars2, df['Test_R2']):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=30)
    
    ax2.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=20)
    
    ax3 = axes[1, 0]
    bars3 = ax3.bar(df['Model'], df['Test_RMSE'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax3.set_title('Test RMSE (Lower is Better)', fontsize=30, fontweight='bold', pad=30)
    ax3.set_ylabel('RMSE', fontsize=22)
    ax3.set_ylim(min(df['Test_RMSE']) * 0.9, max(df['Test_RMSE']) * 1.05)
    ax3.grid(axis='y', alpha=0.3)
    ax3.tick_params(axis='both', which='major', labelsize=24)
    
    for bar, value in zip(bars3, df['Test_RMSE']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=30)
    
    ax3.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=20)
    
    ax4 = axes[1, 1]
    bars4 = ax4.bar(df['Model'], df['Test_MAE'], color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_title('Test MAE (Lower is Better)', fontsize=30, fontweight='bold', pad=30)
    ax4.set_ylabel('MAE', fontsize=22)
    ax4.set_ylim(min(df['Test_MAE']) * 0.9, max(df['Test_MAE']) * 1.05)
    ax4.grid(axis='y', alpha=0.3)
    ax4.tick_params(axis='both', which='major', labelsize=24)
    
    for bar, value in zip(bars4, df['Test_MAE']):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=30)
    
    ax4.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=20)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90, bottom=0.20)
    
    plt.savefig(f'plots/model_performance_comparison_{kfold_splits}fold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    create_summary_comparison_chart(df, kfold_splits)

def create_summary_comparison_chart(df, kfold_splits):
    
    fig, ax = plt.subplots(figsize=(22, 12))
    
    x = np.arange(len(df['Model']))
    width = 0.2
    
    rmse_score = 1 / (1 + df['Test_RMSE'] / min(df['Test_RMSE']))
    mae_score = 1 / (1 + df['Test_MAE'] / min(df['Test_MAE']))
    
    bars1 = ax.bar(x - 1.5*width, df['CV_R2'], width, label='Cross-Val R²', alpha=0.8)
    bars2 = ax.bar(x - 0.5*width, df['Test_R2'], width, label='Test R²', alpha=0.8)
    bars3 = ax.bar(x + 0.5*width, rmse_score, width, label='RMSE Score (inverted)', alpha=0.8)
    bars4 = ax.bar(x + 1.5*width, mae_score, width, label='MAE Score (inverted)', alpha=0.8)
    
    ax.set_title(f'Model Performance Summary - {kfold_splits}-Fold Cross-Validation\n(All metrics normalized - Higher is Better)', 
                fontsize=26, fontweight='bold', pad=40)
    ax.set_ylabel('Performance Score', fontsize=22)
    ax.set_xlabel('Models', fontsize=22)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Model'], rotation=45, ha='right', fontsize=20)
    ax.tick_params(axis='y', which='major', labelsize=24)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=30)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)
    plt.savefig(f'plots/model_performance_summary_{kfold_splits}fold.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*70)
    print(f"MODEL RANKING SUMMARY ({kfold_splits}-Fold Cross-Validation)")
    print("="*70)
    print(f"Best CV R²: {df.loc[df['CV_R2'].idxmax(), 'Model']}")
    print(f"Best Test R²: {df.loc[df['Test_R2'].idxmax(), 'Model']}")
    print(f"Best RMSE (lowest): {df.loc[df['Test_RMSE'].idxmin(), 'Model']}")
    print(f"Best MAE (lowest): {df.loc[df['Test_MAE'].idxmin(), 'Model']}")

if __name__ == "__main__":
    create_model_performance_charts() 