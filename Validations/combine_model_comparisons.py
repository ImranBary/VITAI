
# Author: Imran Feisal
# Date: 29/01/2025

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
VALIDATIONS_DIR = os.path.join(DATA_DIR, "validations")

MODEL_NAMES = {
    "combined_diabetes_tabnet": "Diabetes Model",
    "combined_all_ckd_tabnet": "CKD Model",
    "combined_none_tabnet": "Catch-all Model"
}

def load_validation_data():
    """Load all correlation and ANOVA results from validation directories."""
    combined_corrs = []
    combined_anova = []
    
    for model_id, model_name in MODEL_NAMES.items():
        # Load correlations
        corr_path = os.path.join(VALIDATIONS_DIR, model_id, f"{model_id}_correlations.csv")
        if os.path.exists(corr_path):
            df_corr = pd.read_csv(corr_path)
            df_corr["Model"] = model_name
            combined_corrs.append(df_corr)
        
        # Load ANOVA results
        anova_path = os.path.join(VALIDATIONS_DIR, model_id, f"{model_id}_anova.csv")
        if os.path.exists(anova_path):
            df_anova = pd.read_csv(anova_path)
            df_anova["Model"] = model_name
            combined_anova.append(df_anova)
    
    return (
        pd.concat(combined_corrs, ignore_index=True) if combined_corrs else None,
        pd.concat(combined_anova, ignore_index=True) if combined_anova else None
    )

def create_combined_correlation_plot(df_corr):
    """Create combined correlation visualization."""
    plt.figure(figsize=(12, 6))
    
    # Filter only relevant comparisons
    plot_df = df_corr[df_corr["VarX"] == "Predicted_Health_Index"].copy()
    plot_df["Comparison"] = plot_df["VarY"].replace({
        "CharlsonIndex": "Charlson Index",
        "ElixhauserIndex": "Elixhauser Index"
    })
    
    # Create bar plot
    sns.barplot(data=plot_df, x="Comparison", y="Pearson_R", hue="Model", 
                palette="Set2", errorbar=None)
    
    # Add annotations
    for i, p in enumerate(plt.gca().patches):
        height = p.get_height()
        plt.gca().text(p.get_x() + p.get_width()/2., height + 0.02,
                     f'{height:.2f}', ha='center', va='bottom')
    
    plt.title("Pearson Correlation with Comorbidity Indices by Model")
    plt.ylabel("Correlation Coefficient (R)")
    plt.xlabel("")
    plt.ylim(-0.3, 1.0)
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    return plt

def create_statistical_comparison_plot(df_anova):
    """Create visualization for ANOVA/Kruskal results."""
    plt.figure(figsize=(12, 8))
    
    # Melt and clean the data
    melt_df = pd.melt(df_anova, id_vars=["Model", "Measure"], 
                     value_vars=["ANOVA_F", "Kruskal_H"],
                     var_name="Test", value_name="Value")
    
    # Convert test names to readable format
    melt_df["Test"] = melt_df["Test"].replace({
        "ANOVA_F": "ANOVA",
        "Kruskal_H": "Kruskal-Wallis"
    })
    
    # Create grouped bar plot
    g = sns.catplot(data=melt_df, kind="bar",
                   x="Test", y="Value", hue="Model",
                   col="Measure", palette="Set2",
                   height=5, aspect=0.8, sharey=False)
    
    # Add styling
    g.set_titles("{col_name}")
    g.fig.suptitle("Cluster Comparison Statistics Across Models", y=1.05)
    
    # Add value annotations
    for ax in g.axes.flat:
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.1f}", 
                       (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')
    
    # Adjust labels
    g.set_axis_labels("", "Test Statistic")
    g._legend.set_title("Model")
    
    return g.fig

def main():
    # Load all validation results
    df_corr, df_anova = load_validation_data()
    
    # Create output directory
    output_dir = os.path.join(VALIDATIONS_DIR, "combined_results")
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save plots
    if df_corr is not None:
        corr_plot = create_combined_correlation_plot(df_corr)
        corr_plot.savefig(os.path.join(output_dir, "combined_correlations.png"), dpi=300)
        plt.close()
    
    if df_anova is not None:
        stats_plot = create_statistical_comparison_plot(df_anova)
        stats_plot.savefig(os.path.join(output_dir, "combined_statistical_tests.png"), 
                          bbox_inches="tight", dpi=300)
        plt.close()
    
    print(f"Combined visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()