#!/usr/bin/env python3
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_metrics(
    df,
    metrics,
    explanation_text,
    out_prefix,
    highlight_list,
    subfolder=None
):
    """
    Generates bar charts for each metric in 'metrics',
    sorted appropriately. Saves them with filenames starting
    with 'out_prefix' in the given 'subfolder' if provided.
    """
    # If a subfolder is specified, ensure it exists
    if subfolder:
        os.makedirs(subfolder, exist_ok=True)

    for metric in metrics:
        # Filter out rows with no value for this metric
        df_metric = df.dropna(subset=[metric]).copy()
        if df_metric.empty:
            print(f"[INFO] No valid data for metric={metric}, skipping.")
            continue

        # Decide how to sort & whether "higher" or "lower" is better
        if metric in ["final_davies_bouldin", "dbscan_davies_bouldin", "tabnet_mse"]:
            sort_ascending = True
            better_text = "(lower is better)"
        else:
            sort_ascending = False
            better_text = "(higher is better)"

        df_metric.sort_values(by=metric, ascending=sort_ascending, inplace=True)

        # Make figure dynamically sized based on number of bars
        n_bars = len(df_metric)
        fig_width = max(14, n_bars * 0.6)
        fig, ax = plt.subplots(figsize=(fig_width, 7), dpi=200)
        plt.subplots_adjust(right=0.75, bottom=0.3)

        sns.barplot(
            data=df_metric,
            x="config_id",
            y=metric,
            palette="rocket",
            ax=ax,
            width=0.5
        )

        plt.title(f"{metric} {better_text} (sorted)")
        plt.xticks(rotation=45, ha="right", fontsize=9)

        # Highlight selected models
        for i, patch in enumerate(ax.patches):
            config_id = df_metric.iloc[i]["config_id"]
            if config_id in highlight_list:
                patch.set_color("forestgreen")  # highlight colour

        # Annotate each bar
        for i, patch in enumerate(ax.patches):
            value = patch.get_height()
            offset = 0.01 * abs(value)
            text_y = value + offset if value >= 0 else value - offset
            ax.text(
                patch.get_x() + patch.get_width() / 2.0,
                text_y,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                rotation=90
            )

        # Also colour/bold the x‐label for highlighted models
        for label in ax.get_xticklabels():
            if label.get_text() in highlight_list:
                label.set_color("forestgreen")
                label.set_fontweight("bold")

        # Explanation text box on the right
        ax.text(
            1.02, 0.5,
            explanation_text,
            transform=ax.transAxes,
            fontsize=9,
            va='center',
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", alpha=0.8),
        )

        # Decide final filename
        if subfolder:
            out_name = os.path.join(subfolder, f"{out_prefix}_{metric}.png")
        else:
            out_name = f"{out_prefix}_{metric}.png"

        plt.savefig(out_name, bbox_inches="tight")
        plt.close()
        print(f"[SAVED] {out_name}")

def main(csv_file, out_prefix, highlight_list):
    # 1) Read your final results CSV
    df = pd.read_csv(csv_file)

    # 2) Full set of metrics
    all_metrics = [
        "final_silhouette",
        "final_calinski",
        "final_davies_bouldin",
        "dbscan_silhouette",
        "dbscan_calinski",
        "dbscan_davies_bouldin",
        "tabnet_mse",
        "tabnet_r2"
    ]

    # 3) Chosen “most insightful” metrics for separate presentation folder
    presentation_metrics = [
        "final_silhouette",  # example
        "final_davies_bouldin",
        "tabnet_mse",
        "tabnet_r2"
    ]

    # 4) Explanation text on the right
    explanation_text = (
        "Meaning of config_id segments:\n\n"
        "Feature config:\n"
        " • composite    => uses Health_Index only\n"
        " • cci          => uses CharlsonIndex only\n"
        " • eci          => uses ElixhauserIndex only\n"
        " • combined     => Health_Index + Charlson\n"
        " • combined_eci => Health_Index + Elixhauser\n"
        " • combined_all => Health_Index + Charlson + Elixhauser\n\n"
        "Subset:\n"
        " • none         => entire dataset\n"
        " • diabetes     => diabetic subpopulation\n"
        " • ckd          => CKD subpopulation\n\n"
        "Model approach:\n"
        " • vae     => unsupervised VAE\n"
        " • tabnet  => TabNet regressor (supervised)\n"
        " • hybrid  => combined VAE + TabNet\n"
    )

    # 5) First: produce plots for *all metrics*, saved in current folder
    plot_metrics(
        df=df,
        metrics=all_metrics,
        explanation_text=explanation_text,
        out_prefix=out_prefix,
        highlight_list=highlight_list,
        subfolder=None
    )

    # 6) Then: produce a smaller set of plots for the “presentation” folder
    presentation_folder = "presentation_plots"
    plot_metrics(
        df=df,
        metrics=presentation_metrics,
        explanation_text=explanation_text,
        out_prefix=out_prefix,
        highlight_list=highlight_list,
        subfolder=presentation_folder
    )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate full metric plots plus a smaller subset for presentation.")
    parser.add_argument("--csv-file", required=True, help="Path to the final results CSV.")
    parser.add_argument("--out-prefix", default="vitai_charts", help="Prefix for output image filenames.")
    parser.add_argument(
        "--highlight-models",
        nargs="+",
        default=["combined_diabetes_tabnet", "combined_all_ckd_tabnet", "combined_none_tabnet"],
        help="List of config_id values to highlight in the bar charts."
    )
    args = parser.parse_args()
    main(args.csv_file, args.out_prefix, args.highlight_models)
