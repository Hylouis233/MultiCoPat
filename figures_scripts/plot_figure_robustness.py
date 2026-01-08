import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

from utils import save_figure, setup_plotting_style, PROJECT_ROOT

SCENARIOS = {
    "Baseline": "Baseline",
    "Conservative": "Conservative",
    "Sensitive_Cgau": "Sensitive (Cgau)",
    "Ablation_NoAFD": "Ablation (No AFD)"
}

RESULTS_DIR = PROJECT_ROOT / "Robustness_Results"

def load_data():
    dfs = []
    for folder, label in SCENARIOS.items():
        csv_path = RESULTS_DIR / folder / "wtc_analysis" / "wtc_coherence_features.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["Scenario"] = label
            dfs.append(df)
        else:
            print(f"Warning: {csv_path} not found")
    
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)

def plot_robustness_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    palette = sns.color_palette("Set2", len(SCENARIOS))
    
    sns.boxplot(data=df, x="Scenario", y="mean_coherence", ax=axes[0], palette=palette)
    axes[0].set_title("(A) Distribution of Mean Coherence")
    axes[0].set_ylabel("Mean Coherence")
    axes[0].set_xlabel("")
    
    sns.boxplot(data=df, x="Scenario", y="significant_high_coherence_ratio", ax=axes[1], palette=palette)
    axes[1].set_title("(B) Distribution of Significant Coherence Ratio")
    axes[1].set_ylabel("Significant Ratio")
    axes[1].set_xlabel("")
    
    plt.tight_layout()
    
    save_figure(fig, "figure_robustness_wtc")
    plt.close(fig)

def main():
    print("Generating Figure Robustness...")
    
    if not RESULTS_DIR.exists():
        print(f"Error: Robustness Results directory not found: {RESULTS_DIR}")
        return
        
    df = load_data()
    
    if df is not None:
        print(f"Loaded {len(df)} rows across {df['Scenario'].nunique()} scenarios")
        plot_robustness_comparison(df)
    else:
        print("No data loaded. Skipping...")

if __name__ == "__main__":
    setup_plotting_style()
    main()
