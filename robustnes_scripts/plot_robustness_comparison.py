
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)

BASE_DIR = Path("/home/lh/c/Co_Flu_CN_Wave_Flu")
RESULTS_DIR = BASE_DIR / "Robustness_Results"
OUTPUT_DIR = BASE_DIR / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

SCENARIOS = {
    "Baseline": "Baseline",
    "Conservative": "Conservative",
    "Sensitive_Cgau": "Sensitive (Cgau)",
    "Ablation_NoAFD": "Ablation (No AFD)"
}

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

def plot_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    palette = sns.color_palette("Set2", len(SCENARIOS))
    
    sns.boxplot(data=df, x="Scenario", y="mean_coherence", ax=axes[0], palette=palette, width=0.5)
    axes[0].set_title("Distribution of Mean Coherence")
    axes[0].set_ylabel("Mean Coherence")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis='x', labelsize=10)
    
    sns.boxplot(data=df, x="Scenario", y="significant_high_coherence_ratio", ax=axes[1], palette=palette, width=0.5)
    axes[1].set_title("Distribution of Significant Coherence Ratio")
    axes[1].set_ylabel("Significant Ratio")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis='x', labelsize=10)
    
    plt.tight_layout()
    
    save_path_png = OUTPUT_DIR / "figure_robustness_wtc.png"
    save_path_pdf = OUTPUT_DIR / "figure_robustness_wtc.pdf"
    
    plt.savefig(save_path_png, dpi=600, bbox_inches='tight')
    plt.savefig(save_path_pdf, bbox_inches='tight')
    print(f"Figure saved to {save_path_png}")

def main():
    print("Loading robustness results...")
    df = load_data()
    
    if df is not None:
        print(f"Loaded {len(df)} rows across {df['Scenario'].nunique()} scenarios")
        print("Generating plot...")
        plot_comparison(df)
        print("Done.")
    else:
        print("No data loaded.")

if __name__ == "__main__":
    main()
