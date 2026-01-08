
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import to_rgb, LinearSegmentedColormap

from utils import load_integrated_features, save_figure, setup_plotting_style

def plot_figure6():
    print("Generating Figure 6: Regime Shift Comprehensive Analysis...")
    
    df = load_integrated_features()
        
    feature_cols = [c for c in df.columns if c not in ['Series_Name', 'Region', 'Period_Type', 'Label']]
    feature_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    
    X = df[feature_cols].values
    X = np.nan_to_num(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    df['PC1'] = X_pca[:, 0]
    df['PC2'] = X_pca[:, 1]
    
    kmeans_2 = KMeans(n_clusters=2, random_state=42)
    clusters_2 = kmeans_2.fit_predict(X_pca)
    
    c0_mean = X_pca[clusters_2==0, 0].mean()
    c1_mean = X_pca[clusters_2==1, 0].mean()
    
    if c0_mean < c1_mean:
        ordered_cluster = 0
        chaotic_cluster = 1
    else:
        ordered_cluster = 1
        chaotic_cluster = 0
        
    df['Regime_Cluster'] = clusters_2
    
    LABEL_ORDERED = 'Single-dominant Regime (Ordered)'
    LABEL_CHAOTIC = 'Co-circulation Regime (Chaotic)'
    
    df['Regime_Label'] = df['Regime_Cluster'].map({
        ordered_cluster: LABEL_ORDERED, 
        chaotic_cluster: LABEL_CHAOTIC
    })
    
    COLOR_ORDERED = '#1f77b4'
    COLOR_CHAOTIC = '#d62728'
    
    palette_map = {
        LABEL_ORDERED: COLOR_ORDERED,
        LABEL_CHAOTIC: COLOR_CHAOTIC
    }
    
    marker_map = {
        LABEL_ORDERED: 'X',
        LABEL_CHAOTIC: 'o'
    }
    
    feature_rename_dict = {
        'significant_coherence_ratio': 'High Coupling Ratio (Sig.)',
        'high_coherence_ratio': 'High Coherence Area',
        'mean_coherence': 'Mean Coupling Strength',
        'CWT_freq_drift_rate': 'Seasonal Frequency Drift',
        'CWT_freq_stability': 'Seasonal Frequency Stability',
        'max_coherence': 'Max Coupling Strength',
        'std_phase': 'Phase Instability',
        'T_slope': 'Trend Growth Rate',
        'T_cv': 'Trend Variability',
        'AFD_mean_component_energy': 'Transient Signal Energy',
        'AFD_total_energy': 'Total Residual Energy',
        'EnergyVariability': 'Energy Variability',
        'CWT_seasonal_energy_cv': 'Seasonal Energy Variability',
        'R_mean': 'Residual Mean Level',
        'S_amplitude': 'Seasonal Amplitude',
        'S_std': 'Seasonal Std. Dev.',
        'T_std': 'Trend Std. Dev.',
        'positive_phase_ratio': 'Positive Phase Ratio',
        'negative_phase_ratio': 'Negative Phase Ratio',
        'K_Residual': 'AFD Component Count'
    }
    
    from sklearn.ensemble import RandomForestClassifier
    y = (df['Regime_Cluster'] == chaotic_cluster).astype(int)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_scaled, y)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 15
    top_indices = indices[:top_n]
    top_features_raw = [feature_cols[i] for i in top_indices]
    top_features_clean = [feature_rename_dict.get(f, f.replace('_', ' ').title()) for f in top_features_raw]
    top_importances = importances[top_indices]
    
    direction_colors = []
    chaotic_mask = df['Regime_Cluster'] == chaotic_cluster
    ordered_mask = df['Regime_Cluster'] == ordered_cluster
    
    for feat in top_features_raw:
        mean_chaotic = df.loc[chaotic_mask, feat].mean()
        mean_ordered = df.loc[ordered_mask, feat].mean()
        if mean_chaotic > mean_ordered:
            direction_colors.append(COLOR_CHAOTIC)
        else:
            direction_colors.append(COLOR_ORDERED)

    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(2, 2, height_ratios=[1, 1.2], width_ratios=[1, 1], hspace=0.4, wspace=0.6)
    
    ax_pca = fig.add_subplot(gs[0, 0])
    
    sns.scatterplot(
        data=df, x='PC1', y='PC2', 
        hue='Regime_Label', style='Regime_Label', 
        palette=palette_map, markers=marker_map,
        s=120, alpha=0.9, ax=ax_pca,         edgecolor='w', linewidth=0.5
    )
    
    def plot_confidence_ellipse(data, ax, color, label, y_offset_factor=2.5):
        if len(data) < 3: return
        cov = np.cov(data, rowvar=False)
        lambda_, v = np.linalg.eig(cov)
        lambda_ = np.sqrt(lambda_)
        ell = Ellipse(xy=np.mean(data, axis=0),
                      width=lambda_[0]*4, height=lambda_[1]*4,
                      angle=np.rad2deg(np.arccos(v[0, 0])),
                      color=color, alpha=0.1)
        center = np.mean(data, axis=0)
        ax.text(center[0], center[1] + (lambda_[1]*y_offset_factor if y_offset_factor > 0 else lambda_[1]*y_offset_factor), 
                label, color=color, ha='center', fontweight='bold', fontsize=12,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

    ordered_pca = df.loc[ordered_mask, ['PC1', 'PC2']].values
    plot_confidence_ellipse(ordered_pca, ax_pca, COLOR_ORDERED, "Ordered State", y_offset_factor=-2.5)
    
    chaotic_pca = df.loc[chaotic_mask, ['PC1', 'PC2']].values
    plot_confidence_ellipse(chaotic_pca, ax_pca, COLOR_CHAOTIC, "Disordered State", y_offset_factor=2.5)
    
    xlims = ax_pca.get_xlim()
    ylims = ax_pca.get_ylim()
    arrow_y = ylims[0] - (ylims[1]-ylims[0])*0.12
    ax_pca.annotate('', xy=(xlims[1], arrow_y), xytext=(xlims[0], arrow_y),
                    arrowprops=dict(arrowstyle='<->', lw=2, color='gray'),
                    annotation_clip=False)
    ax_pca.text((xlims[0]+xlims[1])/2, arrow_y - (ylims[1]-ylims[0])*0.05, 
                "Higher Stability      |      Lower Stability", 
                ha='center', va='top', fontsize=11, color='gray', fontweight='bold')

    from sklearn.metrics import silhouette_score
    sil_score = silhouette_score(X_scaled, clusters_2)
    ax_pca.text(0.02, 0.82, f"Silhouette Score: {sil_score:.2f}\n(N={len(df)} samples)", 
                transform=ax_pca.transAxes, fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'), va='top')


    ax_pca.set_title('(A) PCA Dynamics Space: Regime Shift', fontsize=16, fontweight='bold', loc='left')
    ax_pca.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
    ax_pca.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
    ax_pca.legend(loc='upper left', frameon=True, fontsize=10)
    ax_pca.grid(True, linestyle=':', alpha=0.5)
    
    ax_imp = fig.add_subplot(gs[0, 1])
    
    y_pos = np.arange(len(top_features_clean))
    bars = ax_imp.barh(y_pos, top_importances, color=direction_colors, alpha=0.85)
    
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(top_features_clean, fontsize=11)
    ax_imp.invert_yaxis()
    
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color=COLOR_CHAOTIC, lw=6),
                    Line2D([0], [0], color=COLOR_ORDERED, lw=6)]
    ax_imp.legend(custom_lines, ['Higher in Co-circulation', 'Lower in Co-circulation'], 
                  loc='lower right', fontsize=10, frameon=True, title="Direction of Change")
    
    ax_imp.set_title('(B) Top Discriminative Features', fontsize=16, fontweight='bold', loc='left')
    ax_imp.set_xlabel('Relative Importance (Gini)', fontsize=12)
    ax_imp.grid(axis='x', linestyle='--', alpha=0.6)
    
    ax_heat = fig.add_subplot(gs[1, :])
    
    heat_indices = top_indices
    X_heat = X_scaled[:, heat_indices]
    heat_cols_labels = [top_features_clean[i] for i in range(len(heat_indices))]
    
    df['sort_key_regime'] = df['Regime_Cluster'].map({ordered_cluster: 0, chaotic_cluster: 1})
    df['sort_idx'] = df['sort_key_regime'] * 1000 + df['PC1']
    
    sorted_sample_idx = df.sort_values('sort_idx').index
    
    X_heat_sorted = X_heat[sorted_sample_idx, :].T 
    
    im = ax_heat.imshow(X_heat_sorted, aspect='auto', cmap='RdBu_r', vmin=-2.5, vmax=2.5)
    
    ax_heat.set_title('(C) Coupled Heatmap: Feature Patterns across Regimes', fontsize=16, fontweight='bold', loc='left', pad=40)
    ax_heat.set_xlabel('Samples (Sorted by Regime & Stability)', fontsize=12)
    
    ax_heat.set_yticks(np.arange(len(heat_cols_labels)))
    ax_heat.set_yticklabels(heat_cols_labels, fontsize=10)
    
    ax_heat.set_xticks([])
    
    sorted_regimes = df.loc[sorted_sample_idx, 'Regime_Label']
    regime_colors_hex = sorted_regimes.map(palette_map).tolist()
    regime_colors_rgb = [to_rgb(c) for c in regime_colors_hex]
    divider = np.array([regime_colors_rgb])
    
    pos = ax_heat.get_position()
    cbar_height = pos.height * 0.05
    ax_colorbar = fig.add_axes([pos.x0, pos.y1 + 0.005, pos.width, cbar_height])
    ax_colorbar.imshow(divider, aspect='auto')
    ax_colorbar.set_xticks([])
    ax_colorbar.set_yticks([])
    ax_colorbar.set_title("Regime Indicator", fontsize=10, loc='left', pad=5)
    
    cbar_ax = fig.add_axes([pos.x1 + 0.01, pos.y0, 0.015, pos.height])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Z-score Normalized Value', fontsize=10)
    
    sorted_sort_keys = df.loc[sorted_sample_idx, 'sort_key_regime'].values
    boundaries = np.where(sorted_sort_keys[:-1] != sorted_sort_keys[1:])[0]
    
    for b in boundaries:
        ax_heat.axvline(x=b+0.5, color='white', linestyle='-', linewidth=2)
        ax_colorbar.axvline(x=b+0.5, color='white', linestyle='-', linewidth=2)

    save_figure(fig, 'figure6_integrated_stats')
    plt.close()

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure6()
