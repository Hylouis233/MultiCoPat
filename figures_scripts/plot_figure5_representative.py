
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from pathlib import Path
import sys

PROJECT_ROOT = Path('/home/lh/c/Co_Flu_CN_Wave_Flu')
sys.path.append(str(PROJECT_ROOT))

from figures_scripts.utils import save_figure, setup_plotting_style, load_stl_results

CASES_STRUCTURE = [
    {
        'label': 'A. Macro Indicator\n(ILI% vs Humidity)',
        'North': {
            'prefix': 'phase_ILI%北方_vs_ERA5-Land气象再分析数据（北方）_',
            'suffix': 'd2m',
            'region': 'Northern'
        },
        'South': {
            'prefix': 'phase_ILI%南方_vs_ERA5-Land气象再分析数据（南方）_',
            'suffix': 'd2m',
            'region': 'Southern'
        }
    },
    {
        'label': 'B. Influenza A/H1N1\n(vs Temperature)',
        'North': {
            'prefix': 'phase_病毒监测和分型（北方）甲型 H1N1_vs_ERA5-Land气象再分析数据（北方）_',
            'suffix': 't2m',
            'region': 'Northern'
        },
        'South': {
            'prefix': 'phase_病毒监测和分型（南方）甲型 H1N1_vs_ERA5-Land气象再分析数据（南方）_',
            'suffix': 't2m',
            'region': 'Southern'
        }
    },
    {
        'label': 'C. Influenza A/H3N2\n(vs Soil Water)',
        'North': {
            'prefix': 'phase_病毒监测和分型（北方）A(H3N2)_vs_ERA5-Land气象再分析数据（北方）_',
            'suffix': 'swvl4',
            'region': 'Northern'
        },
        'South': {
            'prefix': 'phase_病毒监测和分型（南方）A(H3N2)_vs_ERA5-Land气象再分析数据（南方）_',
            'suffix': 'swvl4',
            'region': 'Southern'
        }
    },
    {
        'label': 'D. Influenza B/Yamagata\n(vs Radiation/Precip)',
        'North': {
            'prefix': 'phase_病毒监测和分型（北方）Yamagata_vs_ERA5-Land气象再分析数据（北方）_',
            'suffix': 'ssrd',
            'region': 'Northern'
        },
        'South': {
            'prefix': 'phase_病毒监测和分型（南方）Yamagata_vs_ERA5-Land气象再分析数据（南方）_',
            'suffix': 'tp',
            'region': 'Southern'
        }
    }
]

def load_data(prefix, suffix, region):
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    
    co_circ_df = pd.read_csv(PROJECT_ROOT / 'co_circulation_periods.csv')
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])
    
    is_co_circ = np.zeros(len(time_index), dtype=bool)
    region_periods = co_circ_df[co_circ_df['Region'] == region]
    for _, row in region_periods.iterrows():
        mask = (time_index >= row['Adjusted Start Date']) & (time_index <= row['Adjusted End Date'])
        is_co_circ |= mask
        
    file_name = f"{prefix}{suffix}.npy"
    file_path = PROJECT_ROOT / 'AFD/wtc_analysis/coherence_data' / file_name
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_name}")
        return None, None
        
    phase_matrix = np.load(file_path)
    
    min_len = min(phase_matrix.shape[1], len(time_index))
    phase_matrix = phase_matrix[:, :min_len]
    current_co_circ = is_co_circ[:min_len]
    
    daily_phases = stats.circmean(phase_matrix, high=np.pi, low=-np.pi, axis=0)
    
    phases_single = daily_phases[~current_co_circ]
    phases_co = daily_phases[current_co_circ]
    
    phases_single = phases_single[~np.isnan(phases_single)]
    phases_co = phases_co[~np.isnan(phases_co)]
    
    return phases_single, phases_co

def plot_rose_enhanced(ax, phases, bins, color, R_val=None, y_limit=None):
    hist, bin_edges = np.histogram(phases, bins=bins, range=(-np.pi, np.pi))
    hist = hist / hist.sum()
    
    width = 2 * np.pi / bins
    theta = bin_edges[:-1] + width/2
    
    bars = ax.bar(theta, hist, width=width, bottom=0.0, color=color, alpha=0.7, edgecolor='black', linewidth=0.5)
    
    ax.grid(True, linestyle=':', alpha=0.5, color='gray')
    
    mean_phase = stats.circmean(phases, high=np.pi, low=-np.pi)
    if R_val is None:
        R_val = np.sqrt(np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2) / len(phases)
    
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
        
    arrow_len = R_val * y_limit
    ax.annotate('', xy=(mean_phase, arrow_len), xytext=(0, 0),
                arrowprops=dict(facecolor='black', edgecolor='black', width=2.0, headwidth=8))
    
    r_step = 0.1
    
    if y_limit is None:
        max_val = hist.max()
        y_limit = np.ceil(max_val * 10) / 10.0
        if y_limit < 0.3: y_limit = 0.3 # Minimum scale
    
    ticks = np.arange(r_step, y_limit + 0.001, r_step)
    ax.set_yticks(ticks)
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', labelsize=6, pad=0)
    
    ax.set_ylim(0, y_limit)
    
    ax.set_rlabel_position(45)

    weeks = [1, 10, 20, 30, 40, 50]
    
    week_angles = [w/52.0 * 2 * np.pi for w in weeks]
    week_labels = [f"{w}w" for w in weeks]
    
    ax.set_xticks(week_angles)
    ax.set_xticklabels(week_labels, fontsize=7)
    ax.tick_params(axis='x', pad=15)
    
    lag_weeks = (mean_phase / (2 * np.pi)) * 52
    if lag_weeks < 0: lag_weeks += 52
    
    stats_text = (f"Phase: {mean_phase:.2f}\nLag: {lag_weeks:.1f} wk\nR: {R_val:.2f}")
    ax.text(0.5, -0.25, stats_text, transform=ax.transAxes,
            ha='center', va='top', fontsize=8, 
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='lightgray', boxstyle='round,pad=0.2'))

def create_representative_grid():
    setup_plotting_style()
    
    fig = plt.figure(figsize=(16, 18))
    
    gs = GridSpec(4, 5, width_ratios=[0.2, 1, 1, 1, 1], hspace=0.4, wspace=0.6)
    
    bins = 100
    color_single = '#3498db'
    color_co = '#e74c3c'
    
    all_max_densities = []
    plot_data = [] # Store loaded data to avoid reloading
    
    print("Loading data and calculating global scale...")
    
    for row_idx, case_pair in enumerate(CASES_STRUCTURE):
        n_case = case_pair['North']
        p_s_n, p_c_n = load_data(n_case['prefix'], n_case['suffix'], n_case['region'])
        
        s_case = case_pair['South']
        p_s_s, p_c_s = load_data(s_case['prefix'], s_case['suffix'], s_case['region'])
        
        row_data = {
            'n_single': p_s_n, 'n_co': p_c_n,
            's_single': p_s_s, 's_co': p_c_s
        }
        plot_data.append(row_data)
    
    for row_idx, data in enumerate(plot_data):
        case_info = CASES_STRUCTURE[row_idx]
        
        ax_label = fig.add_subplot(gs[row_idx, 0])
        ax_label.axis('off')
        ax_label.text(0.5, 0.5, case_info['label'], 
                     ha='center', va='center', rotation=90, 
                     fontsize=11, fontweight='bold')
        
        max_d_n = 0
        if data['n_single'] is not None:
            h, _ = np.histogram(data['n_single'], bins=bins, range=(-np.pi, np.pi))
            max_d_n = max(max_d_n, (h/h.sum()).max())
        if data['n_co'] is not None:
            h, _ = np.histogram(data['n_co'], bins=bins, range=(-np.pi, np.pi))
            max_d_n = max(max_d_n, (h/h.sum()).max())
        
        max_d_s = 0
        if data['s_single'] is not None:
            h, _ = np.histogram(data['s_single'], bins=bins, range=(-np.pi, np.pi))
            max_d_s = max(max_d_s, (h/h.sum()).max())
        if data['s_co'] is not None:
            h, _ = np.histogram(data['s_co'], bins=bins, range=(-np.pi, np.pi))
            max_d_s = max(max_d_s, (h/h.sum()).max())
            
        y_limit_n = max_d_n * 1.25 if max_d_n > 0 else 0.1
        y_limit_s = max_d_s * 1.25 if max_d_s > 0 else 0.1
        
        if data['n_single'] is not None:
            ax = fig.add_subplot(gs[row_idx, 1], projection='polar')
            plot_rose_enhanced(ax, data['n_single'], bins, color_single, y_limit=y_limit_n)
            if row_idx == 0: ax.set_title("Single", fontsize=10, pad=10)
            
        if data['n_co'] is not None:
            ax = fig.add_subplot(gs[row_idx, 2], projection='polar')
            plot_rose_enhanced(ax, data['n_co'], bins, color_co, y_limit=y_limit_n)
            if row_idx == 0: ax.set_title("Co-circulation", fontsize=10, pad=10)

        if data['s_single'] is not None:
            ax = fig.add_subplot(gs[row_idx, 3], projection='polar')
            plot_rose_enhanced(ax, data['s_single'], bins, color_single, y_limit=y_limit_s)
            if row_idx == 0: ax.set_title("Single", fontsize=10, pad=10)

        if data['s_co'] is not None:
            ax = fig.add_subplot(gs[row_idx, 4], projection='polar')
            plot_rose_enhanced(ax, data['s_co'], bins, color_co, y_limit=y_limit_s)
            if row_idx == 0: ax.set_title("Co-circulation", fontsize=10, pad=10)

    fig.text(0.38, 0.96, "Northern China", ha='center', fontsize=14, fontweight='bold')
    fig.text(0.72, 0.96, "Southern China", ha='center', fontsize=14, fontweight='bold')

    save_figure(fig, 'figure5_representative_grid')
    print("Figure generated: figures/figure5_representative_grid.pdf")

if __name__ == "__main__":
    create_representative_grid()
