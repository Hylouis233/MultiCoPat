
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

ENV_VARS = {
    'Temperature': {'label': 'Temperature (t2m)', 'suffix': 't2m'},
    'Humidity': {'label': 'Dewpoint (d2m)', 'suffix': 'd2m'},
    'Radiation': {'label': 'Solar Radiation (ssrd)', 'suffix': 'ssrd'},
    'Precipitation': {'label': 'Precipitation (tp)', 'suffix': 'tp'},
    'WindSpeed': {'label': 'Wind Speed (u10)', 'suffix': 'u10'},
    'Pressure': {'label': 'Surface Pressure (sp)', 'suffix': 'sp'},
    'SoilWater': {'label': 'Soil Water (swvl4)', 'suffix': 'swvl4'}
}

VIRAL_TARGETS = {
    'Macro': [
        {'name': 'ILI% North', 'prefix': 'phase_ILI%北方_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'},
        {'name': 'ILI% South', 'prefix': 'phase_ILI%南方_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'},
        {'name': 'Pos Rate North', 'prefix': 'phase_流感阳性率北方_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'},
        {'name': 'Pos Rate South', 'prefix': 'phase_流感阳性率南方_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'}
    ],
    'North_Strains': [
        {'name': 'A/H1N1 (North)', 'prefix': 'phase_病毒监测和分型（北方）甲型 H1N1_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'},
        {'name': 'A/H3N2 (North)', 'prefix': 'phase_病毒监测和分型（北方）A(H3N2)_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'},
        {'name': 'B/Victoria (North)', 'prefix': 'phase_病毒监测和分型（北方）Victoria_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'},
        {'name': 'B/Yamagata (North)', 'prefix': 'phase_病毒监测和分型（北方）Yamagata_vs_ERA5-Land气象再分析数据（北方）_', 'region': 'Northern'}
    ],
    'South_Strains': [
        {'name': 'A/H1N1 (South)', 'prefix': 'phase_病毒监测和分型（南方）甲型 H1N1_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'},
        {'name': 'A/H3N2 (South)', 'prefix': 'phase_病毒监测和分型（南方）A(H3N2)_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'},
        {'name': 'B/Victoria (South)', 'prefix': 'phase_病毒监测和分型（南方）Victoria_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'},
        {'name': 'B/Yamagata (South)', 'prefix': 'phase_病毒监测和分型（南方）Yamagata_vs_ERA5-Land气象再分析数据（南方）_', 'region': 'Southern'}
    ]
}

def load_data(file_name, region):
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

def plot_rose(ax, phases, bins, color, title, R_val=None, show_stats=True, y_limit=None):
    t_min, t_max = -np.pi, np.pi
        
    hist, bin_edges = np.histogram(phases, bins=bins, range=(-np.pi, np.pi))
    
    hist = hist / hist.sum()
    
    width = 2 * np.pi / bins
    theta = bin_edges[:-1] + width/2
    
    bars = ax.bar(theta, hist, width=width, bottom=0.0, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    
    mean_phase = stats.circmean(phases, high=np.pi, low=-np.pi)
    
    if R_val is None:
        R_val = np.sqrt(np.sum(np.cos(phases))**2 + np.sum(np.sin(phases))**2) / len(phases)
    
    max_hist = np.max(hist)
    
    if y_limit is None:
        if max_hist > 0:
            y_limit = max_hist * 1.25
        else:
            y_limit = 0.1
    
    y_limit = np.ceil(y_limit * 10) / 10.0
    
    ax.set_ylim(0, y_limit)
        
    arrow_len = R_val * y_limit
    
    ax.annotate('', xy=(mean_phase, arrow_len), xytext=(0, 0),
                arrowprops=dict(facecolor='black', edgecolor='black', width=1.5, headwidth=6))
    
    tick_step = np.pi/6
    start_tick = -np.pi
    end_tick = np.pi
    
    tick_vals = np.arange(start_tick, end_tick, tick_step)
    
    tick_labels = []
    for val in tick_vals:
        deg = np.degrees(val)
        tick_labels.append(f"{int(round(deg))}°")
            
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_labels)
    
    ax.set_thetamin(0)
    ax.set_thetamax(360)

    ax.tick_params(axis='y', labelsize=8)
    
    r_step = 0.1
    if y_limit > 0.5:
        r_step = 0.2
    
    ticks = np.arange(r_step, y_limit + 0.001, r_step)
    ax.set_yticks(ticks)
    
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f')) # 保留1位小数
    
    ax.set_rlabel_position(45) 
    
    ax.set_title(title, fontsize=9, fontweight='bold', pad=25)
    
    if show_stats:
        lag_weeks = (mean_phase / (2 * np.pi)) * 52
        stats_text = (f"Phase: {mean_phase:.2f} rad\nLag: ~{lag_weeks:.1f} wks\nR: {R_val:.2f}")
        ax.text(0.5, -0.15, stats_text, transform=ax.transAxes,
                ha='center', va='top', fontsize=8, 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray', boxstyle='round,pad=0.3'))

def create_merged_plot(env_key, env_info):
    groups = ['Macro', 'North_Strains', 'South_Strains']
    n_rows = 4
    n_cols = 6 
    
    fig = plt.figure(figsize=(32, 24)) 
    gs = GridSpec(n_rows, n_cols, hspace=0.5, wspace=0.4)
    
    bins = 60
    
    for g_idx, group_name in enumerate(groups):
        items = VIRAL_TARGETS[group_name]
        col_offset = g_idx * 2
        
        for i, item in enumerate(items):
            if i >= n_rows: break 
            
            file_name = f"{item['prefix']}{env_info['suffix']}.npy"
            print(f"Processing {item['name']} vs {env_info['suffix']}...")
            phases_single, phases_co = load_data(file_name, item['region'])
            
            if phases_single is None:
                continue
                
            R_single = np.sqrt(np.sum(np.cos(phases_single))**2 + np.sum(np.sin(phases_single))**2) / len(phases_single)
            R_co = np.sqrt(np.sum(np.cos(phases_co))**2 + np.sum(np.sin(phases_co))**2) / len(phases_co)
            
            hist_single, _ = np.histogram(phases_single, bins=bins, range=(-np.pi, np.pi))
            hist_single = hist_single / hist_single.sum()
            
            hist_co, _ = np.histogram(phases_co, bins=bins, range=(-np.pi, np.pi))
            hist_co = hist_co / hist_co.sum()
            
            max_density = max(np.max(hist_single), np.max(hist_co))
            
            if max_density > 0:
                y_limit = max_density * 1.25
            else:
                y_limit = 0.1
            
            ax1 = fig.add_subplot(gs[i, col_offset], projection='polar')
            plot_rose(ax1, phases_single, bins, '#3498db', 
                      f"{item['name']} vs {env_info['label']}\nSingle-Dominant", 
                      R_val=R_single, y_limit=y_limit)
            
            ax2 = fig.add_subplot(gs[i, col_offset+1], projection='polar')
            plot_rose(ax2, phases_co, bins, '#e74c3c', 
                      f"{item['name']} vs {env_info['label']}\nCo-circulation", 
                      R_val=R_co, y_limit=y_limit)
                  
    save_figure(fig, f'figure5_wtc_rose_{env_key}')
    plt.close()

def main():
    setup_plotting_style()
    
    for env_key, env_info in ENV_VARS.items():
        print(f"\nGenerating figure for {env_key}...")
        create_merged_plot(env_key, env_info)

if __name__ == "__main__":
    main()
