import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from utils import load_stl_results, load_afd_results, save_figure, setup_plotting_style

NAME_MAPPING = {
    'ILI%北方': 'ILI% (North)',
    'ILI%南方': 'ILI% (South)',
    '流感阳性率北方': 'Positivity Rate (North)',
    '流感阳性率南方': 'Positivity Rate (South)'
}

TARGET_SERIES = list(NAME_MAPPING.keys())

def plot_afd_structure():
    print("Generating AFD fine structure figure (2x2 Layout)...")
    
    stl_array, metadata = load_stl_results()
    afd_data = load_afd_results()
    
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']
    
    from utils import load_co_circulation_periods
    co_circ_df = load_co_circulation_periods()
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    target_indices = []
    valid_targets = []
    
    for name in TARGET_SERIES:
        idx = series_names.index(name)
        target_indices.append(idx)
        valid_targets.append(name)
            
    if not target_indices:
        print("Error: No target series found")
        return

    fig = plt.figure(figsize=(16, 12))
    outer_grid = GridSpec(2, 2, figure=fig, wspace=0.2, hspace=0.3)
    
    for i, (idx, name) in enumerate(zip(target_indices, valid_targets)):
        residual = stl_array[idx, :, 2]
        
        raw_components = afd_data[idx]
        components = raw_components.T
        
        n_time = len(time_index)
        if components.shape[1] != n_time:
            if components.shape[1] > n_time:
                components = components[:, :n_time]
            else:
                pad_width = n_time - components.shape[1]
                components = np.pad(components, ((0,0), (0, pad_width)), constant_values=np.nan)
        
        row = i // 2
        col = i % 2
        
        from matplotlib.gridspec import GridSpecFromSubplotSpec
        sub_gs = GridSpecFromSubplotSpec(4, 1, subplot_spec=outer_grid[row, col], hspace=0.05, height_ratios=[1.5, 1, 1, 1])
        
        eng_name = NAME_MAPPING.get(name, name)
        
        if 'North' in eng_name:
            region_key = 'Northern'
        elif 'South' in eng_name:
            region_key = 'Southern'
        else:
            region_key = None
            
        if co_circ_df is not None and region_key:
            periods = co_circ_df[co_circ_df['Region'] == region_key]
        else:
            periods = None
            
        plot_single_region(fig, sub_gs, time_index, residual, components, eng_name, periods)
        
    save_figure(fig, 'figure_afd_components_structure')
    plt.close()

def filter_components(components, top_k=3):
    variances = np.nanvar(components, axis=1)
    
    valid_indices = np.where(variances > 1e-10)[0]
    
    if len(valid_indices) == 0:
        return [], []
    
    sorted_valid_indices = valid_indices[np.argsort(variances[valid_indices])[::-1]]
    
    top_indices = sorted_valid_indices[:top_k]
    top_components = components[top_indices]
    
    return top_components, top_indices

def plot_single_region(fig, grid_spec, time, residual, components, title, co_circ_periods=None):
    ax_res = fig.add_subplot(grid_spec[0])
    ax_res.plot(time, residual, color='black', lw=1, label='STL Residual')
    ax_res.set_title(f"STL Residual: {title}", loc='left', fontsize=12, fontweight='bold')
    ax_res.grid(True, linestyle='--', alpha=0.5)
    ax_res.set_xticklabels([])
    
    from matplotlib.ticker import FormatStrFormatter
    ax_res.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    
    axes_to_highlight = [ax_res]
    
    res_range = np.max(residual) - np.min(residual)
    if res_range < 1e-10:
        ax_res.set_yticks([])
        ax_res.text(0.5, 0.5, "Near Zero Residual", transform=ax_res.transAxes, ha='center')
    
    target_indices = [1, 2, 3]
    top_components = []
    
    for idx in target_indices:
        if idx < components.shape[0]:
            top_components.append(components[idx])
        else:
            top_components.append(np.full_like(time, np.nan))
            
    colors = sns.color_palette("husl", 3)
    
    for k in range(3):
        ax = fig.add_subplot(grid_spec[k+1], sharex=ax_res)
        axes_to_highlight.append(ax)
        
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        
        comp = top_components[k]
        
        if np.isnan(comp).all():
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
        else:
            ax.plot(time, comp, color=colors[k], lw=1.2)
            
            y_range = np.nanmax(comp) - np.nanmin(comp)
            if y_range < 1e-10:
                 ax.set_yticks([])
                 ax.text(0.5, 0.5, "≈ 0", transform=ax.transAxes, ha='center')
            
        label_idx = k + 1
        ax.set_ylabel(f"AFD\nComp {label_idx}", fontsize=8, rotation=0, labelpad=20, va='center')
        ax.yaxis.set_label_coords(-0.15, 0.5)
        
        ax.grid(True, linestyle=':', alpha=0.5)
        
        if k < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            import matplotlib.dates as mdates
            ax.xaxis.set_major_locator(mdates.YearLocator(2))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    if co_circ_periods is not None:
        for _, row in co_circ_periods.iterrows():
            start_date = row['Adjusted Start Date']
            end_date = row['Adjusted End Date']
            
            if pd.isna(start_date) or pd.isna(end_date):
                continue
                
            for ax in axes_to_highlight:
                ax.axvspan(start_date, end_date, color='red', alpha=0.3, lw=0)

if __name__ == "__main__":
    setup_plotting_style()
    plot_afd_structure()
