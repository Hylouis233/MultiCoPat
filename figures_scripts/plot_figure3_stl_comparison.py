import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from utils import load_stl_results, load_co_circulation_periods, save_figure, setup_plotting_style

def plot_figure3():
    print("Generating Figure 3: STL Dynamics Comparison...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']

    target_series = ['ILI%北方', 'ILI%南方', '流感阳性率北方', '流感阳性率南方']
    eng_names = ['ILI% (North)', 'ILI% (South)', 'Positivity Rate (North)', 'Positivity Rate (South)']
    
    target_indices = []
    for name in target_series:
        target_indices.append(series_names.index(name))

    from scripts.data_portal import UnifiedDataPortal
    portal = UnifiedDataPortal()
    flu_data = portal.load_flu_data()
    
    flu_data['start_date'] = pd.to_datetime(flu_data['start_date'])
    flu_data['stop_date'] = pd.to_datetime(flu_data['stop_date'])
    
    co_circ_df = load_co_circulation_periods()
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    def find_longest_period(region, period_type='co_circulation'):
        if period_type == 'co_circulation':
            region_periods = co_circ_df[co_circ_df['Region'] == region]
            if region_periods.empty:
                return None
            
            longest = region_periods.loc[region_periods['Duration (weeks)'].idxmax()]
            return (longest['Adjusted Start Date'], longest['Adjusted End Date'])
            
        elif period_type == 'single_dominant':
            co_periods = co_circ_df[co_circ_df['Region'] == region]
            
            full_dates = pd.date_range(start=flu_data['start_date'].min(), end=flu_data['stop_date'].max(), freq='W-MON')
            is_co_circ = np.zeros(len(full_dates), dtype=bool)
            
            for _, row in co_periods.iterrows():
                mask = (full_dates >= row['Adjusted Start Date']) & (full_dates <= row['Adjusted End Date'])
                is_co_circ |= mask
                
            if region == 'Northern':
                ili_col = 'ILI%北方'
                pos_col = '流感阳性率北方'
            else:
                ili_col = 'ILI%南方'
                pos_col = '流感阳性率南方'
            
            ili_threshold = flu_data[ili_col].quantile(0.6)
            pos_threshold = flu_data[pos_col].quantile(0.6)
            
            df_region = flu_data.copy()
            df_region['is_co_circ'] = False
            
            for _, row in co_periods.iterrows():
                mask = (df_region['start_date'] >= row['Adjusted Start Date']) & (df_region['stop_date'] <= row['Adjusted End Date'])
                df_region.loc[mask, 'is_co_circ'] = True
                
            candidates = df_region[
                (~df_region['is_co_circ']) & 
                ((df_region[ili_col] > ili_threshold) | (df_region[pos_col] > pos_threshold))
            ]
            
            if candidates.empty:
                return None
                
            candidates = candidates.sort_values('start_date')
            candidates['date_diff'] = candidates['start_date'].diff().dt.days
            candidates['group'] = (candidates['date_diff'] > 14).cumsum()
            
            group_sizes = candidates.groupby('group').size()
            longest_group_id = group_sizes.idxmax()
            
            longest_period_df = candidates[candidates['group'] == longest_group_id]
            
            return (longest_period_df['start_date'].min(), longest_period_df['stop_date'].max())

    fig = plt.figure(figsize=(24, 12))
    gs = GridSpec(2, 4, height_ratios=[1, 1], hspace=0.5, wspace=0.4)
    
    regions = ['Northern', 'Southern', 'Northern', 'Southern']
    
    for col_idx, (series_idx, eng_name, region) in enumerate(zip(target_indices, eng_names, regions)):
        
        sd_start, sd_end = find_longest_period(region, 'single_dominant')
        
        co_start, co_end = find_longest_period(region, 'co_circulation')
        
        if sd_start and sd_end:
            plot_stl_panel(fig, gs[0, col_idx], stl_array[series_idx], time_index, 
                           f"{eng_name}\nSingle-Dominant", 
                           color_base='#3498db',
                           time_range=(sd_start, sd_end))
        else:
            ax = fig.add_subplot(gs[0, col_idx])
            ax.text(0.5, 0.5, "No Data", ha='center')
            
        if co_start and co_end:
            plot_stl_panel(fig, gs[1, col_idx], stl_array[series_idx], time_index, 
                           f"{eng_name}\nCo-circulation", 
                           color_base='#e74c3c',
                           time_range=(co_start, co_end))
        else:
            ax = fig.add_subplot(gs[1, col_idx])
            ax.text(0.5, 0.5, "No Data", ha='center')

    save_figure(fig, 'figure3_stl_comparison')
    plt.close()

def plot_stl_panel(fig, gs_slot, data, time_index, title, color_base, time_range=None):
    
    if time_range:
        mask = (time_index >= pd.to_datetime(time_range[0])) & (time_index <= pd.to_datetime(time_range[1]))
        if not mask.any():
            ax = fig.add_subplot(gs_slot)
            ax.text(0.5, 0.5, "No Data in Range", ha='center')
            return
            
        plot_time = time_index[mask]
        plot_data = data[mask]
    else:
        plot_time = time_index
        plot_data = data
        
    from matplotlib.gridspec import GridSpecFromSubplotSpec
    gs_inner = GridSpecFromSubplotSpec(4, 1, subplot_spec=gs_slot, hspace=0.1, height_ratios=[1, 1, 1, 1])
    
    trend = plot_data[:, 0]
    seasonal = plot_data[:, 1]
    residual = plot_data[:, 2]
    observed = trend + seasonal + residual
    
    ax_obs = fig.add_subplot(gs_inner[0])
    ax_trend = fig.add_subplot(gs_inner[1], sharex=ax_obs)
    ax_season = fig.add_subplot(gs_inner[2], sharex=ax_obs)
    ax_resid = fig.add_subplot(gs_inner[3], sharex=ax_obs)
    
    axes = [ax_obs, ax_trend, ax_season, ax_resid]
    
    from matplotlib.ticker import FormatStrFormatter
    
    ax_obs.plot(plot_time, observed, color='black', lw=1.5, alpha=0.8)
    ax_obs.set_ylabel('Observed', rotation=0, ha='right', va='center', fontweight='bold', fontsize=10)
    ax_obs.yaxis.set_label_coords(-0.12, 0.5)
    
    ax_trend.plot(plot_time, trend, color=color_base, lw=2)
    ax_trend.set_ylabel('Trend', rotation=0, ha='right', va='center', fontweight='bold', color=color_base, fontsize=10)
    ax_trend.yaxis.set_label_coords(-0.12, 0.5)
    
    ax_season.plot(plot_time, seasonal, color=color_base, lw=1.5)
    ax_season.set_ylabel('Seasonal', rotation=0, ha='right', va='center', fontweight='bold', color=color_base, fontsize=10)
    ax_season.yaxis.set_label_coords(-0.12, 0.5)
    
    ax_resid.scatter(plot_time, residual, color='gray', s=5, alpha=0.6) 
    ax_resid.plot(plot_time, residual, color='gray', lw=1, alpha=0.4)
    ax_resid.axhline(0, color='black', linestyle='--', lw=1)
    ax_resid.set_ylabel('Residual', rotation=0, ha='right', va='center', fontweight='bold', color='gray', fontsize=10)
    ax_resid.yaxis.set_label_coords(-0.12, 0.5)
    
    import matplotlib.dates as mdates
    for ax in axes:
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.grid(True, linestyle=':', alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m'))
        
    for ax in axes[:-1]:
        plt.setp(ax.get_xticklabels(), visible=False)
        
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
    ax_obs.set_title(title, loc='center', fontsize=12, fontweight='bold')
    
    resid_range = np.max(np.abs(residual))
    if resid_range < 1e-10:
         ax_resid.set_yticks([])
         ax_resid.text(0.5, 0.5, "≈ 0", transform=ax_resid.transAxes, ha='center')
    else:
         ax_resid.set_ylim(-resid_range*1.2, resid_range*1.2)

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure3()
