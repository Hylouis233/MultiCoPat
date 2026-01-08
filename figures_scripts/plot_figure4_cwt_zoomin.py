import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pywt
import matplotlib.dates as mdates

from matplotlib.patches import ConnectionPatch
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter, FuncFormatter
from scipy.signal import savgol_filter

from utils import load_stl_results, load_co_circulation_periods, save_figure, setup_plotting_style, PROJECT_ROOT

def plot_figure4_zoomin():
    print("Generating CWT frequency drift zoom-in figure (4 Panels)...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']

    from scripts.data_portal import UnifiedDataPortal
    portal = UnifiedDataPortal()
    flu_data = portal.load_flu_data()
    flu_data['start_date'] = pd.to_datetime(flu_data['start_date'])
    flu_data['stop_date'] = pd.to_datetime(flu_data['stop_date'])
    
    co_circ_df = load_co_circulation_periods()
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    target_series = [
        'ILI%北方',
        '病毒监测和分型（北方）甲型 H1N1',
        '病毒监测和分型（北方）A(H3N2)',
        '病毒监测和分型（北方）Victoria',
        '病毒监测和分型（北方）Yamagata',
        'ILI%南方',
        '病毒监测和分型（南方）甲型 H1N1',
        '病毒监测和分型（南方）A(H3N2)',
        '病毒监测和分型（南方）Victoria',
        '病毒监测和分型（南方）Yamagata'
    ]
    eng_names = [
        'Northern China (ILI%) - Macro Stability',
        'Northern China (A/H1N1)',
        'Northern China (A/H3N2)',
        'Northern China (B/Victoria)',
        'Northern China (B/Yamagata)',
        'Southern China (ILI%) - Macro Stability',
        'Southern China (A/H1N1)',
        'Southern China (A/H3N2)',
        'Southern China (B/Victoria)',
        'Southern China (B/Yamagata)'
    ]
    regions = ['Northern']*5 + ['Southern']*5
    
    target_indices = []
    for name in target_series:
        target_indices.append(series_names.index(name))

    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 2, hspace=0.35, wspace=0.15)
    
    def find_longest_period(region, period_type='co_circulation'):
        if period_type == 'co_circulation':
            region_periods = co_circ_df[co_circ_df['Region'] == region]
            if region_periods.empty: return None
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
                ili_col, pos_col = 'ILI%北方', '流感阳性率北方'
            else:
                ili_col, pos_col = 'ILI%南方', '流感阳性率南方'
            
            ili_threshold = flu_data[ili_col].quantile(0.6)
            pos_threshold = flu_data[pos_col].quantile(0.6)
            
            df_region = flu_data.copy()
            df_region['is_co_circ'] = False
            for _, row in co_periods.iterrows():
                mask = (df_region['start_date'] >= row['Adjusted Start Date']) & (df_region['stop_date'] <= row['Adjusted End Date'])
                df_region.loc[mask, 'is_co_circ'] = True
                
            candidates = df_region[(~df_region['is_co_circ']) & ((df_region[ili_col] > ili_threshold) | (df_region[pos_col] > pos_threshold))]
            if candidates.empty: return None
            
            candidates = candidates.sort_values('start_date')
            candidates['date_diff'] = candidates['start_date'].diff().dt.days
            candidates['group'] = (candidates['date_diff'] > 14).cumsum()
            group_sizes = candidates.groupby('group').size()
            longest_period_df = candidates[candidates['group'] == group_sizes.idxmax()]
            return (longest_period_df['start_date'].min(), longest_period_df['stop_date'].max())

    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 100)
    frequencies = pywt.scale2frequency(wavelet, scales)
    periods = 1.0 / frequencies

    for idx, (series_idx, title, region) in enumerate(zip(target_indices, eng_names, regions)):
        col = 0 if idx < 5 else 1
        row = idx if idx < 5 else idx - 5
        
        ax = fig.add_subplot(gs[row, col])
        
        seasonal = stl_array[series_idx, :, 1]
        cwtmatr, freqs = pywt.cwt(seasonal, scales, wavelet)
        power = np.abs(cwtmatr)**2
        
        T, P = np.meshgrid(time_index, periods)
        im = ax.pcolormesh(T, P, power, cmap='jet', shading='auto')
        
        max_power_idx = np.argmax(power, axis=0)
        ridge_periods = periods[max_power_idx]
        
        valid_ridge_mask = (ridge_periods >= 4) & (ridge_periods <= 100)
        
        ax.plot(time_index[valid_ridge_mask], ridge_periods[valid_ridge_mask], color='white', lw=2.0, alpha=0.6, label='Instability')
        
        ax.axhline(52, color='white', linestyle='-', alpha=1.0, lw=1.0)
        
        ax.set_yscale('log')
        ax.set_ylim(4, 100)
        
        custom_ticks = [4, 12, 24, 36, 48, 60, 72, 84]
        ax.set_yticks(custom_ticks)
        ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
        ax.yaxis.set_minor_locator(plt.NullLocator())
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        if col == 0:
            ax.set_ylabel('Period (Weeks)')
        else:
            ax.set_yticklabels([])
        
        sd_range = find_longest_period(region, 'single_dominant')
        co_range = find_longest_period(region, 'co_circulation')
        
        import matplotlib.patches as mpatches
        
        if sd_range:
            start_date, end_date = sd_range
            start_num = mdates.date2num(start_date)
            end_num = mdates.date2num(end_date)
            width = end_num - start_num
            
            rect = mpatches.Rectangle((start_num, 30), width, 50, 
                                      linewidth=1.5, edgecolor='white', facecolor='none', linestyle='-')
            ax.add_patch(rect)
            
            mid_x = start_num + width/2
            ax.text(mid_x, 85, 'Single-Dominant', color='white', fontsize=10, fontweight='bold', 
                    ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))
        
        if co_range:
            start_date, end_date = co_range
            start_num = mdates.date2num(start_date)
            end_num = mdates.date2num(end_date)
            width = end_num - start_num
            
            rect = mpatches.Rectangle((start_num, 30), width, 50, 
                                      linewidth=1.5, edgecolor='white', facecolor='none', linestyle='--')
            ax.add_patch(rect)
            
            mid_x = start_num + width/2
            ax.text(mid_x, 85, 'Co-circulation', color='white', fontsize=10, fontweight='bold', 
                    ha='center', va='bottom', bbox=dict(facecolor='black', alpha=0.4, edgecolor='none', pad=1))

    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cbar_ax)
    cbar.set_label('Wavelet Power Spectrum', fontsize=12)

    save_figure(fig, 'figure4_cwt_zoomin')
    plt.close()

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure4_zoomin()

