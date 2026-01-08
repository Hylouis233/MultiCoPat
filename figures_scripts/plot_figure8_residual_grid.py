import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from matplotlib.ticker import FuncFormatter

from utils import load_stl_results, load_co_circulation_periods, save_figure, setup_plotting_style

def get_region_mask(time_index, region, co_circ_df):
    mask = np.zeros(len(time_index), dtype=bool)
    region_periods = co_circ_df[co_circ_df['Region'] == region]
    
    for _, row in region_periods.iterrows():
        start = pd.to_datetime(row['Adjusted Start Date'])
        end = pd.to_datetime(row['Adjusted End Date'])
        period_mask = (time_index >= start) & (time_index <= end)
        mask |= period_mask
        
    return mask

def calculate_residual_gws(series, scales, wavelet):
    cwtmatr, freqs = pywt.cwt(series, scales, wavelet)
    power = np.abs(cwtmatr)**2
    return power

def plot_residual_panel(ax, periods, power_matrix, title, color):
    mean_power = np.mean(power_matrix, axis=1)
    sem_power = np.std(power_matrix, axis=1) / np.sqrt(power_matrix.shape[1])
    
    ax.plot(periods, mean_power, color=color, lw=2.5)
    ax.fill_between(periods, mean_power - 1.96*sem_power, mean_power + 1.96*sem_power, 
                    color=color, alpha=0.3)
    
    ax.axvline(13, color='gray', linestyle=':', alpha=0.8, lw=1.5, label='13w (Quarterly)')
    ax.axvline(4, color='gray', linestyle='--', alpha=0.8, lw=1.5, label='4w (Monthly)')
    
    ax.set_xscale('linear')
    custom_ticks = [2, 4, 8, 13, 20, 26]
    ax.set_xticks(custom_ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.set_xlim(1, 26)
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.grid(True, which="both", ls="-", alpha=0.2)
    
    total_energy = np.trapz(mean_power, periods)
    ax.text(0.95, 0.95, f'Total Energy: {total_energy:.1f}', transform=ax.transAxes, 
            ha='right', va='top', fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

def plot_figure8():
    print("Generating Figure 8 (Residual Grid)...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']
    co_circ_df = load_co_circulation_periods()
    if 'Region' not in co_circ_df.columns and 'region' in co_circ_df.columns:
        co_circ_df['Region'] = co_circ_df['region']
        
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    target_strains = {
        'North': '病毒监测和分型（北方）A(H3N2)',
        'South': '病毒监测和分型（南方）A(H3N2)'
    }
    
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 40)
    frequencies = pywt.scale2frequency(wavelet, scales)
    periods = 1.0 / frequencies
    mask = (periods >= 0.8) & (periods <= 28)
    scales = scales[mask]
    periods = periods[mask]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey='row')
    
    north_idx = series_names.index(target_strains['North'])
    north_residual = stl_array[north_idx, :, 2]
    north_power = calculate_residual_gws(north_residual, scales, wavelet)
    
    north_mask = get_region_mask(time_index, 'Northern', co_circ_df)
    
    plot_residual_panel(axes[0, 0], periods, north_power[:, ~north_mask], 
                   'A. North Residual - Single', '#3498db')
    axes[0, 0].set_ylabel('Residual Wavelet Power')
    
    plot_residual_panel(axes[0, 1], periods, north_power[:, north_mask], 
                   'B. North Residual - Co-circ', '#e74c3c')
    
    south_idx = series_names.index(target_strains['South'])
    south_residual = stl_array[south_idx, :, 2]
    south_power = calculate_residual_gws(south_residual, scales, wavelet)
    
    south_mask = get_region_mask(time_index, 'Southern', co_circ_df)
    
    plot_residual_panel(axes[1, 0], periods, south_power[:, ~south_mask], 
                   'C. South Residual - Single', '#3498db')
    axes[1, 0].set_ylabel('Residual Wavelet Power')
    axes[1, 0].set_xlabel('Period (Weeks)')
    
    plot_residual_panel(axes[1, 1], periods, south_power[:, south_mask], 
                   'D. South Residual - Co-circ', '#e74c3c')
    axes[1, 1].set_xlabel('Period (Weeks)')
    
    plt.tight_layout()
    save_figure(fig, 'figure8_residual_grid')
    plt.close(fig)

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure8()
