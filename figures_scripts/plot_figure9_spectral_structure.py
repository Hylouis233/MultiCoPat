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

def plot_spectral_comparison(ax, periods, power_co, sem_co, power_single, sem_single, title):
    ax.plot(periods, power_single, color='#1f77b4', lw=2.5, label='Single-Dominant')
    ax.fill_between(periods, power_single - 1.96*sem_single, power_single + 1.96*sem_single, 
                    color='#1f77b4', alpha=0.2)
                    
    ax.plot(periods, power_co, color='#d62728', lw=2.5, label='Co-circulation')
    ax.fill_between(periods, power_co - 1.96*sem_co, power_co + 1.96*sem_co, 
                    color='#d62728', alpha=0.2)
    
    setup_spectral_axis(ax, title)
    ax.set_ylabel('Global Wavelet Power')
    ax.legend(loc='upper left', fontsize=8)

def plot_strain_spectrum(ax, periods, gws, color, label, title):
    ax.plot(periods, gws, color=color, lw=2.5, label=label)
    
    setup_spectral_axis(ax, title)

def setup_spectral_axis(ax, title):
    ax.axvline(52, color='gray', linestyle='--', alpha=0.6, lw=1.5)
    
    ax.axvline(26, color='gray', linestyle=':', alpha=0.6, lw=1.5)
    
    ax.set_xscale('log')
    custom_ticks = [4, 12, 26, 52, 104]
    ax.set_xticks(custom_ticks)
    ax.get_xaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.set_xlim(4, 104)
    
    ax.set_xlabel('Period (Weeks)')
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    ax.grid(True, which="both", ls="-", alpha=0.2)

def calculate_gws(series, scales, wavelet):
    cwtmatr, freqs = pywt.cwt(series, scales, wavelet)
    power = np.abs(cwtmatr)**2
    return power

def plot_figure9():
    print("Generating Figure 9 (Comprehensive Spectral Structure) - All Subtypes...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']
    co_circ_df = load_co_circulation_periods()
    if 'Region' not in co_circ_df.columns and 'region' in co_circ_df.columns:
        co_circ_df['Region'] = co_circ_df['region']
    
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    regions = [
        {
            'name': 'Northern China',
            'region_key': 'Northern',
            'overall_target': '流感阳性率北方', 
            'strains': [
                {'name': '病毒监测和分型（北方）甲型 H1N1', 'label': 'A(H1N1)', 'color': '#1f77b4'},
                {'name': '病毒监测和分型（北方）A(H3N2)', 'label': 'A(H3N2)', 'color': '#d62728'},
                {'name': '病毒监测和分型（北方）Victoria', 'label': 'B(Victoria)', 'color': '#2ca02c'},
                {'name': '病毒监测和分型（北方）Yamagata', 'label': 'B(Yamagata)', 'color': '#9467bd'}
            ]
        },
        {
            'name': 'Southern China',
            'region_key': 'Southern',
            'overall_target': '流感阳性率南方',
            'strains': [
                {'name': '病毒监测和分型（南方）甲型 H1N1', 'label': 'A(H1N1)', 'color': '#1f77b4'},
                {'name': '病毒监测和分型（南方）A(H3N2)', 'label': 'A(H3N2)', 'color': '#d62728'},
                {'name': '病毒监测和分型（南方）Victoria', 'label': 'B(Victoria)', 'color': '#2ca02c'},
                {'name': '病毒监测和分型（南方）Yamagata', 'label': 'B(Yamagata)', 'color': '#9467bd'}
            ]
        }
    ]
    
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 120)
    frequencies = pywt.scale2frequency(wavelet, scales)
    periods = 1.0 / frequencies
    
    fig, axes = plt.subplots(2, 5, figsize=(24, 10), constrained_layout=True)
    
    for i, region_config in enumerate(regions):
        ax_main = axes[i, 0]
        target_name = region_config['overall_target']
        
        if target_name in series_names:
            idx = series_names.index(target_name)
            seasonal = stl_array[idx, :, 1]
            power = calculate_gws(seasonal, scales, wavelet)
            
            co_mask = get_region_mask(time_index, region_config['region_key'], co_circ_df)
            single_mask = ~co_mask
            
            if np.any(co_mask):
                gws_co = np.mean(power[:, co_mask], axis=1)
                sem_co = np.std(power[:, co_mask], axis=1) / np.sqrt(np.sum(co_mask))
            else:
                gws_co = np.zeros_like(periods)
                sem_co = np.zeros_like(periods)
                
            if np.any(single_mask):
                gws_single = np.mean(power[:, single_mask], axis=1)
                sem_single = np.std(power[:, single_mask], axis=1) / np.sqrt(np.sum(single_mask))
            else:
                gws_single = np.zeros_like(periods)
                sem_single = np.zeros_like(periods)
                
            plot_spectral_comparison(ax_main, periods, gws_co, sem_co, gws_single, sem_single, 
                                     f"{region_config['name']}\nPos Rate: Co-circ vs Single")
        else:
            print(f"Warning: Series '{target_name}' not found")
        
        for j, strain in enumerate(region_config['strains']):
            ax_strain = axes[i, j+1]
            s_name = strain['name']
            
            if s_name in series_names:
                idx = series_names.index(s_name)
                seasonal = stl_array[idx, :, 1]
                power = calculate_gws(seasonal, scales, wavelet)
                
                co_mask = get_region_mask(time_index, region_config['region_key'], co_circ_df)
                single_mask = ~co_mask
                
                if np.any(co_mask):
                    gws_co = np.mean(power[:, co_mask], axis=1)
                    sem_co = np.std(power[:, co_mask], axis=1) / np.sqrt(np.sum(co_mask))
                else:
                    gws_co = np.zeros_like(periods)
                    sem_co = np.zeros_like(periods)
                    
                if np.any(single_mask):
                    gws_single = np.mean(power[:, single_mask], axis=1)
                    sem_single = np.std(power[:, single_mask], axis=1) / np.sqrt(np.sum(single_mask))
                else:
                    gws_single = np.zeros_like(periods)
                    sem_single = np.zeros_like(periods)
                
                plot_spectral_comparison(ax_strain, periods, gws_co, sem_co, gws_single, sem_single, 
                                         f"{strain['label']} Spectrum")
                
                ax_strain.set_ylabel('')
                
            else:
                print(f"Warning: Series '{s_name}' not found")
                ax_strain.text(0.5, 0.5, 'Data Not Available', ha='center', va='center')

    save_figure(fig, 'figure9_spectral_structure')
    plt.close(fig)

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure9()
