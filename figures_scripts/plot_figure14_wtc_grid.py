import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.ndimage import uniform_filter1d
from matplotlib.ticker import FuncFormatter

from utils import load_stl_results, load_co_circulation_periods, save_figure, setup_plotting_style

def add_co_circ_shading(ax, time_index, region, co_circ_df):
    region_periods = co_circ_df[co_circ_df['Region'] == region]
    
    if hasattr(time_index, 'iloc'):
        t_min, t_max = time_index.iloc[0], time_index.iloc[-1]
    else:
        t_min, t_max = time_index[0], time_index[-1]
    
    for _, row in region_periods.iterrows():
        start = pd.to_datetime(row['Adjusted Start Date'])
        end = pd.to_datetime(row['Adjusted End Date'])
        
        if end < t_min or start > t_max:
            continue
            
        start = max(start, t_min)
        end = min(end, t_max)
        
        ax.axvspan(start, end, color='gray', alpha=0.3, zorder=2, label='Co-circulation')

def smooth_wavelet_transform(W, time_window=10, scale_window=3):
    W_smooth = uniform_filter1d(W, size=time_window, axis=1, mode='nearest')
    W_smooth = uniform_filter1d(W_smooth, size=scale_window, axis=0, mode='nearest')
    return W_smooth

def compute_wtc(x, y, scales, wavelet='cmor1.5-1.0', sampling_period=1.0):
    Wx, _ = pywt.cwt(x, scales, wavelet, sampling_period)
    Wy, _ = pywt.cwt(y, scales, wavelet, sampling_period)
    
    Wxy = Wx * np.conj(Wy)
    
    S_Wxy = smooth_wavelet_transform(Wxy)
    S_Wx2 = smooth_wavelet_transform(np.abs(Wx)**2)
    S_Wy2 = smooth_wavelet_transform(np.abs(Wy)**2)
    
    denominator = np.sqrt(S_Wx2 * S_Wy2)
    denominator = np.where(denominator == 0, 1e-10, denominator)
    coherence = np.abs(S_Wxy) / denominator
    
    phase = np.arctan2(np.imag(S_Wxy), np.real(S_Wxy))
    
    return coherence, phase

def compute_wtc_significance(x, y, scales, n_surrogates=100):
    coherence_actual, _ = compute_wtc(x, y, scales)
    
    n_time = len(x)
    surrogate_coherence = np.zeros((n_surrogates, coherence_actual.shape[0], coherence_actual.shape[1]))
    
    def ar1_surrogate(ts, n):
        if len(ts) < 2: return np.random.randn(n)
        lag1 = np.corrcoef(ts[:-1], ts[1:])[0, 1]
        if np.isnan(lag1): lag1 = 0.5
        sigma = np.std(ts) * np.sqrt(1 - lag1**2)
        if np.isnan(sigma) or sigma == 0: sigma = 1
        noise = np.random.normal(0, sigma, n)
        surr = np.zeros(n)
        surr[0] = ts[0]
        for i in range(1, n):
            surr[i] = lag1 * surr[i-1] + noise[i]
        return surr
    
    for i in range(n_surrogates):
        x_surr = ar1_surrogate(x, n_time)
        y_surr = ar1_surrogate(y, n_time)
        coh, _ = compute_wtc(x_surr, y_surr, scales)
        surrogate_coherence[i] = coh
        
    threshold = np.percentile(surrogate_coherence, 95, axis=0)
    
    significance_mask = coherence_actual > threshold
    return significance_mask

def plot_single_wtc(ax, time_index, periods, coherence, phase, mask, title):
    T, P = np.meshgrid(time_index, periods)
    
    im = ax.pcolormesh(T, P, coherence, cmap='jet', shading='auto', vmin=0, vmax=1)
    
    if mask is not None:
        ax.contour(T, P, mask, levels=[0.5], colors='black', linewidths=0.8, alpha=0.6)
        
    n_t = len(time_index)
    n_s = len(periods)
    
    step_t = max(1, n_t // 30)
    step_s = max(1, n_s // 15)
    
    idx_t = np.arange(0, n_t, step_t)
    idx_s = np.arange(0, n_s, step_s)
    
    mesh_t, mesh_p = np.meshgrid(time_index[idx_t], periods[idx_s])
    
    phase_sub = phase[np.ix_(idx_s, idx_t)]
    coherence_sub = coherence[np.ix_(idx_s, idx_t)]
    
    arrow_mask = coherence_sub > 0.5
    
    u = np.cos(phase_sub)
    v = np.sin(phase_sub)
    
    u[~arrow_mask] = np.nan
    v[~arrow_mask] = np.nan
    
    ax.quiver(mesh_t, mesh_p, u, v, color='white', scale=25, width=0.003, headwidth=3, alpha=0.8)
    
    ax.set_yscale('log')
    ax.set_ylim(4, 100)
    custom_ticks = [4, 12, 26, 52, 100]
    ax.set_yticks(custom_ticks)
    ax.get_yaxis().set_major_formatter(FuncFormatter(lambda x, p: f'{int(x)}'))
    ax.set_title(title, fontsize=11, fontweight='bold')
    
    return im

def plot_figure14():
    print("Generating Figure 14 (WTC Grid)...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']
    
    co_circ_df = load_co_circulation_periods()
    if 'Region' not in co_circ_df.columns and 'region' in co_circ_df.columns:
        co_circ_df['Region'] = co_circ_df['region']
    co_circ_df['Adjusted Start Date'] = pd.to_datetime(co_circ_df['Adjusted Start Date'])
    co_circ_df['Adjusted End Date'] = pd.to_datetime(co_circ_df['Adjusted End Date'])

    configs = [
        {
            'x_name': '病毒监测和分型（北方）Yamagata',
            'y_name': 'ERA5-Land气象再分析数据（北方）_t2m',
            'title': '(a) North: B/Yamagata vs Temp (Lock-in)'
        },
        {
            'x_name': '病毒监测和分型（南方）Yamagata',
            'y_name': 'ERA5-Land气象再分析数据（南方）_tp',
            'title': '(b) South: B/Yamagata vs Precip (Lock-in)'
        },
        {
            'x_name': '病毒监测和分型（南方）A(H3N2)',
            'y_name': 'ERA5-Land气象再分析数据（南方）_t2m',
            'title': '(c) South: A/H3N2 vs Temp (Steady State)'
        },
        {
            'x_name': '病毒监测和分型（北方）甲型 H1N1',
            'y_name': 'ERA5-Land气象再分析数据（北方）_t2m',
            'title': '(d) North: A/H1N1 vs Temp (Steady State)'
        }
    ]
    
    wavelet = 'cmor1.5-1.0'
    scales = np.arange(1, 100)
    frequencies = pywt.scale2frequency(wavelet, scales)
    periods = 1.0 / frequencies
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    
    for i, config in enumerate(configs):
        ax = axes[i]
        x_name = config['x_name']
        y_name = config['y_name']
        
        if x_name not in series_names or y_name not in series_names:
            print(f"Warning: Series '{x_name}' or '{y_name}' not found, skipping...")
            ax.text(0.5, 0.5, 'Data Missing', ha='center', va='center')
            continue
            
        idx_x = series_names.index(x_name)
        idx_y = series_names.index(y_name)
        
        x = stl_array[idx_x, :, 1]
        y = stl_array[idx_y, :, 1]
        
        x = (x - np.mean(x)) / np.std(x)
        y = (y - np.mean(y)) / np.std(y)
        
        print(f"Calculating WTC for {config['title']}...")
        coherence, phase = compute_wtc(x, y, scales, wavelet)
        mask = compute_wtc_significance(x, y, scales, n_surrogates=50) 
        
        im = plot_single_wtc(ax, time_index, periods, coherence, phase, mask, config['title'])
        
        region = 'Northern' if 'North' in config['title'] else 'Southern'
        add_co_circ_shading(ax, time_index, region, co_circ_df)
        
        if i >= 2:
            ax.set_xlabel('Time')
        if i % 2 == 0:
            ax.set_ylabel('Period (Weeks)')
            
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Coherence')

    save_figure(fig, 'figure14_wtc_grid')
    plt.close(fig)

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure14()
