#!/usr/bin/env python3

import gc
import os
import sys
import time
import warnings
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import label
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.multitest import multipletests
import pywt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

DEFAULT_CWT_CONFIG_TREND = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (52, 500),
    'n_scales': 100,
    'sampling_period': 1.0
}

DEFAULT_CWT_CONFIG_SEASONAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (26, 52),
    'n_scales': 50,
    'sampling_period': 1.0
}

DEFAULT_CWT_CONFIG_RESIDUAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (1, 26),
    'n_scales': 50,
    'sampling_period': 1.0
}

DEFAULT_PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 10
}

DEFAULT_VISUALIZATION_CONFIG = {
    'max_series_to_plot': None,
    'batch_size': 10,
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

DEFAULT_ANALYSIS_CONFIG = {
    'anomaly_threshold_percentile': 95,
    'enable_trend_analysis': True,
    'enable_seasonal_analysis': True,
    'enable_residual_analysis': True,
    'reconstruct_r_from_afd': True,
}

DEFAULT_OUTPUT_CONFIG = {
    'base_dir': 'AFD/cwt_analysis',
    'time_frequency_plots_dir': 'time_frequency_plots',
    'frequency_drift_dir': 'frequency_drift',
    'trend_rate_dir': 'trend_rate',
    'intensity_variation_dir': 'intensity_variation',
    'anomalous_events_dir': 'anomalous_events',
    'comparison_analysis_dir': 'comparison_analysis'
}

CWT_CONFIG_TREND = DEFAULT_CWT_CONFIG_TREND
CWT_CONFIG_SEASONAL = DEFAULT_CWT_CONFIG_SEASONAL
CWT_CONFIG_RESIDUAL = DEFAULT_CWT_CONFIG_RESIDUAL
PARALLEL_CONFIG = DEFAULT_PARALLEL_CONFIG
VISUALIZATION_CONFIG = DEFAULT_VISUALIZATION_CONFIG
ANALYSIS_CONFIG = DEFAULT_ANALYSIS_CONFIG
OUTPUT_CONFIG = DEFAULT_OUTPUT_CONFIG

DEFAULT_STL_DATA_PATH = {
    'stl_results_npy': 'AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': 'AFD/stl_decomposition/arrays/stl_results.pkl'
}

DEFAULT_AFD_DATA_PATH = {
    'residual_components_npy': 'AFD/afd_residual_components.npy',
    'component_numbers_csv': 'AFD/afd_component_numbers.csv',
    'afd_metadata_pkl': 'AFD/afd_stl_metadata.pkl'
}

STL_DATA_PATH = DEFAULT_STL_DATA_PATH
AFD_DATA_PATH = DEFAULT_AFD_DATA_PATH

CO_CIRCULATION_PATH = 'co_circulation_periods.csv'

def load_stl_data():
    print("=" * 80)
    print("Loading STL Decomposition Output (T/S/R)")
    print("=" * 80)
    
    stl_results_npy = Path(STL_DATA_PATH['stl_results_npy'])
    stl_results_pkl = Path(STL_DATA_PATH['stl_results_pkl'])
    
    if not stl_results_npy.exists() or not stl_results_pkl.exists():
        raise FileNotFoundError(f"STL output files not found: {stl_results_npy} or {stl_results_pkl}")
    
    print("Loading from structured arrays...")
    stl_3d_array = np.load(stl_results_npy)
    with open(stl_results_pkl, 'rb') as f:
        metadata = pickle.load(f)
    
    time_index = metadata['time_index']
    series_names = metadata['series_names']
    
    if isinstance(time_index, pd.Series):
        time_index = pd.to_datetime(time_index.values)
    elif not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    
    trend_dict = {}
    seasonal_dict = {}
    residual_dict = {}
    
    for i, series_name in enumerate(series_names):
        trend_dict[series_name] = stl_3d_array[i, :, 0]
        seasonal_dict[series_name] = stl_3d_array[i, :, 1]
        residual_dict[series_name] = stl_3d_array[i, :, 2]
    
    print(f"Successfully loaded {len(series_names)} series from arrays")
    
    return {
        'trend': trend_dict,
        'seasonal': seasonal_dict,
        'residual': residual_dict
    }, time_index, series_names, metadata

def load_afd_data():
    print("=" * 80)
    print("Loading AFD Decomposition Output (R(t) components)")
    print("=" * 80)
    
    residual_components_npy = Path(AFD_DATA_PATH['residual_components_npy'])
    component_numbers_csv = Path(AFD_DATA_PATH['component_numbers_csv'])
    afd_metadata_pkl = Path(AFD_DATA_PATH['afd_metadata_pkl'])
    
    if not residual_components_npy.exists():
        raise FileNotFoundError(f"AFD residual components file not found: {residual_components_npy}")
    
    if not component_numbers_csv.exists():
        raise FileNotFoundError(f"AFD component numbers file not found: {component_numbers_csv}")
    
    print("Loading AFD residual components...")
    afd_residual_array = np.load(residual_components_npy)
    
    component_numbers_df = pd.read_csv(component_numbers_csv)
    component_numbers = dict(zip(component_numbers_df['Series_Name'], 
                                 component_numbers_df['K_Residual']))
    
    metadata = {}
    if afd_metadata_pkl.exists():
        with open(afd_metadata_pkl, 'rb') as f:
            metadata = pickle.load(f)
    
    print(f"Successfully loaded AFD data")
    return afd_residual_array, component_numbers, metadata

def reconstruct_r_from_afd(afd_residual_array, component_numbers, series_names):
    print("=" * 80)
    print("Reconstructing R(t) from AFD Components")
    print("=" * 80)
    
    reconstructed_r = {}
    
    for i, series_name in enumerate(series_names):
        if series_name not in component_numbers:
            continue
        
        k_r = int(component_numbers[series_name])
        series_afd_data = afd_residual_array[i, :, :]
        
        if k_r > 0:
            reconstructed = np.nansum(series_afd_data[:, :k_r], axis=1)
        else:
            reconstructed = np.zeros(series_afd_data.shape[0])
        
        reconstructed_r[series_name] = reconstructed
    
    print(f"Successfully reconstructed R(t) for {len(reconstructed_r)} series")
    return reconstructed_r

def load_co_circulation_periods(file_path=None):
    if file_path is None:
        file_path = Path(CO_CIRCULATION_PATH)
    else:
        file_path = Path(file_path)
    
    if not file_path.exists():
        print(f"Warning: Co-circulation periods file not found: {file_path}")
        return None
    
    co_circulation_df = pd.read_csv(file_path)
    print(f"Loaded co-circulation periods: {len(co_circulation_df)} records")
    return co_circulation_df

def compute_scales(scale_range, n_scales, sampling_period=1.0):
    min_scale, max_scale = scale_range
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
    return scales

def scales_to_frequencies(scales, sampling_period=1.0, wavelet='cmor', wavelet_params=(1.5, 1.0)):
    if wavelet == 'cmor':
        Fb, Fc = wavelet_params
        frequencies = Fc / (scales * sampling_period)
        periods = scales * sampling_period / Fc
    elif wavelet == 'paul':
        m = wavelet_params if isinstance(wavelet_params, (int, float)) else wavelet_params[0]
        fc = (2*m + 1) / (4 * np.pi)
        frequencies = fc / (scales * sampling_period)
        periods = 1.0 / frequencies
    elif wavelet == 'dog':
        m = wavelet_params if isinstance(wavelet_params, (int, float)) else wavelet_params[0]
        fc = np.sqrt(m + 0.5) / (2 * np.pi)
        frequencies = fc / (scales * sampling_period)
        periods = 1.0 / frequencies
    else:
        w = pywt.ContinuousWavelet(wavelet)
        fc = w.center_frequency
        frequencies = fc / (scales * sampling_period)
        periods = 1.0 / frequencies
            
    return frequencies, periods

def compute_cwt(signal, scales, wavelet='cmor', wavelet_params=(1.5, 1.0), 
                sampling_period=1.0):
    if wavelet == 'cmor':
        Fb, Fc = wavelet_params
        w_str = f"cmor{Fb}-{Fc}"
    elif wavelet == 'paul':
        m = wavelet_params if isinstance(wavelet_params, (int, float)) else wavelet_params[0]
        w_str = f"paul{m}"
    elif wavelet == 'dog':
        m = wavelet_params if isinstance(wavelet_params, (int, float)) else wavelet_params[0]
        w_str = f"gaus{m}"
    else:
        w_str = wavelet
        
    coefficients, frequencies = pywt.cwt(
        signal, 
        scales, 
        w_str, 
        sampling_period=sampling_period
    )
    
    frequencies_calc, periods = scales_to_frequencies(
        scales, sampling_period, wavelet, wavelet_params
    )
    
    return coefficients, frequencies_calc, periods

def identify_frequency_drift(cwt_result, frequencies, time_index, component_type='seasonal'):
    energy_density = np.abs(cwt_result)**2
    dominant_freq_indices = np.argmax(energy_density, axis=0)
    dominant_frequencies = frequencies[dominant_freq_indices]
    
    with np.errstate(divide='ignore'):
        dominant_periods = 1.0 / dominant_frequencies
    
    freq_std = np.std(dominant_frequencies)
    freq_mean = np.mean(dominant_frequencies)
    freq_cv = freq_std / freq_mean if freq_mean > 0 else 0
    
    if len(dominant_frequencies) > 1:
        time_points = np.arange(len(dominant_frequencies))
        drift_rate = np.polyfit(time_points, dominant_frequencies, 1)[0]
    else:
        drift_rate = 0.0
    
    return {
        'dominant_frequencies': dominant_frequencies,
        'dominant_periods': dominant_periods,
        'frequency_stability': {
            'std': freq_std,
            'mean': freq_mean,
            'cv': freq_cv
        },
        'drift_rate': drift_rate
    }

def analyze_trend_rate(cwt_result, frequencies, time_index):
    energy_density = np.abs(cwt_result)**2
    low_freq_mask = frequencies < 1/52
    if np.any(low_freq_mask):
        low_freq_energy = np.sum(energy_density[low_freq_mask, :], axis=0)
    else:
        low_freq_energy = np.sum(energy_density, axis=0)
    
    trend_rate = np.gradient(low_freq_energy)
    rate_std = np.std(trend_rate)
    rate_mean = np.mean(np.abs(trend_rate))
    rate_cv = rate_std / rate_mean if rate_mean > 0 else 0
    acceleration = np.gradient(trend_rate)
    
    return {
        'trend_rate': trend_rate,
        'rate_variability': {
            'std': rate_std,
            'mean': rate_mean,
            'cv': rate_cv
        },
        'acceleration': acceleration
    }

def analyze_intensity_variation(cwt_result, time_index, component_type='seasonal'):
    energy_density = np.abs(cwt_result)**2
    total_energy = np.sum(energy_density, axis=0)
    energy_std = np.std(total_energy)
    energy_mean = np.mean(total_energy)
    energy_cv = energy_std / energy_mean if energy_mean > 0 else 0
    
    energy_sum = np.sum(energy_density)
    if energy_sum > 0:
        prob_dist = energy_density / energy_sum
        prob_dist = prob_dist[prob_dist > 0]
        entropy = -np.sum(prob_dist * np.log2(prob_dist))
    else:
        entropy = 0.0
    
    max_energy = np.max(energy_density)
    concentration = max_energy / energy_sum if energy_sum > 0 else 0
    
    return {
        'energy_density': energy_density,
        'total_energy': total_energy,
        'energy_variability': {
            'std': energy_std,
            'mean': energy_mean,
            'cv': energy_cv
        },
        'energy_complexity': {
            'entropy': entropy,
            'concentration': concentration
        }
    }

def detect_anomalous_events(cwt_result, frequencies, time_index, threshold_percentile=95):
    energy_density = np.abs(cwt_result)**2
    energy_threshold = np.percentile(energy_density, threshold_percentile)
    anomalous_mask = energy_density > energy_threshold
    labeled_regions, num_regions = label(anomalous_mask)
    
    anomalous_regions = []
    event_times = []
    
    for region_id in range(1, num_regions + 1):
        region_mask = labeled_regions == region_id
        region_energy = energy_density[region_mask]
        region_indices = np.where(region_mask)
        scale_indices = region_indices[0]
        time_indices = region_indices[1]
        
        if len(time_indices) > 0:
            t_start_idx = np.min(time_indices)
            t_end_idx = np.max(time_indices)
            f_start_idx = np.min(scale_indices)
            f_end_idx = np.max(scale_indices)
            
            t_start = time_index[t_start_idx] if isinstance(time_index, pd.DatetimeIndex) else time_index[t_start_idx]
            t_end = time_index[t_end_idx] if isinstance(time_index, pd.DatetimeIndex) else time_index[t_end_idx]
            f_start = frequencies[f_start_idx]
            f_end = frequencies[f_end_idx]
            peak_energy = np.max(region_energy)
            
            anomalous_regions.append({
                't_start': t_start,
                't_end': t_end,
                'f_start': f_start,
                'f_end': f_end,
                'peak_energy': peak_energy,
                'mean_energy': np.mean(region_energy)
            })
            
            peak_time_idx = time_indices[np.argmax(region_energy)]
            event_times.append(time_index[peak_time_idx] if isinstance(time_index, pd.DatetimeIndex) else time_index[peak_time_idx])
    
    event_statistics = {
        'total_events': num_regions,
        'events_per_year': num_regions / (len(time_index) / 52) if len(time_index) > 0 else 0,
        'mean_event_energy': np.mean([r['peak_energy'] for r in anomalous_regions]) if anomalous_regions else 0,
        'max_event_energy': np.max([r['peak_energy'] for r in anomalous_regions]) if anomalous_regions else 0
    }
    
    return {
        'anomalous_regions': anomalous_regions,
        'event_times': event_times,
        'event_statistics': event_statistics,
        'threshold': energy_threshold
    }

def process_single_series_cwt(series_info):
    series_name, component_dict, component_type, config, time_index = series_info
    
    if component_type == 'trend':
        signal_data = component_dict['trend'].get(series_name)
    elif component_type == 'seasonal':
        signal_data = component_dict['seasonal'].get(series_name)
    elif component_type == 'residual':
        signal_data = component_dict['residual'].get(series_name)
    else:
        raise ValueError(f"Unknown component type: {component_type}")
    
    if signal_data is None:
        return None
    
    signal_data = signal_data[np.isfinite(signal_data)]
    
    if len(signal_data) < 10:
        return None
    
    scales = compute_scales(
        config['scale_range'],
        config['n_scales'],
        config['sampling_period']
    )
    
    cwt_result, frequencies, periods = compute_cwt(
        signal_data,
        scales,
        config['wavelet'],
        config['wavelet_params'],
        config['sampling_period']
    )
    
    features = {
        'series_name': series_name,
        'component_type': component_type,
        'cwt_result': cwt_result,
        'frequencies': frequencies,
        'periods': periods,
        'time_index': time_index[:len(signal_data)]
    }
    
    if component_type == 'seasonal':
        freq_drift = identify_frequency_drift(cwt_result, frequencies, time_index, component_type)
        intensity = analyze_intensity_variation(cwt_result, time_index, component_type)
        features.update({
            'frequency_drift': freq_drift,
            'intensity_variation': intensity
        })
    elif component_type == 'trend':
        trend_rate = analyze_trend_rate(cwt_result, frequencies, time_index)
        features.update({'trend_rate': trend_rate})
    elif component_type == 'residual':
        anomalies = detect_anomalous_events(
            cwt_result, frequencies, time_index,
            threshold_percentile=ANALYSIS_CONFIG['anomaly_threshold_percentile']
        )
        intensity = analyze_intensity_variation(cwt_result, time_index, component_type)
        features.update({
            'anomalous_events': anomalies,
            'intensity_variation': intensity
        })
    
    return features

def process_all_series_cwt(component_dict, component_type, config, time_index, series_names):
    print("=" * 80)
    print(f"CWT Analysis for {component_type.upper()} Components")
    print("=" * 80)
    
    series_info_list = [
        (series_name, component_dict, component_type, config, time_index)
        for series_name in series_names
    ]
    
    results = {}
    max_workers = PARALLEL_CONFIG.get('max_workers', None)
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    print(f"Using parallel processing with {max_workers} workers")
    start_time = time.time()
    successful = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_series_cwt, info): info[0] 
                  for info in series_info_list}
        
        for future in as_completed(futures):
            series_name = futures[future]
            result = future.result()
            if result is not None:
                results[series_name] = result
                successful += 1
    
    elapsed_time = time.time() - start_time
    print(f"\nCWT analysis completed in {elapsed_time/60:.2f} minutes")
    print(f"  Successfully processed: {successful}/{len(series_info_list)} series")
    
    return results

def run_cwt(config: Optional[Dict] = None, output_dir: Optional[Union[str, Path]] = None):
    global CWT_CONFIG_TREND, CWT_CONFIG_SEASONAL, CWT_CONFIG_RESIDUAL
    global PARALLEL_CONFIG, VISUALIZATION_CONFIG, ANALYSIS_CONFIG, OUTPUT_CONFIG
    global STL_DATA_PATH, AFD_DATA_PATH
    
    if config:
        if 'cwt_config_trend' in config:
            CWT_CONFIG_TREND.update(config['cwt_config_trend'])
        if 'cwt_config_seasonal' in config:
            CWT_CONFIG_SEASONAL.update(config['cwt_config_seasonal'])
        if 'cwt_config_residual' in config:
            CWT_CONFIG_RESIDUAL.update(config['cwt_config_residual'])
        if 'parallel_config' in config:
            PARALLEL_CONFIG.update(config['parallel_config'])
        if 'visualization_config' in config:
            VISUALIZATION_CONFIG.update(config['visualization_config'])
        if 'analysis_config' in config:
            ANALYSIS_CONFIG.update(config['analysis_config'])
        if 'output_config' in config:
            OUTPUT_CONFIG.update(config['output_config'])
        if 'stl_data_path' in config:
            STL_DATA_PATH.update(config['stl_data_path'])
        if 'afd_data_path' in config:
            AFD_DATA_PATH.update(config['afd_data_path'])
            
    if output_dir:
        OUTPUT_CONFIG['base_dir'] = str(output_dir)
        
    print("=" * 80)
    print("CWT Feature Identification Analysis (Robustness Version)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {OUTPUT_CONFIG['base_dir']}")
    print(f"Wavelet Trend: {CWT_CONFIG_TREND.get('wavelet')} {CWT_CONFIG_TREND.get('wavelet_params')}")
    print(f"Wavelet Seasonal: {CWT_CONFIG_SEASONAL.get('wavelet')} {CWT_CONFIG_SEASONAL.get('wavelet_params')}")
    
    out_path = Path(OUTPUT_CONFIG['base_dir'])
    out_path.mkdir(parents=True, exist_ok=True)
    
    stl_dict, time_index, series_names, stl_metadata = load_stl_data()
        
    if ANALYSIS_CONFIG['reconstruct_r_from_afd'] and ANALYSIS_CONFIG['enable_residual_analysis']:
        afd_residual_array, component_numbers, afd_metadata = load_afd_data()
        reconstructed_r = reconstruct_r_from_afd(afd_residual_array, component_numbers, series_names)
        stl_dict['residual'] = reconstructed_r
        print("Using reconstructed R(t) from AFD components")
            
    co_circulation_df = load_co_circulation_periods()
    
    all_results = {}
    
    if ANALYSIS_CONFIG['enable_trend_analysis']:
        all_results['trend'] = process_all_series_cwt(
            stl_dict, 'trend', CWT_CONFIG_TREND, time_index, series_names
        )
        
    if ANALYSIS_CONFIG['enable_seasonal_analysis']:
        all_results['seasonal'] = process_all_series_cwt(
            stl_dict, 'seasonal', CWT_CONFIG_SEASONAL, time_index, series_names
        )
        
    if ANALYSIS_CONFIG['enable_residual_analysis']:
        all_results['residual'] = process_all_series_cwt(
            stl_dict, 'residual', CWT_CONFIG_RESIDUAL, time_index, series_names
        )
    
    metadata = {
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'trend': CWT_CONFIG_TREND,
            'seasonal': CWT_CONFIG_SEASONAL,
            'residual': CWT_CONFIG_RESIDUAL
        }
    }
    with open(out_path / 'cwt_metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)
        
    print("\nCWT Analysis Complete")
    return all_results

if __name__ == '__main__':
    run_cwt()
