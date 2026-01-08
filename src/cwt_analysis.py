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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.ndimage import label
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.multitest import multipletests
import pywt

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

CWT_CONFIG_TREND = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (52, 500),
    'n_scales': 100,
    'sampling_period': 1.0
}

CWT_CONFIG_SEASONAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (26, 52),
    'n_scales': 50,
    'sampling_period': 1.0
}

CWT_CONFIG_RESIDUAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (1, 26),
    'n_scales': 50,
    'sampling_period': 1.0
}

PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 10
}

VISUALIZATION_CONFIG = {
    'max_series_to_plot': None,
    'batch_size': 10,
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

ANALYSIS_CONFIG = {
    'anomaly_threshold_percentile': 95,
    'enable_trend_analysis': True,
    'enable_seasonal_analysis': True,
    'enable_residual_analysis': True,
    'reconstruct_r_from_afd': True,
}

# 输出目录配置
OUTPUT_CONFIG = {
    'base_dir': 'AFD/cwt_analysis',
    'time_frequency_plots_dir': 'time_frequency_plots',
    'frequency_drift_dir': 'frequency_drift',
    'trend_rate_dir': 'trend_rate',
    'intensity_variation_dir': 'intensity_variation',
    'anomalous_events_dir': 'anomalous_events',
    'comparison_analysis_dir': 'comparison_analysis'
}

# STL和AFD数据路径
STL_DATA_PATH = {
    'stl_results_npy': 'AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': 'AFD/stl_decomposition/arrays/stl_results.pkl'
}

AFD_DATA_PATH = {
    'residual_components_npy': 'AFD/afd_residual_components.npy',
    'component_numbers_csv': 'AFD/afd_component_numbers.csv',
    'afd_metadata_pkl': 'AFD/afd_stl_metadata.pkl'
}

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
    try:
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
        print(f"Data shape: {stl_3d_array.shape}")
        print(f"Time range: {time_index[0]} to {time_index[-1]}")
        
        return {
            'trend': trend_dict,
            'seasonal': seasonal_dict,
            'residual': residual_dict
        }, time_index, series_names, metadata
        
    except Exception as e:
        print(f"Error loading STL data: {e}")
        raise


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
    try:
        afd_residual_array = np.load(residual_components_npy)
        
        component_numbers_df = pd.read_csv(component_numbers_csv)
        component_numbers = dict(zip(component_numbers_df['Series_Name'], 
                                     component_numbers_df['K_Residual']))
        
        metadata = {}
        if afd_metadata_pkl.exists():
            with open(afd_metadata_pkl, 'rb') as f:
                metadata = pickle.load(f)
        
        print(f"Successfully loaded AFD data")
        print(f"AFD array shape: {afd_residual_array.shape}")
        print(f"Component numbers: {len(component_numbers)} series")
        
        return afd_residual_array, component_numbers, metadata
        
    except Exception as e:
        print(f"Error loading AFD data: {e}")
        raise

def reconstruct_r_from_afd(afd_residual_array, component_numbers, series_names):
    print("=" * 80)
    print("Reconstructing R(t) from AFD Components")
    print("=" * 80)
    
    reconstructed_r = {}
    
    for i, series_name in enumerate(series_names):
        if series_name not in component_numbers:
            print(f"Warning: {series_name} not found in component_numbers, skipping")
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
    
    try:
        co_circulation_df = pd.read_csv(file_path)
        print(f"Loaded co-circulation periods: {len(co_circulation_df)} records")
        return co_circulation_df
    except Exception as e:
        print(f"Error loading co-circulation periods: {e}")
        return None



def compute_scales(scale_range, n_scales, sampling_period=1.0):
    min_scale, max_scale = scale_range
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
    return scales

def scales_to_frequencies(scales, sampling_period=1.0, wavelet_params=(1.5, 1.0)):
    Fb, Fc = wavelet_params
    frequencies = Fc / (scales * sampling_period)
    periods = scales * sampling_period / Fc
    return frequencies, periods

def compute_cwt(signal, scales, wavelet='cmor', wavelet_params=(1.5, 1.0), 
                sampling_period=1.0):
    coefficients, frequencies = pywt.cwt(
        signal, 
        scales, 
        wavelet, 
        sampling_period=sampling_period
    )
    
    frequencies_calc, periods = scales_to_frequencies(
        scales, sampling_period, wavelet_params
    )
    
    return coefficients, frequencies_calc, periods

def identify_frequency_drift(cwt_result, frequencies, time_index, component_type='seasonal'):
    energy_density = np.abs(cwt_result)**2
    
    dominant_freq_indices = np.argmax(energy_density, axis=0)
    dominant_frequencies = frequencies[dominant_freq_indices]
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
    total_energy_sum = np.sum(energy_density)
    concentration = max_energy / total_energy_sum if total_energy_sum > 0 else 0
    
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
            event_times.append(time_index[peak_time_idx] if isinstance(time_index, pd.DatetimeIndex) 
                             else time_index[peak_time_idx])
    
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
    
    try:
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
            print(f"Warning: {series_name} ({component_type}) has insufficient data points")
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
            features.update({
                'trend_rate': trend_rate
            })
        
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
        
    except Exception as e:
        print(f"Error processing {series_name} ({component_type}): {e}")
        import traceback
        traceback.print_exc()
        return None


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
    failed = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_series_cwt, info): info[0] 
                  for info in series_info_list}
        
        completed = 0
        for future in as_completed(futures):
            completed += 1
            series_name = futures[future]
            
            try:
                result = future.result()
                if result is not None:
                    results[series_name] = result
                    successful += 1
                else:
                    failed.append(series_name)
            except Exception as e:
                print(f"Error processing {series_name}: {e}")
                failed.append(series_name)
            
            if completed % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / completed
                remaining = (len(series_info_list) - completed) * avg_time
                print(f"  Progress: {completed}/{len(series_info_list)} ({100*completed/len(series_info_list):.1f}%) "
                      f"- ETA: {remaining/60:.1f} minutes")
    
    elapsed_time = time.time() - start_time
    print(f"\nCWT analysis completed in {elapsed_time/60:.2f} minutes")
    print(f"  Successfully processed: {successful}/{len(series_info_list)} series")
    print(f"  Failed: {len(failed)} series")
    
    return results

def main_cwt_analysis():
    print("=" * 80)
    print("CWT Feature Identification Analysis")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    output_dir = Path(OUTPUT_CONFIG['base_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stl_dict, time_index, series_names, stl_metadata = load_stl_data()
    
    if ANALYSIS_CONFIG['reconstruct_r_from_afd'] and ANALYSIS_CONFIG['enable_residual_analysis']:
        afd_residual_array, component_numbers, afd_metadata = load_afd_data()
        
        reconstructed_r = reconstruct_r_from_afd(afd_residual_array, component_numbers, series_names)
        
        stl_dict['residual'] = reconstructed_r
        print("Using reconstructed R(t) from AFD components for CWT analysis")
    
    co_circulation_df = load_co_circulation_periods()
    
    all_results = {}
    
    if ANALYSIS_CONFIG['enable_trend_analysis']:
        trend_results = process_all_series_cwt(
            stl_dict, 'trend', CWT_CONFIG_TREND, time_index, series_names
        )
        all_results['trend'] = trend_results
    
    if ANALYSIS_CONFIG['enable_seasonal_analysis']:
        seasonal_results = process_all_series_cwt(
            stl_dict, 'seasonal', CWT_CONFIG_SEASONAL, time_index, series_names
        )
        all_results['seasonal'] = seasonal_results
    
    if ANALYSIS_CONFIG['enable_residual_analysis']:
        residual_results = process_all_series_cwt(
            stl_dict, 'residual', CWT_CONFIG_RESIDUAL, time_index, series_names
        )
        all_results['residual'] = residual_results
    
    save_cwt_results(all_results, time_index, series_names, output_dir)
    
    if VISUALIZATION_CONFIG['enable_visualization']:
        generate_cwt_visualizations(all_results, time_index, series_names, 
                                   co_circulation_df, output_dir)
    
    if co_circulation_df is not None:
        comparison_results = perform_comparison_analysis(
            all_results, time_index, series_names, co_circulation_df, output_dir
        )
        save_comparison_results(comparison_results, output_dir)
    
    print("\n" + "=" * 80)
    print("CWT Analysis Complete")
    print("=" * 80)
    print(f"Results saved to: {output_dir}")
    print(f"  - Trend analysis: {len(all_results.get('trend', {}))} series")
    print(f"  - Seasonal analysis: {len(all_results.get('seasonal', {}))} series")
    print(f"  - Residual analysis: {len(all_results.get('residual', {}))} series")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return all_results, time_index, series_names, co_circulation_df

def save_cwt_results(all_results, time_index, series_names, output_dir):
    print("\n" + "=" * 80)
    print("Saving CWT Analysis Results")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    
    freq_drift_dir = output_dir / OUTPUT_CONFIG['frequency_drift_dir']
    trend_rate_dir = output_dir / OUTPUT_CONFIG['trend_rate_dir']
    intensity_dir = output_dir / OUTPUT_CONFIG['intensity_variation_dir']
    anomalies_dir = output_dir / OUTPUT_CONFIG['anomalous_events_dir']
    
    for dir_path in [freq_drift_dir, trend_rate_dir, intensity_dir, anomalies_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    if 'seasonal' in all_results:
        save_frequency_drift_results(all_results['seasonal'], time_index, freq_drift_dir)
    
    if 'trend' in all_results:
        save_trend_rate_results(all_results['trend'], time_index, trend_rate_dir)
    
    if 'seasonal' in all_results:
        save_intensity_variation_results(all_results['seasonal'], time_index, intensity_dir, 'seasonal')
    if 'residual' in all_results:
        save_intensity_variation_results(all_results['residual'], time_index, intensity_dir, 'residual')
    
    if 'residual' in all_results:
        save_anomalous_events_results(all_results['residual'], time_index, anomalies_dir)
    
    metadata = {
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_series': len(series_names),
        'n_timepoints': len(time_index),
        'time_index': time_index,
        'series_names': series_names,
        'cwt_config_trend': CWT_CONFIG_TREND,
        'cwt_config_seasonal': CWT_CONFIG_SEASONAL,
        'cwt_config_residual': CWT_CONFIG_RESIDUAL,
        'analysis_config': ANALYSIS_CONFIG
    }
    
    metadata_path = output_dir / 'metadata.pkl'
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {metadata_path}")

def save_frequency_drift_results(seasonal_results, time_index, output_dir):
    print("Saving frequency drift results...")
    
    all_data = []
    
    for series_name, result in seasonal_results.items():
        if 'frequency_drift' not in result:
            continue
        
        freq_drift = result['frequency_drift']
        series_time_index = result.get('time_index', time_index[:len(freq_drift['dominant_frequencies'])])
        
        for i, (t, freq, period) in enumerate(zip(
            series_time_index,
            freq_drift['dominant_frequencies'],
            freq_drift['dominant_periods']
        )):
            all_data.append({
                'Series_Name': series_name,
                'TimeIndex': t,
                'DominantFrequency': freq,
                'DominantPeriod': period,
                'FrequencyStability': freq_drift['frequency_stability']['cv']
            })
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        dominant_freq_path = output_dir / 'dominant_frequencies.csv'
        df[['Series_Name', 'TimeIndex', 'DominantFrequency', 'DominantPeriod']].to_csv(
            dominant_freq_path, index=False, encoding='utf-8-sig'
        )
        
        stability_data = df.groupby('Series_Name')['FrequencyStability'].first().reset_index()
        stability_data['DriftRate'] = [
            seasonal_results[name]['frequency_drift']['drift_rate'] 
            if name in seasonal_results and 'frequency_drift' in seasonal_results[name]
            else np.nan
            for name in stability_data['Series_Name']
        ]
        
        stability_path = output_dir / 'frequency_stability.csv'
        stability_data.to_csv(stability_path, index=False, encoding='utf-8-sig')
        
        drift_rates_path = output_dir / 'drift_rates.csv'
        stability_data[['Series_Name', 'DriftRate']].to_csv(
            drift_rates_path, index=False, encoding='utf-8-sig'
        )
        
        print(f"  Saved frequency drift results to: {output_dir}")


def save_trend_rate_results(trend_results, time_index, output_dir):
    print("Saving trend rate results...")
    
    all_data = []
    
    for series_name, result in trend_results.items():
        if 'trend_rate' not in result:
            continue
        
        trend_rate = result['trend_rate']
        series_time_index = result.get('time_index', time_index[:len(trend_rate['trend_rate'])])
        
        for i, (t, rate, accel) in enumerate(zip(
            series_time_index,
            trend_rate['trend_rate'],
            trend_rate['acceleration']
        )):
            all_data.append({
                'Series_Name': series_name,
                'TimeIndex': t,
                'TrendRate': rate,
                'Acceleration': accel
            })
    
    if all_data:
        df = pd.DataFrame(all_data)
        
        trend_rates_path = output_dir / 'trend_rates.csv'
        df[['Series_Name', 'TimeIndex', 'TrendRate']].to_csv(
            trend_rates_path, index=False, encoding='utf-8-sig'
        )
        
        rate_var_data = df.groupby('Series_Name')['TrendRate'].agg(['mean', 'std']).reset_index()
        rate_var_data['RateVariability'] = rate_var_data['std'] / rate_var_data['mean'].abs()
        rate_var_data['RateVariability'] = rate_var_data['RateVariability'].fillna(0)
        
        rate_var_path = output_dir / 'rate_variability.csv'
        rate_var_data[['Series_Name', 'RateVariability']].to_csv(
            rate_var_path, index=False, encoding='utf-8-sig'
        )
        
        accel_path = output_dir / 'acceleration.csv'
        df[['Series_Name', 'TimeIndex', 'Acceleration']].to_csv(
            accel_path, index=False, encoding='utf-8-sig'
        )
        
        print(f"  Saved trend rate results to: {output_dir}")

def save_intensity_variation_results(results, time_index, output_dir, component_type):
    print(f"Saving intensity variation results for {component_type}...")
    
    all_energy_data = []
    energy_density_list = []
    
    for series_name, result in results.items():
        if 'intensity_variation' not in result:
            continue
        
        intensity = result['intensity_variation']
        series_time_index = result.get('time_index', time_index[:len(intensity['total_energy'])])
        
        for i, (t, energy) in enumerate(zip(series_time_index, intensity['total_energy'])):
            all_energy_data.append({
                'Series_Name': series_name,
                'TimeIndex': t,
                'TotalEnergy': energy
            })
        
        energy_density_list.append({
            'series_name': series_name,
            'energy_density': intensity['energy_density']
        })
    
    if all_energy_data:
        df = pd.DataFrame(all_energy_data)
        
        total_energy_path = output_dir / f'total_energy_{component_type}.csv'
        df.to_csv(total_energy_path, index=False, encoding='utf-8-sig')
        
        var_data = df.groupby('Series_Name')['TotalEnergy'].agg(['mean', 'std']).reset_index()
        var_data['EnergyVariability'] = var_data['std'] / var_data['mean']
        var_data['EnergyVariability'] = var_data['EnergyVariability'].fillna(0)
        
        var_path = output_dir / f'energy_variability_{component_type}.csv'
        var_data[['Series_Name', 'EnergyVariability']].to_csv(
            var_path, index=False, encoding='utf-8-sig'
        )
        
        print(f"  Saved intensity variation results to: {output_dir}")

def save_anomalous_events_results(residual_results, time_index, output_dir):
    print("Saving anomalous events results...")
    
    all_events = []
    
    for series_name, result in residual_results.items():
        if 'anomalous_events' not in result:
            continue
        
        anomalies = result['anomalous_events']
        
        for event_idx, region in enumerate(anomalies['anomalous_regions']):
            all_events.append({
                'Series_Name': series_name,
                'EventIndex': event_idx + 1,
                'StartTime': region['t_start'],
                'EndTime': region['t_end'],
                'FrequencyRange': f"{region['f_start']:.6f}-{region['f_end']:.6f}",
                'PeakEnergy': region['peak_energy'],
                'MeanEnergy': region['mean_energy'],
                'EventType': 'high_energy_burst'
            })
    
    if all_events:
        df = pd.DataFrame(all_events)
        events_path = output_dir / 'event_locations.csv'
        df.to_csv(events_path, index=False, encoding='utf-8-sig')
        
        stats_data = []
        for series_name, result in residual_results.items():
            if 'anomalous_events' in result:
                stats = result['anomalous_events']['event_statistics']
                stats_data.append({
                    'Series_Name': series_name,
                    'TotalEvents': stats['total_events'],
                    'EventsPerYear': stats['events_per_year'],
                    'MeanEventEnergy': stats['mean_event_energy'],
                    'MaxEventEnergy': stats['max_event_energy']
                })
        
        if stats_data:
            stats_df = pd.DataFrame(stats_data)
            stats_path = output_dir / 'event_statistics.csv'
            stats_df.to_csv(stats_path, index=False, encoding='utf-8-sig')
        
        print(f"  Saved anomalous events results to: {output_dir}")

def plot_time_frequency_spectrum(result, series_name, component_type, time_index, 
                                  co_circulation_df=None, output_path=None):
    if output_path is None:
        return
    
    output_path = Path(output_path)
    
    cwt_result = result['cwt_result']
    frequencies = result['frequencies']
    periods = result['periods']
    series_time_index = result.get('time_index', time_index[:cwt_result.shape[1]])
    
    energy_density = np.abs(cwt_result)**2
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    im = ax.contourf(
        pd.to_datetime(series_time_index),
        periods,
        energy_density,
        levels=50,
        cmap='viridis',
        extend='both'
    )
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Energy Density |W|²', rotation=270, labelpad=20)
    
    if component_type == 'seasonal' and 'frequency_drift' in result:
        dominant_freqs = result['frequency_drift']['dominant_frequencies']
        dominant_periods = 1.0 / dominant_freqs
        ax.plot(pd.to_datetime(series_time_index), dominant_periods, 
               'r-', linewidth=2, label='Dominant Period', alpha=0.8)
        ax.legend()
    
    if co_circulation_df is not None:
        pass
    
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Period (weeks)', fontsize=12)
    ax.set_title(f'{series_name} - {component_type.upper()} Component CWT', fontsize=14, fontweight='bold')
    
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    for fmt in VISUALIZATION_CONFIG['format']:
        output_file = output_path.with_suffix(f'.{fmt}')
        fig.savefig(output_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    
    plt.close(fig)
    gc.collect()

def generate_cwt_visualizations(all_results, time_index, series_names, 
                                co_circulation_df, output_dir):
    print("\n" + "=" * 80)
    print("Generating CWT Visualization Plots")
    print("=" * 80)
    
    output_dir = Path(output_dir)
    viz_dir = output_dir / OUTPUT_CONFIG['time_frequency_plots_dir']
    
    trend_viz_dir = viz_dir / 'trend' / 'individual'
    seasonal_viz_dir = viz_dir / 'seasonal' / 'individual'
    residual_viz_dir = viz_dir / 'residual' / 'individual'
    
    for dir_path in [trend_viz_dir, seasonal_viz_dir, residual_viz_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    plot_tasks = []
    
    if 'trend' in all_results:
        for series_name, result in all_results['trend'].items():
            safe_name = series_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = trend_viz_dir / f"{safe_name}_cwt_trend"
            plot_tasks.append((result, series_name, 'trend', time_index, co_circulation_df, output_path))
    
    if 'seasonal' in all_results:
        for series_name, result in all_results['seasonal'].items():
            safe_name = series_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = seasonal_viz_dir / f"{safe_name}_cwt_seasonal"
            plot_tasks.append((result, series_name, 'seasonal', time_index, co_circulation_df, output_path))
    
    if 'residual' in all_results:
        for series_name, result in all_results['residual'].items():
            safe_name = series_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            output_path = residual_viz_dir / f"{safe_name}_cwt_residual"
            plot_tasks.append((result, series_name, 'residual', time_index, co_circulation_df, output_path))
    
    print(f"Generating {len(plot_tasks)} plots...")
    
    def plot_wrapper(plot_info):
        try:
            plot_time_frequency_spectrum(*plot_info)
            return (True, plot_info[1], None)
        except Exception as e:
            return (False, plot_info[1], str(e))
    
    successful = 0
    failed = []
    
    max_workers = min(os.cpu_count() or 4, len(plot_tasks))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plot = {executor.submit(plot_wrapper, plot_info): plot_info 
                          for plot_info in plot_tasks}
        
        completed = 0
        for future in as_completed(future_to_plot):
            completed += 1
            success, series_name, error = future.result()
            
            if success:
                successful += 1
            else:
                print(f"Error plotting {series_name}: {error}")
                failed.append(series_name)
            
            if completed % 10 == 0 or completed == len(plot_tasks):
                print(f"  Progress: {completed}/{len(plot_tasks)} ({100*completed/len(plot_tasks):.1f}%)")
    
    print(f"\nVisualization completed: {successful}/{len(plot_tasks)} successful, {len(failed)} failed")

def classify_time_periods(time_index, co_circulation_df):
    if co_circulation_df is None:
        return None
    
    if isinstance(time_index, pd.DatetimeIndex):
        time_dates = time_index
    else:
        time_dates = pd.to_datetime(time_index)
    
    is_co_circulation = np.zeros(len(time_dates), dtype=bool)
    
    date_cols = ['Actual Start Date', 'Actual End Date', 'Adjusted Start Date', 'Adjusted End Date', 
                 'start_date', 'end_date', 'Start Date', 'End Date']
    
    start_col = None
    end_col = None
    
    for col in date_cols:
        if col in co_circulation_df.columns:
            if 'start' in col.lower():
                start_col = col
            elif 'end' in col.lower():
                end_col = col
    
    if start_col is None or end_col is None:
        print(f"Warning: Could not find date columns in co_circulation_df. Available columns: {co_circulation_df.columns.tolist()}")
        return None
    
    for _, period in co_circulation_df.iterrows():
        try:
            start_date = pd.to_datetime(period[start_col])
            end_date = pd.to_datetime(period[end_col])
            
            mask = (time_dates >= start_date) & (time_dates <= end_date)
            is_co_circulation[mask] = True
        except Exception as e:
            print(f"Warning: Error processing period {period.get('Region', 'unknown')}: {e}")
            continue
    
    co_count = np.sum(is_co_circulation)
    single_count = len(time_dates) - co_count
    
    print(f"Time period classification:")
    print(f"  Co-circulation periods: {co_count} time points ({100*co_count/len(time_dates):.1f}%)")
    print(f"  Single-dominant periods: {single_count} time points ({100*single_count/len(time_dates):.1f}%)")
    
    return is_co_circulation

def perform_comparison_analysis(all_results, time_index, series_names, 
                                co_circulation_df, output_dir):
    print("\n" + "=" * 80)
    print("Performing Comparison Analysis")
    print("=" * 80)
    
    is_co_circulation = classify_time_periods(time_index, co_circulation_df)
    
    if is_co_circulation is None:
        print("Warning: Cannot perform comparison analysis without co-circulation periods")
        return None
    
    comparison_results = {}
    
    if 'seasonal' in all_results:
        comparison_results['seasonal'] = compare_seasonal_features(
            all_results['seasonal'], is_co_circulation, time_index
        )
    
    if 'trend' in all_results:
        comparison_results['trend'] = compare_trend_features(
            all_results['trend'], is_co_circulation, time_index
        )
    
    if 'residual' in all_results:
        comparison_results['residual'] = compare_residual_features(
            all_results['residual'], is_co_circulation, time_index
        )
    
    return comparison_results


def calculate_cohens_d(group1, group2):
    if len(group1) == 0 or len(group2) == 0:
        return np.nan
    
    mean1, mean2 = np.mean(group1), np.mean(group2)
    std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    
    pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                         (len(group1) + len(group2) - 2))
    
    if pooled_std == 0:
        return np.nan
    
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d

def perform_statistical_test(data1, data2, test_type='variance'):
    if len(data1) == 0 or len(data2) == 0:
        return np.nan, 'N/A', False
    
    if test_type == 'variance':
        try:
            levene_stat, p_value = levene(data1, data2)
            return p_value, 'Levene', False
        except:
            return np.nan, 'Levene', False
    
    elif test_type == 'mean':
        try:
            _, p_norm1 = shapiro(data1)
            _, p_norm2 = shapiro(data2)
            is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        except:
            is_normal = False
        
        try:
            if is_normal:
                t_stat, p_value = ttest_ind(data1, data2)
                return p_value, 't-test', True
            else:
                u_stat, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
                return p_value, 'Mann-Whitney U', False
        except:
            return np.nan, 'N/A', False
    
    else:
        return np.nan, 'N/A', False

def compare_seasonal_features(seasonal_results, is_co_circulation, time_index):
    comparison_data = []
    
    for series_name, result in seasonal_results.items():
        if 'frequency_drift' not in result or 'intensity_variation' not in result:
            continue
        
        freq_drift = result['frequency_drift']
        intensity = result['intensity_variation']
        series_time_index = result.get('time_index', time_index[:len(freq_drift['dominant_frequencies'])])
        
        series_is_co = is_co_circulation[:len(series_time_index)]
        
        co_mask = series_is_co
        single_mask = ~series_is_co
        
        if len(freq_drift['dominant_frequencies']) == len(series_is_co):
            co_freqs = freq_drift['dominant_frequencies'][co_mask]
            single_freqs = freq_drift['dominant_frequencies'][single_mask]
            
            if len(co_freqs) > 0 and len(single_freqs) > 0:
                co_freq_cv = np.std(co_freqs) / np.mean(co_freqs) if np.mean(co_freqs) > 0 else 0
                single_freq_cv = np.std(single_freqs) / np.mean(single_freqs) if np.mean(single_freqs) > 0 else 0
                
                p_value, test_method, is_normal = perform_statistical_test(
                    co_freqs, single_freqs, test_type='variance'
                )
                
                cohens_d = calculate_cohens_d(co_freqs, single_freqs)
                
                comparison_data.append({
                    'Series_Name': series_name,
                    'Feature': 'FrequencyStability',
                    'Component': 'Seasonal',
                    'CoPeriod_Mean': np.mean(co_freqs),
                    'CoPeriod_Std': np.std(co_freqs),
                    'SinglePeriod_Mean': np.mean(single_freqs),
                    'SinglePeriod_Std': np.std(single_freqs),
                    'CoPeriod_CV': co_freq_cv,
                    'SinglePeriod_CV': single_freq_cv,
                    'Difference': co_freq_cv - single_freq_cv,
                    'P_Value': p_value,
                    'Test_Method': test_method,
                    'Cohen_D': cohens_d
                })
        
        if len(intensity['total_energy']) == len(series_is_co):
            co_energy = intensity['total_energy'][co_mask]
            single_energy = intensity['total_energy'][single_mask]
            
            if len(co_energy) > 0 and len(single_energy) > 0:
                co_energy_cv = np.std(co_energy) / np.mean(co_energy) if np.mean(co_energy) > 0 else 0
                single_energy_cv = np.std(single_energy) / np.mean(single_energy) if np.mean(single_energy) > 0 else 0
                
                p_value, test_method, is_normal = perform_statistical_test(
                    co_energy, single_energy, test_type='variance'
                )
                
                cohens_d = calculate_cohens_d(co_energy, single_energy)
                
                comparison_data.append({
                    'Series_Name': series_name,
                    'Feature': 'EnergyVariability',
                    'Component': 'Seasonal',
                    'CoPeriod_Mean': np.mean(co_energy),
                    'CoPeriod_Std': np.std(co_energy),
                    'SinglePeriod_Mean': np.mean(single_energy),
                    'SinglePeriod_Std': np.std(single_energy),
                    'CoPeriod_CV': co_energy_cv,
                    'SinglePeriod_CV': single_energy_cv,
                    'Difference': co_energy_cv - single_energy_cv,
                    'P_Value': p_value,
                    'Test_Method': test_method,
                    'Cohen_D': cohens_d
                })
    
    return comparison_data


def compare_trend_features(trend_results, is_co_circulation, time_index):
    comparison_data = []
    
    for series_name, result in trend_results.items():
        if 'trend_rate' not in result:
            continue
        
        trend_rate = result['trend_rate']
        series_time_index = result.get('time_index', time_index[:len(trend_rate['trend_rate'])])
        
        series_is_co = is_co_circulation[:len(series_time_index)]
        
        co_mask = series_is_co
        single_mask = ~series_is_co
        
        if len(trend_rate['trend_rate']) == len(series_is_co):
            co_rates = np.abs(trend_rate['trend_rate'][co_mask])
            single_rates = np.abs(trend_rate['trend_rate'][single_mask])
            
            if len(co_rates) > 0 and len(single_rates) > 0:
                co_rate_cv = np.std(co_rates) / np.mean(co_rates) if np.mean(co_rates) > 0 else 0
                single_rate_cv = np.std(single_rates) / np.mean(single_rates) if np.mean(single_rates) > 0 else 0
                
                p_value, test_method, is_normal = perform_statistical_test(
                    co_rates, single_rates, test_type='variance'
                )
                
                cohens_d = calculate_cohens_d(co_rates, single_rates)
                
                comparison_data.append({
                    'Series_Name': series_name,
                    'Feature': 'RateVariability',
                    'Component': 'Trend',
                    'CoPeriod_Mean': np.mean(co_rates),
                    'CoPeriod_Std': np.std(co_rates),
                    'SinglePeriod_Mean': np.mean(single_rates),
                    'SinglePeriod_Std': np.std(single_rates),
                    'CoPeriod_CV': co_rate_cv,
                    'SinglePeriod_CV': single_rate_cv,
                    'Difference': co_rate_cv - single_rate_cv,
                    'P_Value': p_value,
                    'Test_Method': test_method,
                    'Cohen_D': cohens_d
                })
    
    return comparison_data


def compare_residual_features(residual_results, is_co_circulation, time_index):
    comparison_data = []
    
    for series_name, result in residual_results.items():
        if 'anomalous_events' not in result or 'intensity_variation' not in result:
            continue
        
        anomalies = result['anomalous_events']
        intensity = result['intensity_variation']
        series_time_index = result.get('time_index', time_index[:len(intensity['total_energy'])])
        
        series_is_co = is_co_circulation[:len(series_time_index)]
        
        co_mask = series_is_co
        single_mask = ~series_is_co
        
        co_events = 0
        single_events = 0
        
        for region in anomalies['anomalous_regions']:
            event_time = pd.to_datetime(region['t_start'])
            if event_time in pd.to_datetime(series_time_index):
                event_idx = pd.to_datetime(series_time_index).get_loc(event_time)
                if event_idx < len(series_is_co):
                    if series_is_co[event_idx]:
                        co_events += 1
                    else:
                        single_events += 1
        
        co_period_years = np.sum(co_mask) / 52.0 if np.sum(co_mask) > 0 else 0
        single_period_years = np.sum(single_mask) / 52.0 if np.sum(single_mask) > 0 else 0
        
        co_events_per_year = co_events / co_period_years if co_period_years > 0 else 0
        single_events_per_year = single_events / single_period_years if single_period_years > 0 else 0
        
        if len(intensity['total_energy']) == len(series_is_co):
            co_energy = intensity['total_energy'][co_mask]
            single_energy = intensity['total_energy'][single_mask]
            
            if len(co_energy) > 0 and len(single_energy) > 0:
                co_energy_cv = np.std(co_energy) / np.mean(co_energy) if np.mean(co_energy) > 0 else 0
                single_energy_cv = np.std(single_energy) / np.mean(single_energy) if np.mean(single_energy) > 0 else 0
                
                p_value, test_method, is_normal = perform_statistical_test(
                    co_energy, single_energy, test_type='variance'
                )
                
                cohens_d = calculate_cohens_d(co_energy, single_energy)
                
                comparison_data.append({
                    'Series_Name': series_name,
                    'Feature': 'EnergyVariability',
                    'Component': 'Residual',
                    'CoPeriod_Mean': np.mean(co_energy),
                    'CoPeriod_Std': np.std(co_energy),
                    'SinglePeriod_Mean': np.mean(single_energy),
                    'SinglePeriod_Std': np.std(single_energy),
                    'CoPeriod_CV': co_energy_cv,
                    'SinglePeriod_CV': single_energy_cv,
                    'Difference': co_energy_cv - single_energy_cv,
                    'P_Value': p_value,
                    'Test_Method': test_method,
                    'Cohen_D': cohens_d
                })
        
        comparison_data.append({
            'Series_Name': series_name,
            'Feature': 'AnomalyFrequency',
            'Component': 'Residual',
            'CoPeriod_Mean': co_events_per_year,
            'CoPeriod_Std': 0.0,
            'SinglePeriod_Mean': single_events_per_year,
            'SinglePeriod_Std': 0.0,
            'CoPeriod_Events': co_events,
            'SinglePeriod_Events': single_events,
            'Difference': co_events_per_year - single_events_per_year,
            'P_Value': np.nan,
            'Test_Method': 'N/A',
            'Cohen_D': np.nan
        })
    
    return comparison_data


def save_comparison_results(comparison_results, output_dir):
    if comparison_results is None:
        return
    
    print("Saving comparison analysis results...")
    
    output_dir = Path(output_dir)
    comparison_dir = output_dir / OUTPUT_CONFIG['comparison_analysis_dir']
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    all_comparison_data = []
    
    for component_type, comparison_data in comparison_results.items():
        for item in comparison_data:
            if 'Component' not in item:
                item['Component'] = component_type
            all_comparison_data.append(item)
    
    if all_comparison_data:
        df = pd.DataFrame(all_comparison_data)
        
        if 'P_Value' in df.columns:
            valid_mask = ~df['P_Value'].isna()
            p_values = df.loc[valid_mask, 'P_Value'].values
            
            if len(p_values) > 0:
                rejected, p_corrected, alpha_sidak, alpha_bonf = multipletests(
                    p_values,
                    alpha=0.05,
                    method='fdr_bh'
                )
                
                df['P_Value_Corrected'] = np.nan
                df.loc[valid_mask, 'P_Value_Corrected'] = p_corrected
                
                df['Significant_Uncorrected'] = False
                df.loc[valid_mask, 'Significant_Uncorrected'] = df.loc[valid_mask, 'P_Value'] < 0.05
                
                df['Significant_FDR'] = False
                df.loc[valid_mask, 'Significant_FDR'] = rejected
                
                n_total = len(df)
                n_valid = len(p_values)
                n_sig_uncorrected = df['Significant_Uncorrected'].sum()
                n_sig_fdr = df['Significant_FDR'].sum()
                
                print(f"\nMultiple Comparison Correction Summary:")
                print(f"  Total comparisons: {n_total}")
                print(f"  Valid comparisons: {n_valid}")
                print(f"  Significant (uncorrected, P<0.05): {n_sig_uncorrected} ({100*n_sig_uncorrected/n_valid:.1f}%)")
                print(f"  Significant (FDR corrected, α=0.05): {n_sig_fdr} ({100*n_sig_fdr/n_valid:.1f}%)")
        
        comparison_path = comparison_dir / 'feature_comparison.csv'
        df.to_csv(comparison_path, index=False, encoding='utf-8-sig')
        print(f"  Saved feature comparison to: {comparison_path}")
        
        if 'P_Value' in df.columns and 'Significant_FDR' in df.columns:
            significant_features = df[df['Significant_FDR']].copy()
            if len(significant_features) > 0:
                stats_path = comparison_dir / 'statistical_tests.csv'
                significant_features.to_csv(stats_path, index=False, encoding='utf-8-sig')
                print(f"  Saved FDR-corrected statistical tests to: {stats_path}")
            
            if 'Cohen_D' in df.columns:
                effect_sizes = df[['Series_Name', 'Component', 'Feature', 'Cohen_D', 
                                  'P_Value', 'P_Value_Corrected', 'Significant_FDR']].copy()
                effect_sizes_path = comparison_dir / 'effect_sizes.csv'
                effect_sizes.to_csv(effect_sizes_path, index=False, encoding='utf-8-sig')
                print(f"  Saved effect sizes to: {effect_sizes_path}")


if __name__ == '__main__':
    results = main_cwt_analysis()


