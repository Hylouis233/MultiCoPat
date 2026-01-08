#!/usr/bin/env python3

import gc
import os
import re
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
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.multitest import multipletests
import pywt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts')))
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

DEFAULT_WTC_CONFIG_SEASONAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (26, 52),
    'n_scales': 50,
    'sampling_period': 1.0,
    'smooth_time_window': 10,
    'smooth_scale_window': 3,
    'n_surrogates': 1000,
    'significance_alpha': 0.05,
    'high_coherence_threshold': 0.5,
    'bootstrap_repeats': 100
}

DEFAULT_WTC_CONFIG_TREND = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (52, 500),
    'n_scales': 100,
    'sampling_period': 1.0,
    'smooth_time_window': 10,
    'smooth_scale_window': 3,
    'n_surrogates': 1000,
    'significance_alpha': 0.05,
    'high_coherence_threshold': 0.5,
    'bootstrap_repeats': 100
}

DEFAULT_PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 10
}

DEFAULT_VISUALIZATION_CONFIG = {
    'max_pairs_to_plot': None,
    'batch_size': 10,
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

DEFAULT_ANALYSIS_CONFIG = {
    'enable_seasonal_wtc': True,
    'enable_trend_wtc': False,
    'enable_phase_analysis': True,
    'enable_comparison_analysis': True,
    'max_flu_series': None,
    'max_env_series': None,
    'enable_bootstrap_phase': True
}

DEFAULT_OUTPUT_CONFIG = {
    'base_dir': 'AFD/wtc_analysis',
    'coherence_data_dir': 'coherence_data',
    'coherence_features_dir': 'coherence_features',
    'phase_features_dir': 'phase_features',
    'comparison_analysis_dir': 'comparison_analysis',
    'coherence_plots_dir': 'coherence_plots',
    'comparison_plots_dir': 'comparison_plots'
}

WTC_CONFIG_SEASONAL = DEFAULT_WTC_CONFIG_SEASONAL
WTC_CONFIG_TREND = DEFAULT_WTC_CONFIG_TREND
PARALLEL_CONFIG = DEFAULT_PARALLEL_CONFIG
VISUALIZATION_CONFIG = DEFAULT_VISUALIZATION_CONFIG
ANALYSIS_CONFIG = DEFAULT_ANALYSIS_CONFIG
OUTPUT_CONFIG = DEFAULT_OUTPUT_CONFIG

DEFAULT_STL_DATA_PATH = {
    'stl_results_npy': 'AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': 'AFD/stl_decomposition/arrays/stl_results.pkl'
}

STL_DATA_PATH = DEFAULT_STL_DATA_PATH

CO_CIRCULATION_PATH = 'co_circulation_periods.csv'

def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

def create_output_directory():
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_name in OUTPUT_CONFIG.values():
        if dir_name != base_dir.name and dir_name.endswith('_dir'):
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def load_flu_seasonal_components():
    print("=" * 80)
    print("Loading Flu Seasonal Components from STL Decomposition")
    print("=" * 80)
    
    stl_results_npy = Path(STL_DATA_PATH['stl_results_npy'])
    stl_results_pkl = Path(STL_DATA_PATH['stl_results_pkl'])
    
    if not stl_results_npy.exists() or not stl_results_pkl.exists():
        raise FileNotFoundError(f"STL output files not found: {stl_results_npy} or {stl_results_pkl}")
    
    stl_3d_array = np.load(stl_results_npy)
    with open(stl_results_pkl, 'rb') as f:
        metadata = pickle.load(f)
    
    time_index = metadata['time_index']
    series_names = metadata['series_names']
    
    if isinstance(time_index, pd.Series):
        time_index = pd.to_datetime(time_index.values)
    elif not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    
    seasonal_dict = {}
    for i, series_name in enumerate(series_names):
        seasonal_dict[series_name] = stl_3d_array[i, :, 1]
    
    print(f"Successfully loaded {len(series_names)} flu series seasonal components")
    return seasonal_dict, time_index, series_names, metadata

def load_environmental_seasonal_components():
    print("=" * 80)
    print("Loading Environmental Seasonal Components from STL Decomposition")
    print("=" * 80)
    
    stl_results_npy = Path(STL_DATA_PATH['stl_results_npy'])
    stl_results_pkl = Path(STL_DATA_PATH['stl_results_pkl'])
    
    if not stl_results_npy.exists() or not stl_results_pkl.exists():
        raise FileNotFoundError(f"STL output files not found: {stl_results_npy} or {stl_results_pkl}")
    
    stl_3d_array = np.load(stl_results_npy)
    with open(stl_results_pkl, 'rb') as f:
        metadata = pickle.load(f)
    
    time_index = metadata['time_index']
    series_names = metadata['series_names']
    
    if isinstance(time_index, pd.Series):
        time_index = pd.to_datetime(time_index.values)
    elif not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    
    env_keywords = ['ERA5', 'meteorology', 't2m', 'd2m', 'temperature', 'humidity', 'temp', 'humid']
    env_series_names = [s for s in series_names if any(kw in str(s) for kw in env_keywords)]
    
    seasonal_dict = {}
    time_index_dict = {}
    series_name_to_index = {name: i for i, name in enumerate(series_names)}
    
    for env_name in env_series_names:
        if env_name in series_name_to_index:
            idx = series_name_to_index[env_name]
            seasonal_dict[env_name] = stl_3d_array[idx, :, 1]
            time_index_dict[env_name] = time_index
    
    print(f"Successfully loaded {len(seasonal_dict)} environmental series seasonal components")
    return seasonal_dict, time_index_dict, list(seasonal_dict.keys())

def load_search_index_seasonal_components():
    print("=" * 80)
    print("Loading Search Index Seasonal Components from STL Decomposition")
    print("=" * 80)
    
    stl_results_npy = Path(STL_DATA_PATH['stl_results_npy'])
    stl_results_pkl = Path(STL_DATA_PATH['stl_results_pkl'])
    
    if not stl_results_npy.exists() or not stl_results_pkl.exists():
        raise FileNotFoundError(f"STL output files not found: {stl_results_npy} or {stl_results_pkl}")
    
    stl_3d_array = np.load(stl_results_npy)
    with open(stl_results_pkl, 'rb') as f:
        metadata = pickle.load(f)
    
    time_index = metadata['time_index']
    series_names = metadata['series_names']
    
    if isinstance(time_index, pd.Series):
        time_index = pd.to_datetime(time_index.values)
    elif not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    
    search_keywords = ['baidu_index', 'baidu', 'search_index', 'search']
    search_series_names = [s for s in series_names if any(kw in str(s) for kw in search_keywords)]
    
    seasonal_dict = {}
    time_index_dict = {}
    series_name_to_index = {name: i for i, name in enumerate(series_names)}
    
    for search_name in search_series_names:
        if search_name in series_name_to_index:
            idx = series_name_to_index[search_name]
            seasonal_dict[search_name] = stl_3d_array[idx, :, 1]
            time_index_dict[search_name] = time_index
    
    print(f"Successfully loaded {len(seasonal_dict)} search index series seasonal components")
    return seasonal_dict, time_index_dict, list(seasonal_dict.keys())

def align_time_series(x: np.ndarray, y: np.ndarray, 
                     time_index_x: pd.DatetimeIndex, 
                     time_index_y: pd.DatetimeIndex) -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    min_len_x = min(len(x), len(time_index_x))
    min_len_y = min(len(y), len(time_index_y))
    
    x = x[:min_len_x]
    y = y[:min_len_y]
    time_index_x = time_index_x[:min_len_x]
    time_index_y = time_index_y[:min_len_y]
    
    if not time_index_x.is_monotonic_increasing:
        df_x = pd.DataFrame({'value': x, 'time': time_index_x})
        if time_index_x.duplicated().any():
            df_x = df_x.groupby('time')['value'].mean().reset_index()
        df_x = df_x.sort_values('time')
        time_index_x = pd.to_datetime(df_x['time'].values)
        x = df_x['value'].values
    
    if not time_index_y.is_monotonic_increasing:
        df_y = pd.DataFrame({'value': y, 'time': time_index_y})
        if time_index_y.duplicated().any():
            df_y = df_y.groupby('time')['value'].mean().reset_index()
        df_y = df_y.sort_values('time')
        time_index_y = pd.to_datetime(df_y['time'].values)
        y = df_y['value'].values

    start_time = max(time_index_x[0], time_index_y[0])
    end_time = min(time_index_x[-1], time_index_y[-1])
    
    if start_time >= end_time:
        raise ValueError(f"No overlapping time range")
    
    time_index_aligned = pd.date_range(start=start_time, end=end_time, freq='W')
    
    x_series = pd.Series(x, index=time_index_x)
    x_aligned = x_series.reindex(time_index_aligned, method='nearest').values
    
    y_series = pd.Series(y, index=time_index_y)
    y_aligned = y_series.reindex(time_index_aligned, method='nearest').values
    
    valid_mask = ~(np.isnan(x_aligned) | np.isnan(y_aligned))
    x_aligned = x_aligned[valid_mask]
    y_aligned = y_aligned[valid_mask]
    time_index_aligned = time_index_aligned[valid_mask]
    
    return x_aligned, y_aligned, time_index_aligned

def compute_scales(scale_range: Tuple[float, float], n_scales: int, sampling_period: float = 1.0) -> np.ndarray:
    min_scale, max_scale = scale_range
    scales = np.logspace(np.log10(min_scale), np.log10(max_scale), n_scales)
    return scales

def smooth_wavelet_transform(W: np.ndarray, time_window: int = 10, scale_window: int = 3) -> np.ndarray:
    W_smooth = uniform_filter1d(W, size=time_window, axis=1, mode='nearest')
    W_smooth = uniform_filter1d(W_smooth, size=scale_window, axis=0, mode='nearest')
    return W_smooth

def compute_wtc(x: np.ndarray, y: np.ndarray, scales: np.ndarray, 
                wavelet: str = 'cmor', wavelet_params: Tuple[float, float] = (1.5, 1.0),
                sampling_period: float = 1.0,                 smooth_time_window: int = 10, 
                smooth_scale_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    if wavelet == 'cmor':
        Fb, Fc = wavelet_params
        w_str = f"cmor{Fb}-{Fc}"
    else:
        w_str = wavelet

    Wx, _ = pywt.cwt(x, scales, w_str, sampling_period)
    Wy, _ = pywt.cwt(y, scales, w_str, sampling_period)
    
    Wxy = Wx * np.conj(Wy)
    
    S_Wxy = smooth_wavelet_transform(Wxy, time_window=smooth_time_window, scale_window=smooth_scale_window)
    S_Wx2 = smooth_wavelet_transform(np.abs(Wx)**2, time_window=smooth_time_window, scale_window=smooth_scale_window)
    S_Wy2 = smooth_wavelet_transform(np.abs(Wy)**2, time_window=smooth_time_window, scale_window=smooth_scale_window)
    
    denominator = np.sqrt(S_Wx2 * S_Wy2)
    denominator = np.where(denominator == 0, np.finfo(float).eps, denominator)
    coherence = np.abs(S_Wxy) / denominator
    phase = np.arctan2(np.imag(S_Wxy), np.real(S_Wxy))
    
    return coherence, phase

def compute_wtc_significance(x: np.ndarray, y: np.ndarray, scales: np.ndarray,
                            wavelet: str = 'cmor', wavelet_params: Tuple[float, float] = (1.5, 1.0),
                            sampling_period: float = 1.0, n_surrogates: int = 1000,
                            alpha: float = 0.05, smooth_time_window: int = 10,
                            smooth_scale_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    coherence_actual, _ = compute_wtc(x, y, scales, wavelet, wavelet_params, 
                                     sampling_period, smooth_time_window, smooth_scale_window)
    
    coherence_surrogates = []
    np.random.seed(42)
    
    for _ in range(n_surrogates):
        y_shuffled = np.random.permutation(y)
        coherence_surr, _ = compute_wtc(x, y_shuffled, scales, wavelet, wavelet_params,
                                        sampling_period, smooth_time_window, smooth_scale_window)
        coherence_surrogates.append(coherence_surr)
    
    coherence_surrogates = np.array(coherence_surrogates)
    upper_bound = np.percentile(coherence_surrogates, (1-alpha/2)*100, axis=0)
    lower_bound = np.percentile(coherence_surrogates, (alpha/2)*100, axis=0)
    
    significance_mask = (coherence_actual > upper_bound) | (coherence_actual < lower_bound)
    
    return coherence_actual, significance_mask

def compute_bootstrap_phase_uncertainty(x: np.ndarray, y: np.ndarray, scales: np.ndarray,
                                      wavelet: str = 'cmor', wavelet_params: Tuple[float, float] = (1.5, 1.0),
                                      sampling_period: float = 1.0, n_repeats: int = 100,
                                      smooth_time_window: int = 10, smooth_scale_window: int = 3) -> np.ndarray:
    n_samples = len(x)
    phases_bootstrap = []
    
    np.random.seed(42)
    
    for _ in range(n_repeats):
        indices = np.random.randint(0, n_samples, n_samples)
        indices.sort()
        
        x_boot = x[indices]
        y_boot = y[indices]
        
        _, phase_boot = compute_wtc(x_boot, y_boot, scales, wavelet, wavelet_params,
                                   sampling_period, smooth_time_window, smooth_scale_window)
        phases_bootstrap.append(phase_boot)
    
    phases_bootstrap = np.array(phases_bootstrap)
    
    sin_sum = np.sum(np.sin(phases_bootstrap), axis=0)
    cos_sum = np.sum(np.cos(phases_bootstrap), axis=0)
    R = np.sqrt(sin_sum**2 + cos_sum**2) / n_repeats
    
    R = np.clip(R, 0, 0.999999)
    phase_std = np.sqrt(-2 * np.log(R))
    
    return phase_std

def extract_coherence_features(coherence: np.ndarray, significance_mask: Optional[np.ndarray] = None,
                               high_coherence_threshold: float = 0.5) -> Dict[str, float]:
    features = {}
    features['mean_coherence'] = np.nanmean(coherence)
    features['std_coherence'] = np.nanstd(coherence)
    
    high_coherence_mask = coherence > high_coherence_threshold
    features['high_coherence_ratio'] = np.nanmean(high_coherence_mask.astype(float))
    
    if significance_mask is not None:
        significant_high_coherence = high_coherence_mask & significance_mask
        features['significant_high_coherence_ratio'] = np.nanmean(significant_high_coherence.astype(float))
    
    return features

def extract_phase_features(phase: np.ndarray, significance_mask: Optional[np.ndarray] = None,
                          phase_uncertainty: Optional[np.ndarray] = None) -> Dict[str, float]:
    features = {}
    features['mean_phase'] = np.nanmean(phase)
    features['std_phase'] = np.nanstd(phase)
    
    positive_phase_mask = phase > 0
    features['positive_phase_ratio'] = np.nanmean(positive_phase_mask.astype(float))
    
    if significance_mask is not None:
        significant_phase = phase[significance_mask]
        if len(significant_phase) > 0:
            features['mean_significant_phase'] = np.nanmean(significant_phase)
            
    if phase_uncertainty is not None:
        features['mean_phase_uncertainty'] = np.nanmean(phase_uncertainty)
        if significance_mask is not None:
            features['mean_significant_phase_uncertainty'] = np.nanmean(phase_uncertainty[significance_mask])
            
    return features

def process_single_wtc_pair_wrapper(pair_info: Tuple):
    flu_name, env_name, flu_seasonal, env_seasonal, flu_time_index, env_time_index, config, cache_dir = pair_info
    pair_name = f"{flu_name}_vs_{env_name}"
    
    cache_path = None
    if cache_dir:
        cache_path = Path(cache_dir) / f"{pair_name}.pkl"
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                result_dict = pickle.load(f)
            
            light_result = {
                'coherence_features': result_dict.get('coherence_features', {}),
                'phase_features': result_dict.get('phase_features', {}),
                'from_cache': True
            }
            return (pair_name, light_result, None)

    x_aligned, y_aligned, time_index_aligned = align_time_series(
        flu_seasonal, env_seasonal, flu_time_index, env_time_index
    )
    
    if len(x_aligned) < config['scale_range'][0]:
        return (pair_name, None, f"insufficient_data_length")
    
    scales = compute_scales(config['scale_range'], config['n_scales'], config['sampling_period'])
    
    coherence, significance_mask = compute_wtc_significance(
        x_aligned, y_aligned, scales,
        config['wavelet'], config['wavelet_params'],
        config['sampling_period'], config['n_surrogates'],
        config['significance_alpha'], config['smooth_time_window'],
        config['smooth_scale_window']
    )
    
    _, phase = compute_wtc(x_aligned, y_aligned, scales,
                          config['wavelet'], config['wavelet_params'],
                          config['sampling_period'],                               config['smooth_time_window'],
                          config['smooth_scale_window'])
    
    phase_uncertainty = None
    if ANALYSIS_CONFIG.get('enable_bootstrap_phase', False):
        phase_uncertainty = compute_bootstrap_phase_uncertainty(
            x_aligned, y_aligned, scales,
            config['wavelet'], config['wavelet_params'],
            config['sampling_period'], config['bootstrap_repeats'],
            config['smooth_time_window'], config['smooth_scale_window']
        )
    
    coherence_features = extract_coherence_features(
        coherence, significance_mask, config['high_coherence_threshold']
    )
    phase_features = extract_phase_features(phase, significance_mask, phase_uncertainty)
    
    result_dict = {
        'coherence': coherence,
        'phase': phase,
        'significance_mask': significance_mask,
        'phase_uncertainty': phase_uncertainty,
        'scales': scales,
        'time_index': time_index_aligned,
        'coherence_features': coherence_features,
        'phase_features': phase_features
    }
    
    if cache_path:
        with open(cache_path, 'wb') as f:
            pickle.dump(result_dict, f)

    light_result = {
        'coherence_features': coherence_features,
        'phase_features': phase_features,
        'from_cache': False
    }
    
    return (pair_name, light_result, None)

def process_all_wtc_pairs(flu_seasonal_dict: Dict, flu_time_index: pd.DatetimeIndex,
                         env_seasonal_dict: Dict, env_time_index_dict: Dict,
                         config: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("Starting WTC Analysis for All Variable Pairs")
    print("=" * 80)
    
    max_flu_series = ANALYSIS_CONFIG.get('max_flu_series', None)
    max_env_series = ANALYSIS_CONFIG.get('max_env_series', None)

    flu_items = list(flu_seasonal_dict.items())
    env_items = list(env_seasonal_dict.items())
    
    if max_flu_series is not None and max_flu_series < len(flu_items):
        print(f"Sampling {max_flu_series} flu series from {len(flu_items)} total.")
        flu_items = flu_items[:max_flu_series]
        
    if max_env_series is not None and max_env_series < len(env_items):
        print(f"Sampling {max_env_series} environmental series from {len(env_items)} total.")
        env_items = env_items[:max_env_series]

    cache_dir = Path(OUTPUT_CONFIG['base_dir']) / 'cache'
    cache_dir.mkdir(exist_ok=True)
    print(f"Using cache directory: {cache_dir}")

    pair_list = []
    for flu_name, flu_seasonal in flu_items:
        for env_name, env_seasonal in env_items:
            env_time_index = env_time_index_dict.get(env_name, flu_time_index)
            pair_list.append((flu_name, env_name, flu_seasonal, env_seasonal, 
                           flu_time_index, env_time_index, config, str(cache_dir)))
    
    wtc_results = {}
    
    pairs_to_process = []
    cached_results = {}
    
    print("Checking for cached results...")
    for pair_info in pair_list:
        flu_name, env_name, _, _, _, _, _, cache_dir_str = pair_info
        pair_name = f"{flu_name}_vs_{env_name}"
        cache_path = Path(cache_dir_str) / f"{pair_name}.pkl"
        
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                result_dict = pickle.load(f)
            
            light_result = {
                'coherence_features': result_dict.get('coherence_features', {}),
                'phase_features': result_dict.get('phase_features', {}),
                'from_cache': True
            }
            cached_results[pair_name] = light_result
        else:
            pairs_to_process.append(pair_info)
            
    print(f"Found {len(cached_results)} cached results. {len(pairs_to_process)} pairs to process.")
    wtc_results.update(cached_results)

    if not pairs_to_process:
        print("All pairs have been processed/cached.")
        return wtc_results

    enable_parallel = PARALLEL_CONFIG.get('enable_parallel', True)
    max_workers = PARALLEL_CONFIG.get('max_workers', None) or (os.cpu_count() or 4)
    
    if enable_parallel:
        print(f"Processing {len(pairs_to_process)} pairs with {max_workers} workers")
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {executor.submit(process_single_wtc_pair_wrapper, pair_info): pair_info for pair_info in pairs_to_process}
            
            for future in as_completed(future_to_pair):
                pair_name, result, error_msg = future.result()
                if result is not None:
                    wtc_results[pair_name] = result
                elif error_msg:
                    print(f"Failed {pair_name}: {error_msg}")
    else:
        print(f"Processing {len(pairs_to_process)} pairs sequentially")
        for pair_info in pairs_to_process:
            pair_name, result, error_msg = process_single_wtc_pair_wrapper(pair_info)
            if result is not None:
                wtc_results[pair_name] = result
            elif error_msg:
                print(f"Failed {pair_name}: {error_msg}")
                
    return wtc_results

def run_wtc(config: Optional[Dict] = None, output_dir: Optional[Union[str, Path]] = None):
    global WTC_CONFIG_SEASONAL, WTC_CONFIG_TREND
    global PARALLEL_CONFIG, VISUALIZATION_CONFIG, ANALYSIS_CONFIG, OUTPUT_CONFIG
    global STL_DATA_PATH
    
    if config:
        if 'wtc_config_seasonal' in config:
            WTC_CONFIG_SEASONAL.update(config['wtc_config_seasonal'])
        if 'wtc_config_trend' in config:
            WTC_CONFIG_TREND.update(config['wtc_config_trend'])
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
            
    if output_dir:
        OUTPUT_CONFIG['base_dir'] = str(output_dir)
        
    print("=" * 80)
    print("WTC Coupling Analysis (Robustness Version)")
    print("=" * 80)
    print(f"Output directory: {OUTPUT_CONFIG['base_dir']}")
    
    create_output_directory()
    setup_matplotlib()
    
    flu_seasonal_dict, flu_time_index, _, _ = load_flu_seasonal_components()
    env_seasonal_dict, env_time_index_dict, _ = load_environmental_seasonal_components()
    search_seasonal_dict, search_time_index_dict, _ = load_search_index_seasonal_components()
    
    all_env_seasonal_dict = {**env_seasonal_dict, **search_seasonal_dict}
    all_env_time_index_dict = {**env_time_index_dict, **search_time_index_dict}
        
    wtc_results = {}
    
    if ANALYSIS_CONFIG['enable_seasonal_wtc']:
        wtc_results = process_all_wtc_pairs(
            flu_seasonal_dict, flu_time_index,
            all_env_seasonal_dict, all_env_time_index_dict,
            WTC_CONFIG_SEASONAL
        )
        
        out_path = Path(OUTPUT_CONFIG['base_dir'])
        
        features_list = []
        for pair_name, res in wtc_results.items():
            feat = res['coherence_features'].copy()
            feat['pair_name'] = pair_name
            features_list.append(feat)
            
        if features_list:
            pd.DataFrame(features_list).to_csv(out_path / 'wtc_coherence_features.csv', index=False)
            
        metadata = {
            'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': {
                'seasonal': WTC_CONFIG_SEASONAL,
                'analysis': ANALYSIS_CONFIG
            }
        }
        with open(out_path / 'wtc_metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
            
    print("\nWTC Analysis Complete")
    return wtc_results

if __name__ == '__main__':
    run_wtc()
