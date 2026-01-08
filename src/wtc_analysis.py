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
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats
from scipy.ndimage import uniform_filter1d
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.multitest import multipletests
import pywt

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

WTC_CONFIG_SEASONAL = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (26, 52),
    'n_scales': 50,
    'sampling_period': 1.0,
    'smooth_time_window': 10,
    'smooth_scale_window': 3,
    'n_surrogates': 1000,
    'significance_alpha': 0.05,
    'high_coherence_threshold': 0.5
}

WTC_CONFIG_TREND = {
    'wavelet': 'cmor',
    'wavelet_params': (1.5, 1.0),
    'scale_range': (52, 500),
    'n_scales': 100,
    'sampling_period': 1.0,
    'smooth_time_window': 10,
    'smooth_scale_window': 3,
    'n_surrogates': 1000,
    'significance_alpha': 0.05,
    'high_coherence_threshold': 0.5
}

PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 10
}

VISUALIZATION_CONFIG = {
    'max_pairs_to_plot': None,
    'batch_size': 10,
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

ANALYSIS_CONFIG = {
    'enable_seasonal_wtc': True,
    'enable_trend_wtc': False,
    'enable_phase_analysis': True,
    'enable_comparison_analysis': True,
    'max_flu_series': None,
    'max_env_series': None,
}

OUTPUT_CONFIG = {
    'base_dir': 'AFD/wtc_analysis',
    'coherence_data_dir': 'coherence_data',
    'coherence_features_dir': 'coherence_features',
    'phase_features_dir': 'phase_features',
    'comparison_analysis_dir': 'comparison_analysis',
    'coherence_plots_dir': 'coherence_plots',
    'comparison_plots_dir': 'comparison_plots'
}

STL_DATA_PATH = {
    'stl_results_npy': 'AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': 'AFD/stl_decomposition/arrays/stl_results.pkl'
}

CO_CIRCULATION_PATH = 'co_circulation_periods.csv'

COLUMN_MAPPING = {
    'year': 'year',
    'week': 'week', 
    'start_date': 'start_date',
    'stop_date': 'stop_date',
    
    'ILI%北方': 'northern_ili_rate',
    'ILI%南方': 'southern_ili_rate',
    
    '流感阳性率北方': 'northern_flu_positive_rate',
    '流感阳性率南方': 'southern_flu_positive_rate',
    
    '病毒监测和分型（北方）检测数': 'northern_detection_count',
    '病毒监测和分型（北方）阳性数': 'northern_positive_count',
    '病毒监测和分型（北方）A型': 'northern_type_a',
    '病毒监测和分型（北方）A(H3N2)': 'northern_h3n2',
    '病毒监测和分型（北方）甲型 H1N1': 'northern_h1n1',
    '病毒监测和分型（北方）A(H7N9)': 'northern_h7n9',
    '病毒监测和分型（北方）A未分系': 'northern_type_a_untyped',
    '病毒监测和分型（北方）B型': 'northern_type_b',
    '病毒监测和分型（北方）B未分系': 'northern_type_b_untyped',
    '病毒监测和分型（北方）Victoria': 'northern_victoria',
    '病毒监测和分型（北方）Yamagata': 'northern_yamagata',
    
    '病毒监测和分型（南方）检测数': 'southern_detection_count',
    '病毒监测和分型（南方）阳性数': 'southern_positive_count',
    '病毒监测和分型（南方）A型': 'southern_type_a',
    '病毒监测和分型（南方）A(H3N2)': 'southern_h3n2',
    '病毒监测和分型（南方）甲型 H1N1': 'southern_h1n1',
    '病毒监测和分型（南方）A(H7N9)': 'southern_h7n9',
    '病毒监测和分型（南方）A未分系': 'southern_type_a_untyped',
    '病毒监测和分型（南方）B型': 'southern_type_b',
    '病毒监测和分型（南方）B未分系': 'southern_type_b_untyped',
    '病毒监测和分型（南方）Victoria': 'southern_victoria',
    '病毒监测和分型（南方）Yamagata': 'southern_yamagata',
}

def get_english_name(chinese_name: str) -> str:
    if chinese_name in COLUMN_MAPPING:
        return COLUMN_MAPPING[chinese_name]
    
    english_name = chinese_name
    
    if '百度指数_' in english_name:
        keyword = english_name.replace('百度指数_', '')
        english_name = f'baidu_{keyword}'
    
    elif 'ERA5-Land气象再分析数据（北方）' in english_name:
        param = english_name.split('）_')[-1] if '）_' in english_name else english_name.split('）')[-1]
        english_name = f'met_north_{param}'
    elif 'ERA5-Land气象再分析数据（南方）' in english_name:
        param = english_name.split('）_')[-1] if '）_' in english_name else english_name.split('）')[-1]
        english_name = f'met_south_{param}'
    
    elif '北方气象_' in english_name:
        param = english_name.split('北方气象_')[-1]
        english_name = f'met_north_surface_{param}'
    elif '南方气象_' in english_name:
        param = english_name.split('南方气象_')[-1]
        english_name = f'met_south_surface_{param}'
    
    elif '北方空气质量_' in english_name:
        param = english_name.split('北方空气质量_')[-1]
        english_name = f'met_north_air_{param}'
    elif '南方空气质量_' in english_name:
        param = english_name.split('南方空气质量_')[-1]
        english_name = f'met_south_air_{param}'
    
    elif '空气质量（国泰安）北方' in english_name:
        english_name = 'air_quality_north'
    elif '空气质量（国泰安）南方' in english_name:
        english_name = 'air_quality_south'
    
    elif '_舆情' in english_name:
        keyword = english_name.replace('_舆情', '')
        english_name = f'sentiment_{keyword}'
    
    elif '相对湿度%' in english_name:
        english_name = 'relative_humidity_pct'
    elif '6小时雨量' in english_name:
        if '.1' in english_name:
            english_name = 'precipitation_6h_alt'
        else:
            english_name = 'precipitation_6h'
    
    if english_name == chinese_name:
        english_name = re.sub(r'[^\w\s-]', '_', english_name)
        english_name = re.sub(r'[\s_]+', '_', english_name)
        english_name = english_name.lower()
        english_name = english_name.strip('_')
    
    return english_name

def convert_pair_name_to_english(pair_name: str) -> str:
    if '_vs_' not in pair_name:
        return get_english_name(pair_name)
    
    parts = pair_name.split('_vs_', 1)
    if len(parts) == 2:
        flu_name = parts[0]
        env_name = parts[1]
        flu_english = get_english_name(flu_name)
        env_english = get_english_name(env_name)
        return f'{flu_english}_vs_{env_english}'
    else:
        return get_english_name(pair_name)

def create_output_directory():
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for dir_name in OUTPUT_CONFIG.values():
        if dir_name != base_dir.name and dir_name.endswith('_dir'):
            dir_path = base_dir / dir_name
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return base_dir

def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['text.usetex'] = False
    plt.rcParams['mathtext.default'] = 'regular'

def load_flu_seasonal_components():
    print("=" * 80)
    print("Loading Flu Seasonal Components from STL Decomposition")
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
        
        seasonal_dict = {}
        
        for i, series_name in enumerate(series_names):
            seasonal_dict[series_name] = stl_3d_array[i, :, 1]
        
        print(f"Successfully loaded {len(series_names)} flu series seasonal components")
        print(f"Data shape: {stl_3d_array.shape}")
        print(f"Time range: {time_index[0]} to {time_index[-1]}")
        
        return seasonal_dict, time_index, series_names, metadata
        
    except Exception as e:
        print(f"Error loading flu seasonal components: {e}")
        raise

def load_environmental_seasonal_components():
    print("=" * 80)
    print("Loading Environmental Seasonal Components from STL Decomposition")
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
        
        env_keywords = ['ERA5', '气象', 't2m', 'd2m', 'temperature', 'humidity', 'temp', 'humid']
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
        print(f"Total environmental series found: {len(env_series_names)}")
        print(f"Time range: {time_index[0]} to {time_index[-1]}")
        
        return seasonal_dict, time_index_dict, list(seasonal_dict.keys())
        
    except Exception as e:
        print(f"Error loading environmental seasonal components: {e}")
        raise

def load_search_index_seasonal_components():
    print("=" * 80)
    print("Loading Search Index Seasonal Components from STL Decomposition")
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
        
        search_keywords = ['百度指数', 'baidu', '搜索', 'search']
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
        print(f"Total search index series found: {len(search_series_names)}")
        print(f"Time range: {time_index[0]} to {time_index[-1]}")
        
        return seasonal_dict, time_index_dict, list(seasonal_dict.keys())
        
    except Exception as e:
        print(f"Error loading search index seasonal components: {e}")
        raise

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
        
        if not time_index_x.is_monotonic_increasing:
            raise ValueError(f"time_index_x is not monotonic after fixing: duplicates={time_index_x.duplicated().sum()}, "
                           f"is_sorted={time_index_x.is_monotonic_increasing}, "
                           f"length={len(time_index_x)}, "
                           f"first={time_index_x[0]}, last={time_index_x[-1]}")
    
    if not time_index_y.is_monotonic_increasing:
        df_y = pd.DataFrame({'value': y, 'time': time_index_y})
        
        if time_index_y.duplicated().any():
            df_y = df_y.groupby('time')['value'].mean().reset_index()
        
        df_y = df_y.sort_values('time')
        
        time_index_y = pd.to_datetime(df_y['time'].values)
        y = df_y['value'].values
        
        if not time_index_y.is_monotonic_increasing:
            raise ValueError(f"time_index_y is not monotonic after fixing: duplicates={time_index_y.duplicated().sum()}, "
                           f"is_sorted={time_index_y.is_monotonic_increasing}, "
                           f"length={len(time_index_y)}, "
                           f"first={time_index_y[0]}, last={time_index_y[-1]}")
    
    start_time = max(time_index_x[0], time_index_y[0])
    end_time = min(time_index_x[-1], time_index_y[-1])
    
    if start_time >= end_time:
        raise ValueError(f"No overlapping time range: x_range=[{time_index_x[0]}, {time_index_x[-1]}], "
                         f"y_range=[{time_index_y[0]}, {time_index_y[-1]}]")
    
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
                sampling_period: float = 1.0, smooth_time_window: int = 10, 
                smooth_scale_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    Wx, _ = pywt.cwt(x, scales, wavelet, sampling_period)
    Wy, _ = pywt.cwt(y, scales, wavelet, sampling_period)
    
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
    lower_bound = np.percentile(coherence_surrogates, (alpha/2)*100, axis=0)
    upper_bound = np.percentile(coherence_surrogates, (1-alpha/2)*100, axis=0)
    
    significance_mask = (coherence_actual > upper_bound) | (coherence_actual < lower_bound)
    
    return coherence_actual, significance_mask

def extract_coherence_features(coherence: np.ndarray, significance_mask: Optional[np.ndarray] = None,
                               high_coherence_threshold: float = 0.5) -> Dict[str, float]:
    features = {}
    
    features['mean_coherence'] = np.nanmean(coherence)
    features['max_coherence'] = np.nanmax(coherence)
    features['min_coherence'] = np.nanmin(coherence)
    features['std_coherence'] = np.nanstd(coherence)
    features['cv_coherence'] = features['std_coherence'] / (features['mean_coherence'] + np.finfo(float).eps)
    
    high_coherence_mask = coherence > high_coherence_threshold
    features['high_coherence_ratio'] = np.nanmean(high_coherence_mask.astype(float))
    
    if significance_mask is not None:
        significant_high_coherence = high_coherence_mask & significance_mask
        features['significant_high_coherence_ratio'] = np.nanmean(significant_high_coherence.astype(float))
        features['significant_coherence_ratio'] = np.nanmean(significance_mask.astype(float))
    
    return features

def extract_phase_features(phase: np.ndarray, significance_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    features = {}
    
    features['mean_phase'] = np.nanmean(phase)
    features['std_phase'] = np.nanstd(phase)
    
    positive_phase_mask = phase > 0
    negative_phase_mask = phase < 0
    
    features['positive_phase_ratio'] = np.nanmean(positive_phase_mask.astype(float))
    features['negative_phase_ratio'] = np.nanmean(negative_phase_mask.astype(float))
    
    if significance_mask is not None:
        significant_phase = phase[significance_mask]
        if len(significant_phase) > 0:
            features['mean_significant_phase'] = np.nanmean(significant_phase)
            features['std_significant_phase'] = np.nanstd(significant_phase)
            
            significant_positive = positive_phase_mask & significance_mask
            significant_negative = negative_phase_mask & significance_mask
            features['significant_positive_phase_ratio'] = np.nanmean(significant_positive.astype(float))
            features['significant_negative_phase_ratio'] = np.nanmean(significant_negative.astype(float))
    
    return features


def load_co_circulation_periods() -> pd.DataFrame:
    co_circulation_path = Path(CO_CIRCULATION_PATH)
    
    if not co_circulation_path.exists():
        print(f"Warning: Co-circulation periods file not found: {co_circulation_path}")
        return pd.DataFrame()
    
    try:
        co_circulation_df = pd.read_csv(co_circulation_path)
        return co_circulation_df
    except Exception as e:
        print(f"Error loading co-circulation periods: {e}")
        return pd.DataFrame()

def identify_periods(time_index: pd.DatetimeIndex, co_circulation_df: pd.DataFrame) -> np.ndarray:
    if co_circulation_df.empty:
        return np.zeros(len(time_index), dtype=bool)
    
    is_co_circulation = np.zeros(len(time_index), dtype=bool)
    
    if 'start_date' in co_circulation_df.columns and 'end_date' in co_circulation_df.columns:
        for _, row in co_circulation_df.iterrows():
            start_date = pd.to_datetime(row['start_date'])
            end_date = pd.to_datetime(row['end_date'])
            
            mask = (time_index >= start_date) & (time_index <= end_date)
            is_co_circulation[mask] = True
    
    return is_co_circulation

def perform_statistical_test(data1: np.ndarray, data2: np.ndarray, test_type: str = 'mean') -> Tuple[float, str, bool]:
    data1 = data1[~np.isnan(data1)]
    data2 = data2[~np.isnan(data2)]
    
    if len(data1) < 3 or len(data2) < 3:
        return np.nan, 'insufficient_data', False
    
    if test_type == 'variance':
        statistic, p_value = levene(data1, data2)
        return p_value, 'Levene', False
    else:
        _, p_norm1 = shapiro(data1) if len(data1) <= 5000 else (0, 0.05)
        _, p_norm2 = shapiro(data2) if len(data2) <= 5000 else (0, 0.05)
        
        is_normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)
        
        if is_normal:
            statistic, p_value = ttest_ind(data1, data2)
            return p_value, 't-test', True
        else:
            statistic, p_value = mannwhitneyu(data1, data2, alternative='two-sided')
            return p_value, 'Mann-Whitney U', False

def compare_wtc_features(wtc_results: Dict, is_co_circulation: np.ndarray, 
                        time_index: pd.DatetimeIndex) -> pd.DataFrame:
    comparison_results = []
    
    for pair_name, result in wtc_results.items():
        coherence = result['coherence']
        phase = result['phase']
        significance_mask = result.get('significance_mask', None)
        
        co_mask = is_co_circulation[:coherence.shape[1]]
        single_mask = ~co_mask
        
        if np.sum(co_mask) < 3 or np.sum(single_mask) < 3:
            continue
        
        co_coherence = coherence[:, co_mask]
        single_coherence = coherence[:, single_mask]
        co_phase = phase[:, co_mask]
        single_phase = phase[:, single_mask]
        
        co_features = extract_coherence_features(co_coherence, 
                                                significance_mask[:, co_mask] if significance_mask is not None else None)
        single_features = extract_coherence_features(single_coherence,
                                                    significance_mask[:, single_mask] if significance_mask is not None else None)
        
        co_phase_features = extract_phase_features(co_phase,
                                                    significance_mask[:, co_mask] if significance_mask is not None else None)
        single_phase_features = extract_phase_features(single_phase,
                                                       significance_mask[:, single_mask] if significance_mask is not None else None)
        
        for feature_name in ['mean_coherence', 'max_coherence', 'std_coherence', 'cv_coherence', 
                            'high_coherence_ratio', 'mean_phase', 'std_phase']:
            co_value = co_features.get(feature_name, np.nan)
            single_value = single_features.get(feature_name, np.nan)
            
            if np.isnan(co_value) or np.isnan(single_value):
                continue
            
            if 'coherence' in feature_name:
                co_data = co_coherence.flatten()
                single_data = single_coherence.flatten()
            else:
                co_data = co_phase.flatten()
                single_data = single_phase.flatten()
            
            test_type = 'variance' if 'cv' in feature_name or 'std' in feature_name else 'mean'
            p_value, test_method, is_normal = perform_statistical_test(co_data, single_data, test_type=test_type)
            
            mean_diff = co_value - single_value
            pooled_std = np.sqrt((np.var(co_data) + np.var(single_data)) / 2)
            cohens_d = mean_diff / (pooled_std + np.finfo(float).eps)
            
            comparison_results.append({
                'Variable_Pair': pair_name,
                'Feature': feature_name,
                'Co_Circulation_Mean': co_value,
                'Single_Dominant_Mean': single_value,
                'Difference': mean_diff,
                'P_Value': p_value,
                'Test_Method': test_method,
                'Cohen_D': cohens_d
            })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if 'P_Value' in comparison_df.columns:
        valid_mask = ~comparison_df['P_Value'].isna()
        p_values = comparison_df.loc[valid_mask, 'P_Value'].values
        if len(p_values) > 0:
            rejected, p_corrected, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
            comparison_df['P_Value_Corrected'] = np.nan
            comparison_df.loc[valid_mask, 'P_Value_Corrected'] = p_corrected
            comparison_df['Significant_Uncorrected'] = False
            comparison_df.loc[valid_mask, 'Significant_Uncorrected'] = comparison_df.loc[valid_mask, 'P_Value'] < 0.05
            comparison_df['Significant_FDR'] = False
            comparison_df.loc[valid_mask, 'Significant_FDR'] = rejected
    
    return comparison_df


def plot_wtc_coherence(coherence: np.ndarray, phase: np.ndarray, significance_mask: Optional[np.ndarray],
                       time_index: pd.DatetimeIndex, scales: np.ndarray, 
                       pair_name: str, output_path: Path, config: Dict):
    plt.rcParams['text.usetex'] = False
    plt.rcParams['text.latex.preamble'] = ''
    
    fig, axes = plt.subplots(2, 1, figsize=VISUALIZATION_CONFIG['figsize'])
    
    if not isinstance(time_index, pd.DatetimeIndex):
        time_index = pd.to_datetime(time_index)
    
    periods = scales * config['sampling_period']
    
    if coherence.shape[1] != len(time_index):
        min_len = min(coherence.shape[1], len(time_index))
        coherence = coherence[:, :min_len]
        phase = phase[:, :min_len]
        time_index = time_index[:min_len]
        if significance_mask is not None:
            significance_mask = significance_mask[:, :min_len]
    
    english_pair_name = convert_pair_name_to_english(pair_name)
    safe_title = english_pair_name.replace('%', r'\%').replace('_', r'\_').replace('$', r'\$')
    
    from matplotlib.ticker import ScalarFormatter, LogFormatter, FuncFormatter
    
    def log_formatter_func(x, pos):
        try:
            if x <= 0 or np.isnan(x) or np.isinf(x):
                return ''
            if x >= 100:
                return f'{x:.0f}'
            elif x >= 10:
                return f'{x:.1f}'
            elif x >= 1:
                return f'{x:.2f}'
            else:
                exp = int(np.floor(np.log10(x)))
                coeff = x / (10 ** exp)
                if abs(coeff) >= 10:
                    coeff = coeff / 10
                    exp = exp + 1
                return f'{coeff:.1f}e{exp}'
        except Exception:
            return ''
    
    log_formatter = FuncFormatter(log_formatter_func)
    
    ax1 = axes[0]
    im1 = ax1.contourf(time_index, periods, coherence, levels=20, cmap='jet')
    if significance_mask is not None:
        ax1.contour(time_index, periods, significance_mask.astype(float), levels=[0.5], 
                   colors='black', linewidths=1.5, linestyles='--')
    ax1.set_ylabel('Period (weeks)', fontsize=12)
    ax1.set_title(f'WTC Coherence: {safe_title}', fontsize=14, fontweight='bold', usetex=False)
    ax1.set_yscale('log')
    ax1.yaxis.set_major_formatter(log_formatter)
    ax1.yaxis.set_minor_formatter(log_formatter)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Coherence')
    
    ax2 = axes[1]
    im2 = ax2.contourf(time_index, periods, phase, levels=20, cmap='hsv')
    if significance_mask is not None:
        ax2.contour(time_index, periods, significance_mask.astype(float), levels=[0.5],
                   colors='black', linewidths=1.5, linestyles='--')
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_ylabel('Period (weeks)', fontsize=12)
    ax2.set_title(f'Phase Difference: {safe_title}', fontsize=14, fontweight='bold', usetex=False)
    ax2.set_yscale('log')
    ax2.yaxis.set_major_formatter(log_formatter)
    ax2.yaxis.set_minor_formatter(log_formatter)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Phase (radians)')
    
    plt.tight_layout()
    
    for fmt in VISUALIZATION_CONFIG['format']:
        output_file = output_path.with_suffix(f'.{fmt}')
        plt.savefig(output_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    
    plt.close()
    gc.collect()

def process_single_wtc_pair_wrapper(pair_info: Tuple):
    flu_name, env_name, flu_seasonal, env_seasonal, flu_time_index, env_time_index, config = pair_info
    pair_name = f"{flu_name}_vs_{env_name}"
    
    try:
        x_aligned, y_aligned, time_index_aligned = align_time_series(
            flu_seasonal, env_seasonal, flu_time_index, env_time_index
        )
        
        if len(x_aligned) < config['scale_range'][0]:
            return (pair_name, None, f"insufficient_data_length: {len(x_aligned)} < {config['scale_range'][0]}")
        
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
                              config['sampling_period'], config['smooth_time_window'],
                              config['smooth_scale_window'])
        
        coherence_features = extract_coherence_features(
            coherence, significance_mask, config['high_coherence_threshold']
        )
        phase_features = extract_phase_features(phase, significance_mask)
        
        result_dict = {
            'coherence': coherence,
            'phase': phase,
            'significance_mask': significance_mask,
            'scales': scales,
            'time_index': time_index_aligned,
            'coherence_features': coherence_features,
            'phase_features': phase_features
        }
        
        return (pair_name, result_dict, None)
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        return (pair_name, None, error_msg)

def process_all_wtc_pairs(flu_seasonal_dict: Dict, flu_time_index: pd.DatetimeIndex,
                         env_seasonal_dict: Dict, env_time_index_dict: Dict,
                         config: Dict) -> Dict:
    print("\n" + "=" * 80)
    print("Starting WTC Analysis for All Variable Pairs")
    print("=" * 80)
    
    pair_list = []
    for flu_name, flu_seasonal in flu_seasonal_dict.items():
        for env_name, env_seasonal in env_seasonal_dict.items():
            env_time_index = env_time_index_dict.get(env_name, flu_time_index)
            pair_list.append((flu_name, env_name, flu_seasonal, env_seasonal, 
                           flu_time_index, env_time_index, config))
    
    total_pairs = len(pair_list)
    print(f"Total variable pairs to process: {total_pairs}")
    
    wtc_results = {}
    failed_pairs = []
    
    max_workers = PARALLEL_CONFIG.get('max_workers', None)
    if max_workers is None:
        max_workers = os.cpu_count() or 4
    
    print(f"Using {max_workers} workers for parallel processing")
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {executor.submit(process_single_wtc_pair_wrapper, pair_info): pair_info for pair_info in pair_list}
        
        completed = 0
        error_summary = {}
        error_examples = {}
        
        for future in as_completed(future_to_pair):
            completed += 1
            pair_name, result, error_msg = future.result()
            
            if result is not None:
                wtc_results[pair_name] = result
            else:
                failed_pairs.append(pair_name)
                if error_msg:
                    error_type = error_msg.split(':')[0] if ':' in error_msg else error_msg
                    error_summary[error_type] = error_summary.get(error_type, 0) + 1
                    if error_type not in error_examples or len(error_examples[error_type]) < 3:
                        if error_type not in error_examples:
                            error_examples[error_type] = []
                        error_examples[error_type].append((pair_name, error_msg))
            
            if completed % 100 == 0:
                print(f"Progress: {completed}/{total_pairs} pairs processed ({completed/total_pairs*100:.1f}%)")
                if len(wtc_results) > 0:
                    print(f"  Current success rate: {len(wtc_results)/completed*100:.1f}%")
    
    print(f"\nWTC analysis completed:")
    print(f"  Successful: {len(wtc_results)}")
    print(f"  Failed: {len(failed_pairs)}")
    
    if failed_pairs:
        print(f"\n  Failed pairs (first 10): {failed_pairs[:10]}")
        
        if error_summary:
            print(f"\n  Error Summary (Total: {len(error_summary)} error types):")
            print("=" * 80)
            for error_type, count in sorted(error_summary.items(), key=lambda x: x[1], reverse=True):
                percentage = count / len(failed_pairs) * 100
                print(f"\n  {error_type}: {count} pairs ({percentage:.1f}%)")
                
                if error_type in error_examples:
                    print(f"    Examples:")
                    for example_pair, example_msg in error_examples[error_type]:
                        print(f"      - {example_pair}")
                        print(f"        {example_msg}")
            print("=" * 80)
    
    return wtc_results


def save_wtc_results(wtc_results: Dict, output_dir: Path, config: Dict):
    print("\n" + "=" * 80)
    print("Saving WTC Analysis Results")
    print("=" * 80)
    
    coherence_data_dir = output_dir / OUTPUT_CONFIG['coherence_data_dir']
    coherence_data_dir.mkdir(parents=True, exist_ok=True)
    
    saved_count = 0
    for pair_name, result in wtc_results.items():
        try:
            coherence_file = coherence_data_dir / f"coherence_{pair_name}.npy"
            np.save(coherence_file, result['coherence'])
            
            phase_file = coherence_data_dir / f"phase_{pair_name}.npy"
            np.save(phase_file, result['phase'])
            
            if result['significance_mask'] is not None:
                significance_file = coherence_data_dir / f"significance_{pair_name}.npy"
                np.save(significance_file, result['significance_mask'])
            
            saved_count += 1
        except Exception as e:
            print(f"Error saving {pair_name}: {e}")
            continue
    
    print(f"Saved {saved_count} WTC time-frequency data files")
    
    coherence_features_dir = output_dir / OUTPUT_CONFIG['coherence_features_dir']
    coherence_features_dir.mkdir(parents=True, exist_ok=True)
    
    phase_features_dir = output_dir / OUTPUT_CONFIG['phase_features_dir']
    phase_features_dir.mkdir(parents=True, exist_ok=True)
    
    coherence_features_list = []
    phase_features_list = []
    
    for pair_name, result in wtc_results.items():
        coherence_features = result['coherence_features'].copy()
        coherence_features['Variable_Pair'] = pair_name
        coherence_features_list.append(coherence_features)
        
        phase_features = result['phase_features'].copy()
        phase_features['Variable_Pair'] = pair_name
        phase_features_list.append(phase_features)
    
    if coherence_features_list:
        coherence_features_df = pd.DataFrame(coherence_features_list)
        coherence_features_file = coherence_features_dir / 'coherence_features.csv'
        coherence_features_df.to_csv(coherence_features_file, index=False)
        print(f"Saved coherence features: {coherence_features_file}")
    
    if phase_features_list:
        phase_features_df = pd.DataFrame(phase_features_list)
        phase_features_file = phase_features_dir / 'phase_features.csv'
        phase_features_df.to_csv(phase_features_file, index=False)
        print(f"Saved phase features: {phase_features_file}")

def save_comparison_results(comparison_df: pd.DataFrame, output_dir: Path):
    comparison_dir = output_dir / OUTPUT_CONFIG['comparison_analysis_dir']
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_file = comparison_dir / 'coherence_comparison.csv'
    comparison_df.to_csv(comparison_file, index=False)
    print(f"Saved comparison results: {comparison_file}")
    
    if 'Significant_FDR' in comparison_df.columns:
        significant_df = comparison_df[comparison_df['Significant_FDR'] == True]
        significant_file = comparison_dir / 'statistical_tests.csv'
        significant_df.to_csv(significant_file, index=False)
        print(f"Saved significant results: {significant_file} ({len(significant_df)} rows)")
    
    if 'Cohen_D' in comparison_df.columns:
        effect_sizes_df = comparison_df[['Variable_Pair', 'Feature', 'Cohen_D', 
                                        'P_Value', 'P_Value_Corrected', 'Significant_FDR']].copy()
        effect_sizes_file = comparison_dir / 'effect_sizes.csv'
        effect_sizes_df.to_csv(effect_sizes_file, index=False)
        print(f"Saved effect sizes: {effect_sizes_file}")

def generate_wtc_visualizations(wtc_results: Dict, output_dir: Path, config: Dict):
    if not VISUALIZATION_CONFIG['enable_visualization']:
        return
    
    print("\n" + "=" * 80)
    print("Generating WTC Visualizations")
    print("=" * 80)
    
    coherence_plots_dir = output_dir / OUTPUT_CONFIG['coherence_plots_dir']
    coherence_plots_dir.mkdir(parents=True, exist_ok=True)
    
    plot_tasks = []
    for pair_name, result in wtc_results.items():
        safe_name = pair_name.replace('/', '_').replace('\\', '_').replace(':', '_').replace('(', '_').replace(')', '_').replace(' ', '_')
        output_path = coherence_plots_dir / f"wtc_{safe_name}"
        plot_tasks.append({
            'coherence': result['coherence'],
            'phase': result['phase'],
            'significance_mask': result.get('significance_mask'),
            'time_index': result['time_index'],
            'scales': result['scales'],
            'pair_name': pair_name,
            'output_path': output_path,
            'config': config
        })
    
    max_pairs = VISUALIZATION_CONFIG.get('max_pairs_to_plot')
    if max_pairs is not None:
        plot_tasks = plot_tasks[:max_pairs]
    
    if len(plot_tasks) == 0:
        print("No plots to generate (no WTC results available)")
        return
    
    print(f"Generating {len(plot_tasks)} WTC plots...")
    
    max_workers = min(VISUALIZATION_CONFIG.get('batch_size', 10), len(plot_tasks))
    if max_workers <= 0:
        max_workers = 1
    
    def plot_wrapper(plot_info):
        try:
            plot_wtc_coherence(
                plot_info['coherence'],
                plot_info['phase'],
                plot_info['significance_mask'],
                plot_info['time_index'],
                plot_info['scales'],
                plot_info['pair_name'],
                plot_info['output_path'],
                plot_info['config']
            )
            return True
        except Exception as e:
            print(f"Error plotting {plot_info['pair_name']}: {e}")
            return False
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_plot = {executor.submit(plot_wrapper, plot_info): plot_info for plot_info in plot_tasks}
        
        completed = 0
        successful = 0
        for future in as_completed(future_to_plot):
            completed += 1
            if future.result():
                successful += 1
            
            if completed % 50 == 0:
                print(f"Progress: {completed}/{len(plot_tasks)} plots generated ({completed/len(plot_tasks)*100:.1f}%)")
    
    print(f"Generated {successful} WTC visualization plots")


def main():
    print("=" * 80)
    print("WTC Coupling Analysis Module")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Seasonal WTC: {ANALYSIS_CONFIG['enable_seasonal_wtc']}")
    print(f"  Trend WTC: {ANALYSIS_CONFIG['enable_trend_wtc']}")
    print(f"  Phase Analysis: {ANALYSIS_CONFIG['enable_phase_analysis']}")
    print(f"  Comparison Analysis: {ANALYSIS_CONFIG['enable_comparison_analysis']}")
    print(f"  Parallel Processing: {PARALLEL_CONFIG['enable_parallel']}")
    print("=" * 80)
    
    setup_matplotlib()
    output_dir = create_output_directory()
    
    print("\n" + "=" * 80)
    print("Step 1: Loading Data")
    print("=" * 80)
    
    flu_seasonal_dict, flu_time_index, flu_series_names, flu_metadata = load_flu_seasonal_components()
    
    env_seasonal_dict, env_time_index_dict, env_series_names = load_environmental_seasonal_components()
    
    search_seasonal_dict, search_time_index_dict, search_series_names = load_search_index_seasonal_components()
    
    all_env_seasonal_dict = {**env_seasonal_dict, **search_seasonal_dict}
    all_env_time_index_dict = {**env_time_index_dict, **search_time_index_dict}
    all_env_series_names = env_series_names + search_series_names
    
    max_flu = ANALYSIS_CONFIG.get('max_flu_series', None)
    max_env = ANALYSIS_CONFIG.get('max_env_series', None)
    
    if max_flu is not None:
        flu_series_names = flu_series_names[:max_flu]
        flu_seasonal_dict = {k: flu_seasonal_dict[k] for k in flu_series_names if k in flu_seasonal_dict}
        print(f"\n⚠️  Limited to first {max_flu} flu series for testing")
    
    if max_env is not None:
        all_env_series_names = all_env_series_names[:max_env]
        all_env_seasonal_dict = {k: all_env_seasonal_dict[k] for k in all_env_series_names if k in all_env_seasonal_dict}
        all_env_time_index_dict = {k: all_env_time_index_dict[k] for k in all_env_series_names if k in all_env_time_index_dict}
        env_series_names = [name for name in env_series_names if name in all_env_series_names]
        search_series_names = [name for name in search_series_names if name in all_env_series_names]
        print(f"⚠️  Limited to first {max_env} environmental series for testing")
    
    print(f"\nData Summary:")
    print(f"  Flu series: {len(flu_series_names)}")
    print(f"  Environmental series: {len(all_env_series_names)}")
    print(f"    - Meteorological: {len(env_series_names)}")
    print(f"    - Search index: {len(search_series_names)}")
    print(f"  Total variable pairs: {len(flu_series_names)} × {len(all_env_series_names)} = {len(flu_series_names) * len(all_env_series_names)}")
    
    if ANALYSIS_CONFIG['enable_seasonal_wtc']:
        print("\n" + "=" * 80)
        print("Step 2: WTC Analysis (Seasonal Components)")
        print("=" * 80)
        
        wtc_results = process_all_wtc_pairs(
            flu_seasonal_dict, flu_time_index,
            all_env_seasonal_dict, all_env_time_index_dict,
            WTC_CONFIG_SEASONAL
        )
        
        print("\n" + "=" * 80)
        print("Step 3: Saving Results")
        print("=" * 80)
        save_wtc_results(wtc_results, output_dir, WTC_CONFIG_SEASONAL)
        
        print("\n" + "=" * 80)
        print("Step 4: Generating Visualizations")
        print("=" * 80)
        generate_wtc_visualizations(wtc_results, output_dir, WTC_CONFIG_SEASONAL)
        
        if ANALYSIS_CONFIG['enable_comparison_analysis']:
            print("\n" + "=" * 80)
            print("Step 5: Comparison Analysis")
            print("=" * 80)
            
            co_circulation_df = load_co_circulation_periods()
            
            is_co_circulation = identify_periods(flu_time_index, co_circulation_df)
            
            comparison_df = compare_wtc_features(wtc_results, is_co_circulation, flu_time_index)
            
            save_comparison_results(comparison_df, output_dir)
    
    print("\n" + "=" * 80)
    print("WTC Analysis Completed Successfully!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print("=" * 80)

if __name__ == "__main__":
    main()

