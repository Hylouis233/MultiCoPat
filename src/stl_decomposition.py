import gc
import os
import re
import sys
import time
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

STL_CONFIG = {
    'seasonal_period': 52,
    'seasonal': 13,
    'trend': None,
    'robust': True,
    'low_pass': None,
    'seasonal_deg': 1,
    'trend_deg': 1,
    'low_pass_deg': 1,
    'seasonal_jump': 1,
    'trend_jump': 1,
    'low_pass_jump': 1
}

PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 10
}

VISUALIZATION_CONFIG = {
    'max_series_to_plot': 50,
    'batch_size': 10,
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

OUTPUT_CONFIG = {
    'base_dir': 'AFD/stl_decomposition',
    'csv_dir': 'csv',
    'arrays_dir': 'arrays',
    'viz_dir': 'visualization'
}

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

def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figsize']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2

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

def create_output_directory():
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    csv_dir = base_dir / OUTPUT_CONFIG['csv_dir']
    arrays_dir = base_dir / OUTPUT_CONFIG['arrays_dir']
    viz_dir = base_dir / OUTPUT_CONFIG['viz_dir']
    
    dirs_to_create = [
        csv_dir / 'stl_trend',
        csv_dir / 'stl_seasonal',
        csv_dir / 'stl_resid',
        arrays_dir,
        viz_dir / 'individual',
        viz_dir / 'summary'
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory structure created: {base_dir}")
    return base_dir

def load_flu_data():
    print("=" * 80)
    print("Loading data using UnifiedDataPortal...")
    print("=" * 80)
    try:
        portal = UnifiedDataPortal()
        flu_data = portal.load_flu_data()
        
        if flu_data is not None:
            print(f"Successfully loaded flu data: {len(flu_data)} rows, {len(flu_data.columns)} columns")
            print(f"Time range: {flu_data['start_date'].min()} to {flu_data['stop_date'].max()}")
            return flu_data
        else:
            print("Data loading failed: returned None")
            return None
            
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None

def prepare_stl_data(flu_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    exclude_columns = [
        'year', 'week', 'start_date', 'stop_date', 
        'year_week', 'index', 'Unnamed: 0'
    ]
    
    all_columns = flu_data.columns.tolist()
    data_columns = [col for col in all_columns if col not in exclude_columns]
    
    print(f"\nPreparing STL analysis, processing {len(data_columns)} data columns")
    print("-" * 80)
    
    stl_series = {}
    skipped_series = []
    
    for col in data_columns:
        try:
            raw_data = flu_data[col]
            
            if not pd.api.types.is_numeric_dtype(raw_data):
                skipped_series.append(f"{col} (non-numeric)")
                continue
            
            processed_data = raw_data.ffill().bfill().fillna(0)
            
            series_data = processed_data.values.astype(float)
            
            if len(series_data) < STL_CONFIG['seasonal_period']:
                skipped_series.append(f"{col} (insufficient data length: {len(series_data)} < {STL_CONFIG['seasonal_period']})")
                continue
                
            if np.all(series_data == 0):
                skipped_series.append(f"{col} (all-zero sequence)")
                continue
            
            if np.any(np.isinf(series_data)) or np.any(np.isnan(series_data)):
                skipped_series.append(f"{col} (contains invalid values)")
                continue
            
            stl_series[col] = series_data
            
        except Exception as e:
            skipped_series.append(f"{col} (processing error: {str(e)})")
            continue
    
    print(f"\nSuccessfully prepared {len(stl_series)} time series for STL analysis")
    
    categories = {
        'ILI-related indicators': [k for k in stl_series.keys() if 'ILI%' in k],
        'Flu positive rate': [k for k in stl_series.keys() if '流感阳性率' in k or 'positive' in k.lower()],
        'Virus typing-Northern': [k for k in stl_series.keys() if '病毒监测和分型（北方）' in k],
        'Virus typing-Southern': [k for k in stl_series.keys() if '病毒监测和分型（南方）' in k],
        'Other': []
    }
    
    for k in stl_series.keys():
        if not any(k in cat_list for cat_list in categories.values() if cat_list is not categories['Other']):
            categories['Other'].append(k)
    
    for category, series_list in categories.items():
        if series_list:
            print(f"\n  {category} ({len(series_list)} series):")
            for series in series_list[:5]:
                print(f"    - {series}: length {len(stl_series[series])}")
            if len(series_list) > 5:
                print(f"    ... and {len(series_list) - 5} more series")
    
    if skipped_series:
        print(f"\nSkipped series ({len(skipped_series)} series):")
        for skip_info in skipped_series[:20]:
            print(f"  - {skip_info}")
        if len(skipped_series) > 20:
            print(f"  ... and {len(skipped_series) - 20} more skipped series")
    
    return stl_series

def process_single_stl_series(series_info: Tuple[str, np.ndarray]) -> Tuple[str, Optional[Dict]]:
    series_name, series_data = series_info
    
    try:
        series_pd = pd.Series(series_data)
        
        stl_result = STL(
            series_pd,
            seasonal=STL_CONFIG['seasonal'],
            trend=STL_CONFIG['trend'],
            robust=STL_CONFIG['robust'],
            period=STL_CONFIG['seasonal_period'],
            low_pass=STL_CONFIG['low_pass'],
            seasonal_deg=STL_CONFIG['seasonal_deg'],
            trend_deg=STL_CONFIG['trend_deg'],
            low_pass_deg=STL_CONFIG['low_pass_deg'],
            seasonal_jump=STL_CONFIG['seasonal_jump'],
            trend_jump=STL_CONFIG['trend_jump'],
            low_pass_jump=STL_CONFIG['low_pass_jump']
        ).fit()
        
        trend = stl_result.trend.values
        seasonal = stl_result.seasonal.values
        resid = stl_result.resid.values
        
        original_reconstructed = trend + seasonal + resid
        reconstruction_error = np.mean(np.abs(series_data - original_reconstructed))
        
        if reconstruction_error > 1e-6:
            print(f"Warning: {series_name} reconstruction error = {reconstruction_error:.2e}")
        
        result_dict = {
            'trend': trend,
            'seasonal': seasonal,
            'resid': resid,
            'original': series_data,
            'reconstruction_error': reconstruction_error
        }
        
        return (series_name, result_dict)
        
    except Exception as e:
        print(f"Error processing {series_name}: {str(e)}")
        return (series_name, None)

def batch_stl_decomposition(stl_series: Dict[str, np.ndarray], 
                            enable_parallel: bool = True) -> Dict[str, Dict]:
    print("\n" + "=" * 80)
    print("Starting STL decomposition...")
    print("=" * 80)
    
    if not stl_series:
        print("No time series to process")
        return {}
    
    series_items = list(stl_series.items())
    total_series = len(series_items)
    
    stl_results = {}
    failed_series = []
    
    if enable_parallel and PARALLEL_CONFIG['enable_parallel']:
        print(f"Using parallel processing with {PARALLEL_CONFIG['max_workers'] or 'auto'} workers")
        
        max_workers = PARALLEL_CONFIG['max_workers']
        if max_workers is None:
            max_workers = os.cpu_count()
        
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_stl_series, item): item[0] 
                      for item in series_items}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                series_name, result = future.result()
                
                if result is not None:
                    stl_results[series_name] = result
                else:
                    failed_series.append(series_name)
                
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (total_series - completed) * avg_time
                    print(f"  Progress: {completed}/{total_series} ({100*completed/total_series:.1f}%) "
                          f"- ETA: {remaining/60:.1f} minutes")
        
        elapsed_time = time.time() - start_time
        
    else:
        print("Using sequential processing")
        start_time = time.time()
        
        for i, (series_name, series_data) in enumerate(series_items):
            _, result = process_single_stl_series((series_name, series_data))
            
            if result is not None:
                stl_results[series_name] = result
            else:
                failed_series.append(series_name)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (total_series - (i + 1)) * avg_time
                print(f"  Progress: {i+1}/{total_series} ({100*(i+1)/total_series:.1f}%) "
                      f"- ETA: {remaining/60:.1f} minutes")
        
        elapsed_time = time.time() - start_time
    
    print(f"\nSTL decomposition completed in {elapsed_time/60:.2f} minutes")
    print(f"  Successfully processed: {len(stl_results)}/{total_series} series")
    print(f"  Failed: {len(failed_series)} series")
    
    if failed_series:
        print(f"\nFailed series:")
        for series_name in failed_series[:10]:
            print(f"  - {series_name}")
        if len(failed_series) > 10:
            print(f"  ... and {len(failed_series) - 10} more")
    
    return stl_results

# =============================================================================
# OUTPUT AND SAVING FUNCTIONS
# =============================================================================

def save_stl_results_csv(stl_results: Dict[str, Dict], 
                         output_dir: Path, 
                         time_index: pd.DatetimeIndex):
    print("\n" + "=" * 80)
    print("Saving STL results as CSV files...")
    print("=" * 80)
    
    csv_base = output_dir / OUTPUT_CONFIG['csv_dir']
    trend_dir = csv_base / 'stl_trend'
    seasonal_dir = csv_base / 'stl_seasonal'
    resid_dir = csv_base / 'stl_resid'
    
    saved_count = 0
    
    for series_name, result in stl_results.items():
        try:
            english_name = get_english_name(series_name)
            safe_name = english_name.replace('/', '_').replace('\\', '_').replace(':', '_')
            
            data_length = len(result['trend'])
            if len(time_index) >= data_length:
                time_axis = time_index[:data_length]
            else:
                time_axis = pd.date_range(start=time_index[0], periods=data_length, freq='W')
            
            df = pd.DataFrame({
                'time_index': time_axis,
                'Trend': result['trend'],
                'Seasonal': result['seasonal'],
                'Residual': result['resid'],
                'Original': result['original']
            })
            
            df[['time_index', 'Trend']].to_csv(
                trend_dir / f"{safe_name}_trend.csv", 
                index=False, encoding='utf-8-sig'
            )
            df[['time_index', 'Seasonal']].to_csv(
                seasonal_dir / f"{safe_name}_seasonal.csv", 
                index=False, encoding='utf-8-sig'
            )
            df[['time_index', 'Residual']].to_csv(
                resid_dir / f"{safe_name}_resid.csv", 
                index=False, encoding='utf-8-sig'
            )
            
            saved_count += 1
            
        except Exception as e:
            print(f"Error saving CSV for {series_name}: {str(e)}")
            continue
    
    print(f"Saved {saved_count} series as CSV files")

def save_stl_arrays(stl_results: Dict[str, Dict], 
                   output_dir: Path,
                   time_index: pd.DatetimeIndex,
                   series_names: List[str]):
    print("\n" + "=" * 80)
    print("Saving STL results as structured arrays...")
    print("=" * 80)
    
    arrays_dir = output_dir / OUTPUT_CONFIG['arrays_dir']
    
    min_length = min(len(result['trend']) for result in stl_results.values())
    n_series = len(stl_results)
    
    print(f"Creating 3D array: shape = ({n_series}, {min_length}, 3)")
    
    stl_3d_array = np.full((n_series, min_length, 3), np.nan)
    
    for i, series_name in enumerate(series_names):
        if series_name in stl_results:
            result = stl_results[series_name]
            stl_3d_array[i, :, 0] = result['trend'][:min_length]
            stl_3d_array[i, :, 1] = result['seasonal'][:min_length]
            stl_3d_array[i, :, 2] = result['resid'][:min_length]
    
    npy_path = arrays_dir / 'stl_results.npy'
    np.save(npy_path, stl_3d_array)
    print(f"Saved 3D array to: {npy_path}")
    
    metadata = {
        'time_index': time_index[:min_length],
        'series_names': series_names,
        'stl_config': STL_CONFIG,
        'data_shape': stl_3d_array.shape,
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_series': n_series,
        'n_timepoints': min_length,
        'reconstruction_errors': {name: stl_results[name]['reconstruction_error'] 
                                for name in series_names if name in stl_results}
    }
    
    pkl_path = arrays_dir / 'stl_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {pkl_path}")
    
    summary_data = {
        'Series_Name': series_names,
        'Length': [min_length] * n_series,
        'Trend_Mean': [np.nanmean(stl_3d_array[i, :, 0]) for i in range(n_series)],
        'Seasonal_Mean': [np.nanmean(stl_3d_array[i, :, 1]) for i in range(n_series)],
        'Residual_Std': [np.nanstd(stl_3d_array[i, :, 2]) for i in range(n_series)],
        'Reconstruction_Error': [metadata['reconstruction_errors'].get(name, np.nan) 
                                 for name in series_names]
    }
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = arrays_dir / 'stl_summary.csv'
    summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
    print(f"Saved summary to: {summary_path}")
    
    return stl_3d_array, metadata

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_stl_decomposition(series_name: str,
                           english_name: str,
                           result: Dict,
                           time_index: pd.DatetimeIndex,
                           output_path: Path):
    matplotlib.use('Agg')
    
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = VISUALIZATION_CONFIG['figsize']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 11
    plt.rcParams['figure.titlesize'] = 18
    plt.rcParams['figure.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.dpi'] = VISUALIZATION_CONFIG['dpi']
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.pad_inches'] = 0.2
    
    fig, axes = plt.subplots(4, 1, figsize=VISUALIZATION_CONFIG['figsize'], 
                            sharex=True)
    
    data_length = len(result['trend'])
    if len(time_index) >= data_length:
        time_axis = time_index[:data_length]
    else:
        time_axis = pd.date_range(start=time_index[0], periods=data_length, freq='W')
    
    axes[0].plot(time_axis, result['original'], 'k-', linewidth=1.5, label='Original')
    axes[0].set_ylabel('Original')
    axes[0].set_title(f'STL Decomposition: {english_name}', fontsize=16, pad=20)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')
    
    axes[1].plot(time_axis, result['trend'], 'b-', linewidth=1.5, label='Trend')
    axes[1].set_ylabel('Trend (T)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper left')
    
    axes[2].plot(time_axis, result['seasonal'], 'g-', linewidth=1.5, label='Seasonal')
    axes[2].set_ylabel('Seasonal (S)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper left')
    
    axes[3].plot(time_axis, result['resid'], 'r-', linewidth=1, label='Residual')
    axes[3].set_ylabel('Residual (R)')
    axes[3].set_xlabel('Time')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='upper left')
    
    plt.tight_layout()
    
    for fmt in VISUALIZATION_CONFIG['format']:
        save_path = output_path.with_suffix(f'.{fmt}')
        plt.savefig(save_path, format=fmt, dpi=VISUALIZATION_CONFIG['dpi'], 
                   bbox_inches='tight')
    
    plt.close()
    fig.clf()
    plt.close('all')

def plot_single_stl_visualization_wrapper(plot_info: Tuple) -> Tuple[str, bool]:
    series_name, english_name, result, time_index, output_path = plot_info
    
    try:
        plot_stl_decomposition(series_name, english_name, result, time_index, output_path)
        return (series_name, True)
    except Exception as e:
        print(f"Error plotting {series_name}: {str(e)}")
        return (series_name, False)

def generate_visualizations(stl_results: Dict[str, Dict],
                           output_dir: Path,
                           time_index: pd.DatetimeIndex,
                           max_series: Optional[int] = None):
    if not VISUALIZATION_CONFIG['enable_visualization']:
        print("Visualization disabled in configuration")
        return
    
    print("\n" + "=" * 80)
    print("Generating visualization plots...")
    print("=" * 80)
    
    viz_dir = output_dir / OUTPUT_CONFIG['viz_dir'] / 'individual'
    
    series_to_plot = list(stl_results.items())
    if max_series is not None:
        series_to_plot = series_to_plot[:max_series]
    
    if VISUALIZATION_CONFIG.get('max_series_to_plot') is not None:
        series_to_plot = series_to_plot[:VISUALIZATION_CONFIG['max_series_to_plot']]
    
    print(f"Plotting {len(series_to_plot)} time series")
    
    plot_tasks = []
    for series_name, result in series_to_plot:
        english_name = get_english_name(series_name)
        safe_name = english_name.replace('/', '_').replace('\\', '_').replace(':', '_')
        output_path = viz_dir / f"{safe_name}_stl_decomposition"
        plot_tasks.append((series_name, english_name, result, time_index, output_path))
    
    if PARALLEL_CONFIG['enable_parallel']:
        max_workers = PARALLEL_CONFIG['max_workers']
        if max_workers is None:
            max_workers = os.cpu_count()
        
        print(f"Using parallel processing with {max_workers} workers")
        start_time = time.time()
        
        successful_plots = 0
        failed_plots = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(plot_single_stl_visualization_wrapper, task): task[0] 
                      for task in plot_tasks}
            
            completed = 0
            for future in as_completed(futures):
                completed += 1
                series_name, success = future.result()
                
                if success:
                    successful_plots += 1
                else:
                    failed_plots.append(series_name)
                
                if completed % 10 == 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed
                    remaining = (len(plot_tasks) - completed) * avg_time
                    print(f"  Progress: {completed}/{len(plot_tasks)} ({100*completed/len(plot_tasks):.1f}%) "
                          f"- ETA: {remaining/60:.1f} minutes")
        
        elapsed_time = time.time() - start_time
        print(f"\nVisualization completed in {elapsed_time/60:.2f} minutes")
        print(f"  Successfully generated: {successful_plots}/{len(plot_tasks)} plots")
        print(f"  Failed: {len(failed_plots)} plots")
        
        if failed_plots:
            print(f"\nFailed plots:")
            for series_name in failed_plots[:10]:
                print(f"  - {series_name}")
            if len(failed_plots) > 10:
                print(f"  ... and {len(failed_plots) - 10} more")
    
    else:
        print("Using sequential processing")
        start_time = time.time()
        
        successful_plots = 0
        failed_plots = []
        
        for i, plot_info in enumerate(plot_tasks):
            series_name, success = plot_single_stl_visualization_wrapper(plot_info)
            
            if success:
                successful_plots += 1
            else:
                failed_plots.append(series_name)
            
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / (i + 1)
                remaining = (len(plot_tasks) - (i + 1)) * avg_time
                print(f"  Progress: {i+1}/{len(plot_tasks)} ({100*(i+1)/len(plot_tasks):.1f}%) "
                      f"- ETA: {remaining/60:.1f} minutes")
        
        elapsed_time = time.time() - start_time
        print(f"\nVisualization completed in {elapsed_time/60:.2f} minutes")
        print(f"  Successfully generated: {successful_plots}/{len(plot_tasks)} plots")
        print(f"  Failed: {len(failed_plots)} plots")
    
    print(f"Generated {successful_plots} visualization plots")

def main():
    print("=" * 80)
    print("STL Decomposition Module")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Seasonal period: {STL_CONFIG['seasonal_period']}")
    print(f"  Seasonal window: {STL_CONFIG['seasonal']}")
    print(f"  Robust mode: {STL_CONFIG['robust']}")
    print(f"  Parallel processing: {PARALLEL_CONFIG['enable_parallel']}")
    print("=" * 80)
    
    setup_matplotlib()
    output_dir = create_output_directory()
    
    flu_data = load_flu_data()
    if flu_data is None:
        print("Failed to load data. Exiting.")
        return
    
    stl_series = prepare_stl_data(flu_data)
    if not stl_series:
        print("No valid time series found. Exiting.")
        return
    
    time_index = pd.to_datetime(flu_data['start_date'])
    
    stl_results = batch_stl_decomposition(stl_series, 
                                         enable_parallel=PARALLEL_CONFIG['enable_parallel'])
    
    if not stl_results:
        print("No STL results generated. Exiting.")
        return
    
    save_stl_results_csv(stl_results, output_dir, time_index)
    
    series_names = list(stl_results.keys())
    stl_3d_array, metadata = save_stl_arrays(stl_results, output_dir, time_index, series_names)
    
    generate_visualizations(stl_results, output_dir, time_index)
    
    print("\n" + "=" * 80)
    print("STL Decomposition Completed Successfully!")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Total series processed: {len(stl_results)}")
    print(f"Data shape: {metadata['data_shape']}")
    print(f"Time range: {time_index[0]} to {time_index[len(time_index)-1]}")
    print("=" * 80)

if __name__ == "__main__":
    main()

