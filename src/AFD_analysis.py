
import gc
import os
import sys
import time
import warnings
import pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import partial
import multiprocessing as mp
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import acorr_ljungbox
from tslearn.metrics import dtw

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

VISUALIZATION_CONFIG = {
    'max_variables': 50,
    'batch_size': 10,
    'enable_visualization': True
}

PARALLEL_CONFIG = {
    'enable_parallel': True,
    'max_workers': None,
    'chunk_size': 1,
    'timeout': 300
}

AFD_STL_CONFIG = {
    'max_levels': 100,
    'white_noise_test_alpha': 0.1,
    'white_noise_test_lags': 20,
    'stl_output_dir': 'AFD/stl_decomposition',
    'afd_method': 2,
    'dic_gen_method': 2,
    'dic_params': (1/50, 1),
    'linear_trend_adjust': False
}

CHINESE_TO_ENGLISH = {
    'ILI%北方': 'Northern ILI%',
    'ILI%南方': 'Southern ILI%',
    '流感阳性率北方': 'Northern Flu Positive Rate',
    '流感阳性率南方': 'Southern Flu Positive Rate',
    '病毒监测和分型（北方）检测数': 'Northern Virus Detection Count',
    '病毒监测和分型（南方）检测数': 'Southern Virus Detection Count',
    '病毒监测和分型（北方）阳性数': 'Northern Virus Positive Count',
    '病毒监测和分型（南方）阳性数': 'Southern Virus Positive Count',
    '病毒监测和分型（北方）A(H3N2)': 'Northern A(H3N2)',
    '病毒监测和分型（南方）A(H3N2)': 'Southern A(H3N2)',
    '病毒监测和分型（北方）甲型 H1N1': 'Northern Type A H1N1',
    '病毒监测和分型（南方）甲型 H1N1': 'Southern Type A H1N1',
    '病毒监测和分型（北方）A(H7N9)': 'Northern A(H7N9)',
    '病毒监测和分型（南方）A(H7N9)': 'Southern A(H7N9)',
    '病毒监测和分型（北方）A未分系': 'Northern A Untyped',
    '病毒监测和分型（南方）A未分系': 'Southern A Untyped',
    '病毒监测和分型（北方）Victoria': 'Northern Victoria',
    '病毒监测和分型（南方）Victoria': 'Southern Victoria',
    '病毒监测和分型（北方）Yamagata': 'Northern Yamagata',
    '病毒监测和分型（南方）Yamagata': 'Southern Yamagata',
    '病毒监测和分型（北方）B未分系': 'Northern B Untyped',
    '病毒监测和分型（南方）B未分系': 'Southern B Untyped',
    '病毒监测和分型（北方）A型': 'Northern Type A',
    '病毒监测和分型（南方）A型': 'Southern Type A',
    '病毒监测和分型（北方）B型': 'Northern Type B',
    '病毒监测和分型（南方）B型': 'Southern Type B',
}

flu_data = None

def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('seaborn-v0_8-darkgrid')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['legend.fontsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600

def translate_text(chinese_text):
    return CHINESE_TO_ENGLISH.get(chinese_text, chinese_text)

def save_figure(fig_name, func_name):
    save_dir = f"./AFD/{func_name}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    png_path = os.path.join(save_dir, f"{fig_name}.png")
    plt.savefig(png_path, dpi=600, bbox_inches='tight')
    print(f"Figure saved to: {png_path}")
    
    pdf_path = os.path.join(save_dir, f"{fig_name}.pdf")
    plt.savefig(pdf_path, dpi=600, bbox_inches='tight')
    print(f"Figure saved to: {pdf_path}")

def set_xticks_display(ax, data_col, interval=None, subplot_count=1):
    data_length = len(data_col)
    
    if interval is None:
        if data_length > 500:
            interval = 52 * max(1, subplot_count // 4)
        elif data_length > 300:
            interval = 26 * max(1, subplot_count // 4)
        elif data_length > 150:
            interval = 13 * max(1, subplot_count // 6)
        elif data_length > 50:
            interval = 8 * max(1, subplot_count // 8)
        else:
            interval = 4 * max(1, subplot_count // 10)
    
    indices = range(0, data_length, interval)
    
    if hasattr(data_col, 'iloc'):
        tick_positions = [data_col.iloc[i] for i in indices]
    else:
        tick_positions = [data_col[i] for i in indices]
    
    tick_labels = []
    for i in indices:
        if hasattr(data_col, 'iloc'):
            date_val = data_col.iloc[i]
        else:
            date_val = data_col[i]
        
        if hasattr(date_val, 'strftime'):
            tick_labels.append(date_val.strftime('%Y-%m'))
        else:
            str_val = str(date_val)
            if len(str_val) > 7:
                tick_labels.append(str_val[:7])
            else:
                tick_labels.append(str_val)
    
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    
    fontsize = 9 if subplot_count <= 4 else 8
    plt.setp(ax.xaxis.get_majorticklabels(), fontsize=fontsize)

def load_data():
    print("Loading data using UnifiedDataPortal...")
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

def load_stl_output(stl_output_dir=None):
    if stl_output_dir is None:
        stl_output_dir = Path(AFD_STL_CONFIG['stl_output_dir'])
    else:
        stl_output_dir = Path(stl_output_dir)
    
    print("=" * 80)
    print("Loading STL Decomposition Output (S/R)")
    print("=" * 80)
    
    arrays_dir = stl_output_dir / 'arrays'
    npy_path = arrays_dir / 'stl_results.npy'
    pkl_path = arrays_dir / 'stl_results.pkl'
    
    if npy_path.exists() and pkl_path.exists():
        print("Loading from structured arrays...")
        try:
            stl_3d_array = np.load(npy_path)
            with open(pkl_path, 'rb') as f:
                metadata = pickle.load(f)
            
            time_index = metadata['time_index']
            series_names = metadata['series_names']
            
            if isinstance(time_index, pd.Series):
                time_index = pd.to_datetime(time_index.values)
            elif not isinstance(time_index, pd.DatetimeIndex):
                time_index = pd.to_datetime(time_index)
            
            seasonal_dict = {}
            residual_dict = {}
            
            for i, series_name in enumerate(series_names):
                seasonal_dict[series_name] = stl_3d_array[i, :, 1]
                residual_dict[series_name] = stl_3d_array[i, :, 2]
            
            print(f"Successfully loaded {len(series_names)} series from arrays")
            print(f"Data shape: {stl_3d_array.shape}")
            print(f"Time range: {time_index[0]} to {time_index[-1]}")
            
            return {
                'seasonal': seasonal_dict,
                'residual': residual_dict
            }, time_index, metadata
            
        except Exception as e:
            print(f"Error loading from arrays: {e}")
            print("Falling back to CSV files...")
    
    csv_dir = stl_output_dir / 'csv'
    seasonal_csv_dir = csv_dir / 'stl_seasonal'
    residual_csv_dir = csv_dir / 'stl_resid'
    
    if not seasonal_csv_dir.exists() or not residual_csv_dir.exists():
        raise FileNotFoundError(f"STL output directory not found: {stl_output_dir}")
    
    print("Loading from CSV files...")
    
    seasonal_dict = {}
    residual_dict = {}
    series_names = set()
    
    for csv_file in seasonal_csv_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'time_index' in df.columns and 'Seasonal' in df.columns:
                series_name = csv_file.stem.replace('_seasonal', '').replace('_trend', '').replace('_resid', '')
                series_names.add(series_name)
                seasonal_dict[series_name] = df['Seasonal'].values
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    for csv_file in residual_csv_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'time_index' in df.columns and 'Residual' in df.columns:
                series_name = csv_file.stem.replace('_seasonal', '').replace('_trend', '').replace('_resid', '')
                series_names.add(series_name)
                residual_dict[series_name] = df['Residual'].values
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue
    
    common_series = set(seasonal_dict.keys()) & set(residual_dict.keys())
    seasonal_dict = {k: seasonal_dict[k] for k in common_series}
    residual_dict = {k: residual_dict[k] for k in common_series}
    
    time_index = None
    for csv_file in seasonal_csv_dir.glob('*.csv'):
        try:
            df = pd.read_csv(csv_file, encoding='utf-8-sig')
            if 'time_index' in df.columns:
                time_index = pd.to_datetime(df['time_index'])
                break
        except:
            continue
    
    if time_index is None:
        if seasonal_dict:
            first_series = list(seasonal_dict.keys())[0]
            data_length = len(seasonal_dict[first_series])
            time_index = pd.date_range(start='2011-01-01', periods=data_length, freq='W')
    
    metadata = {
        'series_names': list(common_series),
        'time_index': time_index,
        'n_series': len(common_series),
        'n_timepoints': len(time_index) if time_index is not None else None
    }
    
    print(f"Successfully loaded {len(common_series)} series from CSV files")
    if time_index is not None:
        print(f"Time range: {time_index[0]} to {time_index[-1]}")
    
    return {
        'seasonal': seasonal_dict,
        'residual': residual_dict
    }, time_index, metadata

def white_noise_test(residual, lags=None, alpha=None):
    if lags is None:
        lags = AFD_STL_CONFIG['white_noise_test_lags']
    if alpha is None:
        alpha = AFD_STL_CONFIG['white_noise_test_alpha']
    
    residual_clean = residual[np.isfinite(residual)]
    
    if len(residual_clean) < lags + 1:
        return False, 0.0, 0.0
    
    lb_result = acorr_ljungbox(residual_clean, lags=lags)
    
    p_value = lb_result['lb_pvalue'].iloc[-1]
    lb_stat = lb_result['lb_stat'].iloc[-1]
    
    is_white_noise = p_value > alpha
    
    return is_white_noise, p_value, lb_stat

def process_single_afd_series_adaptive(series_info, component_type='seasonal', linear_trend_adjust=None):
    series_name, series_data = series_info
    
    if linear_trend_adjust is None:
        linear_trend_adjust = AFD_STL_CONFIG['linear_trend_adjust']
    
    max_levels = AFD_STL_CONFIG['max_levels']
    optimal_K = None
    
    try:
        if 'AFDCal' not in sys.modules:
            sys.path.append('AFDCal')
        from AFDCal import AFDCal
        
        series_normalized = (series_data - np.mean(series_data)) / (np.std(series_data) + 1e-8)
        
        x = np.arange(len(series_normalized))
        if linear_trend_adjust:
            slope = (series_normalized[-1] - series_normalized[0]) / (len(series_normalized) - 1)
            linear_trend = series_normalized[0] + slope * x
            series_detrended = series_normalized - linear_trend
        else:
            series_detrended = series_normalized
            linear_trend = np.zeros_like(series_normalized)
        
        processed_signal = series_detrended
        
        decomposed_components_list = []
        residue_history = []
        p_value_history = []
        
        for K in range(1, max_levels + 1):
            try:
                afdcal = AFDCal()
                
                signal_2d = processed_signal.reshape(1, -1)
                afdcal.loadInputSignal(signal_2d)
                
                afdcal.setDecompMethod(AFD_STL_CONFIG['afd_method'])
                afdcal.setDicGenMethod(AFD_STL_CONFIG['dic_gen_method'])
                afdcal.genDic(*AFD_STL_CONFIG['dic_params'])
                afdcal.genEva()
                afdcal.init_decomp()
                
                for level_num in range(K):
                    afdcal.nextDecomp()
                
                current_level = getattr(afdcal, 'level', 0)
                components_current = []
                
                if hasattr(afdcal, 'deComp') and \
                   isinstance(afdcal.deComp, list) and \
                   len(afdcal.deComp) > 0 and \
                   isinstance(afdcal.deComp[0], list) and \
                   current_level > 0:
                    
                    components_from_deComp = afdcal.deComp[0]
                    num_actual_components = len(components_from_deComp)
                    actual_levels_to_extract = min(current_level, num_actual_components)
                    
                    for l_idx in range(actual_levels_to_extract):
                        if l_idx < num_actual_components:
                            component_complex = components_from_deComp[l_idx]
                            
                            if isinstance(component_complex, np.ndarray):
                                if component_complex.ndim == 2 and component_complex.shape[0] == 1:
                                    component = np.real(component_complex.ravel())
                                elif component_complex.ndim == 1:
                                    component = np.real(component_complex)
                                else:
                                    continue
                                
                                comp_len = len(component)
                                sig_len = len(processed_signal)
                                if comp_len == sig_len:
                                    components_current.append(component)
                                elif comp_len > sig_len:
                                    components_current.append(component[:sig_len])
                
                residue_current = None
                if components_current and len(processed_signal) > 0:
                    sum_components = np.sum(np.array(components_current), axis=0)
                    if len(sum_components) == len(processed_signal):
                        residue_current = processed_signal - sum_components
                else:
                    residue_current = np.copy(processed_signal)
                
                is_white_noise, p_value, lb_stat = white_noise_test(residue_current)
                
                residue_history.append(residue_current)
                p_value_history.append(p_value)
                decomposed_components_list = components_current.copy()
                
                if is_white_noise:
                    optimal_K = K
                    break
                
            except Exception as decomp_error:
                if decomposed_components_list:
                    optimal_K = len(decomposed_components_list)
                break
        
        if optimal_K is None:
            optimal_K = len(decomposed_components_list) if decomposed_components_list else max_levels
            if residue_history:
                residue_current = residue_history[-1]
            else:
                residue_current = np.copy(processed_signal)
        
        result = {
            'optimal_K': optimal_K,
            'decomposed_components_processed': decomposed_components_list,
            'residue_component_processed': residue_current,
            'processed_signal': processed_signal,
            'linear_trend': linear_trend,
            'mean': np.mean(series_data),
            'std': np.std(series_data) + 1e-8,
            'original_data': series_data,
            'normalized_data': series_normalized,
            'p_value_history': p_value_history,
            'component_type': component_type
        }
        
        return (series_name, result)
        
    except Exception as e:
        return (series_name, f"Error: {str(e)}")

def display_system_info():
    print("="*60)
    print("System Information & Performance Recommendations")
    print("="*60)
    
    cpu_count = mp.cpu_count()
    print(f"CPU cores available: {cpu_count}")
    
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Total memory: {memory.total / (1024**3):.1f} GB")
        print(f"Available memory: {memory.available / (1024**3):.1f} GB")
        print(f"Memory usage: {memory.percent:.1f}%")
    except ImportError:
        print("Memory information not available (psutil not installed)")
    
    print(f"\nCurrent parallel configuration:")
    print(f"  - Parallel processing: {'Enabled' if PARALLEL_CONFIG['enable_parallel'] else 'Disabled'}")
    print(f"  - Max workers: {PARALLEL_CONFIG['max_workers'] or 'Auto (CPU cores)'}")
    print(f"  - Timeout: {PARALLEL_CONFIG['timeout']} seconds")
    
    print(f"\nPerformance recommendations:")
    if PARALLEL_CONFIG['enable_parallel']:
        if PARALLEL_CONFIG['max_workers'] is None:
            recommended_workers = min(cpu_count, 8)
            print(f"  - Recommended max_workers: {recommended_workers} (leaving {cpu_count - recommended_workers} cores for system)")
        else:
            if PARALLEL_CONFIG['max_workers'] > cpu_count:
                print(f"  - Warning: max_workers ({PARALLEL_CONFIG['max_workers']}) > CPU cores ({cpu_count})")
                print(f"  - Consider reducing to {cpu_count} or fewer")
            else:
                print(f"  - Current max_workers setting looks good")
    else:
        print(f"  - Consider enabling parallel processing for better performance")
        print(f"  - Expected speedup: {cpu_count}x (theoretical maximum)")
    
    print("="*60)

def perform_afd_stl_analysis_parallel(stl_series_dict, component_type='seasonal'):
    if not PARALLEL_CONFIG['enable_parallel']:
        return perform_afd_stl_analysis_sequential(stl_series_dict, component_type)
    
    print("=" * 80)
    print(f"Starting Parallel AFD Analysis for {component_type} components")
    print("=" * 80)
    
    series_dict = stl_series_dict[component_type]
    series_items = list(series_dict.items())
    total_series = len(series_items)
    
    print(f"Total series to process: {total_series}")
    print(f"Component type: {component_type}")
    print(f"Max workers: {PARALLEL_CONFIG['max_workers'] or 'CPU cores'}")
    
    max_workers = PARALLEL_CONFIG['max_workers']
    if max_workers is None:
        max_workers = min(mp.cpu_count(), total_series)
    
    results = {}
    successful_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    process_func = partial(process_single_afd_series_adaptive, component_type=component_type)
    
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_series = {
                executor.submit(process_func, (series_name, series_data)): series_name
                for series_name, series_data in series_items
            }
            
            print(f"\nSubmitted {len(future_to_series)} tasks to {max_workers} workers...")
            
            completed = 0
            for future in as_completed(future_to_series, timeout=PARALLEL_CONFIG['timeout']):
                series_name = future_to_series[future]
                completed += 1
                
                try:
                    result_series_name, result_data = future.result()
                    
                    if isinstance(result_data, str) and result_data.startswith("Error:"):
                        print(f"[{completed}/{total_series}] ✗ {series_name}: {result_data}")
                        failed_count += 1
                    else:
                        results[result_series_name] = result_data
                        optimal_K = result_data.get('optimal_K', 0)
                        print(f"[{completed}/{total_series}] ✓ {series_name}: optimal K = {optimal_K}")
                        successful_count += 1
                        
                except Exception as e:
                    print(f"[{completed}/{total_series}] ✗ {series_name}: Exception: {str(e)}")
                    failed_count += 1
                
                if completed % 10 == 0:
                    elapsed_time = time.time() - start_time
                    avg_time_per_task = elapsed_time / completed
                    remaining_tasks = total_series - completed
                    estimated_remaining_time = remaining_tasks * avg_time_per_task
                    print(f"Progress: {completed}/{total_series} ({completed/total_series*100:.1f}%) - "
                          f"Elapsed: {elapsed_time:.1f}s - ETA: {estimated_remaining_time:.1f}s")
    
    except Exception as e:
        print(f"Error in parallel processing: {str(e)}")
        print("Falling back to sequential processing...")
        return perform_afd_stl_analysis_sequential(stl_series_dict, component_type)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\n" + "=" * 80)
    print(f"Parallel AFD Analysis for {component_type} Completed")
    print("=" * 80)
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per series: {total_time/total_series:.2f} seconds")
    print(f"Success: {successful_count} series")
    print(f"Failed: {failed_count} series")
    print(f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    
    return results

def perform_afd_stl_analysis_sequential(stl_series_dict, component_type='seasonal'):
    print("=" * 80)
    print(f"Starting Sequential AFD Analysis for {component_type} components")
    print("=" * 80)
    
    series_dict = stl_series_dict[component_type]
    series_items = list(series_dict.items())
    total_series = len(series_items)
    
    results = {}
    successful_count = 0
    failed_count = 0
    
    start_time = time.time()
    
    for i, (series_name, series_data) in enumerate(series_items, 1):
        print(f"\n[{i}/{total_series}] Processing: {series_name}")
        
        try:
            series_info = (series_name, series_data)
            result_series_name, result_data = process_single_afd_series_adaptive(
                series_info, component_type=component_type
            )
            
            if isinstance(result_data, str) and result_data.startswith("Error:"):
                print(f"  ✗ Decomposition failed: {result_data}")
                failed_count += 1
            else:
                results[result_series_name] = result_data
                optimal_K = result_data.get('optimal_K', 0)
                print(f"  ✓ Successfully completed, optimal K = {optimal_K}")
                successful_count += 1
            
        except Exception as e:
            print(f"  ✗ Decomposition failed: {str(e)}")
            failed_count += 1
            continue
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nAFD decomposition completion statistics:")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Average time per series: {total_time/total_series:.2f} seconds")
    print(f"Success: {successful_count} series")
    print(f"Failed: {failed_count} series")
    print(f"Success rate: {successful_count/(successful_count+failed_count)*100:.1f}%")
    
    return results

def save_afd_stl_results(seasonal_results, residual_results, time_index, series_names, output_dir='AFD'):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("Saving AFD-STL Analysis Results (Residual Components Only)")
    print("=" * 80)
    
    component_numbers = {
        'Series_Name': series_names,
        'K_Residual': []
    }
    
    for series_name in series_names:
        if residual_results and series_name in residual_results:
            K_R = residual_results[series_name].get('optimal_K', 0)
        else:
            K_R = 0
        
        component_numbers['K_Residual'].append(K_R)
    
    component_numbers_df = pd.DataFrame(component_numbers)
    component_numbers_csv = output_path / 'afd_component_numbers.csv'
    component_numbers_df.to_csv(component_numbers_csv, index=False, encoding='utf-8-sig')
    print(f"Saved component numbers to: {component_numbers_csv}")
    
    max_K_R = max(component_numbers['K_Residual']) if component_numbers['K_Residual'] else 0
    
    n_series = len(series_names)
    n_timepoints = len(time_index) if time_index is not None else 0
    
    if max_K_R > 0 and n_timepoints > 0:
        residual_3d_array = np.full((n_series, n_timepoints, max_K_R + 1), np.nan)
        
        for i, series_name in enumerate(series_names):
            if residual_results and series_name in residual_results:
                result = residual_results[series_name]
                components = result.get('decomposed_components_processed', [])
                residue = result.get('residue_component_processed', None)
                
                for k, comp in enumerate(components):
                    if k < max_K_R and len(comp) == n_timepoints:
                        residual_3d_array[i, :, k] = comp
                
                if residue is not None and len(residue) == n_timepoints:
                    residual_3d_array[i, :, max_K_R] = residue
        
        residual_npy = output_path / 'afd_residual_components.npy'
        np.save(residual_npy, residual_3d_array)
        print(f"Saved residual 3D array to: {residual_npy}")
        print(f"  Shape: {residual_3d_array.shape}")
    
    metadata = {
        'time_index': time_index,
        'series_names': series_names,
        'component_numbers': component_numbers,
        'afd_stl_config': AFD_STL_CONFIG,
        'processing_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'n_series': n_series,
        'n_timepoints': n_timepoints,
        'max_K_Residual': max_K_R
    }
    
    metadata_pkl = output_path / 'afd_stl_metadata.pkl'
    with open(metadata_pkl, 'wb') as f:
        pickle.dump(metadata, f)
    print(f"Saved metadata to: {metadata_pkl}")
    
    print("=" * 80)
    print("AFD-STL Results Saved Successfully!")
    print("=" * 80)

def main_afd_stl_analysis():
    print("=" * 80)
    print("AFD-STL Analysis: Fine Decomposition of Residual Components (R)")
    print("=" * 80)
    print(f"Configuration:")
    print(f"  Max levels: {AFD_STL_CONFIG['max_levels']}")
    print(f"  White noise test alpha: {AFD_STL_CONFIG['white_noise_test_alpha']}")
    print(f"  White noise test lags: {AFD_STL_CONFIG['white_noise_test_lags']}")
    print(f"  STL output directory: {AFD_STL_CONFIG['stl_output_dir']}")
    print(f"  Parallel processing: {PARALLEL_CONFIG['enable_parallel']}")
    print("\nNote: Only residual components R(t) will be decomposed using AFD.")
    print("Seasonal components S(t) are not decomposed.")
    print("=" * 80)
    
    try:
        stl_series_dict, time_index, stl_metadata = load_stl_output()
    except Exception as e:
        print(f"Error loading STL output: {e}")
        print("Please run STL decomposition first (stl_decomposition.py)")
        return None, None, None, None
    
    series_names = stl_metadata['series_names']
    
    print("\n" + "=" * 80)
    print("AFD Decomposition of Residual Components (R)")
    print("=" * 80)
    
    if PARALLEL_CONFIG['enable_parallel']:
        residual_results = perform_afd_stl_analysis_parallel(stl_series_dict, component_type='residual')
    else:
        residual_results = perform_afd_stl_analysis_sequential(stl_series_dict, component_type='residual')
    
    print("\n" + "=" * 80)
    print("Saving Results")
    print("=" * 80)
    
    save_afd_stl_results(None, residual_results, time_index, series_names)
    
    print("\n" + "=" * 80)
    print("AFD-STL Analysis Completed Successfully!")
    print("=" * 80)
    print(f"Total series processed: {len(series_names)}")
    print(f"Time range: {time_index[0]} to {time_index[-1]}")
    print("=" * 80)
    
    return None, residual_results, time_index, series_names

def main_stl_afd():
    print("=" * 80)
    print("AFD-STL Analysis: Adaptive Fine Decomposition of Residual Components")
    print("=" * 80)
    
    setup_matplotlib()
    
    print(f"\nConfiguration:")
    print(f"  - Parallel processing: {'Enabled' if PARALLEL_CONFIG['enable_parallel'] else 'Disabled'}")
    print(f"  - Max workers: {PARALLEL_CONFIG['max_workers'] or 'Auto (CPU cores)'}")
    print(f"  - Max levels: {AFD_STL_CONFIG['max_levels']}")
    print(f"  - White noise test alpha: {AFD_STL_CONFIG['white_noise_test_alpha']}")
    print(f"  - White noise test lags: {AFD_STL_CONFIG['white_noise_test_lags']}")
    print(f"\nNote: Only residual components R(t) will be decomposed using AFD.")
    
    seasonal_results, residual_results, time_index, series_names = main_afd_stl_analysis()
    
    if residual_results is None:
        print("AFD-STL analysis failed. Please check the error messages above.")
        return None
    
    print("\n" + "=" * 80)
    print("AFD-STL Analysis Completed!")
    print("=" * 80)
    print("Generated files:")
    print("- AFD/afd_component_numbers.csv (Component numbers K_R for each series)")
    print("- AFD/afd_residual_components.npy (3D array for residual components)")
    print("- AFD/afd_stl_metadata.pkl (Metadata including time index, series names, etc.)")
    print("=" * 80)
    
    return seasonal_results, residual_results, time_index, series_names

if __name__ == '__main__':
    np.random.seed(42)
    
    results = main_stl_afd()