import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib import font_manager

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

DATA_PATHS = {
    'stl_results_npy': PROJECT_ROOT / 'data/AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': PROJECT_ROOT / 'data/AFD/stl_decomposition/arrays/stl_results.pkl',
    'afd_residual_npy': PROJECT_ROOT / 'data/AFD/afd_residual_components.npy',
    'integrated_features_csv': PROJECT_ROOT / 'data/AFD/comprehensive_analysis/integrated_features/integrated_features.csv',
    'wtc_phase_csv': PROJECT_ROOT / 'data/AFD/wtc_analysis/phase_features/phase_features.csv',
    'co_circulation_csv': PROJECT_ROOT / 'data/co_circulation_periods.csv',
    'cwt_results_dir': PROJECT_ROOT / 'data/AFD/cwt_analysis'
}

def setup_plotting_style():
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)
    
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=sns.color_palette("deep"))
    
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def load_integrated_features():
    path = DATA_PATHS['integrated_features_csv']
    if not path.exists():
        raise FileNotFoundError(f"Integrated features file not found: {path}")
    return pd.read_csv(path)

def load_stl_results():
    npy_path = DATA_PATHS['stl_results_npy']
    pkl_path = DATA_PATHS['stl_results_pkl']
    
    if not npy_path.exists() or not pkl_path.exists():
        raise FileNotFoundError("STL results file not found")
        
    stl_array = np.load(npy_path)
    with open(pkl_path, 'rb') as f:
        metadata = pickle.load(f)
        
    return stl_array, metadata

def load_afd_results():
    path = DATA_PATHS['afd_residual_npy']
    if not path.exists():
        raise FileNotFoundError(f"AFD results file not found: {path}")
    return np.load(path)

def load_co_circulation_periods():
    path = DATA_PATHS['co_circulation_csv']
    if not path.exists():
        raise FileNotFoundError(f"Co-circulation periods file not found: {path}")
    return pd.read_csv(path)

def get_co_circulation_mask(series_name, time_index, co_circulation_df):
    mask = np.zeros(len(time_index), dtype=bool)
    
    if 'Northern' in series_name or 'North' in series_name:
        region = 'Northern'
    elif 'Southern' in series_name or 'South' in series_name:
        region = 'Southern'
    else:
        return mask
        
    region_periods = co_circulation_df[co_circulation_df['region'] == region]
    
    for _, row in region_periods.iterrows():
        start = pd.to_datetime(row['start_date'])
        end = pd.to_datetime(row['end_date'])
        mask |= (time_index >= start) & (time_index <= end)
        
    return mask

def save_figure(fig, filename):
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    fig.savefig(output_dir / f"{filename}.png")
    fig.savefig(output_dir / f"{filename}.pdf")
    print(f"Figure saved: {output_dir}/{filename}")

setup_plotting_style()
