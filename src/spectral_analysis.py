#!/usr/bin/env python3

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.stats import ttest_ind, mannwhitneyu
from scipy.fft import fft, fftfreq

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

COLUMN_MAPPING = {
    'year': 'year', 'week': 'week', 'start_date': 'start_date', 'stop_date': 'stop_date',
    'ILI%北方': 'northern_ili_rate', 'ILI%南方': 'southern_ili_rate',
    '流感阳性率北方': 'northern_flu_positive_rate',     '流感阳性率南方': 'southern_flu_positive_rate',
}

ANALYSIS_CONFIG = {
    'high_circulation_percentile': 0.60,
    'sampling_frequency': 1.0,
    'frequency_bands': {
        'long_term': (0, 1/52),
        'annual': (1/52, 1/26),
        'semi_annual': (1/26, 1/13),
        'seasonal': (1/13, 1/8),
        'short_term': (1/8, 0.5)
    },
    'output_dir': 'AFD/spectral_analysis'
}

VIZ_CONFIG = {
    'figure_size': (20, 12),
    'dpi': 600,
    'style': 'seaborn-v0_8-whitegrid',
}

def create_output_directory():
    output_dir = ANALYSIS_CONFIG['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def setup_matplotlib():
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['figure.figsize'] = VIZ_CONFIG['figure_size']
    plt.rcParams['figure.dpi'] = VIZ_CONFIG['dpi']
    plt.style.use(VIZ_CONFIG['style'])

def load_co_circulation_data():
    try:
        co_circulation_df = pd.read_csv("co_circulation_periods.csv")
        date_columns = ['Adjusted Start Date', 'Adjusted End Date', 'Actual Start Date', 'Actual End Date']
        for col in date_columns:
            if col in co_circulation_df.columns:
                co_circulation_df[col] = pd.to_datetime(co_circulation_df[col])
        return co_circulation_df
    except Exception as e:
        print(f"Error loading co-circulation data: {e}")
        return None

def load_flu_data():
    try:
        portal = UnifiedDataPortal()
        flu_data = portal.load_flu_data()
        if flu_data is not None:
            existing_columns = {k: v for k, v in COLUMN_MAPPING.items() if k in flu_data.columns}
            flu_data = flu_data.rename(columns=existing_columns)
        return flu_data
    except Exception as e:
        print(f"Error loading flu data: {e}")
        return None

def classify_time_periods(flu_data, co_circulation_df):
    flu_data_copy = flu_data.copy()
    flu_data_copy['period_type'] = 'low_circulation'
    
    ili_threshold_north = flu_data_copy['northern_ili_rate'].quantile(ANALYSIS_CONFIG['high_circulation_percentile'])
    ili_threshold_south = flu_data_copy['southern_ili_rate'].quantile(ANALYSIS_CONFIG['high_circulation_percentile'])
    pos_threshold_north = flu_data_copy['northern_flu_positive_rate'].quantile(ANALYSIS_CONFIG['high_circulation_percentile'])
    pos_threshold_south = flu_data_copy['southern_flu_positive_rate'].quantile(ANALYSIS_CONFIG['high_circulation_percentile'])
    
    high_circulation_mask = (
        (flu_data_copy['northern_ili_rate'] >= ili_threshold_north) |
        (flu_data_copy['southern_ili_rate'] >= ili_threshold_south)
    ) & (
        (flu_data_copy['northern_flu_positive_rate'] >= pos_threshold_north) |
        (flu_data_copy['southern_flu_positive_rate'] >= pos_threshold_south)
    )
    flu_data_copy.loc[high_circulation_mask, 'period_type'] = 'single_dominant'
    
    for _, period in co_circulation_df.iterrows():
        mask = (flu_data_copy['start_date'] >= period['Actual Start Date']) & \
               (flu_data_copy['start_date'] <= period['Actual End Date'])
        flu_data_copy.loc[mask, 'period_type'] = 'co_circulation'
    
    return flu_data_copy

def calculate_psd(time_series, fs=1.0, method='welch'):
    time_series_clean = np.array(time_series)
    time_series_clean = time_series_clean[~np.isnan(time_series_clean)]
    
    if len(time_series_clean) < 10:
        return None, None
    
    time_series_detrended = signal.detrend(time_series_clean)
    
    if method == 'welch':
        nperseg = min(len(time_series_detrended) // 4, 52)
        freqs, psd = signal.welch(time_series_detrended, fs=fs, nperseg=nperseg, 
                                   scaling='density', window='hann')
    else:
        freqs, psd = signal.periodogram(time_series_detrended, fs=fs, scaling='density')
    
    return freqs, psd

def extract_spectral_features(freqs, psd):
    if freqs is None or psd is None or len(psd) == 0:
        return None
    
    non_dc_mask = freqs > 0
    freqs_no_dc = freqs[non_dc_mask]
    psd_no_dc = psd[non_dc_mask]
    
    if len(psd_no_dc) == 0:
        return None
    
    dominant_freq_idx = np.argmax(psd_no_dc)
    dominant_frequency = freqs_no_dc[dominant_freq_idx]
    dominant_period = 1 / dominant_frequency if dominant_frequency > 0 else np.inf
    dominant_power = psd_no_dc[dominant_freq_idx]
    
    total_power = np.sum(psd_no_dc)
    spectral_centroid = np.sum(freqs_no_dc * psd_no_dc) / total_power if total_power > 0 else 0
    
    psd_normalized = psd_no_dc / total_power if total_power > 0 else psd_no_dc
    psd_normalized = psd_normalized[psd_normalized > 0]
    spectral_entropy = -np.sum(psd_normalized * np.log2(psd_normalized)) if len(psd_normalized) > 0 else 0
    
    band_energies = {}
    for band_name, (f_low, f_high) in ANALYSIS_CONFIG['frequency_bands'].items():
        band_mask = (freqs_no_dc >= f_low) & (freqs_no_dc < f_high)
        band_energy = np.sum(psd_no_dc[band_mask])
        band_energies[band_name] = band_energy
        band_energies[f'{band_name}_ratio'] = band_energy / total_power if total_power > 0 else 0
    
    total_spectral_power = total_power
    
    return {
        'dominant_frequency': dominant_frequency,
        'dominant_period_weeks': dominant_period,
        'dominant_power': dominant_power,
        'spectral_centroid': spectral_centroid,
        'spectral_entropy': spectral_entropy,
        'total_power': total_spectral_power,
        **band_energies
    }

def analyze_series_spectral_features(series_data, period_classification, series_name):
    print(f"  Analyzing: {series_name}")
    
    co_mask = period_classification['period_type'] == 'co_circulation'
    single_mask = period_classification['period_type'] == 'single_dominant'
    
    co_data = series_data[co_mask]
    single_data = series_data[single_mask]
    
    if len(co_data) < 10 or len(single_data) < 10:
        print(f"    Insufficient data: co={len(co_data)}, single={len(single_data)}")
        return None
    
    fs = ANALYSIS_CONFIG['sampling_frequency']
    
    co_freqs, co_psd = calculate_psd(co_data, fs=fs)
    single_freqs, single_psd = calculate_psd(single_data, fs=fs)
    
    if co_freqs is None or single_freqs is None:
        return None
    
    co_features = extract_spectral_features(co_freqs, co_psd)
    single_features = extract_spectral_features(single_freqs, single_psd)
    
    if co_features is None or single_features is None:
        return None
    
    return {
        'series_name': series_name,
        'co_features': co_features,
        'single_features': single_features,
        'co_psd': (co_freqs, co_psd),
        'single_psd': (single_freqs, single_psd),
        'co_n': len(co_data),
        'single_n': len(single_data)
    }

def perform_spectral_analysis(flu_data, period_classification):
    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS: FREQUENCY DOMAIN COMPARISON")
    print("="*80)
    
    results = []
    
    key_variables = ['northern_ili_rate', 'southern_ili_rate', 
                     'northern_flu_positive_rate', 'southern_flu_positive_rate']
    
    for var in key_variables:
        if var in flu_data.columns:
            result = analyze_series_spectral_features(
                flu_data[var].values, 
                period_classification, 
                var
            )
            if result is not None:
                results.append(result)
    
    return results

def compare_spectral_features(spectral_results):
    print("\nComparing spectral features...")
    
    comparison_data = []
    
    for result in spectral_results:
        series_name = result['series_name']
        co_feat = result['co_features']
        single_feat = result['single_features']
        
        features_to_compare = [
            'dominant_frequency', 'dominant_period_weeks', 'dominant_power',
            'spectral_centroid', 'spectral_entropy', 'total_power'
        ]
        
        for feat_name in features_to_compare:
            comparison_data.append({
                'series': series_name,
                'feature': feat_name,
                'co_value': co_feat[feat_name],
                'single_value': single_feat[feat_name],
                'difference': co_feat[feat_name] - single_feat[feat_name],
                    'ratio': co_feat[feat_name] / single_feat[feat_name] if single_feat[feat_name] != 0 else np.inf
            })
        
        for band_name in ANALYSIS_CONFIG['frequency_bands'].keys():
            ratio_key = f'{band_name}_ratio'
            if ratio_key in co_feat and ratio_key in single_feat:
                comparison_data.append({
                    'series': series_name,
                    'feature': ratio_key,
                    'co_value': co_feat[ratio_key],
                    'single_value': single_feat[ratio_key],
                    'difference': co_feat[ratio_key] - single_feat[ratio_key],
                    'ratio': co_feat[ratio_key] / single_feat[ratio_key] if single_feat[ratio_key] != 0 else np.inf
                })
    
    comparison_df = pd.DataFrame(comparison_data)
    return comparison_df

def create_spectral_visualizations(spectral_results, comparison_df, output_dir):
    print("\nCreating visualizations...")
    
    setup_matplotlib()
    
    for result in spectral_results:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        series_name = result['series_name']
        
        co_freqs, co_psd = result['co_psd']
        single_freqs, single_psd = result['single_psd']
        
        co_periods = 1 / co_freqs[co_freqs > 0]
        co_psd_periods = co_psd[co_freqs > 0]
        single_periods = 1 / single_freqs[single_freqs > 0]
        single_psd_periods = single_psd[single_freqs > 0]
        
        axes[0, 0].semilogy(co_freqs, co_psd, label='Co-circulation', linewidth=2)
        axes[0, 0].semilogy(single_freqs, single_psd, label='Single-dominant', linewidth=2, alpha=0.7)
        axes[0, 0].set_xlabel('Frequency (cycles/week)')
        axes[0, 0].set_ylabel('Power Spectral Density')
        axes[0, 0].set_title(f'Power Spectral Density: {series_name}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].semilogy(co_periods, co_psd_periods, label='Co-circulation', linewidth=2)
        axes[0, 1].semilogy(single_periods, single_psd_periods, label='Single-dominant', linewidth=2, alpha=0.7)
        axes[0, 1].set_xlabel('Period (weeks)')
        axes[0, 1].set_ylabel('Power Spectral Density')
        axes[0, 1].set_title(f'PSD vs Period: {series_name}')
        axes[0, 1].set_xlim(0, 104)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        band_names = list(ANALYSIS_CONFIG['frequency_bands'].keys())
        co_band_energies = [result['co_features'][f'{band}_ratio'] for band in band_names]
        single_band_energies = [result['single_features'][f'{band}_ratio'] for band in band_names]
        
        x = np.arange(len(band_names))
        width = 0.35
        axes[1, 0].bar(x - width/2, co_band_energies, width, label='Co-circulation', alpha=0.8)
        axes[1, 0].bar(x + width/2, single_band_energies, width, label='Single-dominant', alpha=0.8)
        axes[1, 0].set_xlabel('Frequency Band')
        axes[1, 0].set_ylabel('Energy Ratio')
        axes[1, 0].set_title('Frequency Band Energy Distribution')
        axes[1, 0].set_xticks(x)
        axes[1, 0].set_xticklabels(band_names, rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3, axis='y')
        
        axes[1, 1].axis('off')
        feature_comparison = [
            ['Feature', 'Co-circulation', 'Single-dominant', 'Ratio'],
            ['Dominant Period (weeks)', 
             f"{result['co_features']['dominant_period_weeks']:.1f}",
             f"{result['single_features']['dominant_period_weeks']:.1f}",
             f"{result['co_features']['dominant_period_weeks']/result['single_features']['dominant_period_weeks']:.2f}"],
            ['Spectral Centroid',
             f"{result['co_features']['spectral_centroid']:.4f}",
             f"{result['single_features']['spectral_centroid']:.4f}",
             f"{result['co_features']['spectral_centroid']/result['single_features']['spectral_centroid']:.2f}"],
            ['Spectral Entropy',
             f"{result['co_features']['spectral_entropy']:.3f}",
             f"{result['single_features']['spectral_entropy']:.3f}",
             f"{result['co_features']['spectral_entropy']/result['single_features']['spectral_entropy']:.2f}"],
            ['Total Power',
             f"{result['co_features']['total_power']:.2e}",
             f"{result['single_features']['total_power']:.2e}",
             f"{result['co_features']['total_power']/result['single_features']['total_power']:.2f}"]
        ]
        
        table = axes[1, 1].table(cellText=feature_comparison, cellLoc='left',
                                loc='center', colWidths=[0.3, 0.23, 0.23, 0.15])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        plt.tight_layout()
        
        safe_name = series_name.replace('/', '_').replace('\\', '_')
        plt.savefig(f'{output_dir}/spectral_{safe_name}.png', dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.savefig(f'{output_dir}/spectral_{safe_name}.pdf', dpi=VIZ_CONFIG['dpi'], bbox_inches='tight')
        plt.close()
    
    print(f"Visualizations saved to: {output_dir}")

def generate_spectral_report(spectral_results, comparison_df, output_dir):
    print("\nGenerating report...")
    
    report_path = os.path.join(output_dir, 'spectral_feature_comparison.csv')
    comparison_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    summary_path = os.path.join(output_dir, 'spectral_analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("Spectral Analysis Summary: Co-circulation vs Single-dominant Periods\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. ANALYZED TIME SERIES\n")
        f.write("-" * 80 + "\n")
        for result in spectral_results:
            f.write(f"  - {result['series_name']}: ")
            f.write(f"Co-circulation N={result['co_n']}, Single-dominant N={result['single_n']}\n")
        
        f.write("\n2. KEY FINDINGS\n")
        f.write("-" * 80 + "\n")
        
        period_comparison = comparison_df[comparison_df['feature'] == 'dominant_period_weeks']
        f.write("Dominant Period Changes:\n")
        for _, row in period_comparison.iterrows():
            f.write(f"  {row['series']}: {row['single_value']:.1f}w → {row['co_value']:.1f}w ")
            f.write(f"(change: {row['difference']:.1f}w, ratio: {row['ratio']:.2f})\n")
        
        f.write("\nSpectral Complexity (Entropy) Changes:\n")
        entropy_comparison = comparison_df[comparison_df['feature'] == 'spectral_entropy']
        for _, row in entropy_comparison.iterrows():
            change_pct = (row['co_value'] - row['single_value']) / row['single_value'] * 100
            f.write(f"  {row['series']}: {row['single_value']:.3f} → {row['co_value']:.3f} ")
            f.write(f"({change_pct:+.1f}%)\n")
        
        f.write("\nTotal Spectral Power Changes:\n")
        power_comparison = comparison_df[comparison_df['feature'] == 'total_power']
        for _, row in power_comparison.iterrows():
            f.write(f"  {row['series']}: ratio = {row['ratio']:.2f}\n")
    
    print(f"Report saved to: {summary_path}")

def main_spectral_analysis():
    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS: FREQUENCY DOMAIN COMPARISON")
    print("="*80)
    
    output_dir = create_output_directory()
    
    flu_data = load_flu_data()
    if flu_data is None:
        return
    
    co_circulation_df = load_co_circulation_data()
    if co_circulation_df is None:
        return
    
    period_classification = classify_time_periods(flu_data, co_circulation_df)
    
    spectral_results = perform_spectral_analysis(flu_data, period_classification)
    
    if not spectral_results:
        print("No valid spectral analysis results.")
        return
    
    comparison_df = compare_spectral_features(spectral_results)
    
    create_spectral_visualizations(spectral_results, comparison_df, output_dir)
    
    generate_spectral_report(spectral_results, comparison_df, output_dir)
    
    print("\n" + "="*80)
    print("SPECTRAL ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    return spectral_results, comparison_df

if __name__ == '__main__':
    results = main_spectral_analysis()

