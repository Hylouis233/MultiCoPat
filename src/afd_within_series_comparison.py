#!/usr/bin/env python3

import os
import sys
import pickle
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu

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
    'significance_level': 0.05,
    'min_component_variance': 1e-10,
    'top_n_components': 10,
    'output_dir': 'AFD/within_series_comparison'
}

def create_output_directory():
    output_dir = ANALYSIS_CONFIG['output_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_afd_stl_results():
    print("Loading AFD-STL results...")
    try:
        seasonal_data = np.load("AFD/afd_seasonal_components.npy", allow_pickle=False)
        
        residual_data = np.load("AFD/afd_residual_components.npy", allow_pickle=False)
        with open("AFD/afd_stl_metadata.pkl", "rb") as f:
            try:
                metadata = pickle.load(f, encoding='latin1')
        except:
                metadata = pickle.load(f)
        
        time_index = metadata['time_index']
        series_names = metadata['series_names']
        component_numbers = metadata['component_numbers']
        
        n_series, n_timepoints, n_s_components = seasonal_data.shape
        _, _, n_r_components = residual_data.shape
        
        combined_data = np.full((n_series, n_timepoints, n_s_components + n_r_components), np.nan)
        combined_data[:, :, :n_s_components] = seasonal_data
        combined_data[:, :, n_s_components:] = residual_data
        
        afd_result_types = []
        max_K_S = metadata.get('max_K_Seasonal', 0)
        max_K_R = metadata.get('max_K_Residual', 0)
        
        for k in range(max_K_S):
            afd_result_types.append(f'seasonal_component_{k+1}')
        afd_result_types.append('seasonal_residue')
        
        for k in range(max_K_R):
            afd_result_types.append(f'residual_component_{k+1}')
        afd_result_types.append('residual_residue')
        
        print(f"Loaded AFD-STL results:")
        print(f"  Series: {len(series_names)}")
        print(f"  Time points: {n_timepoints}")
        print(f"  Seasonal components: {max_K_S} (+ 1 residue)")
        print(f"  Residual components: {max_K_R} (+ 1 residue)")
        print(f"  Total components: {len(afd_result_types)}")
        
        return combined_data, time_index, series_names, afd_result_types, component_numbers
        
    except Exception as e:
        print(f"Error loading AFD-STL results: {e}")
        print("Please run AFD-STL analysis first (python scripts/AFD_analysis.py --stl)")
        return None, None, None, None, None

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

def analyze_single_series_afd(series_afd_data, period_classification, series_name, afd_result_types, component_numbers=None):
    print(f"  Analyzing: {series_name}")
    
    component_indices = [i for i, col in enumerate(afd_result_types) 
                        if col.startswith('seasonal_component_') or col.startswith('residual_component_')]
    
    if len(series_afd_data) != len(period_classification):
        print(f"    Warning: Length mismatch. Skipping.")
        return None
    
    co_mask = period_classification['period_type'] == 'co_circulation'
    single_mask = period_classification['period_type'] == 'single_dominant'
    
    co_count = np.sum(co_mask)
    single_count = np.sum(single_mask)
    
    if co_count < 3 or single_count < 3:
        print(f"    Insufficient data: co={co_count}, single={single_count}")
        return None
    
    co_components = series_afd_data[co_mask][:, component_indices]
    single_components = series_afd_data[single_mask][:, component_indices]
    
    component_results = []
    
    for comp_idx, comp_col_idx in enumerate(component_indices):
        comp_name = afd_result_types[comp_col_idx]
        if 'seasonal_component_' in comp_name:
            comp_num = int(comp_name.split('_')[-1])
            comp_type = 'seasonal'
        elif 'residual_component_' in comp_name:
            comp_num = int(comp_name.split('_')[-1])
            comp_type = 'residual'
        else:
            comp_num = comp_idx + 1
            comp_type = 'unknown'
        
        co_data = co_components[:, comp_idx]
        single_data = single_components[:, comp_idx]
        
        co_energy = np.var(co_data)
        single_energy = np.var(single_data)
        
        if co_energy < ANALYSIS_CONFIG['min_component_variance'] and \
           single_energy < ANALYSIS_CONFIG['min_component_variance']:
            continue
        
        try:
            t_stat, t_pvalue = ttest_ind(co_data, single_data)
        except:
            t_stat, t_pvalue = np.nan, np.nan
        
        try:
            u_stat, u_pvalue = mannwhitneyu(co_data, single_data, alternative='two-sided')
        except:
            u_stat, u_pvalue = np.nan, np.nan
        
        try:
            pooled_std = np.sqrt(((len(co_data) - 1) * np.var(co_data, ddof=1) + 
                                (len(single_data) - 1) * np.var(single_data, ddof=1)) / 
                               (len(co_data) + len(single_data) - 2))
            cohens_d = (np.mean(co_data) - np.mean(single_data)) / pooled_std if pooled_std > 0 else 0
        except:
            cohens_d = np.nan
        
        component_results.append({
            'component_number': comp_num,
            'component_type': comp_type,
            'component_name': comp_name,
            'co_mean': np.mean(co_data),
            'co_std': np.std(co_data),
            'co_energy': co_energy,
            'single_mean': np.mean(single_data),
            'single_std': np.std(single_data),
            'single_energy': single_energy,
            'energy_ratio': co_energy / single_energy if single_energy > 0 else np.inf,
            't_statistic': t_stat,
            't_pvalue': t_pvalue,
            'u_statistic': u_stat,
            'u_pvalue': u_pvalue,
            'cohens_d': cohens_d,
            'significant': t_pvalue < ANALYSIS_CONFIG['significance_level'] if not np.isnan(t_pvalue) else False
        })
    
    if not component_results:
        return None
    
    component_df = pd.DataFrame(component_results)
    
    co_total_energy = component_df['co_energy'].sum()
    single_total_energy = component_df['single_energy'].sum()
    
    top_n = ANALYSIS_CONFIG['top_n_components']
    co_top_n_energy = component_df.nlargest(top_n, 'co_energy')['co_energy'].sum()
    single_top_n_energy = component_df.nlargest(top_n, 'single_energy')['single_energy'].sum()
    
    co_concentration = co_top_n_energy / co_total_energy if co_total_energy > 0 else 0
    single_concentration = single_top_n_energy / single_total_energy if single_total_energy > 0 else 0
    
    n_significant = component_df['significant'].sum()
    
    return {
        'series_name': series_name,
        'component_results': component_df,
        'co_n': co_count,
        'single_n': single_count,
        'co_total_energy': co_total_energy,
        'single_total_energy': single_total_energy,
        'total_energy_ratio': co_total_energy / single_total_energy if single_total_energy > 0 else np.inf,
        'co_energy_concentration': co_concentration,
        'single_energy_concentration': single_concentration,
        'concentration_difference': co_concentration - single_concentration,
        'n_components_analyzed': len(component_results),
        'n_significant_changes': n_significant,
            'significant_change_rate': n_significant / len(component_results) if len(component_results) > 0 else 0
    }

def perform_within_series_analysis(afd_data, series_names, afd_result_types, period_classification, component_numbers=None):
    print("\n" + "="*80)
    print("AFD WITHIN-SERIES COMPARISON (AFD-STL Results)")
    print("="*80)
    print("Analyzing how each series' AFD structure changes between periods...")
    print("Note: Using AFD-STL results (S/R decomposition)")
    
    results = []
    
    for series_idx, series_name in enumerate(series_names):
        series_afd_data = afd_data[series_idx, :, :]
        
        result = analyze_single_series_afd(
            series_afd_data, 
            period_classification, 
            series_name, 
            afd_result_types,
            component_numbers
        )
        
        if result is not None:
            results.append(result)
    
    print(f"\nSuccessfully analyzed {len(results)} time series")
    
    return results


def generate_comprehensive_report(all_results, output_dir):
    print("\nGenerating comprehensive report...")
    
    all_component_data = []
    for result in all_results:
        series_name = result['series_name']
        for _, row in result['component_results'].iterrows():
            all_component_data.append({
                'series_name': series_name,
                **row.to_dict()
            })
    
    component_df = pd.DataFrame(all_component_data)
    component_path = os.path.join(output_dir, 'component_level_comparison.csv')
    component_df.to_csv(component_path, index=False, encoding='utf-8-sig')
    
    series_summary = []
    for result in all_results:
        series_summary.append({
            'series_name': result['series_name'],
            'co_n': result['co_n'],
            'single_n': result['single_n'],
            'total_energy_ratio': result['total_energy_ratio'],
            'co_energy_concentration': result['co_energy_concentration'],
            'single_energy_concentration': result['single_energy_concentration'],
            'concentration_difference': result['concentration_difference'],
            'n_components_analyzed': result['n_components_analyzed'],
            'n_significant_changes': result['n_significant_changes'],
            'significant_change_rate': result['significant_change_rate']
        })
    
    series_df = pd.DataFrame(series_summary)
    series_path = os.path.join(output_dir, 'series_level_summary.csv')
    series_df.to_csv(series_path, index=False, encoding='utf-8-sig')
    
    summary_path = os.path.join(output_dir, 'analysis_summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("AFD Within-Series Comparison Analysis Summary\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("1. OVERALL STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Total series analyzed: {len(all_results)}\n")
        f.write(f"Total component comparisons: {len(component_df)}\n")
        f.write(f"Overall significant change rate: {component_df['significant'].mean():.1%}\n\n")
        
        f.write("2. ENERGY RATIO DISTRIBUTION\n")
        f.write("-" * 80 + "\n")
        high = (series_df['total_energy_ratio'] > 1.5).sum()
        moderate = ((series_df['total_energy_ratio'] > 1.0) & (series_df['total_energy_ratio'] <= 1.5)).sum()
        low = (series_df['total_energy_ratio'] <= 1.0).sum()
        f.write(f"Higher energy in co-circulation (>1.5): {high} ({high/len(series_df):.1%})\n")
        f.write(f"Moderately higher (1.0-1.5): {moderate} ({moderate/len(series_df):.1%})\n")
        f.write(f"Lower or equal (≤1.0): {low} ({low/len(series_df):.1%})\n")
        f.write(f"Mean energy ratio: {series_df['total_energy_ratio'].mean():.2f}\n")
        f.write(f"Median energy ratio: {series_df['total_energy_ratio'].median():.2f}\n\n")
        
        f.write("3. TOP 10 SERIES WITH LARGEST ENERGY CHANGES\n")
        f.write("-" * 80 + "\n")
        top_10 = series_df.nlargest(10, 'total_energy_ratio')[['series_name', 'total_energy_ratio', 'n_significant_changes']]
        for idx, row in top_10.iterrows():
            f.write(f"  {row['series_name']}: ratio={row['total_energy_ratio']:.2f}, ")
            f.write(f"significant_changes={row['n_significant_changes']}\n")
        
        f.write("\n4. SERIES WITH HIGHEST SIGNIFICANT CHANGE RATES\n")
        f.write("-" * 80 + "\n")
        top_sig = series_df.nlargest(10, 'significant_change_rate')[['series_name', 'significant_change_rate', 'n_significant_changes', 'n_components_analyzed']]
        for idx, row in top_sig.iterrows():
            f.write(f"  {row['series_name']}: {row['significant_change_rate']:.1%} ")
            f.write(f"({row['n_significant_changes']}/{row['n_components_analyzed']} components)\n")
        
        f.write("\n5. ENERGY CONCENTRATION PATTERNS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean concentration difference: {series_df['concentration_difference'].mean():+.3f}\n")
        more_concentrated = (series_df['concentration_difference'] > 0.05).sum()
        less_concentrated = (series_df['concentration_difference'] < -0.05).sum()
        similar = len(series_df) - more_concentrated - less_concentrated
        f.write(f"More concentrated in co-circulation: {more_concentrated}\n")
        f.write(f"Less concentrated in co-circulation: {less_concentrated}\n")
        f.write(f"Similar concentration: {similar}\n")
    
    print(f"Reports saved to: {output_dir}")

def main_within_series_analysis():
    print("\n" + "="*80)
    print("AFD WITHIN-SERIES COMPARISON ANALYSIS (AFD-STL)")
    print("="*80)
    
    output_dir = create_output_directory()
    
    afd_data, time_index, series_names, afd_result_types, component_numbers = load_afd_stl_results()
    if afd_data is None:
        return
    
    flu_data = load_flu_data()
    if flu_data is None:
        return
    
    co_circulation_df = load_co_circulation_data()
    if co_circulation_df is None:
        return
    
    period_classification = classify_time_periods(flu_data, co_circulation_df)
    
    all_results = perform_within_series_analysis(
        afd_data, series_names, afd_result_types, period_classification, component_numbers
    )
    
    if not all_results:
        print("No valid results.")
        return
    
    generate_comprehensive_report(all_results, output_dir)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    return all_results

if __name__ == '__main__':
    results = main_within_series_analysis()

