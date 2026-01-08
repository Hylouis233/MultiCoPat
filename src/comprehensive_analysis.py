#!/usr/bin/env python3

import gc
import os
import sys
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
from scipy.stats import ttest_ind, mannwhitneyu, levene, shapiro
from statsmodels.stats.multitest import multipletests

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

sys.path.append('.')
from data_portal import UnifiedDataPortal

warnings.filterwarnings('ignore')

COMPREHENSIVE_CONFIG = {
    'enable_stl_features': True,
    'enable_afd_features': True,
    'enable_cwt_features': True,
    'enable_wtc_features': True,
    'normalize_features': True,
    'feature_selection': True,
    'n_components_pca': 10,
    'n_clusters': None,
    'n_bootstrap': 1000,
    'alpha': 0.05,
    'fdr_method': 'fdr_bh',
    'random_state': 42
}

OUTPUT_CONFIG = {
    'base_dir': 'data/AFD/comprehensive_analysis',
    'integrated_features_dir': 'integrated_features',
    'statistical_tests_dir': 'statistical_tests',
    'pattern_analysis_dir': 'pattern_analysis',
    'robustness_analysis_dir': 'robustness_analysis',
    'visualization_dir': 'visualization'
}

DATA_PATHS = {
    'stl_results_npy': 'data/AFD/stl_decomposition/arrays/stl_results.npy',
    'stl_results_pkl': 'data/AFD/stl_decomposition/arrays/stl_results.pkl',
    'stl_summary_csv': 'data/AFD/stl_decomposition/arrays/stl_summary.csv',
    'afd_component_numbers_csv': 'data/AFD/afd_component_numbers.csv',
    'afd_residual_array_npy': 'data/AFD/afd_residual_components.npy',
    'afd_metadata_pkl': 'data/AFD/afd_stl_metadata.pkl',
    'afd_comparison_dir': 'data/AFD/within_series_comparison',
    'cwt_analysis_dir': 'data/AFD/cwt_analysis',
    'wtc_coherence_csv': 'data/AFD/wtc_analysis/coherence_features/coherence_features.csv',
    'wtc_phase_csv': 'data/AFD/wtc_analysis/phase_features/phase_features.csv',
    'co_circulation_csv': 'data/co_circulation_periods.csv'
}

VISUALIZATION_CONFIG = {
    'enable_visualization': True,
    'dpi': 600,
    'figsize': (14, 10),
    'format': ['png', 'pdf']
}

def create_output_directory():
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for key, dir_name in OUTPUT_CONFIG.items():
        if key != 'base_dir':
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

def load_stl_features() -> pd.DataFrame:
    print("="*80)
    print("Loading STL Features")
    print("="*80)
    
    stl_summary_path = Path(DATA_PATHS['stl_summary_csv'])
    if not stl_summary_path.exists():
        print(f"Error: STL summary file not found: {stl_summary_path}")
        return pd.DataFrame()
    
    stl_summary = pd.read_csv(stl_summary_path)
    
    stl_npy_path = Path(DATA_PATHS['stl_results_npy'])
    stl_pkl_path = Path(DATA_PATHS['stl_results_pkl'])
    
    if stl_npy_path.exists() and stl_pkl_path.exists():
        stl_array = np.load(stl_npy_path)
        with open(stl_pkl_path, 'rb') as f:
            stl_metadata = pickle.load(f)
        
        series_names = stl_metadata['series_names']
        n_series = len(series_names)
        
        stl_features = []
        for i in range(n_series):
            trend = stl_array[i, :, 0]
            seasonal = stl_array[i, :, 1]
            residual = stl_array[i, :, 2]
            
            t_mean = np.nanmean(trend)
            t_std = np.nanstd(trend)
            t_cv = t_std / abs(t_mean) if abs(t_mean) > 1e-10 else 0
            
            if len(trend) > 1:
                time_points = np.arange(len(trend))
                t_slope = np.polyfit(time_points, trend, 1)[0]
            else:
                t_slope = 0.0
            
            s_mean = np.nanmean(seasonal)
            s_std = np.nanstd(seasonal)
            s_cv = s_std / abs(s_mean) if abs(s_mean) > 1e-10 else 0
            s_amplitude = np.nanmax(seasonal) - np.nanmin(seasonal)
            
            r_std = np.nanstd(residual)
            r_mean = np.nanmean(residual)
            r_cv = r_std / abs(r_mean) if abs(r_mean) > 1e-10 else 0
            
            stl_features.append({
                'Series_Name': series_names[i],
                'T_mean': t_mean,
                'T_slope': t_slope,
                'T_std': t_std,
                'T_cv': t_cv,
                'S_mean': s_mean,
                'S_amplitude': s_amplitude,
                'S_std': s_std,
                'S_cv': s_cv,
                'R_mean': r_mean,
                'R_std': r_std,
                'R_cv': r_cv
            })
        
        stl_features_df = pd.DataFrame(stl_features)
        print(f"Loaded STL features for {len(stl_features_df)} series")
        return stl_features_df
    else:
        print("Warning: STL array files not found, using summary data only")
        return stl_summary

def load_afd_features() -> pd.DataFrame:
    print("\n" + "="*80)
    print("Loading AFD Features")
    print("="*80)
    
    component_numbers_path = Path(DATA_PATHS['afd_component_numbers_csv'])
    residual_array_path = Path(DATA_PATHS['afd_residual_array_npy'])
    metadata_path = Path(DATA_PATHS['afd_metadata_pkl'])
    
    if not component_numbers_path.exists():
        print(f"Warning: AFD component numbers file not found: {component_numbers_path}")
        return pd.DataFrame()
    
    component_numbers_df = pd.read_csv(component_numbers_path)
    
    if residual_array_path.exists() and metadata_path.exists():
        try:
            residual_array = np.load(residual_array_path)
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            series_names = metadata['series_names']
            n_series, n_timepoints, n_components = residual_array.shape
            
            afd_features = []
            for i, series_name in enumerate(series_names):
                series_components = residual_array[i, :, :]
                
                component_energies = []
                for k in range(n_components - 1):
                    comp = series_components[:, k]
                    if not np.all(np.isnan(comp)):
                        energy = np.nanvar(comp)
                        component_energies.append(energy)
                    else:
                        component_energies.append(0.0)
                
                total_energy = np.sum(component_energies) if component_energies else 0.0
                
                if total_energy > 0:
                    max_energy = np.max(component_energies) if component_energies else 0.0
                    energy_concentration = max_energy / total_energy
                else:
                    energy_concentration = 0.0
                
                if total_energy > 0 and component_energies:
                    energy_probs = np.array(component_energies) / total_energy
                    energy_probs = energy_probs[energy_probs > 0]
                    if len(energy_probs) > 0:
                        energy_entropy = -np.sum(energy_probs * np.log2(energy_probs + 1e-10))
                    else:
                        energy_entropy = 0.0
                else:
                    energy_entropy = 0.0
                
                k_residual = component_numbers_df[component_numbers_df['Series_Name'] == series_name]['K_Residual'].values
                k_residual = k_residual[0] if len(k_residual) > 0 else 0
                
                afd_features.append({
                    'Series_Name': series_name,
                    'K_Residual': k_residual,
                    'AFD_total_energy': total_energy,
                    'AFD_energy_concentration': energy_concentration,
                    'AFD_energy_entropy': energy_entropy,
                    'AFD_max_component_energy': np.max(component_energies) if component_energies else 0.0,
                    'AFD_mean_component_energy': np.mean(component_energies) if component_energies else 0.0
                })
            
            afd_features_df = pd.DataFrame(afd_features)
            print(f"Loaded AFD features for {len(afd_features_df)} series")
            return afd_features_df
            
        except Exception as e:
            print(f"Error loading AFD array data: {e}")
            return component_numbers_df
    else:
        print("Warning: AFD array files not found, using component numbers only")
        return component_numbers_df

def load_cwt_features() -> pd.DataFrame:
    print("\n" + "="*80)
    print("Loading CWT Features")
    print("="*80)
    
    cwt_dir = Path(DATA_PATHS['cwt_analysis_dir'])
    if not cwt_dir.exists():
        print(f"Warning: CWT analysis directory not found: {cwt_dir}")
        return pd.DataFrame()
    
    cwt_features_list = []
    
    freq_drift_dir = cwt_dir / 'frequency_drift'
    if freq_drift_dir.exists():
        freq_stability_path = freq_drift_dir / 'frequency_stability.csv'
        if freq_stability_path.exists():
            try:
                freq_stability_df = pd.read_csv(freq_stability_path, encoding='utf-8-sig')
                freq_stability_df = freq_stability_df.rename(columns={
                    'FrequencyStability': 'CWT_freq_stability',
                    'DriftRate': 'CWT_freq_drift_rate'
                })
                cwt_features_list.append(freq_stability_df)
                print(f"  Loaded frequency stability: {len(freq_stability_df)} series")
            except Exception as e:
                print(f"  Error loading frequency stability: {e}")
    
    intensity_dir = cwt_dir / 'intensity_variation'
    if intensity_dir.exists():
        total_energy_seasonal_path = intensity_dir / 'total_energy_seasonal.csv'
        if total_energy_seasonal_path.exists():
            try:
                total_energy_seasonal_df = pd.read_csv(total_energy_seasonal_path, encoding='utf-8-sig')
                seasonal_energy_agg = total_energy_seasonal_df.groupby('Series_Name')['TotalEnergy'].agg([
                    ('CWT_seasonal_energy_mean', 'mean'),
                    ('CWT_seasonal_energy_std', 'std'),
                    ('CWT_seasonal_energy_cv', lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
                ]).reset_index()
                cwt_features_list.append(seasonal_energy_agg)
                print(f"  Loaded seasonal energy variation: {len(seasonal_energy_agg)} series")
            except Exception as e:
                print(f"  Error loading seasonal energy: {e}")
        
        total_energy_residual_path = intensity_dir / 'total_energy_residual.csv'
        if total_energy_residual_path.exists():
            try:
                total_energy_residual_df = pd.read_csv(total_energy_residual_path, encoding='utf-8-sig')
                residual_energy_agg = total_energy_residual_df.groupby('Series_Name')['TotalEnergy'].agg([
                    ('CWT_residual_energy_mean', 'mean'),
                    ('CWT_residual_energy_std', 'std'),
                    ('CWT_residual_energy_cv', lambda x: x.std() / x.mean() if x.mean() > 0 else 0)
                ]).reset_index()
                cwt_features_list.append(residual_energy_agg)
                print(f"  Loaded residual energy variation: {len(residual_energy_agg)} series")
            except Exception as e:
                print(f"  Error loading residual energy: {e}")
        
        energy_variability_seasonal_path = intensity_dir / 'energy_variability_seasonal.csv'
        if energy_variability_seasonal_path.exists():
            try:
                energy_variability_seasonal_df = pd.read_csv(energy_variability_seasonal_path, encoding='utf-8-sig')
                energy_variability_seasonal_df = energy_variability_seasonal_df.rename(columns={
                    'EnergyStd': 'CWT_seasonal_energy_std_alt',
                    'EnergyCV': 'CWT_seasonal_energy_cv_alt'
                })
                cwt_features_list.append(energy_variability_seasonal_df)
                print(f"  Loaded seasonal energy variability: {len(energy_variability_seasonal_df)} series")
            except Exception as e:
                print(f"  Error loading seasonal energy variability: {e}")
    
    anomalies_dir = cwt_dir / 'anomalous_events'
    if anomalies_dir.exists():
        event_stats_path = anomalies_dir / 'event_statistics.csv'
        if event_stats_path.exists():
            try:
                event_stats_df = pd.read_csv(event_stats_path, encoding='utf-8-sig')
                event_stats_df = event_stats_df.rename(columns={
                    'TotalEvents': 'CWT_anomaly_total_events',
                    'EventsPerYear': 'CWT_anomaly_events_per_year',
                    'MeanEventEnergy': 'CWT_anomaly_mean_energy',
                    'MaxEventEnergy': 'CWT_anomaly_max_energy'
                })
                cwt_features_list.append(event_stats_df)
                print(f"  Loaded anomalous events: {len(event_stats_df)} series")
            except Exception as e:
                print(f"  Error loading anomalous events: {e}")
    
    if cwt_features_list:
        cwt_features = cwt_features_list[0]
        for df in cwt_features_list[1:]:
            if 'Series_Name' in df.columns:
                cwt_features = cwt_features.merge(df, on='Series_Name', how='outer')
        
        print(f"Loaded CWT features for {len(cwt_features)} series")
        return cwt_features
    else:
        print("Warning: No CWT feature files found")
        return pd.DataFrame()

def load_wtc_features() -> pd.DataFrame:
    print("\n" + "="*80)
    print("Loading WTC Features")
    print("="*80)
    
    coherence_path = Path(DATA_PATHS['wtc_coherence_csv'])
    phase_path = Path(DATA_PATHS['wtc_phase_csv'])
    
    if not coherence_path.exists() or not phase_path.exists():
        print(f"Warning: WTC feature files not found")
        return pd.DataFrame()
    
    coherence_df = pd.read_csv(coherence_path)
    phase_df = pd.read_csv(phase_path)
    
    wtc_features = coherence_df.merge(phase_df, on='Variable_Pair', suffixes=('_coh', '_phase'))
    
    wtc_features['Series_Name'] = wtc_features['Variable_Pair'].str.split('_vs_').str[0]
    
    wtc_agg = wtc_features.groupby('Series_Name').agg({
        'mean_coherence': 'mean',
        'max_coherence': 'mean',
        'high_coherence_ratio': 'mean',
        'significant_coherence_ratio': 'mean',
        'mean_phase': 'mean',
        'std_phase': 'mean',
        'positive_phase_ratio': 'mean',
        'negative_phase_ratio': 'mean'
    }).reset_index()
    
    print(f"Loaded WTC features for {len(wtc_agg)} series")
    return wtc_agg

def load_co_circulation_periods() -> pd.DataFrame:
    co_circulation_path = Path(DATA_PATHS['co_circulation_csv'])
    if not co_circulation_path.exists():
        print(f"Warning: Co-circulation periods file not found: {co_circulation_path}")
        return pd.DataFrame()
    
    co_circulation_df = pd.read_csv(co_circulation_path)
    return co_circulation_df

def integrate_all_features(stl_features: pd.DataFrame,
                          afd_features: pd.DataFrame,
                          cwt_features: pd.DataFrame,
                          wtc_features: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "="*80)
    print("Integrating All Features")
    print("="*80)
    
    integrated = stl_features.copy() if not stl_features.empty else pd.DataFrame()
    
    if not afd_features.empty and 'Series_Name' in afd_features.columns:
        if integrated.empty:
            integrated = afd_features.copy()
        else:
            integrated = integrated.merge(afd_features, on='Series_Name', how='outer')
        print(f"Merged AFD features: {len(afd_features)} series")
    
    if not cwt_features.empty and 'Series_Name' in cwt_features.columns:
        if integrated.empty:
            integrated = cwt_features.copy()
        else:
            integrated = integrated.merge(cwt_features, on='Series_Name', how='outer')
        print(f"Merged CWT features: {len(cwt_features)} series")
    
    if not wtc_features.empty and 'Series_Name' in wtc_features.columns:
        if integrated.empty:
            integrated = wtc_features.copy()
        else:
            integrated = integrated.merge(wtc_features, on='Series_Name', how='outer')
        print(f"Merged WTC features: {len(wtc_features)} series")
    
    print(f"\nFeature integration complete:")
    print(f"  Total series: {len(integrated)}")
    print(f"  Total features: {len(integrated.columns) - 1}")
    
    if not integrated.empty:
        missing_counts = integrated.isna().sum()
        if missing_counts.sum() > 0:
            print(f"  Missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"    - {col}: {count} ({count/len(integrated)*100:.1f}%)")
    
    return integrated

def load_cwt_comparison_results() -> pd.DataFrame:
    print("\n" + "="*80)
    print("Loading CWT Comparison Results")
    print("="*80)
    
    cwt_comparison_dir = Path(DATA_PATHS['cwt_analysis_dir']) / 'comparison_analysis'
    
    if not cwt_comparison_dir.exists():
        print(f"Warning: CWT comparison directory not found: {cwt_comparison_dir}")
        return pd.DataFrame()
    
    feature_comparison_path = cwt_comparison_dir / 'feature_comparison.csv'
    if feature_comparison_path.exists():
        try:
            comparison_df = pd.read_csv(feature_comparison_path, encoding='utf-8-sig')
            print(f"Loaded CWT comparison results: {len(comparison_df)} rows")
            return comparison_df
        except Exception as e:
            print(f"Error loading CWT comparison results: {e}")
            return pd.DataFrame()
    else:
        print("Warning: CWT feature comparison file not found")
        return pd.DataFrame()

def identify_periods(time_index: pd.DatetimeIndex, 
                    co_circulation_df: pd.DataFrame) -> np.ndarray:
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

def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    mean1, mean2 = np.nanmean(group1), np.nanmean(group2)
    std1, std2 = np.nanstd(group1, ddof=1), np.nanstd(group2, ddof=1)
    
    pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return 0.0
    
    cohens_d = (mean1 - mean2) / pooled_std
    return cohens_d

def interpret_effect_size(cohens_d: float) -> str:
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return '小'
    elif abs_d < 0.5:
        return '中等'
    elif abs_d < 0.8:
        return '大'
    else:
        return '非常大'

def analyze_feature_importance(integrated_features: pd.DataFrame,
                              is_co_circulation: np.ndarray) -> pd.DataFrame:
    print("\n" + "="*80)
    print("Analyzing Feature Importance")
    print("="*80)
    
    feature_importance_list = []
    feature_cols = [col for col in integrated_features.columns if col != 'Series_Name']
    
    for feature in feature_cols:
        values = integrated_features[feature].dropna()
        if len(values) < 3:
            continue
        
        feature_importance_list.append({
            'Feature': feature,
            'Variance': float(values.var()),
            'Mean': float(values.mean()),
            'Std': float(values.std())
        })
    
    feature_importance_df = pd.DataFrame(feature_importance_list)
    if len(feature_importance_df) > 0:
        feature_importance_df = feature_importance_df.sort_values('Variance', ascending=False)
    
    print(f"Analyzed {len(feature_importance_df)} features")
    return feature_importance_df

def perform_pca(integrated_features: pd.DataFrame,
                n_components: int = None) -> Dict:
    print("\n" + "="*80)
    print("Performing PCA")
    print("="*80)
    
    if n_components is None:
        n_components = COMPREHENSIVE_CONFIG['n_components_pca']
    
    feature_cols = [col for col in integrated_features.columns if col != 'Series_Name']
    X = integrated_features[feature_cols].values
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    if COMPREHENSIVE_CONFIG['normalize_features']:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = X_imputed
    
    pca = PCA(n_components=n_components, random_state=COMPREHENSIVE_CONFIG['random_state'])
    X_pca = pca.fit_transform(X_scaled)
    
    pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df['Series_Name'] = integrated_features['Series_Name'].values
    
    results = {
        'pca_df': pca_df,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'components': pca.components_,
        'feature_names': feature_cols,
        'n_components': n_components
    }
    
    print(f"PCA completed: {n_components} components")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return results

def perform_clustering(integrated_features: pd.DataFrame,
                      n_clusters: int = None) -> Dict:
    print("\n" + "="*80)
    print("Performing Clustering")
    print("="*80)
    
    feature_cols = [col for col in integrated_features.columns if col != 'Series_Name']
    X = integrated_features[feature_cols].values
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    if COMPREHENSIVE_CONFIG['normalize_features']:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_imputed)
    else:
        X_scaled = X_imputed
    
    if n_clusters is None:
        pca = PCA(n_components=min(10, len(feature_cols)), random_state=COMPREHENSIVE_CONFIG['random_state'])
        X_pca = pca.fit_transform(X_scaled)
        n_clusters = 5
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=COMPREHENSIVE_CONFIG['random_state'], n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    silhouette_avg = silhouette_score(X_scaled, cluster_labels)
    calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)
    
    clustering_df = integrated_features[['Series_Name']].copy()
    clustering_df['Cluster'] = cluster_labels
    
    results = {
        'clustering_df': clustering_df,
        'cluster_labels': cluster_labels,
        'n_clusters': n_clusters,
        'silhouette_score': silhouette_avg,
        'calinski_harabasz_score': calinski_harabasz,
        'cluster_centers': kmeans.cluster_centers_
    }
    
    print(f"Clustering completed: {n_clusters} clusters")
    print(f"Silhouette score: {silhouette_avg:.4f}")
    print(f"Calinski-Harabasz score: {calinski_harabasz:.4f}")
    
    return results

def bootstrap_confidence_intervals(integrated_features: pd.DataFrame,
                                  feature_cols: List[str],
                                  n_bootstrap: int = None) -> pd.DataFrame:
    print("\n" + "="*80)
    print("Bootstrap Confidence Intervals")
    print("="*80)
    
    if n_bootstrap is None:
        n_bootstrap = COMPREHENSIVE_CONFIG['n_bootstrap']
    
    bootstrap_results = []
    
    for feature in feature_cols:
        if feature not in integrated_features.columns:
            continue
        
        values = integrated_features[feature].dropna().values
        if len(values) < 3:
            continue
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        ci_lower = np.percentile(bootstrap_means, 2.5)
        ci_upper = np.percentile(bootstrap_means, 97.5)
        mean_value = np.mean(values)
        
        bootstrap_results.append({
            'Feature': feature,
            'Mean': mean_value,
            'CI_Lower_2.5': ci_lower,
            'CI_Upper_97.5': ci_upper,
            'CI_Width': ci_upper - ci_lower
        })
    
    bootstrap_df = pd.DataFrame(bootstrap_results)
    print(f"Computed Bootstrap CIs for {len(bootstrap_df)} features")
    
    return bootstrap_df

def compare_periods(features_df: pd.DataFrame,
                   is_co_circulation: np.ndarray,
                   feature_cols: List[str]) -> pd.DataFrame:
    print("\n" + "="*80)
    print("Comparing Co-circulation vs Single-dominant Periods")
    print("="*80)
    
    if len(is_co_circulation) != len(features_df):
        print(f"Warning: Period identification length ({len(is_co_circulation)}) "
              f"does not match features length ({len(features_df)})")
        pass
    
    comparison_results = []
    
    for feature in feature_cols:
        if feature not in features_df.columns or feature == 'Series_Name':
            continue
        
        values = features_df[feature].values
        
        co_values = values
        single_values = values
        
        if len(co_values) == 0 or len(single_values) == 0:
            continue
        
        co_values = co_values[~np.isnan(co_values)]
        single_values = single_values[~np.isnan(single_values)]
        
        if len(co_values) < 3 or len(single_values) < 3:
            continue
        
        _, p_norm_co = shapiro(co_values) if len(co_values) <= 5000 else (0, 0.05)
        _, p_norm_single = shapiro(single_values) if len(single_values) <= 5000 else (0, 0.05)
        is_normal = p_norm_co > 0.05 and p_norm_single > 0.05
        
        _, p_levene = levene(co_values, single_values)
        equal_var = p_levene > 0.05
        
        if is_normal and equal_var:
            statistic, p_value = ttest_ind(co_values, single_values, equal_var=True)
            test_name = 't-test'
        elif is_normal:
            statistic, p_value = ttest_ind(co_values, single_values, equal_var=False)
            test_name = "Welch's t-test"
        else:
            statistic, p_value = mannwhitneyu(co_values, single_values, alternative='two-sided')
            test_name = 'Mann-Whitney U'
        
        cohens_d = compute_cohens_d(co_values, single_values)
        effect_interpretation = interpret_effect_size(cohens_d)
        
        comparison_results.append({
            'Feature': feature,
            'Co_Period_Mean': np.nanmean(co_values),
            'Single_Period_Mean': np.nanmean(single_values),
            'Difference': np.nanmean(co_values) - np.nanmean(single_values),
            'Test_Name': test_name,
            'Statistic': statistic,
            'P_Value': p_value,
            'Effect_Size_Cohen_D': cohens_d,
            'Effect_Size_Interpretation': effect_interpretation,
            'Is_Normal': is_normal,
            'Equal_Variance': equal_var
        })
    
    comparison_df = pd.DataFrame(comparison_results)
    
    if len(comparison_df) > 0:
        _, p_corrected, _, _ = multipletests(
            comparison_df['P_Value'].values,
            alpha=COMPREHENSIVE_CONFIG['alpha'],
            method=COMPREHENSIVE_CONFIG['fdr_method']
        )
        comparison_df['P_Value_FDR'] = p_corrected
        comparison_df['Significant_FDR'] = p_corrected < COMPREHENSIVE_CONFIG['alpha']
        
        significant_count = comparison_df['Significant_FDR'].sum()
        print(f"\nStatistical test results:")
        print(f"  Total features tested: {len(comparison_df)}")
        print(f"  Significant features (FDR corrected): {significant_count} ({significant_count/len(comparison_df)*100:.1f}%)")
    
    return comparison_df

def plot_integrated_features_heatmap(integrated_features: pd.DataFrame, output_path: Path):
    print("\n" + "="*80)
    print("Generating Integrated Features Heatmap")
    print("="*80)
    
    feature_cols = [col for col in integrated_features.columns if col != 'Series_Name']
    X = integrated_features[feature_cols].values
    
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    max_series = 100
    if len(integrated_features) > max_series:
        selected_indices = np.random.choice(len(integrated_features), max_series, replace=False)
        X_display = X_scaled[selected_indices, :]
        series_names_display = integrated_features['Series_Name'].iloc[selected_indices].values
    else:
        X_display = X_scaled
        series_names_display = integrated_features['Series_Name'].values
    
    im = ax.imshow(X_display, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    
    ax.set_xlabel('Features', fontsize=12)
    ax.set_ylabel('Series', fontsize=12)
    ax.set_title('Integrated Features Heatmap (Z-score normalized)', fontsize=14)
    
    ax.set_xticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=8)
    
    n_ticks = min(20, len(series_names_display))
    tick_indices = np.linspace(0, len(series_names_display)-1, n_ticks, dtype=int)
    ax.set_yticks(tick_indices)
    ax.set_yticklabels([series_names_display[i] for i in tick_indices], fontsize=8)
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Z-score', fontsize=10)
    
    plt.tight_layout()
    
    for fmt in VISUALIZATION_CONFIG['format']:
        save_path = output_path.with_suffix(f'.{fmt}')
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        print(f"  Saved heatmap: {save_path}")
    
    plt.close()
    
def plot_feature_importance(feature_importance_df: pd.DataFrame, output_path: Path):
    if feature_importance_df.empty:
        return
    
    print("\n" + "="*80)
    print("Generating Feature Importance Plot")
    print("="*80)
    
    top_features = feature_importance_df.head(20)
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['Variance'].values, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
    ax.set_xlabel('Variance', fontsize=12)
    ax.set_title('Top 20 Feature Importance (by Variance)', fontsize=14)
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    for fmt in VISUALIZATION_CONFIG['format']:
        save_path = output_path.with_suffix(f'.{fmt}')
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        print(f"  Saved feature importance plot: {save_path}")
    
    plt.close()

def plot_pca_scatter(pca_results: Dict, output_path: Path):
    if not pca_results or 'pca_df' not in pca_results:
        return
    
    print("\n" + "="*80)
    print("Generating PCA Scatter Plot")
    print("="*80)
    
    pca_df = pca_results['pca_df']
    
    fig, ax = plt.subplots(figsize=VISUALIZATION_CONFIG['figsize'])
    
    if 'PC1' in pca_df.columns and 'PC2' in pca_df.columns:
        ax.scatter(pca_df['PC1'].values, pca_df['PC2'].values, alpha=0.6, s=50)
        ax.set_xlabel(f"PC1 ({pca_results['explained_variance_ratio'][0]*100:.1f}% variance)", fontsize=12)
        ax.set_ylabel(f"PC2 ({pca_results['explained_variance_ratio'][1]*100:.1f}% variance)", fontsize=12)
        ax.set_title('PCA Scatter Plot (PC1 vs PC2)', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    for fmt in VISUALIZATION_CONFIG['format']:
        save_path = output_path.with_suffix(f'.{fmt}')
        plt.savefig(save_path, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
        print(f"  Saved PCA scatter plot: {save_path}")
    
    plt.close()

def generate_summary_report(integrated_features: pd.DataFrame,
                           feature_importance_df: pd.DataFrame,
                           pca_results: Dict,
                           clustering_results: Dict,
                           bootstrap_results: pd.DataFrame,
                           cwt_comparison_df: pd.DataFrame,
                           output_dir: Path):
    print("\n" + "="*80)
    print("Generating Summary Report")
    print("="*80)
    
    report_file = output_dir / 'comprehensive_report.txt'
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("Comprehensive Statistical Analysis Report\n")
        f.write("="*80 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("1. Data Summary\n")
        f.write("-"*80 + "\n")
        f.write(f"Total series: {len(integrated_features)}\n")
        f.write(f"Total features: {len(integrated_features.columns) - 1}\n\n")
        
        f.write("2. Feature Importance (Top 10)\n")
        f.write("-"*80 + "\n")
        if not feature_importance_df.empty:
            top_10 = feature_importance_df.head(10)
            for idx, row in top_10.iterrows():
                f.write(f"{row['Feature']}: Variance={row['Variance']:.6f}\n")
        f.write("\n")
        
        f.write("3. PCA Analysis\n")
        f.write("-"*80 + "\n")
        if pca_results:
            f.write(f"Number of components: {pca_results['n_components']}\n")
            f.write(f"Total explained variance: {pca_results['explained_variance_ratio'].sum()*100:.2f}%\n")
            f.write(f"PC1 explained variance: {pca_results['explained_variance_ratio'][0]*100:.2f}%\n")
            f.write(f"PC2 explained variance: {pca_results['explained_variance_ratio'][1]*100:.2f}%\n")
        f.write("\n")
        
        f.write("4. Clustering Analysis\n")
        f.write("-"*80 + "\n")
        if clustering_results:
            f.write(f"Number of clusters: {clustering_results['n_clusters']}\n")
            f.write(f"Silhouette score: {clustering_results['silhouette_score']:.4f}\n")
            f.write(f"Calinski-Harabasz score: {clustering_results['calinski_harabasz_score']:.4f}\n")
        f.write("\n")
        
        f.write("5. CWT Comparison Results\n")
        f.write("-"*80 + "\n")
        if not cwt_comparison_df.empty:
            if 'Significant_FDR' in cwt_comparison_df.columns:
                significant_count = cwt_comparison_df['Significant_FDR'].sum()
                f.write(f"Total comparisons: {len(cwt_comparison_df)}\n")
                f.write(f"Significant features (FDR corrected): {significant_count} ({significant_count/len(cwt_comparison_df)*100:.1f}%)\n")
        f.write("\n")
        
        f.write("="*80 + "\n")
    
    print(f"Saved summary report: {report_file}")

def main():
    print("="*80)
    print("Comprehensive Statistical Analysis")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    setup_matplotlib()
    output_dir = create_output_directory()
    
    stl_features = load_stl_features() if COMPREHENSIVE_CONFIG['enable_stl_features'] else pd.DataFrame()
    afd_features = load_afd_features() if COMPREHENSIVE_CONFIG['enable_afd_features'] else pd.DataFrame()
    cwt_features = load_cwt_features() if COMPREHENSIVE_CONFIG['enable_cwt_features'] else pd.DataFrame()
    wtc_features = load_wtc_features() if COMPREHENSIVE_CONFIG['enable_wtc_features'] else pd.DataFrame()
    
    integrated_features = integrate_all_features(stl_features, afd_features, cwt_features, wtc_features)
    
    if integrated_features.empty:
        print("Error: No features loaded. Exiting.")
        return
    
    print("\n" + "="*80)
    print("Step 1: Saving Integrated Features")
    print("="*80)
    integrated_features_file = output_dir / OUTPUT_CONFIG['integrated_features_dir'] / 'integrated_features.csv'
    integrated_features.to_csv(integrated_features_file, index=False, encoding='utf-8-sig')
    print(f"Saved integrated features: {integrated_features_file}")
    
    co_circulation_df = load_co_circulation_periods()
    
    cwt_comparison_df = load_cwt_comparison_results()
    if not cwt_comparison_df.empty:
        cwt_comparison_file = output_dir / OUTPUT_CONFIG['statistical_tests_dir'] / 'cwt_comparison_results.csv'
        cwt_comparison_df.to_csv(cwt_comparison_file, index=False, encoding='utf-8-sig')
        print(f"Saved CWT comparison results: {cwt_comparison_file}")
    
    print("\n" + "="*80)
    print("Step 2: Pattern Recognition")
    print("="*80)
    
    feature_cols = [col for col in integrated_features.columns if col != 'Series_Name']
    feature_importance_df = analyze_feature_importance(integrated_features, np.array([]))
    if not feature_importance_df.empty:
        feature_importance_file = output_dir / OUTPUT_CONFIG['pattern_analysis_dir'] / 'feature_importance.csv'
        feature_importance_df.to_csv(feature_importance_file, index=False, encoding='utf-8-sig')
        print(f"Saved feature importance: {feature_importance_file}")
    
    pca_results = perform_pca(integrated_features)
    if pca_results:
        pca_df_file = output_dir / OUTPUT_CONFIG['pattern_analysis_dir'] / 'pca_results.csv'
        pca_results['pca_df'].to_csv(pca_df_file, index=False, encoding='utf-8-sig')
        print(f"Saved PCA results: {pca_df_file}")
        
        pca_variance_df = pd.DataFrame({
            'Component': [f'PC{i+1}' for i in range(pca_results['n_components'])],
            'Explained_Variance_Ratio': pca_results['explained_variance_ratio'],
            'Cumulative_Variance_Ratio': np.cumsum(pca_results['explained_variance_ratio'])
        })
        pca_variance_file = output_dir / OUTPUT_CONFIG['pattern_analysis_dir'] / 'pca_variance.csv'
        pca_variance_df.to_csv(pca_variance_file, index=False, encoding='utf-8-sig')
        print(f"Saved PCA variance: {pca_variance_file}")
    
    clustering_results = perform_clustering(integrated_features)
    if clustering_results:
        clustering_df_file = output_dir / OUTPUT_CONFIG['pattern_analysis_dir'] / 'clustering_results.csv'
        clustering_results['clustering_df'].to_csv(clustering_df_file, index=False, encoding='utf-8-sig')
        print(f"Saved clustering results: {clustering_df_file}")
    
    print("\n" + "="*80)
    print("Step 3: Robustness Analysis")
    print("="*80)
    
    bootstrap_results = bootstrap_confidence_intervals(integrated_features, feature_cols)
    if not bootstrap_results.empty:
        bootstrap_file = output_dir / OUTPUT_CONFIG['robustness_analysis_dir'] / 'bootstrap_confidence_intervals.csv'
        bootstrap_results.to_csv(bootstrap_file, index=False, encoding='utf-8-sig')
        print(f"Saved Bootstrap CIs: {bootstrap_file}")
    
    if VISUALIZATION_CONFIG['enable_visualization']:
        print("\n" + "="*80)
        print("Step 4: Generating Visualizations")
        print("="*80)
        
        viz_dir = output_dir / OUTPUT_CONFIG['visualization_dir']
        viz_dir.mkdir(parents=True, exist_ok=True)
        
        heatmap_path = viz_dir / 'integrated_features_heatmap'
        plot_integrated_features_heatmap(integrated_features, heatmap_path)
        
        if not feature_importance_df.empty:
            importance_path = viz_dir / 'feature_importance'
            plot_feature_importance(feature_importance_df, importance_path)
        
        if pca_results:
            pca_scatter_path = viz_dir / 'pca_scatter'
            plot_pca_scatter(pca_results, pca_scatter_path)
    
    print("\n" + "="*80)
    print("Step 5: Generating Summary Report")
    print("="*80)
    
    generate_summary_report(integrated_features, feature_importance_df, pca_results, 
                          clustering_results, bootstrap_results, cwt_comparison_df, output_dir)
    
    print("\n" + "="*80)
    print("Comprehensive Analysis Completed")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()

