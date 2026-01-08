
import sys
import os
import numpy as np
import pandas as pd
from scipy import stats, signal
from pathlib import Path
import pickle

sys.path.append(os.getcwd())

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
    stl_results_npy = Path(STL_DATA_PATH['stl_results_npy'])
    stl_results_pkl = Path(STL_DATA_PATH['stl_results_pkl'])
    
    if not stl_results_npy.exists() or not stl_results_pkl.exists():
        raise FileNotFoundError("STL data not found")
        
    stl_3d_array = np.load(stl_results_npy)
    with open(stl_results_pkl, 'rb') as f:
        metadata = pickle.load(f)
        
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']
    
    trend_dict = {}
    seasonal_dict = {}
    residual_dict = {}
    
    for i, name in enumerate(series_names):
        trend_dict[name] = stl_3d_array[i, :, 0]
        seasonal_dict[name] = stl_3d_array[i, :, 1]
        residual_dict[name] = stl_3d_array[i, :, 2]
        
    return trend_dict, seasonal_dict, residual_dict, time_index, series_names

def load_afd_data():
    residual_components_npy = Path(AFD_DATA_PATH['residual_components_npy'])
    if not residual_components_npy.exists():
        raise FileNotFoundError("AFD data not found")
    return np.load(residual_components_npy)

def get_frequency_stability(seasonal_signal):
    analytic_signal = signal.hilbert(seasonal_signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi))
    return np.std(instantaneous_frequency)

def get_energy_entropy(afd_comps):
    energies = np.sum(afd_comps**2, axis=0)
    
    if np.sum(energies) == 0:
        return 0
        
    probs = energies / np.sum(energies)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return entropy

def get_high_freq_ratio(residual, trend, seasonal):
    total_signal = trend + seasonal + residual
    e_total = np.sum(total_signal**2)
    e_res = np.sum(residual**2)
    
    if e_total < 1e-10:
        return np.nan
    
    ratio = e_res / e_total
    return min(1.0, ratio)

def get_seasonal_strength(seasonal, residual):
    var_r = np.var(residual)
    var_sr = np.var(seasonal + residual)
    if var_sr == 0:
        return 0
    return max(0, 1 - var_r / var_sr)

def get_trend_change_rate(trend):
    # Normalized Mean Absolute Derivative
    # Rate of change relative to the signal's magnitude
    diff = np.diff(trend)
    mean_val = np.mean(np.abs(trend))
    if mean_val == 0:
        return 0
    # Return as percentage change per step
    return (np.mean(np.abs(diff)) / mean_val) * 100

def main():
    T, S, R, time_index, series_names = load_stl_data()
    afd_comps = load_afd_data()
    
    df_cocirc = pd.read_csv(CO_CIRCULATION_PATH)
    df_cocirc['Adjusted Start Date'] = pd.to_datetime(df_cocirc['Adjusted Start Date'])
    df_cocirc['Adjusted End Date'] = pd.to_datetime(df_cocirc['Adjusted End Date'])
    
    results = []
    
    for series_idx, series_name in enumerate(series_names):
        if 'North' in series_name or '北方' in series_name:
            region = 'Northern'
        elif 'South' in series_name or '南方' in series_name:
            region = 'Southern'
        else:
            continue
            
        t_series = T[series_name]
        s_series = S[series_name]
        r_series = R[series_name]
        afd_series = afd_comps[series_idx]
        
        cocirc_periods = df_cocirc[df_cocirc['Region'] == region]
        
        is_cocirc = np.zeros(len(time_index), dtype=bool)
        
        for _, row in cocirc_periods.iterrows():
            mask = (time_index >= row['Adjusted Start Date']) & (time_index <= row['Adjusted End Date'])
            is_cocirc[mask] = True
            
            period_indices = np.where(mask)[0]
            if len(period_indices) < 2: continue
            
            p_s = s_series[period_indices]
            p_r = r_series[period_indices]
            p_t = t_series[period_indices]
            p_afd = afd_series[period_indices]
            
            stats_dict = {
                'Type': 'Co-circ',
                'Energy Entropy': get_energy_entropy(p_afd),
                'Intensity of Irregular Fluctuations': get_high_freq_ratio(p_r, p_t, p_s),
                'Frequency Stability': get_frequency_stability(p_s),
                'Seasonal Strength': get_seasonal_strength(p_s, p_r),
                'Trend Change Rate': get_trend_change_rate(p_t)
            }
            results.append(stats_dict)
            
        from scipy.ndimage import label
        labeled, n_features = label(~is_cocirc)
        
        for i in range(1, n_features+1):
            period_indices = np.where(labeled == i)[0]
            if len(period_indices) < 4: continue
            
            p_s = s_series[period_indices]
            p_r = r_series[period_indices]
            p_t = t_series[period_indices]
            p_afd = afd_series[period_indices]
            
            stats_dict = {
                'Type': 'Single',
                'Energy Entropy': get_energy_entropy(p_afd),
                'Intensity of Irregular Fluctuations': get_high_freq_ratio(p_r, p_t, p_s),
                'Frequency Stability': get_frequency_stability(p_s),
                'Seasonal Strength': get_seasonal_strength(p_s, p_r),
                'Trend Change Rate': get_trend_change_rate(p_t)
            }
            results.append(stats_dict)

    # Aggregate and Stats
    df_res = pd.DataFrame(results)
    
    features = ['Energy Entropy', 'Intensity of Irregular Fluctuations', 'Frequency Stability', 'Seasonal Strength', 'Trend Change Rate']
    
    print(r"\begin{table}[H]")
    print(r"\centering")
    print(r"\caption{Detailed comparison of dynamic features between single-dominant and co-circulation periods. This table details the statistical differences in time-frequency features between the two types of periods. Note the significant increase in frequency instability (Frequency Instability) and significant decrease in seasonal strength (Seasonal Strength) during co-circulation periods, which quantifies the 'disordered' nature of co-circulation events.}")
    print(r"\label{tab:supp_feature_stats}")
    print(r"\begin{tabular}{lcccccc}")
    print(r"\toprule")
    print(r"Feature & Single-dominant Mean (SD) & Co-circulation Mean (SD) & Difference ($\delta$) & $U$ Statistic & $P$-value \\")
    print(r"\midrule")
    
    for feat in features:
        single = df_res[df_res['Type'] == 'Single'][feat].dropna()
        cocirc = df_res[df_res['Type'] == 'Co-circ'][feat].dropna()
        
        mean_s = single.mean()
        std_s = single.std()
        mean_c = cocirc.mean()
        std_c = cocirc.std()
        
        res = stats.mannwhitneyu(cocirc, single, alternative='two-sided')
        u_stat = res.statistic
        p_val = res.pvalue
        
        # Calculate Cliff's Delta
        n1, n2 = len(cocirc), len(single)
        # U is usually U_1 (for cocirc).
        # Cliff's Delta = (2U - n1n2) / (n1n2)
        cliffs_delta = (2 * u_stat - n1 * n2) / (n1 * n2)
        
        if p_val < 0.001:
            p_str = "$<0.001^{***}$"
        elif p_val < 0.01:
            p_str = f"{p_val:.3f}$^{{**}}$"
        elif p_val < 0.05:
            p_str = f"{p_val:.3f}$^{{*}}$"
        else:
            p_str = f"{p_val:.3f}"
            
        print(f"{feat} & {mean_s:.3f} ({std_s:.3f}) & {mean_c:.3f} ({std_c:.3f}) & {cliffs_delta:.3f} & {u_stat:.1f} & {p_str} \\\\")
        
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == '__main__':
    main()
