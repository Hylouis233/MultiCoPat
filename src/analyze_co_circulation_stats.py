
import pandas as pd
from scipy import stats
import numpy as np

df = pd.read_csv('co_circulation_periods.csv')

north = df[df['Region'] == 'Northern']
south = df[df['Region'] == 'Southern']

metrics = [
    {'col': 'Duration (weeks)', 'name': 'Duration (Weeks)'},
    {'col': 'Co-circulation Intensity Score', 'name': 'Intensity Score'},
    {'col': 'ILI Rate (%)', 'name': 'ILI Rate (\%)'},
    {'col': 'Flu Positive Rate (%)', 'name': 'Positive Rate (\%)'},
    {'col': 'Maximum Difference (%)', 'name': 'Strain Evenness (Max Diff \%)'} 
]

print(r"\begin{table}[H]")
print(r"\centering")
print(r"\caption{Comparison of Co-circulation Characteristics between Northern and Southern China (2011-2025)}")
print(r"\label{tab:co_circ_stats}")
print(r"\begin{tabular}{lccccc}")
print(r"\toprule")
print(r"Characteristic & Northern (n=18) & Southern (n=16) & Diff (S-N) & $U$ Statistic & $P$-value \\")
print(r"& Mean (SD) & Mean (SD) & & & \\")
print(r"\midrule")

for metric in metrics:
    col = metric['col']
    name = metric['name']
    
    data_n = north[col]
    data_s = south[col]
    
    mean_n = data_n.mean()
    std_n = data_n.std()
    
    mean_s = data_s.mean()
    std_s = data_s.std()
    
    diff = mean_s - mean_n
    
    res = stats.mannwhitneyu(data_s, data_n, alternative='two-sided')
    u_stat = res.statistic
    p_val = res.pvalue
    
    if p_val < 0.001:
        p_str = "$<0.001^{***}$"
    elif p_val < 0.01:
        p_str = f"{p_val:.3f}$^{{**}}$"
    elif p_val < 0.05:
        p_str = f"{p_val:.3f}$^{{*}}$"
    else:
        p_str = f"{p_val:.3f}"
        
    print(f"{name} & {mean_n:.2f} ({std_n:.2f}) & {mean_s:.2f} ({std_s:.2f}) & {diff:.2f} & {u_stat:.1f} & {p_str} \\\\")

print(r"\bottomrule")
print(r"\end{tabular}")
print(r"\end{table}")
