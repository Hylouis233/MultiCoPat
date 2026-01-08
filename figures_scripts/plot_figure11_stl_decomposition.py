import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import load_stl_results, save_figure, setup_plotting_style

def plot_stl_decomposition(ax_list, time_index, stl_data, title):
    trend = stl_data[:, 0]
    seasonal = stl_data[:, 1]
    residual = stl_data[:, 2]
    original = trend + seasonal + residual
    
    components = [original, trend, seasonal, residual]
    labels = ['Original', 'Trend', 'Seasonal', 'Residual']
    colors = ['black', '#d62728', '#1f77b4', 'gray']
    
    for ax, data, label, color in zip(ax_list, components, labels, colors):
        ax.plot(time_index, data, color=color, lw=1.2)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
        
        if label in ['Seasonal', 'Residual']:
            ax.axhline(0, color='black', lw=0.5, linestyle='--')
            
    ax_list[0].set_title(title, fontsize=14, fontweight='bold')
    
    import matplotlib.dates as mdates
    ax_list[-1].xaxis.set_major_locator(mdates.YearLocator(2))
    ax_list[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

def plot_figure11():
    print("Generating Figure 11 (STL Decomposition)...")
    
    stl_array, metadata = load_stl_results()
    time_index = pd.to_datetime(metadata['time_index'])
    series_names = metadata['series_names']

    configs = [
        {
            'filename': 'figure11_stl_north',
            'target': 'ILI%北方',
            'title': '(a) Northern China: STL Decomposition (ILI%)'
        },
        {
            'filename': 'figure11_stl_south',
            'target': 'ILI%南方',
            'title': '(b) Southern China: STL Decomposition (ILI%)'
        }
    ]
    
    for config in configs:
        target_name = config['target']
        if target_name not in series_names:
            print(f"Warning: Series '{target_name}' not found, skipping...")
            continue
            
        idx = series_names.index(target_name)
        stl_data = stl_array[idx, :, :]
        
        fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        plot_stl_decomposition(axes, time_index, stl_data, config['title'])
        
        plt.tight_layout()
        save_figure(fig, config['filename'])
        plt.close(fig)

if __name__ == "__main__":
    setup_plotting_style()
    plot_figure11()
