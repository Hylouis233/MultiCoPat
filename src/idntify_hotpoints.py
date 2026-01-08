from data_portal import UnifiedDataPortal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import os
import matplotlib.dates as mdates

plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

portal = UnifiedDataPortal()
flu_data = portal.load_flu_data()

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

def standardize_column_names(data):
    rename_dict = {}
    for chinese_col, english_col in COLUMN_MAPPING.items():
        if chinese_col in data.columns:
            rename_dict[chinese_col] = english_col
    
    data_renamed = data.rename(columns=rename_dict)
    
    print(f"Column name standardization completed, converted {len(rename_dict)} column names")
    return data_renamed

# 标准化列名
flu_data = standardize_column_names(flu_data)

# 添加一个转换日期格式的辅助函数
def convert_date_format(date_str):
    if isinstance(date_str, datetime):
        return date_str
    
    if isinstance(date_str, str):
        formats = ['%Y/%m/%d', '%Y-%m-%d']
        for fmt in formats:
            from datetime import datetime as dt
            result = dt.strptime(date_str, fmt)
            return result
    
    return date_str

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

def identify_co_circulation_periods(data):
    ili_cols = ['northern_ili_rate', 'southern_ili_rate']
    pos_rate_cols = ['northern_flu_positive_rate', 'southern_flu_positive_rate']
    
    north_virus_cols = [
        'northern_h3n2',
        'northern_h1n1',
        'northern_h7n9',
        'northern_victoria',
        'northern_yamagata',
        'northern_type_a_untyped',
        'northern_type_b_untyped'
    ]
    
    south_virus_cols = [
        'southern_h3n2',
        'southern_h1n1',
        'southern_h7n9',
        'southern_victoria',
        'southern_yamagata',
        'southern_type_a_untyped',
        'southern_type_b_untyped'
    ]
    
    print("Calculating virus typing proportions...")
    
    north_total = data['northern_positive_count']
    for col in north_virus_cols:
        prop_col = col.replace('northern_', 'northern_') + '_proportion'
        data[prop_col] = (data[col] / north_total * 100).fillna(0)
    
    south_total = data['southern_positive_count']
    for col in south_virus_cols:
        prop_col = col.replace('southern_', 'southern_') + '_proportion'
        data[prop_col] = (data[col] / south_total * 100).fillna(0)
    
    ili_threshold_north = data['northern_ili_rate'].quantile(0.60)
    ili_threshold_south = data['southern_ili_rate'].quantile(0.60)
    pos_threshold_north = data['northern_flu_positive_rate'].quantile(0.60)
    pos_threshold_south = data['southern_flu_positive_rate'].quantile(0.60)
    
    print(f"High circulation thresholds:")
    print(f"  Northern ILI: {ili_threshold_north:.2f}%")
    print(f"  Southern ILI: {ili_threshold_south:.2f}%")
    print(f"  Northern flu positive rate: {pos_threshold_north:.2f}%")
    print(f"  Southern flu positive rate: {pos_threshold_south:.2f}%")
    
    high_circulation_mask = (
        (data['northern_ili_rate'] >= ili_threshold_north) |
        (data['southern_ili_rate'] >= ili_threshold_south)
    ) & (
        (data['northern_flu_positive_rate'] >= pos_threshold_north) |
        (data['southern_flu_positive_rate'] >= pos_threshold_south)
    )
    
    co_circulation_periods = []
    
    virus_prop_cols = [col for col in data.columns if '_proportion' in col]
    
    high_circulation_indices = data[high_circulation_mask].index.tolist()
    
    regions = ['北方', '南方']
    region_mapping = {'北方': 'northern', '南方': 'southern'}
    
    for region in regions:
        region_prefix = region_mapping[region]
        region_cols = [col for col in virus_prop_cols if col.startswith(region_prefix)]
        
        print(f"\nAnalyzing {region} region:")
        print(f"  Found {len(region_cols)} virus strain proportion columns: {region_cols}")
        
        current_co_circulation = None
        
        for idx in high_circulation_indices:
            row = data.loc[idx]
            proportions = [row[col] for col in region_cols if row[col] > 5]
            
            print(f"  Row {idx} (date: {row['start_date']}): found {len(proportions)} valid virus strains")
            if len(proportions) >= 2:
                proportions.sort(reverse=True)
                max_diff = proportions[0] - proportions[1] if len(proportions) >= 2 else 0
                
                print(f"    Virus strain proportions: {[f'{p:.1f}%' for p in proportions]}")
                print(f"    Maximum difference: {max_diff:.1f}%")
                
                if max_diff <= 40 and len(proportions) >= 2:
                    print(f"    ✓ Meets co-circulation condition!")
                    if current_co_circulation is None:
                        start_date_adjusted = row['start_date'] - timedelta(days=7) if isinstance(row['start_date'], datetime) else row['start_date']
                        
                        current_co_circulation = {
                            'start_date': start_date_adjusted,
                            'end_date': row['start_date'],
                            'actual_start_date': row['start_date'],
                            'region': region,
                            'ili_rate': row[f'northern_ili_rate' if region == '北方' else 'southern_ili_rate'],
                            'pos_rate': row[f'northern_flu_positive_rate' if region == '北方' else 'southern_flu_positive_rate'],
                            'virus_proportions': {col.replace(f'{region}', '').replace('_比例', ''): row[col] 
                                                for col in region_cols if row[col] > 5},
                            'max_diff': max_diff,
                            'co_circulation_score': len(proportions) * (30 - max_diff) / 30,
                            'duration_weeks': 1
                        }
                    else:
                        current_co_circulation['end_date'] = row['start_date']
                        current_co_circulation['duration_weeks'] += 1
                        current_co_circulation['virus_proportions'] = {col.replace(f'{region}', '').replace('_比例', ''): row[col] 
                                                                     for col in region_cols if row[col] > 5}
                        current_co_circulation['max_diff'] = max_diff
                        current_co_circulation['co_circulation_score'] = len(proportions) * (30 - max_diff) / 30
                        current_co_circulation['ili_rate'] = row[f'northern_ili_rate' if region == '北方' else 'southern_ili_rate']
                        current_co_circulation['pos_rate'] = row[f'northern_flu_positive_rate' if region == '北方' else 'southern_flu_positive_rate']
                else:
                    print(f"    ✗ Does not meet co-circulation condition (difference > {40}%)")
                    if current_co_circulation is not None:
                        end_date_adjusted = current_co_circulation['end_date'] + timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
                        current_co_circulation['end_date'] = end_date_adjusted
                        current_co_circulation['actual_end_date'] = current_co_circulation['end_date'] - timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
                        co_circulation_periods.append(current_co_circulation)
                        current_co_circulation = None
            else:
                print(f"    ✗ Insufficient virus strains (only {len(proportions)} found)")
                if current_co_circulation is not None:
                    end_date_adjusted = current_co_circulation['end_date'] + timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
                    current_co_circulation['end_date'] = end_date_adjusted
                    current_co_circulation['actual_end_date'] = current_co_circulation['end_date'] - timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
                    co_circulation_periods.append(current_co_circulation)
                    current_co_circulation = None
        
        if current_co_circulation is not None:
            end_date_adjusted = current_co_circulation['end_date'] + timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
            current_co_circulation['end_date'] = end_date_adjusted
            current_co_circulation['actual_end_date'] = current_co_circulation['end_date'] - timedelta(days=7) if isinstance(current_co_circulation['end_date'], datetime) else current_co_circulation['end_date']
            co_circulation_periods.append(current_co_circulation)
    
    return co_circulation_periods, data

def analyze_and_visualize_co_circulation(data, co_circulation_periods):
    log_file_path = "./co_circulation_analysis_log.txt"
    
    log_content = []
    log_content.append(f"Multi-strain Co-circulation Analysis Results Log")
    log_content.append(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_content.append("="*80)
    log_content.append(f"\nFound {len(co_circulation_periods)} potential co-circulation periods:")
    log_content.append("="*80)
    
    print(f"\nFound {len(co_circulation_periods)} potential co-circulation periods:")
    print("="*80)
    
    csv_data = []
    
    for i, period in enumerate(co_circulation_periods, 1):
        period_info = []
        region_name = 'Northern' if period['region'] == '北方' else 'Southern'
        
        period_info.append(f"\nCo-circulation Period {i}:")
        period_info.append(f"  Adjusted Start Date: {period['start_date']}")
        period_info.append(f"  Adjusted End Date: {period['end_date']}")
        if 'actual_start_date' in period:
            period_info.append(f"  Actual Start Date: {period['actual_start_date']}")
        if 'actual_end_date' in period:
            period_info.append(f"  Actual End Date: {period['actual_end_date']}")
        period_info.append(f"  Duration: {period['duration_weeks']} weeks")
        period_info.append(f"  Region: {region_name}")
        period_info.append(f"  ILI Rate: {period['ili_rate']:.2f}%")
        period_info.append(f"  Flu Positive Rate: {period['pos_rate']:.2f}%")
        period_info.append(f"  Virus Strain Proportions:")
        
        virus_proportions_str = "; ".join([f"{virus}: {prop:.2f}%" for virus, prop in period['virus_proportions'].items()])
        
        virus_translation = {
            'A未分系': 'Type A Untyped',
            'B未分系': 'Type B Untyped',
            '甲型 H1N1': 'A(H1N1)'
        }
        
        for virus, prop in period['virus_proportions'].items():
            display_name = virus_translation.get(virus, virus)
            period_info.append(f"    {display_name}: {prop:.2f}%")
        
        period_info.append(f"  Maximum Difference: {period['max_diff']:.2f}%")
        period_info.append(f"  Co-circulation Intensity Score: {period['co_circulation_score']:.2f}")
        period_info.append("-" * 40)
        
        for line in period_info:
            print(line)
            log_content.append(line)
        
        csv_row = {
            'Co-circulation Period #': i,
            'Adjusted Start Date': period['start_date'],
            'Adjusted End Date': period['end_date'],
            'Actual Start Date': period.get('actual_start_date', period['start_date']),
            'Actual End Date': period.get('actual_end_date', period['end_date']),
            'Duration (weeks)': period['duration_weeks'],
            'Region': region_name,
            'ILI Rate (%)': round(period['ili_rate'], 2),
            'Flu Positive Rate (%)': round(period['pos_rate'], 2),
            'Virus Proportions': virus_proportions_str,
            'Maximum Difference (%)': round(period['max_diff'], 2),
            'Co-circulation Intensity Score': round(period['co_circulation_score'], 2),
            'Number of Co-circulating Strains': len(period['virus_proportions'])
        }
        
        for virus, prop in period['virus_proportions'].items():
            display_name = virus_translation.get(virus, virus)
            csv_row[f'{display_name} (%)'] = round(prop, 2)
        
        csv_data.append(csv_row)
    
    if csv_data:
        csv_file_path = "./co_circulation_periods.csv"
        df_csv = pd.DataFrame(csv_data)
        df_csv.to_csv(csv_file_path, index=False, encoding='utf-8-sig')
        print(f"\nCo-circulation period data saved to CSV file: {csv_file_path}")
        log_content.append(f"\nCo-circulation period data saved to CSV file: {csv_file_path}")
    
    if co_circulation_periods:
        summary = []
        summary.append("\nAnalysis Summary:")
        summary.append("="*40)
        summary.append(f"Total co-circulation periods found: {len(co_circulation_periods)}")
        
        north_count = len([p for p in co_circulation_periods if p['region'] == '北方'])
        south_count = len([p for p in co_circulation_periods if p['region'] == '南方'])
        summary.append(f"Northern co-circulation periods: {north_count}")
        summary.append(f"Southern co-circulation periods: {south_count}")
        
        avg_duration = np.mean([p['duration_weeks'] for p in co_circulation_periods])
        summary.append(f"Average duration: {avg_duration:.1f} weeks")
        
        max_duration_period = max(co_circulation_periods, key=lambda x: x['duration_weeks'])
        region_name = 'Northern' if max_duration_period['region'] == '北方' else 'Southern'
        summary.append(f"Longest duration period: {max_duration_period['start_date']} to {max_duration_period['end_date']} ({region_name}) - {max_duration_period['duration_weeks']} weeks")
        
        avg_score = np.mean([p['co_circulation_score'] for p in co_circulation_periods])
        summary.append(f"Average co-circulation intensity score: {avg_score:.2f}")
        
        max_score_period = max(co_circulation_periods, key=lambda x: x['co_circulation_score'])
        region_name = 'Northern' if max_score_period['region'] == '北方' else 'Southern'
        summary.append(f"Highest intensity co-circulation period: {max_score_period['start_date']} to {max_score_period['end_date']} ({region_name}) - Score: {max_score_period['co_circulation_score']:.2f}")
        
        for line in summary:
            print(line)
            log_content.append(line)
    
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(log_content))
    print(f"\nAnalysis results saved to log file: {log_file_path}")
    
    create_co_circulation_plots(data, co_circulation_periods)

def create_co_circulation_plots(data, co_circulation_periods):
    output_dir = "./figures/idntify_hotpoints"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nDiagnostic information:")
    print("All column names in dataframe:", data.columns.tolist())
    
    fig, axes = plt.subplots(2, 2, figsize=(24, 12))
    
    x_axis_dates = []
    for date in data['start_date']:
        x_axis_dates.append(convert_date_format(date))
    
    x_axis = pd.to_datetime(x_axis_dates)
    x_label = 'Date'
    
    axes[0, 0].plot(x_axis, data['northern_ili_rate'], label='Northern', linewidth=2)
    axes[0, 0].plot(x_axis, data['southern_ili_rate'], label='Southern', linewidth=2)
    axes[0, 0].set_title('ILI% Trends')
    axes[0, 0].set_ylabel('ILI%')
    axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 0].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    axes[0, 1].plot(x_axis, data['northern_flu_positive_rate'], label='Northern', linewidth=2)
    axes[0, 1].plot(x_axis, data['southern_flu_positive_rate'], label='Southern', linewidth=2)
    axes[0, 1].set_title('Flu Positive Rate Trends')
    axes[0, 1].set_ylabel('Positive Rate %')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[0, 1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    virus_types = ['h3n2', 'h1n1', 'h7n9', 'victoria', 'yamagata', 'type_a_untyped', 'type_b_untyped']
    display_names = ['A(H3N2)', 'A(H1N1)', 'A(H7N9)', 'Victoria', 'Yamagata', 'Type A Untyped', 'Type B Untyped']
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'darkred', 'darkblue']

    print("\nNorthern virus strain column check:")
    for virus in virus_types:
        col_name = f'northern_{virus}_proportion'
        exists = col_name in data.columns
        print(f"Column '{col_name}' exists: {exists}")

    for virus, display_name, color in zip(virus_types, display_names, colors):
        col_name = f'northern_{virus}_proportion'
        if col_name in data.columns:
            axes[1, 0].plot(x_axis, data[col_name], label=display_name, color=color, linewidth=2)
    
    axes[1, 0].set_title('Northern Major Virus Strain Proportions')
    axes[1, 0].set_ylabel('Proportion %')
    axes[1, 0].set_xlabel(x_label)
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1, 0].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    print("\nSouthern virus strain column check:")
    for virus in virus_types:
        col_name = f'southern_{virus}_proportion'
        exists = col_name in data.columns
        print(f"Column '{col_name}' exists: {exists}")
        
    for virus, display_name, color in zip(virus_types, display_names, colors):
        col_name = f'southern_{virus}_proportion'
        if col_name in data.columns:
            axes[1, 1].plot(x_axis, data[col_name], label=display_name, color=color, linewidth=2)
    
    axes[1, 1].set_title('Southern Major Virus Strain Proportions')
    axes[1, 1].set_ylabel('Proportion %')
    axes[1, 1].set_xlabel(x_label)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1, 1].xaxis.set_major_locator(mdates.YearLocator())
    plt.setp(axes[1, 1].xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=9)
    
    for period in co_circulation_periods:
        x_pos_start = pd.to_datetime(convert_date_format(period['start_date']))
        x_pos_end = pd.to_datetime(convert_date_format(period['end_date']))
        
        if period['region'] == '北方':
            axes[1, 0].axvspan(x_pos_start, x_pos_end, color='red', alpha=0.3, label='Co-circulation Period' if period == co_circulation_periods[0] and period['region'] == '北方' else "")
            axes[1, 0].axvline(x=x_pos_start, color='red', linestyle='--', alpha=0.7, linewidth=2)
        else:
            axes[1, 1].axvspan(x_pos_start, x_pos_end, color='red', alpha=0.3, label='Co-circulation Period' if period == co_circulation_periods[0] and period['region'] == '南方' else "")
            axes[1, 1].axvline(x=x_pos_start, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    plt.tight_layout()
    plt.subplots_adjust(right=0.85, wspace=0.3)
    plt.savefig(f'{output_dir}/co_circulation_period_analysis.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/co_circulation_period_analysis.pdf', dpi=600, bbox_inches='tight')
    plt.show()
    
    if co_circulation_periods:
        create_co_circulation_heatmap(co_circulation_periods, output_dir)

def create_co_circulation_heatmap(co_circulation_periods, output_dir):
    if not co_circulation_periods:
        print("No co-circulation periods found, skipping heatmap creation.")
        return
    
    print(f"Creating heatmap with {len(co_circulation_periods)} co-circulation periods...")
    
    data = []
    for period in co_circulation_periods:
        start_date = convert_date_format(period['start_date'])
        region = 'Northern' if period['region'] == '北方' else 'Southern'
        data.append({
            'Region': region,
            'Start_Date': start_date,
            'Intensity': period['co_circulation_score']
        })
    
    df = pd.DataFrame(data)
    print(f"Heatmap data preview:")
    print(df.head())
    
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], errors='coerce')
    df = df.dropna(subset=['Start_Date'])
    
    df['Year'] = df['Start_Date'].dt.year
    df['Month'] = df['Start_Date'].dt.month
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['Month_Name'] = df['Month'].map(month_names)
    
    print(f"Year and Month columns created:")
    print(df[['Region', 'Year', 'Month_Name', 'Intensity']].head())
    
    pivot = df.pivot_table(values='Intensity', index='Month_Name', columns='Year', fill_value=0)
    
    month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot = pivot.reindex([month for month in month_order if month in pivot.index])
    
    print(f"Pivot table created (Months x Years):")
    print(pivot)
    
    if pivot.empty:
        print("Pivot table is empty, cannot create heatmap")
        return
    
    regions = df['Region'].unique()
    
    if len(regions) > 1:
        fig, axes = plt.subplots(1, len(regions), figsize=(8 * len(regions), 8))
        if len(regions) == 1:
            axes = [axes]
        
        for i, region in enumerate(regions):
            region_data = df[df['Region'] == region]
            region_pivot = region_data.pivot_table(values='Intensity', index='Month_Name', columns='Year', fill_value=0)
            
            region_pivot = region_pivot.reindex([month for month in month_order if month in region_pivot.index])
            
            sns.heatmap(region_pivot, annot=True, cmap='YlOrRd', fmt='.1f', 
                       cbar_kws={'label': 'Co-circulation Intensity Score'}, ax=axes[i])
            axes[i].set_title(f'{region} Region Co-circulation Intensity', fontsize=12, fontweight='bold')
            axes[i].set_xlabel('Year')
            axes[i].set_ylabel('Month')
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].tick_params(axis='y', rotation=0)
        
        plt.suptitle('Multi-strain Co-circulation Intensity Heatmap (Monthly by Year)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
    else:
        plt.figure(figsize=(16, 8))
        sns.heatmap(pivot, annot=True, cmap='YlOrRd', fmt='.1f', 
                    cbar_kws={'label': 'Co-circulation Intensity Score'})
        plt.title('Multi-strain Co-circulation Intensity Heatmap (Monthly by Year)', fontsize=14, fontweight='bold')
        plt.xlabel('Year')
        plt.ylabel('Month')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
    
    plt.savefig(f'{output_dir}/co_circulation_intensity_heatmap.png', dpi=600, bbox_inches='tight')
    plt.savefig(f'{output_dir}/co_circulation_intensity_heatmap.pdf', dpi=600, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    co_circulation_periods, enhanced_data = identify_co_circulation_periods(flu_data)
    
    analyze_and_visualize_co_circulation(enhanced_data, co_circulation_periods)