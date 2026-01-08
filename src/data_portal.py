import os
import glob
import pickle
import logging
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UnifiedDataPortal:
    
    def __init__(self, data_root: str = "."):
        self.data_root = Path(data_root)
        
        self.flu_data = None
        self.baidu_search_data = None
        self.migration_data = {}
        
        self.co_circulation_periods = None
        self.period_classification = None
        self.spectral_features = None
        self.afd_components_array = None
        self.afd_metadata = None
        self.integrated_features = None
        
        self.data_loaded = {
            'flu_data': False,
            'baidu_search': False,
            'migration_data': False,
            'co_circulation_periods': False,
            'spectral_features': False,
            'afd_components': False
        }
        
        self.province_mapping = {
            '110000': 'Beijing', '120000': 'Tianjin', '130000': 'Hebei',
            '140000': 'Shanxi', '150000': 'Inner Mongolia', '210000': 'Liaoning',
            '220000': 'Jilin', '230000': 'Heilongjiang', '310000': 'Shanghai',
            '320000': 'Jiangsu', '330000': 'Zhejiang', '340000': 'Anhui',
            '350000': 'Fujian', '360000': 'Jiangxi', '370000': 'Shandong',
            '410000': 'Henan', '420000': 'Hubei', '430000': 'Hunan',
            '440000': 'Guangdong', '450000': 'Guangxi', '460000': 'Hainan',
            '500000': 'Chongqing', '510000': 'Sichuan', '520000': 'Guizhou',
            '530000': 'Yunnan', '540000': 'Tibet', '610000': 'Shaanxi',
            '620000': 'Gansu', '630000': 'Qinghai', '640000': 'Ningxia',
            '650000': 'Xinjiang', '810000': 'Hong Kong', '820000': 'Macau'
        }
        
        self.province_code_mapping = {v: k for k, v in self.province_mapping.items()}
        
        self.lag_periods = [1, 2, 3, 4, 8, 12]
        
        logger.info("Unified data portal initialized")
    
    def load_flu_data(self, file_path: str = "fludata_ts_20250922.csv") -> pd.DataFrame:
        file_path = self.data_root / file_path
        logger.info(f"Loading flu data: {file_path}")
        
        self.flu_data = pd.read_csv(file_path, encoding='utf-8-sig')
        
        self.flu_data['start_date'] = pd.to_datetime(self.flu_data['start_date'])
        self.flu_data['stop_date'] = pd.to_datetime(self.flu_data['stop_date'])
        self.flu_data['year_week'] = (self.flu_data['year'].astype(str) + '_' + 
                                    self.flu_data['week'].astype(str).str.zfill(2))
        
        self.data_loaded['flu_data'] = True
        logger.info(f"Flu data loaded: {self.flu_data.shape}")
        logger.info(f"Time range: {self.flu_data['start_date'].min()} to {self.flu_data['stop_date'].max()}")
        
        return self.flu_data
    
    def load_baidu_search_data(self, file_path: str = "flu_baidu_index_weekly.csv") -> pd.DataFrame:
        file_path = self.data_root / file_path
        logger.info(f"Loading Baidu search index data: {file_path}")
        
        self.baidu_search_data = pd.read_csv(file_path, encoding='utf-8-sig')
        
        self.baidu_search_data['date'] = pd.to_datetime(self.baidu_search_data['date'], errors='coerce')
        self.baidu_search_data['year'] = self.baidu_search_data['date'].dt.year
        self.baidu_search_data['week'] = self.baidu_search_data['date'].dt.isocalendar().week
        self.baidu_search_data['year_week'] = (self.baidu_search_data['year'].astype(str) + '_' + 
                                             self.baidu_search_data['week'].astype(str).str.zfill(2))
        
        self.data_loaded['baidu_search'] = True
        logger.info(f"Baidu search index data loaded: {self.baidu_search_data.shape}")
        logger.info(f"Time range: {self.baidu_search_data['date'].min()} to {self.baidu_search_data['date'].max()}")
        
        return self.baidu_search_data
    
    def load_migration_data(self, 
                          data_dir: str = "baidu_move",
                          load_province: bool = True,
                          load_city: bool = True,
                          sample_size: Optional[int] = None) -> Dict:
        data_dir = self.data_root / data_dir
        logger.info(f"Loading Baidu migration data: {data_dir}")
        
        migration_data = {
            'province': {},
            'city': {}
        }
        
        if load_province:
            province_dir = data_dir / "province_movein_moveout"
            if province_dir.exists():
                province_files = list(province_dir.glob("*.xlsx"))
                logger.info(f"Found {len(province_files)} province migration data files")
                
                for file_path in province_files[:sample_size] if sample_size else province_files:
                    filename = file_path.stem
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        province_code = parts[0]
                        province_name = parts[2]
                        move_type = parts[3]
                        
                        df = pd.read_excel(file_path)
                        
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                        
                        key = f"{province_code}_{move_type}"
                        migration_data['province'][key] = {
                            'data': df,
                            'province_code': province_code,
                            'province_name': province_name,
                            'move_type': move_type
                        }
        
        if load_city:
            city_dir = data_dir / "city_index_percentage_231225"
            if city_dir.exists():
                city_files = list(city_dir.glob("*.txt"))
                logger.info(f"Found {len(city_files)} city migration data files")
                
                for file_path in city_files[:sample_size] if sample_size else city_files:
                    filename = file_path.stem
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        location_code = parts[0]
                        location_type = parts[1]
                        location_name = parts[2]
                        move_type = parts[3]
                        
                        df = pd.read_csv(file_path)
                        df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                        
                        key = f"{location_code}_{move_type}"
                        migration_data['city'][key] = {
                            'data': df,
                            'location_code': location_code,
                            'location_type': location_type,
                            'location_name': location_name,
                            'move_type': move_type
                        }
        
        self.migration_data = migration_data
        self.data_loaded['migration_data'] = True
        
        logger.info(f"Migration data loaded:")
        logger.info(f"  Province data: {len(migration_data['province'])} files")
        logger.info(f"  City data: {len(migration_data['city'])} files")
        
        return migration_data
    
    def get_flu_data(self, 
                    start_date: Optional[str] = None,
                    end_date: Optional[str] = None,
                    columns: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.data_loaded['flu_data']:
            raise ValueError("Please load flu data first")
        
        data = self.flu_data.copy()
        
        if start_date:
            data = data[data['start_date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['stop_date'] <= pd.to_datetime(end_date)]
        
        if columns:
            available_columns = [col for col in columns if col in data.columns]
            if available_columns:
                data = data[available_columns]
            else:
                logger.warning("Specified column names do not exist")
        
        return data
    
    def get_baidu_search_data(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            provinces: Optional[List[str]] = None,
                            keywords: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.data_loaded['baidu_search']:
            raise ValueError("Please load Baidu search index data first")
        
        data = self.baidu_search_data.copy()
        
        if start_date:
            data = data[data['date'] >= pd.to_datetime(start_date)]
        if end_date:
            data = data[data['date'] <= pd.to_datetime(end_date)]
        
        if provinces:
            data = data[data['Province'].isin(provinces)]
        
        if keywords:
            keyword_mask = data['keyword'].str.contains('|'.join(keywords), na=False)
            data = data[keyword_mask]
        
        return data
    
    def get_migration_data(self,
                         location_type: str = 'province',
                         location_codes: Optional[List[str]] = None,
                         move_type: Optional[str] = None,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Dict:
        if not self.data_loaded['migration_data']:
            raise ValueError("Please load migration data first")
        
        if location_type not in self.migration_data:
            raise ValueError(f"Unsupported location type: {location_type}")
        
        result = {}
        data_dict = self.migration_data[location_type]
        
        for key, data_info in data_dict.items():
            if location_codes:
                location_code = data_info.get('location_code') or data_info.get('province_code')
                if location_code not in location_codes:
                    continue
            
            if move_type and data_info['move_type'] != move_type:
                continue
            
            df = data_info['data'].copy()
            if 'date' in df.columns:
                if start_date:
                    df = df[df['date'] >= pd.to_datetime(start_date)]
                if end_date:
                    df = df[df['date'] <= pd.to_datetime(end_date)]
            else:
                logger.warning(f"Migration data {key} does not have date column, skipping time filtering")
            
            result[key] = {
                **data_info,
                'data': df
            }
        
        return result
    
    def aggregate_baidu_search_by_week(self,
                                     start_date: Optional[str] = None,
                                     end_date: Optional[str] = None,
                                     provinces: Optional[List[str]] = None) -> pd.DataFrame:
        if not self.data_loaded['baidu_search']:
            raise ValueError("Please load Baidu search index data first")
        
        data = self.get_baidu_search_data(start_date, end_date, provinces)
        
        aggregated = data.groupby(['year_week', 'Province']).agg({
            'index': ['mean', 'sum', 'count']
        }).reset_index()
        
        aggregated.columns = ['year_week', 'Province', 'avg_index', 'total_index', 'count']
        
        return aggregated
    
    def aggregate_migration_by_week(self,
                                  location_type: str = 'province',
                                  location_codes: Optional[List[str]] = None,
                                  move_type: Optional[str] = None,
                                  start_date: Optional[str] = None,
                                  end_date: Optional[str] = None) -> pd.DataFrame:
        if not self.data_loaded['migration_data']:
            raise ValueError("Please load migration data first")
        
        migration_data = self.get_migration_data(
            location_type, location_codes, move_type, start_date, end_date
        )
        
        aggregated_data = []
        
        for key, data_info in migration_data.items():
            df = data_info['data'].copy()
            
            if df.empty:
                continue
            
            if 'date' in df.columns:
                df['year'] = df['date'].dt.year
                df['week'] = df['date'].dt.isocalendar().week
                df['year_week'] = df['year'].astype(str) + '_' + df['week'].astype(str).str.zfill(2)
            else:
                logger.warning(f"Migration data {key} does not have date column, skipping aggregation")
                continue
            
            if 'percent' in df.columns:
                weekly_agg = df.groupby('year_week').agg({
                    'percent': ['mean', 'sum', 'count']
                }).reset_index()
                weekly_agg.columns = ['year_week', 'avg_percent', 'total_percent', 'count']
            else:
                weekly_agg = df.groupby('year_week').size().reset_index(name='count')
                weekly_agg['year_week'] = weekly_agg['year_week']
            
            weekly_agg['location_code'] = data_info.get('location_code') or data_info.get('province_code')
            weekly_agg['location_name'] = data_info.get('location_name') or data_info.get('province_name')
            weekly_agg['move_type'] = data_info['move_type']
            weekly_agg['location_type'] = location_type
            
            aggregated_data.append(weekly_agg)
        
        if aggregated_data:
            return pd.concat(aggregated_data, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def create_unified_dataset(self,
                             start_date: Optional[str] = None,
                             end_date: Optional[str] = None,
                             aggregation_level: str = 'national',
                             include_migration: bool = True,
                             include_search: bool = True) -> pd.DataFrame:
        if not self.data_loaded['flu_data']:
            raise ValueError("Please load flu data first")
        
        flu_data = self.get_flu_data(start_date, end_date)
        unified_data = flu_data.copy()
        
        if include_search and self.data_loaded['baidu_search']:
            search_agg = self.aggregate_baidu_search_by_week(start_date, end_date)
            
            if aggregation_level == 'national':
                national_search = search_agg.groupby('year_week').agg({
                    'avg_index': 'mean',
                    'total_index': 'sum',
                    'count': 'sum'
                }).reset_index()
                national_search.columns = ['year_week', 'national_avg_search_index', 
                                        'national_total_search_index', 'national_search_count']
                
                unified_data = unified_data.merge(national_search, on='year_week', how='left')
            
            elif aggregation_level == 'provincial':
                logger.info("Provincial-level search index data requires further development")
        
        if include_migration and self.data_loaded['migration_data']:
            migration_agg = self.aggregate_migration_by_week(
                start_date=start_date, end_date=end_date
            )
            
            if not migration_agg.empty and aggregation_level == 'national':
                national_migration = migration_agg.groupby('year_week').agg({
                    'avg_percent': 'mean',
                    'total_percent': 'sum',
                    'count': 'sum'
                }).reset_index()
                national_migration.columns = ['year_week', 'national_avg_migration', 
                                           'national_total_migration', 'national_migration_count']
                
                unified_data = unified_data.merge(national_migration, on='year_week', how='left')
        
        return unified_data
    
    def get_data_summary(self) -> Dict:
        summary = {
            'data_loaded': self.data_loaded,
            'data_sizes': {},
            'time_ranges': {}
        }
        
        if self.data_loaded['flu_data']:
            summary['data_sizes']['flu_data'] = self.flu_data.shape
            summary['time_ranges']['flu_data'] = {
                'start': self.flu_data['start_date'].min(),
                'end': self.flu_data['stop_date'].max()
            }
        
        if self.data_loaded['baidu_search']:
            summary['data_sizes']['baidu_search'] = self.baidu_search_data.shape
            summary['time_ranges']['baidu_search'] = {
                'start': self.baidu_search_data['date'].min(),
                'end': self.baidu_search_data['date'].max()
            }
        
        if self.data_loaded['migration_data']:
            province_count = len(self.migration_data.get('province', {}))
            city_count = len(self.migration_data.get('city', {}))
            summary['data_sizes']['migration_data'] = {
                'province_files': province_count,
                'city_files': city_count
            }
        
        return summary
    
    def save_unified_data(self, 
                         data: pd.DataFrame, 
                         output_file: str,
                         format: str = 'csv') -> bool:
        output_path = self.data_root / output_file
        
        if format.lower() == 'csv':
            data.to_csv(output_path, index=False, encoding='utf-8-sig')
        elif format.lower() == 'excel':
            data.to_excel(output_path, index=False)
        elif format.lower() == 'parquet':
            data.to_parquet(output_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Data saved to: {output_path}")
        return True
    
    def load_co_circulation_periods(self, 
                                    file_path: str = "co_circulation_periods.csv") -> pd.DataFrame:
        file_path = self.data_root / file_path
        logger.info(f"Loading co-circulation period classification data: {file_path}")
        
        self.co_circulation_periods = pd.read_csv(file_path, encoding='utf-8-sig')
        
        self.co_circulation_periods['Adjusted Start Date'] = pd.to_datetime(
            self.co_circulation_periods['Adjusted Start Date'])
        self.co_circulation_periods['Adjusted End Date'] = pd.to_datetime(
            self.co_circulation_periods['Adjusted End Date'])
        
        self.data_loaded['co_circulation_periods'] = True
        logger.info(f"Co-circulation period data loaded: {self.co_circulation_periods.shape}")
        logger.info(f"Number of co-circulation periods: {len(self.co_circulation_periods)}")
        
        return self.co_circulation_periods
    
    def classify_epidemic_periods(self) -> pd.DataFrame:
        if not self.data_loaded['flu_data']:
            raise ValueError("Please load flu data first")
        if not self.data_loaded['co_circulation_periods']:
            raise ValueError("Please load co-circulation period data first")
        
        logger.info("Starting three-period classification...")
        
        classification = self.flu_data.copy()
        
        column_mapping = {
            'ili_north': 'ILI%北方' if 'ILI%北方' in classification.columns else 'northern_ili_rate',
            'ili_south': 'ILI%南方' if 'ILI%南方' in classification.columns else 'southern_ili_rate',
            'pos_north': '流感阳性率北方' if '流感阳性率北方' in classification.columns else 'northern_flu_positive_rate',
            'pos_south': '流感阳性率南方' if '流感阳性率南方' in classification.columns else 'southern_flu_positive_rate'
        }
        
        logger.info(f"  Detected column name format: {list(column_mapping.values())}")
        
        ili_north_threshold = classification[column_mapping['ili_north']].quantile(0.60)
        ili_south_threshold = classification[column_mapping['ili_south']].quantile(0.60)
        pos_north_threshold = classification[column_mapping['pos_north']].quantile(0.60)
        pos_south_threshold = classification[column_mapping['pos_south']].quantile(0.60)
        
        logger.info(f"Northern ILI% threshold (60th): {ili_north_threshold:.2f}")
        logger.info(f"Southern ILI% threshold (60th): {ili_south_threshold:.2f}")
        logger.info(f"Northern positive rate threshold (60th): {pos_north_threshold:.4f}")
        logger.info(f"Southern positive rate threshold (60th): {pos_south_threshold:.4f}")
        
        classification['period_type_north'] = 'non_epidemic'
        classification['period_type_south'] = 'non_epidemic'
        
        high_epidemic_north = (
            (classification[column_mapping['ili_north']] >= ili_north_threshold) |
            (classification[column_mapping['pos_north']] >= pos_north_threshold)
        )
        high_epidemic_south = (
            (classification[column_mapping['ili_south']] >= ili_south_threshold) |
            (classification[column_mapping['pos_south']] >= pos_south_threshold)
        )
        
        classification.loc[high_epidemic_north, 'period_type_north'] = 'single_dominant'
        classification.loc[high_epidemic_south, 'period_type_south'] = 'single_dominant'
        
        for _, period in self.co_circulation_periods.iterrows():
            start_date = period['Adjusted Start Date']
            end_date = period['Adjusted End Date']
            region = period['Region'].lower()
            
            mask = (
                (classification['start_date'] >= start_date) &
                (classification['stop_date'] <= end_date)
            )
            
            if region == 'northern':
                classification.loc[mask, 'period_type_north'] = 'co_circulation'
            elif region == 'southern':
                classification.loc[mask, 'period_type_south'] = 'co_circulation'
        
        logger.info("\nNorthern period classification statistics:")
        logger.info(classification['period_type_north'].value_counts())
        logger.info("\nSouthern period classification statistics:")
        logger.info(classification['period_type_south'].value_counts())
        
        self.period_classification = classification
        return classification
    
    def load_spectral_features(self, 
                               file_path: str = "AFD/spectral_analysis/spectral_feature_comparison.csv") -> pd.DataFrame:
        file_path = self.data_root / file_path
        logger.info(f"Loading spectral features: {file_path}")
        
        spectral_raw = pd.read_csv(file_path, encoding='utf-8-sig')
        
        spectral_co = spectral_raw.pivot(index='series', columns='feature', values='co_value')
        spectral_co.columns = [f'spectral_{col}_co' for col in spectral_co.columns]
        
        spectral_single = spectral_raw.pivot(index='series', columns='feature', values='single_value')
        spectral_single.columns = [f'spectral_{col}_single' for col in spectral_single.columns]
        
        self.spectral_features = pd.concat([spectral_co, spectral_single], axis=1)
        self.spectral_features.reset_index(inplace=True)
        
        self.data_loaded['spectral_features'] = True
        logger.info(f"Spectral features loaded: {self.spectral_features.shape}")
        logger.info(f"Number of feature columns: {len(self.spectral_features.columns) - 1}")
        
        return self.spectral_features
    
    def load_afd_components(self, 
                           seasonal_npy: str = "afd_seasonal_components.npy",
                           residual_npy: str = "afd_residual_components.npy",
                           metadata_pkl: str = "afd_metadata.pkl") -> Tuple[np.ndarray, np.ndarray, Dict]:
        seasonal_path = self.data_root / seasonal_npy
        residual_path = self.data_root / residual_npy
        metadata_path = self.data_root / metadata_pkl
        
        logger.info(f"Loading AFD-STL component data...")
        logger.info(f"  Seasonal component .npy file: {seasonal_path}")
        logger.info(f"  Residual component .npy file: {residual_path}")
        logger.info(f"  Metadata .pkl file: {metadata_path}")
        
        self.afd_seasonal_components = np.load(seasonal_path, allow_pickle=False)
        self.afd_residual_components = np.load(residual_path, allow_pickle=False)
        
        with open(metadata_path, 'rb') as f:
            self.afd_stl_metadata = pickle.load(f)
        
        logger.info(f"AFD-STL data loaded!")
        logger.info(f"  Seasonal component array shape: {self.afd_seasonal_components.shape}")
        logger.info(f"  Residual component array shape: {self.afd_residual_components.shape}")
        logger.info(f"  Number of series: {self.afd_seasonal_components.shape[0]}")
        logger.info(f"  Number of time points: {self.afd_seasonal_components.shape[1]}")
        logger.info(f"  Series name examples: {self.afd_stl_metadata.get('series_names', [])[:3]}")
        
        self.afd_components_array = np.concatenate([self.afd_seasonal_components, self.afd_residual_components], axis=2)
        self.afd_metadata = self.afd_stl_metadata
        
        self.data_loaded['afd_stl_components'] = True
        self.data_loaded['afd_components'] = True
        return self.afd_seasonal_components, self.afd_residual_components, self.afd_stl_metadata
    
    def extract_meteorological_features(self, region: str = 'north') -> pd.DataFrame:
        if not self.data_loaded['flu_data']:
            raise ValueError("Please load flu data first")
        
        logger.info(f"Extracting {region} meteorological features...")
        
        data = self.flu_data.copy()
        region_suffix = '北方' if region == 'north' else '南方'
        
        era5_columns = [col for col in data.columns if f'ERA5-Land气象再分析数据（{region_suffix}）' in col]
        
        surface_columns = [col for col in data.columns if f'{region_suffix}气象_' in col]
        
        air_quality_columns = [col for col in data.columns if f'{region_suffix}空气质量_' in col]
        
        met_columns = era5_columns + surface_columns + air_quality_columns
        
        logger.info(f"  ERA5-Land indicators: {len(era5_columns)}")
        logger.info(f"  Surface observation indicators: {len(surface_columns)}")
        logger.info(f"  Air quality indicators: {len(air_quality_columns)}")
        logger.info(f"  Total: {len(met_columns)}")
        
        met_features = data[['year_week'] + met_columns].copy()
        
        rename_dict = {}
        for col in met_columns:
            if 'ERA5-Land' in col:
                new_name = col.split('）_')[-1]
                rename_dict[col] = f'met_{region}_{new_name}'
            elif '气象_' in col:
                new_name = col.split('_')[-1]
                rename_dict[col] = f'met_{region}_surface_{new_name}'
            elif '空气质量_' in col:
                new_name = col.split('_')[-1]
                rename_dict[col] = f'met_{region}_air_{new_name}'
        
        met_features.rename(columns=rename_dict, inplace=True)
        
        return met_features
    
    def extract_baidu_search_features(self) -> pd.DataFrame:
        if not self.data_loaded['flu_data']:
            raise ValueError("Please load flu data first")
        
        logger.info("Extracting Baidu search index features...")
        
        data = self.flu_data.copy()
        
        baidu_columns = [col for col in data.columns if '百度指数_' in col]
        
        logger.info(f"  Number of Baidu search keywords: {len(baidu_columns)}")
        
        search_features = data[['year_week'] + baidu_columns].copy()
        
        rename_dict = {col: f"baidu_{col.replace('百度指数_', '')}" for col in baidu_columns}
        search_features.rename(columns=rename_dict, inplace=True)
        
        return search_features
    
    def extract_migration_features(self) -> pd.DataFrame:
        if not self.data_loaded['migration_data']:
            logger.warning("Migration data not loaded, returning empty dataframe")
            if self.data_loaded['flu_data']:
                return pd.DataFrame({'year_week': self.flu_data['year_week']})
            return pd.DataFrame()
        
        logger.info("Extracting migration features...")
        
        migration_agg = self.aggregate_migration_by_week()
        
        if migration_agg.empty:
            logger.warning("Migration data aggregation result is empty")
            return pd.DataFrame({'year_week': self.flu_data['year_week']})
        
        available_columns = migration_agg.columns.tolist()
        logger.info(f"Migration aggregated data columns: {available_columns}")
        
        if 'avg_percent' in available_columns:
            move_in = migration_agg[migration_agg['move_type'] == 'move_in'].groupby('year_week').agg({
                'avg_percent': 'mean'
            }).reset_index()
            move_in.columns = ['year_week', 'migration_in']
            
            move_out = migration_agg[migration_agg['move_type'] == 'move_out'].groupby('year_week').agg({
                'avg_percent': 'mean'
            }).reset_index()
            move_out.columns = ['year_week', 'migration_out']
        elif 'total_percent' in available_columns:
            move_in = migration_agg[migration_agg['move_type'] == 'move_in'].groupby('year_week').agg({
                'total_percent': 'mean'
            }).reset_index()
            move_in.columns = ['year_week', 'migration_in']
            
            move_out = migration_agg[migration_agg['move_type'] == 'move_out'].groupby('year_week').agg({
                'total_percent': 'mean'
            }).reset_index()
            move_out.columns = ['year_week', 'migration_out']
        else:
            move_in = migration_agg[migration_agg['move_type'] == 'move_in'].groupby('year_week').agg({
                'count': 'mean'
            }).reset_index()
            move_in.columns = ['year_week', 'migration_in']
            
            move_out = migration_agg[migration_agg['move_type'] == 'move_out'].groupby('year_week').agg({
                'count': 'mean'
            }).reset_index()
            move_out.columns = ['year_week', 'migration_out']
        
        migration_features = pd.merge(move_in, move_out, on='year_week', how='outer')
        
        logger.info(f"  Migration feature extraction completed: {migration_features.shape}")
        
        return migration_features
    
    def create_lagged_features(self, 
                              df: pd.DataFrame, 
                              feature_columns: List[str],
                              lags: Optional[List[int]] = None,
                              group_by: Optional[str] = None) -> pd.DataFrame:
        if lags is None:
            lags = self.lag_periods
        
        logger.info(f"Creating lagged features: {len(feature_columns)} features × {len(lags)} lag periods")
        
        result = df.copy()
        
        for col in feature_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} does not exist, skipping")
                continue
            
            for lag in lags:
                lag_col_name = f"{col}_lag{lag}"
                
                if group_by and group_by in df.columns:
                    result[lag_col_name] = df.groupby(group_by)[col].shift(lag)
                else:
                    result[lag_col_name] = df[col].shift(lag)
        
        logger.info(f"  Lagged features created, new columns: {len(feature_columns) * len(lags)}")
        
        return result
    
    def create_interaction_features(self, df: pd.DataFrame, region: str = 'north') -> pd.DataFrame:
        logger.info(f"Creating {region} meteorological interaction terms...")
        
        result = df.copy()
        interactions = []
        
        interaction_pairs = [
            (f'met_{region}_t2m', f'met_{region}_d2m', 'temp_humidity'),
            (f'met_{region}_air_AQI', f'met_{region}_air_PM2.5', 'aqi_pm25'),
            (f'met_{region}_t2m', f'met_{region}_tp', 'temp_precip'),
        ]
        
        for col1, col2, name in interaction_pairs:
            if col1 in result.columns and col2 in result.columns:
                interaction_col = f'interact_{region}_{name}'
                result[interaction_col] = result[col1] * result[col2]
                interactions.append(interaction_col)
                logger.info(f"  Created interaction term: {interaction_col}")
        
        logger.info(f"  Interaction terms created, new columns: {len(interactions)}")
        
        return result, interactions
    
    def integrate_all_features(self,
                              start_date: Optional[str] = None,
                              end_date: Optional[str] = None,
                              regions: List[str] = ['north', 'south'],
                              include_lags: bool = False,
                              include_interactions: bool = False,
                              save_intermediate: bool = True) -> pd.DataFrame:
        logger.info("=" * 60)
        logger.info("Starting feature integration workflow...")
        logger.info("=" * 60)
        
        features_dir = self.data_root / "features"
        features_dir.mkdir(exist_ok=True)
        
        logger.info("\n[Step 1/6] Loading base data...")
        if not self.data_loaded['flu_data']:
            self.load_flu_data()
        if not self.data_loaded['co_circulation_periods']:
            self.load_co_circulation_periods()
        if not self.data_loaded['spectral_features']:
            self.load_spectral_features()
        if not self.data_loaded['afd_components']:
            self.load_afd_components()
        
        logger.info("\n[Step 2/6] Performing three-period classification...")
        period_data = self.classify_epidemic_periods()
        
        if save_intermediate:
            period_file = features_dir / "period_classification.csv"
            period_data.to_csv(period_file, index=False, encoding='utf-8-sig')
            logger.info(f"  Saved: {period_file}")
        
        logger.info("\n[Step 3/6] Extracting and integrating various features...")
        
        all_region_data = []
        
        for region in regions:
            logger.info(f"\nProcessing {region} data...")
            
            region_df = period_data[['year_week', 'start_date', 'stop_date']].copy()
            region_df['region'] = region
            
            period_col = f'period_type_{region}'
            region_df['period_type'] = period_data[period_col]
            
            ili_col = 'ILI%北方' if region == 'north' else 'ILI%南方'
            pos_col = '流感阳性率北方' if region == 'north' else '流感阳性率南方'
            region_df['ili_rate'] = period_data[ili_col]
            region_df['flu_positive_rate'] = period_data[pos_col]
            
            logger.info(f"  Adding spectral features...")
            series_name = f"{region}ern_ili_rate"
            
            if series_name in self.spectral_features['series'].values:
                spectral_row = self.spectral_features[self.spectral_features['series'] == series_name]
                
                for col in spectral_row.columns:
                    if col == 'series':
                        continue
                    
                    if '_co' in col:
                        feature_name = col.replace('_co', '')
                        co_value = spectral_row[col].values[0]
                        region_df.loc[region_df['period_type'] == 'co_circulation', feature_name] = co_value
                    
                    elif '_single' in col:
                        feature_name = col.replace('_single', '')
                        single_value = spectral_row[col].values[0]
                        region_df.loc[region_df['period_type'] != 'co_circulation', feature_name] = single_value
            
            logger.info(f"  Adding core variable AFD component features...")
            if self.afd_components_array is not None:
                series_names = self.afd_metadata.get('series_names', [])
                time_index = self.afd_metadata.get('time_index', [])
                
                region_cn = '北方' if region == 'north' else '南方'
                core_variables = [
                    f"流感阳性率{region_cn}",
                    f"病毒监测和分型（{region_cn}）甲型 H1N1",
                    f"病毒监测和分型（{region_cn}）A(H3N2)",
                    f"病毒监测和分型（{region_cn}）Victoria",
                    f"病毒监测和分型（{region_cn}）Yamagata"
                ]
                
                for var_name in core_variables:
                    if var_name not in series_names:
                        logger.warning(f"    Core variable {var_name} not found in AFD data, skipping")
                        continue
                    
                    series_idx = series_names.index(var_name)
                    
                    afd_series_data = self.afd_components_array[series_idx, :, :]
                    n_components = afd_series_data.shape[1]
                    
                    if "ILI%" in var_name:
                        var_prefix = f"ili_{region}"
                    elif "流感阳性率" in var_name:
                        var_prefix = f"flu_pos_{region}"
                    elif "甲型 H1N1" in var_name or "H1N1" in var_name:
                        var_prefix = f"H1N1_{region}"
                    elif "A(H3N2)" in var_name or "H3N2" in var_name:
                        var_prefix = f"H3N2_{region}"
                    elif "Victoria" in var_name:
                        var_prefix = f"Victoria_{region}"
                    elif "Yamagata" in var_name:
                        var_prefix = f"Yamagata_{region}"
                    else:
                        var_prefix = var_name.replace(f"{region}ern_", "").replace("_rate", "").replace("_proportion", "")
                    
                    afd_df = pd.DataFrame(
                        afd_series_data, 
                        columns=[f'afd_{var_prefix}_comp{i+1}' for i in range(n_components)]
                    )
                    
                    comp_cols = [col for col in afd_df.columns if 'comp' in col]
                    afd_df[f'afd_{var_prefix}_total_energy'] = afd_df[comp_cols].sum(axis=1)
                    afd_df[f'afd_{var_prefix}_energy_concentration'] = (
                        afd_df[comp_cols].max(axis=1) / afd_df[f'afd_{var_prefix}_total_energy']
                    ).replace([np.inf, -np.inf], 0)
                    
                    if len(afd_df) == len(region_df):
                        for col in afd_df.columns:
                            region_df[col] = afd_df[col].values
                        logger.info(f"    Added AFD features for {var_name}: {len(afd_df.columns)} columns")
                    else:
                        logger.warning(f"    {var_name} AFD data length mismatch, skipping")
            
            logger.info(f"  Creating lagged AFD features for ILI%...")
            
            ili_var_name = f"ILI%{region_cn}"
            if ili_var_name in series_names:
                series_idx = series_names.index(ili_var_name)
                ili_afd_data = self.afd_components_array[series_idx, :, :]
                n_components = ili_afd_data.shape[1]
                
                ili_afd_df = pd.DataFrame(
                    ili_afd_data, 
                    columns=[f'afd_ili_{region}_comp{i+1}' for i in range(n_components)]
                )
                
                comp_cols = [col for col in ili_afd_df.columns if 'comp' in col]
                ili_afd_df[f'afd_ili_{region}_total_energy'] = ili_afd_df[comp_cols].sum(axis=1)
                ili_afd_df[f'afd_ili_{region}_energy_concentration'] = (
                    ili_afd_df[comp_cols].max(axis=1) / ili_afd_df[f'afd_ili_{region}_total_energy']
                ).replace([np.inf, -np.inf], 0)
                
                lag_periods = [1, 2, 3]
                for lag in lag_periods:
                    for col in ili_afd_df.columns:
                        lag_col_name = f"{col}_lag{lag}"
                        region_df[lag_col_name] = ili_afd_df[col].shift(lag)
                
                logger.info(f"    Added lagged AFD features for ILI%: {len(ili_afd_df.columns) * len(lag_periods)} columns")
            else:
                logger.warning(f"    ILI%{region_cn} not found in AFD data, skipping ILI% AFD features")
            
            logger.info(f"  Creating lagged features for other AFD features...")
            lag_periods = [1, 2, 3]
            
            afd_columns = [col for col in region_df.columns if col.startswith('afd_') and 'ili_' not in col]
            
            for lag in lag_periods:
                for col in afd_columns:
                    lag_col_name = f"{col}_lag{lag}"
                    region_df[lag_col_name] = region_df[col].shift(lag)
                
                logger.info(f"    Added lagged {lag}-week other AFD features: {len(afd_columns)} columns")
            
            region_df = region_df.drop(columns=afd_columns)
            logger.info(f"    Removed original AFD features, keeping only lagged features to avoid data leakage")
            
            logger.info(f"  Adding meteorological features...")
            met_features = self.extract_meteorological_features(region=region)
            region_df = pd.merge(region_df, met_features, on='year_week', how='left')
            
            if region == regions[0]:
                logger.info(f"  Adding Baidu search features...")
                search_features = self.extract_baidu_search_features()
                region_df = pd.merge(region_df, search_features, on='year_week', how='left')
            
            if region == regions[0]:
                logger.info(f"  Adding migration features...")
                migration_features = self.extract_migration_features()
                if not migration_features.empty:
                    region_df = pd.merge(region_df, migration_features, on='year_week', how='left')
            
            if include_interactions:
                logger.info(f"  Adding meteorological interaction terms...")
                region_df, _ = self.create_interaction_features(region_df, region=region)
            
            all_region_data.append(region_df)
            logger.info(f"  {region} data integration completed: {region_df.shape}")
        
        logger.info("\n[Step 4/6] Merging northern and southern data...")
        integrated_df = pd.concat(all_region_data, ignore_index=True)
        logger.info(f"  Merged data shape: {integrated_df.shape}")
        
        if include_lags:
            logger.info("\n[Step 5/6] Creating lagged features...")
            
            lag_columns = []
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('spectral_')])
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('afd_')])
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('met_')])
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('baidu_')])
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('migration_')])
            
            lag_columns.extend([col for col in integrated_df.columns if col.startswith('interact_')])
            
            logger.info(f"  Number of features to lag: {len(lag_columns)}")
            
            integrated_df = self.create_lagged_features(
                integrated_df, 
                lag_columns,
                group_by='region'
            )
            
            logger.info(f"  Data shape after creating lagged features: {integrated_df.shape}")
        
        logger.info("\n[Step 6/6] Data cleaning and filtering...")
        
        if start_date:
            integrated_df = integrated_df[integrated_df['start_date'] >= pd.to_datetime(start_date)]
        if end_date:
            integrated_df = integrated_df[integrated_df['stop_date'] <= pd.to_datetime(end_date)]
        
        logger.info("\n  Data quality diagnostics:")
        missing_ratio = integrated_df.isnull().sum(axis=1) / len(integrated_df.columns)
        logger.info(f"    Overall missing rate: mean={missing_ratio.mean():.2%}, max={missing_ratio.max():.2%}")
        
        for region in integrated_df['region'].unique():
            region_df = integrated_df[integrated_df['region'] == region]
            missing_ratio_region = region_df.isnull().sum(axis=1) / len(region_df.columns)
            logger.info(f"    {region}: samples={len(region_df)}, "
                       f"mean missing rate={missing_ratio_region.mean():.2%}, "
                       f"max missing rate={missing_ratio_region.max():.2%}")
        
        logger.info(f"\n  Final data shape: {integrated_df.shape}")
        logger.info(f"  Sample counts by region:")
        for region in integrated_df['region'].unique():
            region_count = (integrated_df['region'] == region).sum()
            logger.info(f"    {region}: {region_count} samples")
        
        self.integrated_features = integrated_df
        
        if save_intermediate:
            output_file = features_dir / "integrated_features_full.csv"
            integrated_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            logger.info(f"\n✅ Complete feature matrix saved: {output_file}")
            
            output_parquet = features_dir / "integrated_features_full.parquet"
            integrated_df.to_parquet(output_parquet, index=False)
            logger.info(f"✅ Parquet format saved: {output_parquet}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Feature integration completed!")
        logger.info(f"Final number of features: {len(integrated_df.columns)}")
        logger.info(f"Number of samples: {len(integrated_df)}")
        logger.info("=" * 60)
        
        return integrated_df
    
    def prepare_bym2_input(self,
                          target_variable: str = 'ili_rate',
                          feature_prefix_filter: Optional[List[str]] = None,
                          standardize: bool = True,
                          spatial_level: str = 'region',
                          include_od_matrix: bool = False,
                          od_aggregation: str = 'weekly',
                          output_dir: str = "features") -> Dict:
        if self.integrated_features is None:
            raise ValueError("Please run integrate_all_features() method first")
        
        logger.info("\n" + "=" * 60)
        logger.info("Preparing BYM2 model input...")
        logger.info(f"Spatial level: {spatial_level}")
        logger.info("=" * 60)
        
        df = self.integrated_features.copy()
        
        if target_variable not in df.columns:
            raise ValueError(f"Target variable {target_variable} does not exist")
        
        Y = df[target_variable].values
        logger.info(f"\nDependent variable: {target_variable}")
        logger.info(f"  Number of samples: {len(Y)}")
        logger.info(f"  Mean: {Y.mean():.4f}, Std: {Y.std():.4f}")
        
        exclude_cols = ['year_week', 'start_date', 'stop_date', 'region', 'period_type',
                       'ili_rate', 'flu_positive_rate']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        if feature_prefix_filter:
            feature_cols = [col for col in feature_cols 
                          if any(col.startswith(prefix) for prefix in feature_prefix_filter)]
            logger.info(f"\nApplying feature filter: {feature_prefix_filter}")
        
        logger.info(f"\nNumber of features: {len(feature_cols)}")
        
        X = df[feature_cols].values
        
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        logger.info(f"Missing value imputation completed (strategy: median)")
        
        if standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            logger.info(f"Feature standardization completed")
        
        time_index = df['year_week'].values
        spatial_index = df['region'].map({'north': 0, 'south': 1}).values
        period_type = df['period_type'].values
        
        logger.info(f"\nLoading adjacency matrix based on spatial level '{spatial_level}'...")
        
        if spatial_level == 'region':
            adjacency_matrix = np.array([[0, 1], [1, 0]])
            n_regions = 2
            region_names = ['north', 'south']
            logger.info(f"  Using north-south partition adjacency matrix (2×2)")
            
        elif spatial_level == 'province':
            logger.info(f"  Generating provincial adjacency matrix...")
            adj_result = self.generate_adjacency_matrix_from_shapefile(level='province')
            adjacency_matrix = adj_result['adjacency_matrix']
            n_regions = adj_result['n_regions']
            region_names = adj_result['region_names']
            logger.info(f"  Provincial adjacency matrix loaded ({n_regions}×{n_regions})")
            
        elif spatial_level == 'city':
            logger.warning(f"  City-level adjacency matrix not yet implemented, using region level as fallback")
            adjacency_matrix = np.array([[0, 1], [1, 0]])
            n_regions = 2
            region_names = ['north', 'south']
        else:
            raise ValueError(f"Unsupported spatial level: {spatial_level}")
        
        od_matrix = None
        od_metadata = None
        
        if include_od_matrix:
            logger.info(f"\nLoading OD matrix (aggregation method: {od_aggregation})...")
            
            if spatial_level == 'province':
                od_result = self.build_od_matrix_province(
                    aggregation=od_aggregation,
                    save_output=True
                )
                od_matrix = od_result['od_matrix']
                od_metadata = od_result['metadata']
                logger.info(f"  Provincial OD matrix loaded: {od_matrix.shape}")
                
            elif spatial_level == 'city':
                od_result = self.build_od_matrix_city(
                    aggregation=od_aggregation,
                    save_output=True
                )
                od_matrix = od_result['od_matrix']
                od_metadata = od_result['metadata']
                if od_matrix is not None:
                    logger.info(f"  City-level OD matrix loaded: {od_matrix.shape}")
                else:
                    logger.warning("  City-level OD matrix data unavailable")
            else:
                logger.warning(f"  {spatial_level} level does not support OD matrix")
        
        bym2_input = {
            'Y': Y,
            'X': X,
            'feature_names': feature_cols,
            'time_index': time_index,
            'spatial_index': spatial_index,
            'period_type': period_type,
            'adjacency_matrix': adjacency_matrix,
            'n_samples': len(Y),
            'n_features': X.shape[1],
            'n_regions': n_regions,
            'n_timepoints': len(np.unique(time_index)),
            'metadata': {
                'target_variable': target_variable,
                'standardized': standardize,
                'spatial_level': spatial_level,
                'regions': region_names,
                'period_types': ['non_epidemic', 'single_dominant', 'co_circulation'],
                'include_od_matrix': include_od_matrix,
                'od_aggregation': od_aggregation if include_od_matrix else None
            }
        }
        
        if od_matrix is not None:
            bym2_input['od_matrix'] = od_matrix
            bym2_input['od_metadata'] = od_metadata
        
        output_path = self.data_root / output_dir
        output_path.mkdir(exist_ok=True)
        
        output_file = output_path / "bym2_input_data.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(bym2_input, f)
        
        logger.info(f"\n✅ BYM2 input data saved: {output_file}")
        
        summary_file = output_path / "bym2_input_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write("BYM2 Model Input Data Summary\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Target variable: {target_variable}\n")
            f.write(f"Number of samples: {bym2_input['n_samples']}\n")
            f.write(f"Number of features: {bym2_input['n_features']}\n")
            f.write(f"Number of regions: {bym2_input['n_regions']}\n")
            f.write(f"Number of time points: {bym2_input['n_timepoints']}\n\n")
            f.write(f"Feature standardization: {'Yes' if standardize else 'No'}\n\n")
            f.write("Period type distribution:\n")
            for period in ['non_epidemic', 'single_dominant', 'co_circulation']:
                count = (period_type == period).sum()
                pct = count / len(period_type) * 100
                f.write(f"  {period}: {count} ({pct:.1f}%)\n")
            f.write("\nFeature list (first 20):\n")
            for i, feat in enumerate(feature_cols[:20], 1):
                f.write(f"  {i}. {feat}\n")
            if len(feature_cols) > 20:
                f.write(f"  ... {len(feature_cols)-20} more features\n")
        
        logger.info(f"✅ Data summary saved: {summary_file}")
        
        logger.info("\n" + "=" * 60)
        logger.info("BYM2 input preparation completed!")
        logger.info("=" * 60)
        
        return bym2_input
    
    def build_od_matrix_province(self,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None,
                                aggregation: str = 'weekly',
                                save_output: bool = True) -> Dict:
        logger.info("\n" + "=" * 60)
        logger.info("Building provincial OD matrix...")
        logger.info("=" * 60)
        
        province_dir = self.data_root / "baidu_move" / "province_movein_moveout"
        
        if not province_dir.exists():
            raise FileNotFoundError(f"Provincial migration data directory does not exist: {province_dir}")
        
        move_in_files = sorted(province_dir.glob("*_move_in.xlsx"))
        logger.info(f"\nFound {len(move_in_files)} province move-in files")
        
        province_info = []
        for file in move_in_files:
            parts = file.stem.split('_')
            province_code = parts[0]
            province_name = parts[2]
            province_info.append((province_code, province_name))
        
        province_codes = [info[0] for info in province_info]
        province_names = [info[1] for info in province_info]
        n_provinces = len(province_codes)
        
        logger.info(f"Number of provinces: {n_provinces}")
        logger.info(f"Province list: {province_names[:5]}... (showing first 5)")
        
        all_data = []
        
        for i, (code, name) in enumerate(province_info):
            file_path = province_dir / f"{code}_province_{name}_move_in.xlsx"
            
            df = pd.read_excel(file_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            elif 'Date' in df.columns:
                df['date'] = pd.to_datetime(df['Date'])
            
            df['destination_code'] = code
            df['destination_name'] = name
            
            all_data.append(df)
            
            if (i + 1) % 10 == 0:
                logger.info(f"  Read {i+1}/{n_provinces} provinces")
        
        logger.info("\nMerging all province data...")
        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"  Merged data shape: {combined_df.shape}")
        
        if start_date:
            combined_df = combined_df[combined_df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            combined_df = combined_df[combined_df['date'] <= pd.to_datetime(end_date)]
        
        logger.info(f"  Data shape after time filtering: {combined_df.shape}")
        logger.info(f"  Time range: {combined_df['date'].min()} to {combined_df['date'].max()}")
        
        logger.info(f"\nAggregating at {aggregation} level...")
        
        if aggregation == 'daily':
            combined_df['time_key'] = combined_df['date'].dt.strftime('%Y-%m-%d')
        elif aggregation == 'weekly':
            combined_df['year'] = combined_df['date'].dt.year
            combined_df['week'] = combined_df['date'].dt.isocalendar().week
            combined_df['time_key'] = (combined_df['year'].astype(str) + '_' + 
                                      combined_df['week'].astype(str).str.zfill(2))
        elif aggregation == 'monthly':
            combined_df['time_key'] = combined_df['date'].dt.strftime('%Y-%m')
        else:
            raise ValueError(f"Unsupported aggregation method: {aggregation}")
        
        logger.info("\nConstructing OD matrix...")
        
        time_points = sorted(combined_df['time_key'].unique())
        n_times = len(time_points)
        
        logger.info(f"  Number of time points: {n_times}")
        
        od_matrix = np.zeros((n_times, n_provinces, n_provinces))
        
        code_to_idx = {code: i for i, code in enumerate(province_codes)}
        
        for t_idx, time_key in enumerate(time_points):
            time_data = combined_df[combined_df['time_key'] == time_key]
            
            for _, row in time_data.iterrows():
                dest_idx = code_to_idx.get(row['destination_code'])
                
                if dest_idx is None:
                    continue
                
                if 'origin_province_code' in row and 'percent' in row:
                    origin_code = row['origin_province_code']
                    origin_idx = code_to_idx.get(origin_code)
                    
                    if origin_idx is not None:
                        od_matrix[t_idx, origin_idx, dest_idx] = row['percent']
            
            if (t_idx + 1) % 50 == 0:
                logger.info(f"  Processed {t_idx+1}/{n_times} time points")
        
        logger.info(f"\nOD matrix construction completed!")
        logger.info(f"  Matrix shape: {od_matrix.shape}")
        logger.info(f"  Non-zero element ratio: {(od_matrix > 0).sum() / od_matrix.size * 100:.2f}%")
        
        result = {
            'od_matrix': od_matrix,
            'province_codes': province_codes,
            'province_names': province_names,
            'time_index': time_points,
            'aggregation': aggregation,
            'n_provinces': n_provinces,
            'n_timepoints': n_times,
            'metadata': {
                'start_date': combined_df['date'].min(),
                'end_date': combined_df['date'].max(),
                'data_source': 'Baidu Migration Data',
                'level': 'province'
            }
        }
        
        if save_output:
            output_dir = self.data_root / "features" / "od_matrix"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / f"od_matrix_province_{aggregation}.npy", od_matrix)
            
            with open(output_dir / f"od_matrix_province_{aggregation}_metadata.pkl", 'wb') as f:
                pickle.dump({k: v for k, v in result.items() if k != 'od_matrix'}, f)
            
            od_long = []
            for t_idx, time_key in enumerate(time_points):
                for i, origin_code in enumerate(province_codes):
                    for j, dest_code in enumerate(province_codes):
                        flow = od_matrix[t_idx, i, j]
                        if flow > 0:
                            od_long.append({
                                'time': time_key,
                                'origin_code': origin_code,
                                'origin_name': province_names[i],
                                'destination_code': dest_code,
                                'destination_name': province_names[j],
                                'flow': flow
                            })
            
            od_long_df = pd.DataFrame(od_long)
            od_long_df.to_csv(output_dir / f"od_matrix_province_{aggregation}_long.csv", 
                             index=False, encoding='utf-8-sig')
            
            logger.info(f"\n✅ OD matrix saved:")
            logger.info(f"  - {output_dir / f'od_matrix_province_{aggregation}.npy'}")
            logger.info(f"  - {output_dir / f'od_matrix_province_{aggregation}_metadata.pkl'}")
            logger.info(f"  - {output_dir / f'od_matrix_province_{aggregation}_long.csv'}")
        
        return result
    
    def build_od_matrix_city(self,
                            start_date: Optional[str] = None,
                            end_date: Optional[str] = None,
                            aggregation: str = 'weekly',
                            save_output: bool = True) -> Dict:
        logger.info("\n" + "=" * 60)
        logger.info("Building city-level OD matrix...")
        logger.info("=" * 60)
        logger.info("Note: City-level OD matrix computation is intensive, please wait...")
        
        city_dir = self.data_root / "baidu_move" / "city_index_percentage_231225"
        
        if not city_dir.exists():
            logger.warning(f"City-level migration data directory does not exist: {city_dir}")
            logger.warning("Will return empty result")
            return {
                'od_matrix': None,
                'city_codes': [],
                'city_names': [],
                'time_index': [],
                'aggregation': aggregation,
                'metadata': {'level': 'city', 'status': 'data_not_available'}
            }
        
        move_in_files = sorted(city_dir.glob("*_city_*_move_in.txt"))
        
        if len(move_in_files) == 0:
            logger.warning("No city-level move-in data files found")
            return {
                'od_matrix': None,
                'city_codes': [],
                'city_names': [],
                'time_index': [],
                'aggregation': aggregation,
                'metadata': {'level': 'city', 'status': 'no_files_found'}
            }
        
        logger.info(f"Found {len(move_in_files)} city move-in files")
        
        logger.info("City-level OD matrix construction function is reserved, waiting for data format confirmation to complete")
        
        return {
            'od_matrix': None,
            'city_codes': [],
            'city_names': [],
            'time_index': [],
            'aggregation': aggregation,
            'metadata': {'level': 'city', 'status': 'to_be_implemented'}
        }
    
    def generate_adjacency_matrix_from_shapefile(self,
                                                level: str = 'province',
                                                shapefile_path: Optional[str] = None,
                                                save_output: bool = True) -> Dict:
        logger.info("\n" + "=" * 60)
        logger.info(f"Generating {level}-level adjacency matrix...")
        logger.info("=" * 60)
        
        if shapefile_path is None:
            logger.info("No shapefile path provided, using predefined adjacency relationships...")
            
            if level == 'province':
                return self._generate_province_adjacency_manual()
            else:
                logger.warning(f"Unsupported level: {level}, only province level has predefined adjacency relationships")
                raise ValueError(f"Level {level} requires shapefile_path")
        else:
            import geopandas as gpd
            from shapely.geometry import Point, Polygon
            
            logger.info(f"Reading from shapefile: {shapefile_path}")
            gdf = gpd.read_file(shapefile_path)
            
            logger.info(f"  Read {len(gdf)} spatial units")
            logger.info(f"  Column names: {gdf.columns.tolist()}")
            
            adjacency_matrix = self._calculate_adjacency_from_geodataframe(gdf, level)
            
            return adjacency_matrix
    
    def _generate_province_adjacency_manual(self) -> Dict:
        logger.info("Using manually defined provincial adjacency relationships...")
        
        province_codes = list(self.province_mapping.keys())
        province_names = list(self.province_mapping.values())
        n_provinces = len(province_codes)
        
        adjacency = np.zeros((n_provinces, n_provinces), dtype=int)
        
        neighbors = {
            '110000': ['130000', '120000'],
            '120000': ['110000', '130000'],
            '130000': ['110000', '120000', '140000', '150000', '210000', '370000', '410000'],
            '140000': ['130000', '150000', '610000', '410000'],
            '150000': ['130000', '140000', '210000', '220000', '230000', '610000', '620000', '640000'],
            '210000': ['130000', '150000', '220000'],
            '220000': ['150000', '210000', '230000'],
            '230000': ['150000', '220000'],
            '310000': ['320000', '330000'],
            '320000': ['310000', '330000', '340000', '370000'],
            '330000': ['310000', '320000', '340000', '360000', '350000'],
            '340000': ['320000', '330000', '360000', '410000', '420000'],
            '350000': ['330000', '360000', '440000'],
            '360000': ['330000', '340000', '350000', '420000', '430000'],
            '370000': ['130000', '320000', '410000'],
            '410000': ['130000', '140000', '340000', '370000', '420000', '610000'],
            '420000': ['340000', '360000', '410000', '430000', '500000', '610000'],
            '430000': ['360000', '420000', '440000', '450000', '510000', '520000'],
            '440000': ['350000', '430000', '450000', '460000'],
            '450000': ['430000', '440000', '520000', '530000'],
            '460000': ['440000'],
            '500000': ['420000', '510000', '520000', '610000'],
            '510000': ['430000', '500000', '520000', '530000', '540000', '610000', '620000', '630000'],
            '520000': ['430000', '450000', '500000', '510000', '530000'],
            '530000': ['450000', '510000', '520000', '540000'],
            '540000': ['510000', '530000', '620000', '630000', '650000'],
            '610000': ['140000', '150000', '410000', '420000', '500000', '510000', '620000', '640000', '650000'],
            '620000': ['150000', '510000', '540000', '610000', '630000', '640000', '650000'],
            '630000': ['510000', '540000', '620000', '650000'],
            '640000': ['150000', '610000', '620000'],
            '650000': ['540000', '610000', '620000', '630000']
        }
        
        code_to_idx = {code: i for i, code in enumerate(province_codes)}
        
        for code, neighbor_list in neighbors.items():
            if code not in code_to_idx:
                continue
            
            i = code_to_idx[code]
            
            for neighbor_code in neighbor_list:
                if neighbor_code not in code_to_idx:
                    continue
                
                j = code_to_idx[neighbor_code]
                adjacency[i, j] = 1
                adjacency[j, i] = 1
        
        n_neighbors = {province_names[i]: int(adjacency[i].sum()) 
                      for i in range(n_provinces)}
        
        logger.info(f"\nAdjacency matrix generation completed!")
        logger.info(f"  Matrix dimensions: {adjacency.shape}")
        logger.info(f"  Total adjacency relationships: {int(adjacency.sum() / 2)}")
        logger.info(f"  Average number of neighbors: {adjacency.sum(axis=1).mean():.2f}")
        
        result = {
            'adjacency_matrix': adjacency,
            'region_codes': province_codes,
            'region_names': province_names,
            'n_neighbors': n_neighbors,
            'n_regions': n_provinces,
            'metadata': {
                'level': 'province',
                'method': 'manual',
                'source': 'China administrative map'
            }
        }
        
        if True:
            output_dir = self.data_root / "features" / "spatial"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / "adjacency_matrix_province.npy", adjacency)
            
            with open(output_dir / "adjacency_matrix_province_metadata.pkl", 'wb') as f:
                pickle.dump({k: v for k, v in result.items() if k != 'adjacency_matrix'}, f)
            
            adjacency_list = []
            for i, code_i in enumerate(province_codes):
                for j, code_j in enumerate(province_codes):
                    if adjacency[i, j] == 1 and i < j:
                        adjacency_list.append({
                            'region_1_code': code_i,
                            'region_1_name': province_names[i],
                            'region_2_code': code_j,
                            'region_2_name': province_names[j]
                        })
            
            adj_df = pd.DataFrame(adjacency_list)
            adj_df.to_csv(output_dir / "adjacency_list_province.csv", 
                         index=False, encoding='utf-8-sig')
            
            logger.info(f"\n✅ Adjacency matrix saved:")
            logger.info(f"  - {output_dir / 'adjacency_matrix_province.npy'}")
            logger.info(f"  - {output_dir / 'adjacency_matrix_province_metadata.pkl'}")
            logger.info(f"  - {output_dir / 'adjacency_list_province.csv'}")
        
        return result
    
    def _calculate_adjacency_from_geodataframe(self, gdf, level):
        logger.info("Calculating adjacency relationships from GeoDataFrame...")
        
        n = len(gdf)
        adjacency = np.zeros((n, n), dtype=int)
        
        for i in range(n):
            for j in range(i+1, n):
                if gdf.geometry.iloc[i].touches(gdf.geometry.iloc[j]):
                    adjacency[i, j] = 1
                    adjacency[j, i] = 1
        
        logger.info(f"  Adjacency relationship calculation completed, found {int(adjacency.sum()/2)} pairs of adjacency relationships")
        
        return adjacency


def main():
    print("=== Unified Data Portal Demo ===\n")
    
    portal = UnifiedDataPortal()
    
    print("1. Loading data...")
    flu_data = portal.load_flu_data()
    search_data = portal.load_baidu_search_data()
    migration_data = portal.load_migration_data(sample_size=5)
    
    print("\n2. Data summary:")
    summary = portal.get_data_summary()
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n3. Creating unified dataset...")
    unified_data = portal.create_unified_dataset(
        start_date='2022-01-01',
        end_date='2023-12-31',
        aggregation_level='national'
    )
    
    print(f"Unified dataset shape: {unified_data.shape}")
    print(f"Column names: {list(unified_data.columns)}")
    
    print("\n4. Saving unified data...")
    success = portal.save_unified_data(unified_data, 'unified_flu_dataset.csv')
    if success:
        print("Data saved successfully!")
    
    print("\n=== Demo completed ===")


if __name__ == "__main__":
    main()
