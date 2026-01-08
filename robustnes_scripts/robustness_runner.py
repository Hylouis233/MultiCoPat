#!/usr/bin/env python3

import os
import sys
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import robust_stl
import robust_afd
import robust_cwt
import robust_wtc
from data_portal import UnifiedDataPortal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('robustness_experiment.log')
    ]
)
logger = logging.getLogger(__name__)

SKIP_STEPS_BEFORE_WTC = True

BASE_OUTPUT_DIR = Path("Robustness_Results")

BASELINE_CONFIG = {
    'name': 'Baseline',
    'description': 'Standard parameter settings used in main analysis',
    'stl_config': {
        'seasonal': 13,
        'robust': True
    },
    'afd_config': {
        'energy_threshold': 0.95,
        'max_iter': 50,
        'skip_afd': False
    },
    'cwt_config_seasonal': {
        'wavelet': 'cmor',
        'wavelet_params': (1.5, 1.0),
        'scale_range': (26, 52)
    },
    'wtc_config_seasonal': {
        'wavelet': 'cmor',
        'wavelet_params': (1.5, 1.0),
        'significance_alpha': 0.05,
        'n_surrogates': 200
    }
}

CONSERVATIVE_CONFIG = {
    'name': 'Conservative',
    'description': 'Stricter parameters to test result stability',
    'stl_config': {
        'seasonal': 7,
        'robust': True
    },
    'afd_config': {
        'energy_threshold': 0.99,
        'max_iter': 100,
        'skip_afd': False
    },
    'cwt_config_seasonal': {
        'wavelet': 'cmor',
        'wavelet_params': (3.0, 1.0),
        'scale_range': (26, 52)
    },
    'wtc_config_seasonal': {
        'wavelet': 'cmor',
        'wavelet_params': (3.0, 1.0),
        'significance_alpha': 0.01,
        'n_surrogates': 200
    }
}

SENSITIVE_WAVELET_CONFIG = {
    'name': 'Sensitive_Cgau',
    'description': 'Using Complex Gaussian (cgau5) wavelet to test time localization sensitivity',
    'stl_config': BASELINE_CONFIG['stl_config'],
    'afd_config': BASELINE_CONFIG['afd_config'],
    'cwt_config_seasonal': {
        'wavelet': 'cgau5',
        'wavelet_params': None,
        'scale_range': (26, 52)
    },
    'wtc_config_seasonal': {
        'wavelet': 'cgau5',
        'wavelet_params': (1.0, 1.0),
        'significance_alpha': 0.05,
        'n_surrogates': 200
    }
}

SENSITIVE_SHANNON_CONFIG = {
    'name': 'Sensitive_Shannon',
    'description': 'Using Shannon wavelet (high freq resolution) to test spectral sensitivity',
    'stl_config': BASELINE_CONFIG['stl_config'],
    'afd_config': BASELINE_CONFIG['afd_config'],
    'cwt_config_seasonal': {
        'wavelet': 'shan1.5-1.0', 
        'wavelet_params': None,
        'scale_range': (26, 52)
    },
    'wtc_config_seasonal': {
        'wavelet': 'shan1.5-1.0',
        'wavelet_params': None,
        'significance_alpha': 0.05,
        'n_surrogates': 200
    }
}

ABLATION_NO_AFD_CONFIG = {
    'name': 'Ablation_NoAFD',
    'description': 'Skipping AFD decomposition to verify its contribution',
    'stl_config': BASELINE_CONFIG['stl_config'],
    'afd_config': {
        'skip_afd': True
    },
    'cwt_config_seasonal': BASELINE_CONFIG['cwt_config_seasonal'],
    'wtc_config_seasonal': BASELINE_CONFIG['wtc_config_seasonal'],
    'analysis_config': {
        'reconstruct_r_from_afd': False,
        'enable_residual_analysis': True
    }
}

def add_noise_to_data(data_portal, noise_level=0.1):
    logger.warning("Data noise injection not fully implemented yet. Skipping noise addition.")
    return data_portal

def run_scenario(scenario_config):
    scenario_name = scenario_config['name']
    logger.info(f"Starting Scenario: {scenario_name}")
    logger.info(f"Description: {scenario_config['description']}")
    
    output_dir = BASE_OUTPUT_DIR / scenario_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stl_out_dir = output_dir / "stl_decomposition"
    afd_out_dir = output_dir / "afd_analysis"
    cwt_out_dir = output_dir / "cwt_analysis"
    wtc_out_dir = output_dir / "wtc_analysis"

    if not SKIP_STEPS_BEFORE_WTC:
        logger.info(f"[{scenario_name}] Running STL Decomposition...")
        stl_params = {
            'stl_config': scenario_config.get('stl_config', {}),
            'output_config': {'base_dir': str(stl_out_dir)}
        }
        robust_stl.run_stl(config=stl_params, output_dir=stl_out_dir)
    else:
        logger.info(f"[{scenario_name}] Skipping STL Decomposition (SKIP_STEPS_BEFORE_WTC=True)")
        stl_result_file = stl_out_dir / 'arrays/stl_results.npy'
        if not stl_result_file.exists() and scenario_name != 'Baseline':
            baseline_stl_dir = BASE_OUTPUT_DIR / 'Baseline' / 'stl_decomposition'
            if (baseline_stl_dir / 'arrays/stl_results.npy').exists():
                logger.info(f"[{scenario_name}] Missing STL files. Copying from Baseline...")
                if stl_out_dir.exists():
                    shutil.rmtree(stl_out_dir)
                shutil.copytree(baseline_stl_dir, stl_out_dir)
            else:
                logger.warning(f"[{scenario_name}] Missing STL files and Baseline not found.")

    stl_data_path = {
        'stl_results_npy': str(stl_out_dir / 'arrays/stl_results.npy'),
        'stl_results_pkl': str(stl_out_dir / 'arrays/stl_results.pkl')
    }
    
    if not SKIP_STEPS_BEFORE_WTC:
        logger.info(f"[{scenario_name}] Running AFD Analysis...")
        afd_params = {
            'afd_config': scenario_config.get('afd_config', {}),
            'stl_data_path': stl_data_path,
            'output_config': {'base_dir': str(afd_out_dir)}
        }
        robust_afd.run_afd(config=afd_params, output_dir=afd_out_dir)
    else:
        logger.info(f"[{scenario_name}] Skipping AFD Analysis (SKIP_STEPS_BEFORE_WTC=True)")
        afd_result_file = afd_out_dir / 'afd_residual_components.npy'
        if not afd_result_file.exists() and scenario_name != 'Baseline':
             baseline_afd_dir = BASE_OUTPUT_DIR / 'Baseline' / 'afd_analysis'
             if (baseline_afd_dir / 'afd_residual_components.npy').exists():
                 logger.info(f"[{scenario_name}] Missing AFD files. Copying from Baseline...")
                 if afd_out_dir.exists():
                     shutil.rmtree(afd_out_dir)
                 shutil.copytree(baseline_afd_dir, afd_out_dir)
    
    afd_data_path = {
        'residual_components_npy': str(afd_out_dir / 'afd_residual_components.npy'),
        'component_numbers_csv': str(afd_out_dir / 'afd_component_numbers.csv'),
        'afd_metadata_pkl': str(afd_out_dir / 'afd_stl_metadata.pkl')
    }
    
    if not SKIP_STEPS_BEFORE_WTC:
        logger.info(f"[{scenario_name}] Running CWT Analysis...")
        cwt_params = {
            'cwt_config_seasonal': scenario_config.get('cwt_config_seasonal', {}),
            'stl_data_path': stl_data_path,
            'afd_data_path': afd_data_path,
            'output_config': {'base_dir': str(cwt_out_dir)},
            'analysis_config': scenario_config.get('analysis_config', {})
        }
        robust_cwt.run_cwt(config=cwt_params, output_dir=cwt_out_dir)
    else:
        logger.info(f"[{scenario_name}] Skipping CWT Analysis (SKIP_STEPS_BEFORE_WTC=True)")
    
    logger.info(f"[{scenario_name}] Running WTC Analysis...")
    wtc_out_dir = output_dir / "wtc_analysis"
    
    wtc_params = {
        'wtc_config_seasonal': scenario_config.get('wtc_config_seasonal', {}),
        'stl_data_path': stl_data_path,
        'output_config': {'base_dir': str(wtc_out_dir)},
        'analysis_config': {
            'max_flu_series': 10,
            'max_env_series': 10,
            'enable_bootstrap_phase': True,
            'enable_seasonal_wtc': True
        }
    }
    robust_wtc.run_wtc(config=wtc_params, output_dir=wtc_out_dir)
    
    logger.info(f"Scenario {scenario_name} completed successfully.")

def main():
    logger.info("Starting Robustness Experiment Runner")
    
    scenarios = [
        BASELINE_CONFIG,
        CONSERVATIVE_CONFIG,
        SENSITIVE_WAVELET_CONFIG,
        ABLATION_NO_AFD_CONFIG
    ]
    
    for scenario in scenarios:
        run_scenario(scenario)
    
    logger.info("All robustness experiments completed.")

if __name__ == "__main__":
    main()
