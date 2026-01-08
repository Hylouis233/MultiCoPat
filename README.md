# MultiCoPat

**Time-Frequency Dynamics of Influenza Co-circulation based on STL-AFD-CWT-WTC**

[English](README.md) | [简体中文](README_CN.md)

This repository contains the code and analysis pipeline for the study **"Time-Frequency Dynamics of Influenza Co-circulation based on STL-AFD-CWT-WTC"**.

## Abstract

**Objective**: Co-circulation of multiple influenza virus subtypes poses a major challenge to global public health surveillance and prediction. Although spatiotemporal heterogeneity in influenza transmission has been documented, the specific impact of viral co-circulation on non-stationary epidemic sequences remains poorly understood. This study aims to reveal the impact of co-circulation on influenza transmission systems and their coupling relationships with environmental driving factors through time-frequency analysis.

**Methods**: We integrated an analytical framework combining **Seasonal-Trend decomposition using Loess (STL)**, **Adaptive Fourier Decomposition (AFD)**, **Continuous Wavelet Transform (CWT)**, and **Wavelet Coherence (WTC)**. We analyzed 323 weekly influenza surveillance time series from China spanning 2011 to 2025, and quantitatively assessed the time-frequency coupling strength of 63,308 influenza-environment variable pairs.

**Results**: The study found that co-circulation periods and single-dominant periods have fundamentally different dynamic mechanisms, with co-circulation periods exhibiting a regime shift of the influenza system from order to chaos. During co-circulation periods, seasonality shift significantly increased, and high-intensity anomalous fluctuations emerged in the residual component. Furthermore, this study revealed significant north-south mechanism differences from a data perspective: northern regions exhibited "Environmental Locking", where even during co-circulation periods, epidemics remain strongly constrained by harsh climate; while southern regions showed "Environmental Decoupling", particularly during H3N2 co-circulation periods, where ecological competition between viruses surpassed environmental driving forces (such as humidity and temperature), leading to the failure of traditional environmental predictive factors.

**Conclusions**: The results of this study demonstrate the unique time-frequency state of influenza co-circulation periods. Influenza co-circulation is not simply a superposition of strains but triggers changes in the influenza transmission system. This indicates that public health strategies need to be context-adaptive: northern regions can continue to rely on meteorological warnings, while during co-circulation periods in southern regions, reliance on environmental indicators should be reduced, instead strengthening real-time etiological and serological surveillance.

![Framework](figures/figure1_framework.pdf)
*Figure 1: The Four-Step Multi-Scale Analysis Framework (STL-AFD-CWT-WTC).*

## Key Findings

1.  **Regime Shift from Order to Chaos**:
    *   Co-circulation periods exhibit fundamentally different dynamic mechanisms compared to single-dominant periods.
    *   During co-circulation, the influenza system undergoes a regime shift, characterized by significantly increased **Seasonality Shift** and the emergence of high-intensity anomalous fluctuations in the residual component.

    ![STL Dynamics Comparison](figures/figure3_stl_comparison.pdf)
    *Figure 3: Comparison of STL dynamics between single-dominant and co-circulation periods. Note the increased residual volatility during co-circulation.*

2.  **Spatiotemporal Heterogeneity**:
    *   **Northern China ("Environmental Locking")**: Epidemics remain strongly constrained by harsh climate conditions even during co-circulation periods.
    *   **Southern China ("Environmental Decoupling")**: Ecological competition between viruses often surpasses environmental driving forces (such as humidity and temperature). This is particularly evident during H3N2 co-circulation periods, leading to the failure of traditional environmental predictive factors.

    ![Integrated Statistics](figures/figure6_integrated_stats.pdf)
    *Figure 6: Integrated statistical analysis showing distinct patterns between Northern and Southern China.*

3.  **Implications for Public Health**:
    *   Strategies need to be **context-adaptive**.
    *   **North**: Continue relying on meteorological warnings.
    *   **South**: During co-circulation, reduce reliance on environmental indicators and strengthen real-time etiological and serological surveillance.

## Methodology: The Four-Step Framework

The analysis pipeline follows a "divide-and-conquer" strategy:

1.  **STL Decomposition**: Isolates macroscopic trends ($T_t$) and stable seasonality ($S_t$) from the original time series.
2.  **Adaptive Fourier Decomposition (AFD)**: Extracts physically meaningful transient components from high-frequency residuals ($R_t$) to capture nonlinear dynamics.
3.  **Continuous Wavelet Transform (CWT)**: Reveals time-frequency fingerprints of viral spread and transient features of subtype competition.
4.  **Wavelet Transform Coherence (WTC)**: Quantifies the dynamic coupling between influenza transmission and environmental drivers.

## Repository Structure

- `src/`: Core analysis scripts implementing the methodology.
  - `data_portal.py`: Unified interface for loading and preprocessing data.
  - `stl_decomposition.py`: Step 1 - Seasonal-Trend decomposition.
  - `AFD_analysis.py`: Step 2 - Adaptive Fourier Decomposition.
  - `cwt_analysis.py`: Step 3 - Continuous Wavelet Transform.
  - `wtc_analysis.py`: Step 4 - Wavelet Coherence.
  - `comprehensive_analysis.py`: Integrated statistical analysis and pattern recognition.

- `figures/`: Scripts to reproduce the figures in the manuscript.
  - `plot_figure*.py`: Scripts for specific figures (e.g., Figure 3: STL Dynamics Comparison).
  - `utils.py`: Shared plotting utilities.

- `robustness/`: Scripts for sensitivity analysis.
  - `robustness_runner.py`: Driver for robustness experiments (Baseline, Conservative, Sensitive, Ablation scenarios).

- `data/`: Directory for input data and processed results.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Hylouis233/MultiCoPat.git
   cd MultiCoPat
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Data Preparation
Place your raw data in the `data/` directory.

### 2. Run Analysis Pipeline
Execute the core analysis scripts in order:
```bash
python src/stl_decomposition.py
python src/AFD_analysis.py
# ... and so on
```

### 3. Reproduce Figures
To generate the figures (e.g., Figure 3):
```bash
python figures/plot_figure3_stl_comparison.py
```
The outputs will be saved in `results/figures/`.

### 4. Run Robustness Checks
```bash
python robustness/robustness_runner.py
```