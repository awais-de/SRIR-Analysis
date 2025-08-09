# Statistical Analysis of Spatial Room Impulse Responses

## Technologies and Concepts Used

- **Tools and Technologies**:
  - MATLAB (Version R2023a or later)
  - MATLAB Statistics and Machine Learning Toolbox (for functions like `pca`, `fitcdiscr`, `factoran`, and `confusionchart`)
  - SDMtools (Spatial Decomposition Method toolbox for acoustic parameter extraction)
  - Custom MATLAB functions (e.g., `create_RAP_Mask` for RAP selection)

- **Concepts**:
  - Dimensionality Reduction Techniques: Principal Component Analysis (PCA), Linear Discriminant Analysis (LDA), Exploratory Factor Analysis (EFA)
  - Data Preprocessing: Z-score Standardization, Feature Extraction from Spatial Room Impulse Responses (SRIRs)
  - Classification: Supervised Learning with Discriminant Analysis, Train/Test Splitting (70/30 Stratified Hold-Out)
  - Evaluation Metrics: Confusion Matrices, Classification Accuracy
  - Room Acoustics Parameters: EDT (Early Decay Time), DT20m (Decay Time 20 dB), D50 (Definition 50 ms), C50/C80 (Clarity Indices), DRR (Direct-to-Reverberant Ratio), Grel (Relative Strength), TS (Centre Time)
  - Octave Band Analysis (62 Hz to 8 kHz)
  - Visualization: Heatmaps, Scatter Plots, Bar Charts

This project leverages these to analyze and classify room acoustics based on SRIR datasets.

## Table of Contents

- [Overview](#overview)
- [Project Objectives](#project-objectives)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Files in This Repository](#files-in-this-repository)
- [Requirements](#requirements)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Results](#results)
- [Contributors](#contributors)
- [References](#references)
- [License](#license)
- [Contact](#contact)

## Overview

Understanding room acoustics is essential for applications in architectural design, audio engineering, and immersive audio systems. This project analyzes SRIRs to:
- Evaluate acoustic quality across different environments.
- Cluster rooms based on key acoustic parameters.
- Compare statistical methods (PCA, LDA, EFA) for dimensionality reduction and classification.
- Identify parameter sensitivity (e.g., which parameters like DT20m, DRR, or Grel have the highest impact).

The project builds on prior PCA work by the supervising team and extends it with LDA (supervised) and EFA (unsupervised) for deeper insights.

## Project Objectives

- Evaluate and cluster rooms using acoustic parameters.
- Discover parameter sensitivity in PCA, LDA, and EFA.
- Compare classification accuracy across methods.
- Provide visualizations like confusion matrices, heatmaps, and scatter plots for interpretability.

## Dataset

- **Source**: High-resolution SRIR dataset from Stolz et al. (2024), available on Zenodo (DOI: 10.5281/zenodo.10450779).
- **Rooms Analyzed**: H1539b, H2505, H3522-HW, HL-WV, ML2-102.
- **Measurement Setup**: TORY robot with a 7-channel spherical microphone array; exponential sine sweeps.
- **Acoustic Parameters**: EDT, DT20m, D50, C50, C80, DRR, Grel, TS (across octave bands: 62 Hz to 8 kHz).
- **Preprocessing**: Z-score normalization (zero mean, unit variance).

The dataset captures temporal, spatial, and directional sound propagation in coupled rooms. Note that the user must download the dataset from the Zenodo source to run the code, as it is not included in this repository due to size and privacy considerations.

## Methodology

1. **Data Preprocessing**: Load .mat files, extract parameters, standardize features.
2. **Dimensionality Reduction**:
   - **PCA**: Variance-based (baseline from supervising team).
   - **LDA**: Class-separability-based (band-wise and full-matrix).
   - **EFA**: Correlation-based latent factor extraction (Kaiser criterion, Varimax rotation).
3. **Classification**: Stratified 70/30 train-test split; LDA-based classifier on reduced features.
4. **Evaluation**: Confusion matrices, accuracy metrics, parameter sensitivity (exclude parameters sequentially).
5. **Tools**: MATLAB for analysis and visualization.

## Files in This Repository

- **EMT2 Final Presentation.pptx**: PowerPoint slides covering project purpose, background, methodology, results, and parameter sensitivity.
- **EMT2-Final Report.pdf**: Detailed 7-page report including abstract, objectives, dataset description, methodology, results (with figures), and conclusions.
- **srir_analysis.m**: MATLAB code script for comparing PCA, LDA, and EFA, including data loading, standardization, feature extraction, classification, confusion matrices, and accuracy computation. The script is fully commented and structured for clarity.
- **README.md**: This file (project documentation).
- **images/**: (Placeholder for any extracted images from the presentation/report, e.g., image1.png, image2.svg – not included in pasted content but referenced).

Note: The code assumes a `data/` folder with .mat files (not included due to size/privacy; refer to Zenodo dataset).

## Requirements

- **MATLAB**: Version R2023a or later.
- **Toolboxes**: Statistics and Machine Learning Toolbox (for `fitcdiscr`, `factoran`, etc.).
- **Libraries**: SDMtools (Spatial Decomposition Method for acoustic parameter extraction – add via `addpath`).
- No external installations needed beyond MATLAB; code uses built-in functions.

## Installation and Setup

1. Clone the repository:
git clone https://github.com/awais-de/SRIR-Analysis.git
cd SRIR-Analysis

2. Download the SRIR dataset from [Zenodo](https://doi.org/10.5281/zenodo.10450779) and place .mat files in a `data/` folder. The user must download the dataset to proceed with running the analysis script.
3. Add required paths in MATLAB (as per the code: `addpath(genpath('../SDMtools/')); addpath(genpath('functions/'));`).
4. Ensure SDMtools is available (download from relevant sources if needed).

## Usage

1. Open MATLAB and navigate to the repository folder.
2. Run the script:
run('srir_analysis.m');

3. The script will:
- Load and preprocess data from the `data/` folder (requires dataset download from Zenodo).
- Perform PCA, LDA, and EFA for dimensionality reduction.
- Train classifiers and generate confusion matrices for each method.
- Compute and display classification accuracies (e.g., PCA: 99.11%, LDA: 94.04%, EFA: 94.54%).
- Plot a bar chart comparing classification accuracies.
4. View the presentation (.pptx) or report (.pdf) for detailed explanations and figures.

Example Output (from code):
Classification Accuracy Comparison:
PCA Accuracy: 99.11%
LDA Accuracy: 94.04%
EFA Accuracy: 94.54%


## Results

- **Classification Accuracies**:
  | Method | Accuracy (%) |
  |--------|--------------|
  | PCA    | 99.11       |
  | LDA    | 94.04       |
  | EFA    | 94.54       |

- **Key Findings**:
  - PCA excels in classification but is unsupervised.
  - LDA highlights DT20m as most sensitive for classification.
  - EFA identifies DRR and Grel as critical; uncovers latent factors (e.g., temporal decay, clarity).
  - Confusion matrices show minimal misclassifications, with overlaps in acoustically similar rooms.
  - Visualizations: Heatmaps (coefficients/loadings), scatter plots (1D/2D/3D clustering).

For full results, refer to the report's figures (e.g., LDA Coefficient Heatmap, EFA Factor Score Plots).

## Contributors

- Muhammad Awais (muhammad.awais@tu-ilmenau.de)

Supervising Team: G. Stolz, G. Götz, L. Treybig, S. Werner, F. Klein.

## References

1. G. Stolz, G. Götz, L. Treybig, S. Werner, and F. Klein, “Spatial Room Impulse Response Dataset: A Robot’s Journey Through Coupled Rooms of a Reverberant University Building,” 2024.
2. L. Treybig, F. Klein, G. Stolz, and S. Werner, “A high spatial resolution dataset of spatial room impulse responses for different acoustic room configurations,” https://doi.org/10.5281/zenodo.10450779, 2024.
3. Additional references from the report (e.g., Jolliffe on PCA, Fabrigar on EFA).

Full list available in the report's References section.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (create one if needed).

## Contact

For questions or collaborations, reach out to:
- Muhammad Awais: muhammadawais.de@gmail.com

Feel free to fork, star, or contribute!