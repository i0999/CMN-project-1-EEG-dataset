# EEG Motor Task Classification with CSP + LDA

This notebook builds an EEG classification pipeline to distinguish **real vs imagined left-hand motor tasks** using the PhysioNet EEG Motor Movement/Imagery dataset. The workflow includes preprocessing, feature extraction with Common Spatial Patterns (CSP), classification with Linear Discriminant Analysis (LDA), cross-validation, and channel importance analysis.

## Overview

The pipeline performs the following steps:

1. **Data Loading**
   - Downloads EEG data for multiple subjects
   - Selects specific motor task runs (real vs imagined movement)

2. **Preprocessing**
   - Channel selection (motor cortex channels)
   - Average reference
   - Band-pass filtering (8–30 Hz)
   - Epoch extraction
   - Resampling

3. **Feature Extraction**
   - CSP transforms EEG signals into discriminative spatial components

4. **Classification**
   - LDA classifier
   - Stratified group cross-validation (subject-wise separation)

5. **Channel Importance Analysis**
   - Estimates relative channel contributions based on CSP patterns

## Dataset

The notebook automatically downloads EEG recordings from the PhysioNet motor movement/imagery dataset using MNE.

Tasks used:

- Real left-hand movement
- Imagined left-hand movement

## Model Pipeline

```
EEG → Epoching → Filtering → CSP → LDA → Cross-validation
```

Cross-validation is performed with subject grouping to avoid subject leakage.

## Results

The notebook reports:

- Cross-validation accuracy per fold
- Mean classification accuracy
- Ranked EEG channel importance

## Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run notebook

Open the notebook in Jupyter:

```bash
jupyter notebook
```

Execute all cells to:

- Download data
- Train the classifier
- View results and channel ranking


## Output Interpretation

Channel importance values reflect relative contribution to classification — **not causal brain activity**.

## License

For research and educational use. Dataset license follows PhysioNet terms.
