# Thermal Imaging Analysis for Diabetic Foot Assessment

A comprehensive pipeline for processing and analyzing thermal images to assess diabetic foot complications.

## Project Overview

This project implements a machine learning pipeline for analyzing thermal images of diabetic feet. It includes preprocessing, feature extraction, risk assessment, and visualization components to help identify potential complications early.

### Key Features

- Advanced thermal image preprocessing and standardization
- Automated foot region segmentation and alignment
- Feature extraction from multiple anatomical regions
- Multi-model machine learning approach (Random Forest and CNN)
- Risk assessment based on temperature patterns and asymmetry
- Comprehensive visualization tools
- Longitudinal patient monitoring capabilities

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for CNN training)

### Required Libraries

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
python main.py
```

### Advanced Configuration

You can customize the analysis by modifying the configuration parameters:

```python
preprocessing_config = {
    'base_resolution': (224, 224),
    'temp_range': (20, 40)  # Temperature range in Celsius
}

model_config = {
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2
}

dataset = ThermalImageDataset('./data/train', preprocessing_config)
analyzer = ThermalImageAnalyzer(model_config)
```

## Output Structure

The pipeline generates organized outputs:

```
outputs_[timestamp]/
├── models/
│   ├── random_forest_model.joblib
│   ├── cnn_model/
│   └── scaler.joblib
├── visualizations/
│   ├── risk_distribution.png
│   ├── temperature_heatmaps/
│   ├── feature_importance.png
│   └── risk_dashboard.png
└── model_config.json
```

## Model Performance Metrics

The system evaluates models using:

- Classification metrics (precision, recall, F1)
- ROC curves and AUC scores
- Confusion matrices
- Cross-validation scores
- Feature importance rankings

## Data Requirements

- Image Format: 16-bit PNG files
- Resolution: Minimum 224x224 pixels
- Temperature Range: 20-40°C
- Clear foot region visibility
- Consistent imaging conditions

**Database: https://www.kaggle.com/datasets/vuppalaadithyasairam/thermography-images-of-diabetic-foot**

## License

This project is licensed under the MIT License - see the LICENSE file for details.
