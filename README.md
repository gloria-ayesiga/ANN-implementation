# ANN-implementation
Breast Cancer Diagnosis with Artificial Neural Network (TensorFlow/Keras)

A TensorFlow-based Artificial Neural Network (ANN) for binary classification of breast masses as **benign** or **malignant** using radiomics features from the Breast Cancer Wisconsin (Diagnostic) dataset.  

Developed as an educational demonstration of how machine learning can support radiologists in cancer diagnosis (e.g. at institutions like MD Anderson Cancer Institute).  

**Important Disclaimer**: This is an **educational/research project only**. The model is **not** a medical device, has **not** undergone clinical validation, and should **never** be used for real patient diagnosis or treatment decisions.

## Project Highlights

**Dataset**: UCI Breast Cancer Wisconsin (Diagnostic) – 569 samples, 30 numeric features derived from digitized FNA images (nucleus radius, texture, perimeter, area, smoothness, etc.)
**Model**: Feedforward ANN with 3 hidden layers (32 → 16 → 8 neurons), ReLU + BatchNormalization + Dropout

- **Performance on Test Set** (86 samples):
  - Accuracy: **98.84%**
  - AUC: **0.9994**
  - Precision (Malignant): **1.0000** (no false positives)
  - Recall (Malignant): **0.9688** (~96.9% of true cancers detected)
  - F1-score (Malignant): **0.98**
    
 **Training**: Adam optimizer, binary cross-entropy loss, early stopping, class weighting explored
 **Key Insight**: Even a simple ANN achieves near-perfect results on well-extracted radiomics features strong proof-of-concept for imaging-based cancer support tools.

## Features

- Full end-to-end pipeline: data loading → preprocessing → model building → training → evaluation → visualization
- Interactive learning curves (loss & accuracy)
- Confusion matrix, classification report, ROC curve
- Early stopping & batch normalization for stable training
- Unit-test friendly structure

## Requirements

- Python 3.8+
- Libraries:
  
  pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
