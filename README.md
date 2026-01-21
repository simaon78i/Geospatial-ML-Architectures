# Geospatial Intelligence: Machine Learning Model Comparison

A deep dive into geospatial classification, comparing advanced Machine Learning architectures to predict country codes based on latitude and longitude coordinates. This project highlights the performance differences between Neural Networks (MLP) and Ensemble Learning (XGBoost).

## 🚀 Project Overview
This repository evaluates the effectiveness of different models on tabular spatial data. The core challenge was to map continuous coordinates into discrete political boundaries, analyzing how different algorithms handle "jagged" geographical borders and non-linear patterns.

### 🧪 Models Evaluated
* **Multi-Layer Perceptron (MLP)**: A 6-layer deep neural network built with **PyTorch**, featuring **Batch Normalization** and **ReLU** activations for stable and fast convergence.
* **XGBoost**: A gradient-boosted decision tree framework, used as a benchmark for high-performance classification on tabular data.
* **ResNet18 (Fine-tuning)**: An exploration of Transfer Learning for image-based classification (as a comparison point for spatial inductive bias).

## 📊 Performance & Key Results

### 1. Neural Network Optimization (MLP)
The MLP achieved a final test accuracy of **91.33%**. Key engineering insights included:
* **Optimal Learning Rate**: Through Grid Search, **0.001** was identified as the ideal balance between convergence speed and stability.
* **Regularization**: Batch Normalization was critical; without it, the 6-layer architecture suffered from slower training and lower accuracy.

### 2. Decision Boundaries & Inductive Bias
The project demonstrated that while MLPs are highly flexible, **XGBoost** often excels at capturing the axis-aligned nature of coordinate data. Tree-based models create precise "rectangular" partitions that mirror political borders more naturally than the smooth transitions of a neural network.

## 🛠 Tech Stack
* **Deep Learning**: PyTorch
* **Machine Learning**: XGBoost, Scikit-learn
* **Data Processing**: Python, NumPy, Pandas
* **Visualization**: Matplotlib

## 📈 Summary of Findings
1. **Model Selection**: For low-dimensional tabular data like coordinates, MLPs with proper normalization are highly competitive, though tree-based ensembles (XGBoost) remain a robust baseline.
2. **Transfer Learning**: In comparative tasks (Part 2 of the project), fine-tuning pre-trained models (ResNet18) proved vastly superior to training from scratch for high-fidelity tasks.

---
**Developed by Shimon Ifrach** *E2E Software Engineer & CS Student at The Hebrew University of Jerusalem*
