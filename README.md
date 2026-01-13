# IoT Intrusion Detection System

## 🛡️ Overview

This project implements a **Machine Learning-based Intrusion Detection System** for IoT (Internet of Things) networks. The system uses audit data from network traffic to classify whether network activity is **benign** or **malicious**.

## 📊 Dataset

- **Source**: Kaggle IoT Network Intrusion Dataset
- **Size**: ~1 million records with 46 features
- **Labels**: 34 attack types + Benign Traffic

### Attack Categories

| Category | Examples |
|----------|----------|
| DDoS Attacks | ICMP Flood, SYN Flood, UDP Flood, TCP Flood, etc. |
| DoS Attacks | HTTP Flood, SYN Flood, UDP Flood |
| Mirai Botnet | greeth_flood, greip_flood, udpplain |
| Reconnaissance | Port Scan, OS Scan, Host Discovery |
| Web Attacks | SQL Injection, XSS, Command Injection |
| Others | Backdoor, Spoofing, Brute Force |

## 🤖 Machine Learning Models

This system implements **three classification algorithms**:

### 1. Decision Tree Classifier
- **Type**: Supervised learning, tree-based
- **Advantages**: Interpretable, fast training, no feature scaling needed
- **Use Case**: Quick initial classification with explainable results

### 2. AdaBoost (Adaptive Boosting)
- **Type**: Ensemble method (boosting)
- **Advantages**: Combines weak learners, reduces bias, handles imbalanced data
- **Use Case**: Improved accuracy through iterative refinement

### 3. Support Vector Machine (SVM)
- **Type**: Supervised learning, kernel-based
- **Advantages**: Effective in high-dimensional spaces, robust to overfitting
- **Use Case**: Complex decision boundaries for nuanced classification

## 🚀 Getting Started

### Prerequisites

```bash
# Python 3.8 or higher
python --version

# Install dependencies
pip install -r requirements.txt
```

### Running the System

```bash
# Run the complete pipeline
python iot_intrusion_detection.py
```

### Expected Output

1. **Data Loading**: Dataset statistics and label distribution
2. **Preprocessing**: Feature scaling and train/test split
3. **Training**: Three models trained with progress indicators
4. **Evaluation**: Accuracy, Precision, Recall, F1-Score for each model
5. **Visualizations**: Performance charts saved to `model_outputs/`
6. **Models Saved**: Trained models saved as `.joblib` files

## 📈 Performance Metrics

| Metric | Description |
|--------|-------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **F1-Score** | Harmonic mean of Precision and Recall |

## 📁 Project Structure

```
IoT_Intrusion_Detection/
├── IoT_Intrusion.csv           # Dataset
├── iot_intrusion_detection.py  # Main detection system
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── model_outputs/              # Generated outputs
    ├── decision_tree_model.joblib
    ├── adaboost_model.joblib
    ├── svm_model.joblib
    ├── scaler.joblib
    ├── label_encoder.joblib
    ├── performance_comparison.png
    ├── confusion_matrices.png
    ├── radar_chart.png
    └── results_summary.csv
```

## 🔧 Configuration

You can adjust the following parameters in `iot_intrusion_detection.py`:

### Data Loading
```python
detector.load_data(sample_size=100000)  # Adjust sample size
```

### Classification Mode
```python
# Binary: Attack vs Benign
detector.preprocess_data(binary_classification=True)

# Multi-class: All 34 attack types
detector.preprocess_data(binary_classification=False)
```

### Model Hyperparameters
```python
# Decision Tree
detector.train_decision_tree(max_depth=20, min_samples_split=10)

# AdaBoost
detector.train_adaboost(n_estimators=100, learning_rate=1.0)

# SVM
detector.train_svm(kernel='rbf', C=1.0, gamma='scale')
```

## 📊 Making Predictions

```python
import pandas as pd
import joblib

# Load saved model
model = joblib.load('model_outputs/decision_tree_model.joblib')
scaler = joblib.load('model_outputs/scaler.joblib')
label_encoder = joblib.load('model_outputs/label_encoder.joblib')

# Prepare new data (must have same features as training data)
new_data = pd.read_csv('new_network_traffic.csv')
new_data_scaled = scaler.transform(new_data)

# Predict
predictions = model.predict(new_data_scaled)
labels = label_encoder.inverse_transform(predictions)
print(labels)  # ['Attack', 'Benign', 'Attack', ...]
```

## 📚 References

- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Decision Tree Classifier](https://scikit-learn.org/stable/modules/tree.html)
- [AdaBoost Classifier](https://scikit-learn.org/stable/modules/ensemble.html#adaboost)
- [Support Vector Machines](https://scikit-learn.org/stable/modules/svm.html)

## 👨‍💻 Author

Created for IoT Network Security Analysis - Year 3 Course Work

---

**Note**: This system is designed for educational and research purposes. For production environments, additional security measures and validation should be implemented.
