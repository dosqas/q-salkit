# 🧠 QNNs — Saliency & Sensitivity Kit

[![QAMP](https://img.shields.io/badge/QAMP-Qiskit%20Advocate%20Mentorship%20Program-blueviolet)](https://github.com/qiskit-advocate/qamp)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.3.1-6929C4)](https://qiskit.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB)](https://python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A comprehensive toolkit for **interpretability and explainability** in Quantum Neural Networks (QNNs) and Variational Quantum Classifiers (VQCs). This project provides implementations of saliency methods, sensitivity analysis, and faithfulness metrics adapted for quantum machine learning models.

## 📋 Table of Contents

- [Overview](#-overview)
- [Notebooks](#-notebooks)
- [Key Features](#-key-features)
- [Installation](#-installation)
- [Interpretability Methods](#-interpretability-methods)
- [Faithfulness Metrics](#-faithfulness-metrics)
- [Contributing](#-contributing)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

As quantum machine learning models become more powerful, understanding *why* they make certain predictions is crucial. This toolkit addresses the **interpretability gap** in QML by providing:

- **Feature Saliency**: Which input features matter most for predictions?
- **Parameter Sensitivity**: Which quantum gates are most critical?
- **Stability Analysis**: How robust are explanations to noise and randomness?
- **Faithfulness Metrics**: Are the explanations actually meaningful?

## 📓 Notebooks

| Notebook | Architecture | Dataset | Description |
|----------|-------------|---------|-------------|
| [hybrid_qnn_saliency.ipynb](hybrid_qnn_saliency.ipynb) | Hybrid QNN (Qiskit + PyTorch) | Iris (3-class) | Full interpretability suite with EstimatorQNN and TorchConnector |
| [hybrid_qnn_saliency_cleveland.ipynb](hybrid_qnn_saliency_cleveland.ipynb) | Hybrid QNN (Qiskit + PyTorch) | Cleveland Heart Disease | Medical diagnosis with comprehensive saliency analysis |
| [VQC_Iris.ipynb](VQC_Iris.ipynb) | Pure VQC (Qiskit Primitives) | Iris (binary) | Variational classifier with SPSA training |
| [VQC_Cleveland.ipynb](VQC_Cleveland.ipynb) | Pure VQC (Qiskit Primitives) | Cleveland Heart Disease | Binary heart disease classification with VQC |

### Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────┐
│                    HYBRID QNN ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│  Input → [Quantum Circuit (Qiskit)] → [Classical NN (PyTorch)] → Output │
│          EstimatorQNN + TorchConnector    Linear/Softmax        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     PURE VQC ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────┤
│  Input → [Data Encoding (RY)] → [Variational Layers] → ⟨Z⟩ → P(class) │
│          Feature angles        RZ/RY + CNOT ring      Estimator │
└─────────────────────────────────────────────────────────────────┘
```

## ✨ Key Features

### 🔬 Saliency Methods
- **Gradient-based saliency** via parameter-shift rule
- **SmoothGrad** for noise-robust attribution
- **Integrated Gradients** for path-based attribution
- **Gradient × Input** for scaled importance
- **Occlusion** for perturbation-based analysis
- **Noise Sensitivity** per feature

### 📊 Sensitivity Analysis
- **Gate knock-out analysis** — importance of individual quantum gates
- **Layer-wise sensitivity** — contribution of each variational layer
- **Seed stability** — reproducibility across random initializations
- **Shot noise robustness** — impact of finite sampling

### ✅ Faithfulness Metrics
- **Deletion Metric** — Does removing important features reduce confidence?
- **Average Sensitivity** — Are explanations stable under perturbations?
- **Saliency Entropy** — How focused are the explanations?
- **Saliency Sparseness** — L1/L2 ratio for concentration

### 📈 Visualizations
- Decision boundary plots
- Feature importance bar charts
- Training curves (loss & accuracy)
- Confusion matrices
- PCA embeddings of quantum layer outputs
- Partial dependence plots
- Spearman correlation heatmaps for stability

## 🚀 Installation

### Prerequisites
- Python 3.10+
- Conda (recommended) or pip


## 🔍 Interpretability Methods

### Parameter-Shift Rule for Input Saliency

For rotation gates encoding input features, we compute exact gradients:

$$\frac{\partial \langle Z \rangle}{\partial x_j} = \frac{1}{2}\left[\langle Z \rangle_{x_j + \frac{\pi}{2}} - \langle Z \rangle_{x_j - \frac{\pi}{2}}\right]$$

### Integrated Gradients

Path integral of gradients from baseline to input:

$$IG_j(x) = (x_j - x'_j) \times \frac{1}{m}\sum_{k=1}^{m} \frac{\partial f(x' + \frac{k}{m}(x-x'))}{\partial x_j}$$

### SmoothGrad

Noise-averaged gradients for stable explanations:

$$\text{SmoothGrad}(x) = \frac{1}{K} \sum_{k=1}^{K} \left|\nabla_x f(x + \epsilon_k)\right|, \quad \epsilon_k \sim \mathcal{N}(0, \sigma^2)$$

## 📐 Faithfulness Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Deletion** | $\Delta P$ when masking top-k features | Higher = more faithful |
| **Sensitivity** | $\mathbb{E}[\|\nabla f(x) - \nabla f(x+\epsilon)\|]$ | Lower = more stable |
| **Entropy** | $H(s) = -\sum \hat{s}_j \log(\hat{s}_j)$ | Lower = more focused |
| **Sparseness** | $\|s\|_1 / \|s\|_2$ | Lower = sparser |


## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add more datasets (MNIST, Fashion-MNIST quantum embeddings)
- [ ] Implement SHAP values for QNNs
- [ ] Add hardware noise models
- [ ] Create interactive visualizations with Plotly
- [ ] Benchmark against classical XAI methods

## 🙏 Acknowledgments

This project is part of the **Qiskit Advocate Mentorship Program (QAMP)**.

### Technologies Used

- [Qiskit](https://qiskit.org/) — Quantum computing framework
- [Qiskit Machine Learning](https://qiskit-community.github.io/qiskit-machine-learning/) — QNN and VQC implementations
- [PyTorch](https://pytorch.org/) — Deep learning for hybrid models
- [scikit-learn](https://scikit-learn.org/) — Classical ML utilities

### Datasets

- **Iris Dataset** — Fisher, R. A. (1936). UCI Machine Learning Repository
- **Cleveland Heart Disease** — Detrano, R. et al. (1989). UCI Machine Learning Repository

---

<p align="center">
  <i>Part of the QAMP Project: "QNNs — Saliency & Sensitivity Kit"</i>
</p>

<p align="center">
  Made with ❤️ and ⚛️
</p>
