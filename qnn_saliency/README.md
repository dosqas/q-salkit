# QNN Saliency

**Saliency and Sensitivity Toolkit for Quantum Neural Networks**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![Tests](https://img.shields.io/badge/tests-64%20passing-brightgreen.svg)]()

A model-agnostic interpretability library for Quantum Neural Networks (QNNs), providing saliency methods and faithfulness metrics. Designed for seamless integration with Qiskit Machine Learning and potential contribution to the Qiskit ecosystem.

## 🎯 Overview

Understanding *why* a quantum model makes certain predictions is crucial for debugging, improving, and trusting QNNs. This toolkit brings classical interpretability techniques to the quantum domain:

- **Compute feature importance** for any QNN prediction
- **Evaluate explanation quality** with faithfulness metrics
- **Compare different saliency methods** for robustness
- **Diagnose model behavior** with advanced metrics

## ✨ Features

### Saliency Methods

| Method | Description | Best For |
|--------|-------------|----------|
| `GradientSaliency` | Basic gradient-based attribution | Fast baseline |
| `GradientTimesInput` | Gradient × Input product | Signed attribution |
| `SmoothGrad` | Noise-robust gradient averaging | Noisy models |
| `IntegratedGradients` | Path-integral attribution (axiomatic) | Rigorous analysis |
| `Occlusion` | Perturbation-based importance | Model-agnostic |
| `NoiseSensitivity` | Feature-wise noise sensitivity | Robustness testing |

### Gradient Engines

| Engine | Description |
|--------|-------------|
| `FiniteDifferenceGradient` | Numerical gradients (works with any model) |
| `ParameterShiftGradient` | Exact quantum gradients via π/2 shift rule |

### Model Wrappers

| Wrapper | Use Case |
|---------|----------|
| `VQCWrapper` | Qiskit VQC, SamplerQNN, EstimatorQNN |
| `HybridQNNWrapper` | PyTorch hybrid quantum-classical models |

### Metrics

**Faithfulness Metrics** — *"Does the explanation reflect the model?"*
- `deletion_score`: Area Under Deletion Curve (AUDC) — lower is better
- `insertion_score`: Area Under Insertion Curve (AUIC) — higher is better
- `average_sensitivity`: Explanation stability under perturbations

**Distribution Metrics** — *"How is importance distributed?"*
- `saliency_entropy`: Concentration of attribution
- `saliency_sparseness`: L1/L2 sparseness ratio
- `hoyer_sparseness`: Hoyer sparseness measure
- `saliency_concentration`: Top-k feature concentration
- `gini_coefficient`: Inequality of attribution

**Diagnostic Metrics** — *"What can we learn about the model?"*
- `bug_detection_signal`: Compare correct vs incorrect predictions
- `minimum_efficacy`: Single-feature importance test
- `overreliance_risk`: Confidence-fidelity correlation
- `feature_agreement`: Cross-method consistency

## 📦 Installation

```bash
# Clone and install in editable mode
git clone https://github.com/your-org/qnn-saliency.git
cd qnn-saliency
pip install -e .

# With optional dependencies
pip install -e ".[torch]"    # PyTorch support for HybridQNNWrapper
pip install -e ".[dev]"      # Development tools (pytest, black, mypy)
pip install -e ".[all]"      # Everything
```

**Requirements:**
- Python ≥ 3.9
- NumPy ≥ 1.21
- Qiskit ≥ 1.0
- Qiskit Machine Learning ≥ 0.7

## 🚀 Quick Start

### Basic Usage with Qiskit VQC

```python
import numpy as np
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

from qnn_saliency import (
    VQCWrapper,
    GradientSaliency,
    FiniteDifferenceGradient,
    deletion_score,
    saliency_entropy,
)

# 1. Create and train a VQC
feature_map = ZZFeatureMap(feature_dimension=4, reps=2)
ansatz = RealAmplitudes(num_qubits=4, reps=2)
sampler = StatevectorSampler()

vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=COBYLA(maxiter=100),
    sampler=sampler
)
vqc.fit(X_train, y_train)

# 2. Wrap the trained VQC for saliency analysis
theta_trained = vqc._fit_result.x
neural_network = vqc._neural_network

def predict_fn(theta, X):
    X = np.atleast_2d(X)
    probs = neural_network.forward(X, theta_trained)
    return probs[:, 1] if probs.ndim == 2 else probs.flatten()

model = VQCWrapper(
    predict_fn=predict_fn,
    theta=theta_trained,
    n_features=4,
    n_classes=2
)

# 3. Compute saliency
gradient_engine = FiniteDifferenceGradient(delta=1e-3)
saliency = GradientSaliency(model, gradient_engine)
scores = saliency.attribute(X_test[0])

# 4. Evaluate the explanation
audc = deletion_score(model, X_test[0], scores)
entropy = saliency_entropy(scores)

print(f"Feature importance: {scores}")
print(f"Deletion AUDC: {audc:.4f} (lower = more faithful)")
print(f"Saliency entropy: {entropy:.4f}")
```

**Output:**
```
Feature importance: [2.60 0.23 2.71 1.13]
Deletion AUDC: 0.6188 (lower = more faithful)
Saliency entropy: 1.1484
```

### Multiple Saliency Methods

```python
from qnn_saliency import SmoothGrad, IntegratedGradients, Occlusion

# Different methods for robustness
gradient = GradientSaliency(model, gradient_engine)
smoothgrad = SmoothGrad(model, gradient_engine, n_samples=30, sigma=0.1)
integrated = IntegratedGradients(model, gradient_engine, n_steps=25)
occlusion = Occlusion(model, baseline='zero')

# Compare attributions
print("Gradient:    ", gradient.attribute(x))
print("SmoothGrad:  ", smoothgrad.attribute(x))
print("IntegratedIG:", integrated.attribute(x))
print("Occlusion:   ", occlusion.attribute(x))
```

### Batch Analysis

```python
# Compute saliency for multiple samples
batch_scores = saliency.attribute_batch(X_test[:20])

# Average feature importance
avg_importance = np.mean(np.abs(batch_scores), axis=0)
print(f"Average importance: {avg_importance}")
```

### Hybrid PyTorch+Qiskit Models

```python
from qnn_saliency import HybridQNNWrapper
from qnn_saliency.metrics import bug_detection_signal

# Wrap a PyTorch model with quantum layers
model = HybridQNNWrapper(
    model=pytorch_hybrid_model,
    n_features=4,
    n_classes=2
)

# Analyze with SmoothGrad
saliency = SmoothGrad(model, n_samples=50, sigma=0.1)
scores = saliency.attribute_batch(X_test)

# Diagnostic: compare correct vs incorrect predictions
bug_signal = bug_detection_signal(model, X_test, y_test, scores)
print(f"Bug detection ratio: {bug_signal['ratio']:.4f}")
```

## 📐 API Reference

### Saliency Methods

All saliency methods inherit from `BaseSaliency`:

```python
class MySaliency(BaseSaliency):
    def attribute(self, x: np.ndarray, target: int = None) -> np.ndarray:
        """Compute attribution scores for a single input."""
        ...
    
    def attribute_batch(self, X: np.ndarray, target: int = None) -> np.ndarray:
        """Compute attributions for a batch of inputs."""
        ...
```

### Gradient Engines

```python
class MyGradient(BaseGradientEngine):
    def compute_gradient(
        self, 
        x: np.ndarray, 
        forward_fn: Callable, 
        target_idx: int = None
    ) -> np.ndarray:
        """Compute gradient of forward_fn w.r.t. input x."""
        ...
```

### Model Wrappers

```python
class MyWrapper(BaseModelWrapper):
    def predict(self, x: np.ndarray) -> np.ndarray:
        """Get model output (probabilities or expectation values)."""
        ...
    
    def predict_class(self, x: np.ndarray) -> int | np.ndarray:
        """Get predicted class label(s)."""
        ...
    
    @property
    def n_features(self) -> int:
        """Number of input features."""
        ...
```

## 🔧 Supported Model Types

The package is **model-agnostic** and works with any differentiable function:

| Model Type | Wrapper | Notes |
|------------|---------|-------|
| Qiskit `VQC` | `VQCWrapper` | Use `vqc._neural_network.forward()` for continuous outputs |
| Qiskit `SamplerQNN` | `VQCWrapper` | Direct `forward(X, weights)` |
| Qiskit `EstimatorQNN` | `VQCWrapper` | Direct `forward(X, weights)` |
| Qiskit `VQR` | `VQCWrapper` | Same pattern as VQC |
| PyTorch hybrid | `HybridQNNWrapper` | Wrap `nn.Module` |
| Custom QNN | `VQCWrapper` | Any `f(θ, X) → continuous output` |

**Key requirement:** The prediction function must return **continuous outputs** (probabilities, expectation values) — not discrete class labels.

## 🧪 Testing

```bash
# Run all tests (64 tests)
pytest tests/ -v

# With coverage report
pytest tests/ --cov=qnn_saliency --cov-report=html

# Run specific test modules
pytest tests/test_saliency.py
pytest tests/test_faithfulness.py
pytest tests/test_distribution.py
```

## 📖 Examples

See the `examples/` directory:

| File | Description |
|------|-------------|
| `qiskit_vqc_example.py` | Complete VQC training + saliency analysis workflow |
| `usage_examples.py` | API demonstrations for all methods |

Run the full VQC example:
```bash
python examples/qiskit_vqc_example.py
```

## 🎓 Interpreting Results

### Feature Importance Ranking

```
Rank | Feature        | Score
-----|----------------|--------
   1 | petal_length   | +2.7132
   2 | sepal_length   | +2.6046
   3 | petal_width    | +1.1284
   4 | sepal_width    | +0.2261
```

**Insights:**
- Higher absolute scores = more influential features
- Sign indicates direction (positive/negative contribution)
- Low-importance features may be candidates for removal

### Faithfulness Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Deletion AUDC | 0.62 | Moderate — removing important features reduces confidence |
| Insertion AUIC | 0.63 | Moderate — adding important features increases confidence |

- **AUDC < 0.5**: Good faithfulness
- **AUDC ≈ 0.5**: Random baseline
- **AUDC > 0.7**: Poor faithfulness

### Distribution Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Entropy | 1.15 / 2.0 | Moderately concentrated |
| Gini | 0.33 | Some inequality in attribution |
| Hoyer | 0.30 | Moderate sparseness |

## 🤝 Contributing

Contributions are welcome! This project follows Qiskit contribution guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

Apache License 2.0. See [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

Developed as part of the **QAMP (Qiskit Advocate Mentorship Program)** project:
*"QNNs — Saliency & Sensitivity Kit"*

## 📚 References

1. Smilkov, D., et al. "SmoothGrad: removing noise by adding noise." *ICML Workshop* (2017)
2. Sundararajan, M., et al. "Axiomatic Attribution for Deep Networks." *ICML* (2017)
3. Schuld, M., et al. "Evaluating analytic gradients on quantum hardware." *Physical Review A* (2019)
4. Samek, W., et al. "Evaluating the Visualization of What a Deep Neural Network has Learned." *IEEE TNNLS* (2017)
