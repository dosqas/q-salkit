# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""
QNN Saliency Usage Examples
===========================

This module demonstrates how to use the qnn_saliency package with
both VQC and Hybrid QNN models.
"""

import numpy as np


def example_vqc_usage():
    """
    Example: Using qnn_saliency with a pure VQC model.
    
    This demonstrates the typical workflow:
    1. Wrap your trained VQC
    2. Compute saliency maps
    3. Evaluate faithfulness metrics
    """
    from qnn_saliency import (
        VQCWrapper,
        GradientSaliency,
        SmoothGrad,
        IntegratedGradients,
        ParameterShiftGradient,
        deletion_score,
        saliency_entropy,
        saliency_sparseness,
    )
    
    # === Step 1: Define your VQC predict function ===
    # This should wrap your trained circuit and estimator
    
    # Mock example (replace with your actual VQC)
    def vqc_predict_fn(theta, X):
        """
        Your VQC prediction function.
        
        Parameters
        ----------
        theta : np.ndarray
            Trained variational parameters
        X : np.ndarray
            Input data of shape (n_samples, n_features)
            
        Returns
        -------
        np.ndarray
            Raw expectation values of shape (n_samples,)
        """
        # Example: Simple linear combination (replace with actual circuit)
        return np.tanh(X @ theta[:X.shape[1]])
    
    # Example trained parameters and test data
    n_features = 4
    theta_star = np.random.randn(n_features)
    X_test = np.random.randn(10, n_features)
    
    # === Step 2: Wrap the VQC ===
    model = VQCWrapper(
        predict_fn=vqc_predict_fn,
        theta=theta_star,
        n_features=n_features,
        n_classes=2
    )
    
    # === Step 3: Create saliency method ===
    # Option A: Use parameter-shift rule (exact for VQC)
    gradient_engine = ParameterShiftGradient(shift=np.pi/2)
    saliency_method = GradientSaliency(model, gradient_engine=gradient_engine)
    
    # Option B: Use SmoothGrad for noise-robust attribution
    smoothgrad = SmoothGrad(model, n_samples=30, sigma=0.1)
    
    # Option C: Use Integrated Gradients
    ig = IntegratedGradients(model, n_steps=25, baseline='zero')
    
    # === Step 4: Compute saliency for a sample ===
    x = X_test[0]
    
    # Basic gradient saliency
    scores = saliency_method.attribute(x)
    print(f"Gradient saliency: {scores}")
    
    # SmoothGrad
    smooth_scores = smoothgrad.attribute(x)
    print(f"SmoothGrad saliency: {smooth_scores}")
    
    # Integrated Gradients
    ig_scores = ig.attribute(x, baseline=np.zeros(n_features))
    print(f"Integrated Gradients: {ig_scores}")
    
    # === Step 5: Evaluate faithfulness ===
    
    # Deletion metric (lower AUDC = more faithful)
    audc = deletion_score(model, x, scores)
    print(f"Deletion AUDC: {audc:.4f}")
    
    # Saliency distribution metrics
    entropy = saliency_entropy(scores)
    sparseness = saliency_sparseness(scores)
    print(f"Entropy: {entropy:.4f}, Sparseness: {sparseness:.4f}")
    
    return scores


def example_hybrid_usage():
    """
    Example: Using qnn_saliency with a hybrid QNN model.
    
    This works with PyTorch hybrid models that combine
    quantum layers with classical neural networks.
    """
    from qnn_saliency import (
        HybridQNNWrapper,
        GradientSaliency,
        Occlusion,
        NoiseSensitivity,
        FiniteDifferenceGradient,
    )
    from qnn_saliency.metrics import (
        deletion_score,
        insertion_score,
        bug_detection_signal,
        minimum_efficacy,
    )
    
    # === Step 1: Create a mock hybrid model ===
    # (Replace with your actual PyTorch hybrid model)
    
    class MockHybridModel:
        """Mock model for demonstration."""
        def __init__(self, n_features, n_classes):
            self.weights = np.random.randn(n_features, n_classes)
        
        def __call__(self, x):
            x = np.atleast_2d(x)
            logits = x @ self.weights
            return logits
    
    n_features = 4
    n_classes = 3
    mock_model = MockHybridModel(n_features, n_classes)
    
    # === Step 2: Wrap the hybrid model ===
    model = HybridQNNWrapper(
        model=mock_model,
        n_features=n_features,
        n_classes=n_classes
    )
    
    # === Step 3: Create saliency methods ===
    
    # Finite-difference gradient (works for any differentiable model)
    gradient_engine = FiniteDifferenceGradient(delta=1e-3)
    gradient_saliency = GradientSaliency(model, gradient_engine=gradient_engine)
    
    # Occlusion-based saliency
    occlusion = Occlusion(model, baseline='zero', metric='prob_change')
    
    # Noise sensitivity
    noise_sens = NoiseSensitivity(model, sigma=0.1, n_samples=50)
    
    # === Step 4: Compute saliency ===
    X_test = np.random.randn(20, n_features)
    y_test = np.random.randint(0, n_classes, size=20)
    
    x = X_test[0]
    
    grad_scores = gradient_saliency.attribute(x)
    occ_scores = occlusion.attribute(x)
    ns_scores = noise_sens.attribute(x)
    
    print(f"Gradient saliency: {grad_scores}")
    print(f"Occlusion importance: {occ_scores}")
    print(f"Noise sensitivity: {ns_scores}")
    
    # === Step 5: Batch saliency computation ===
    all_saliency = gradient_saliency.attribute_batch(X_test)
    print(f"Batch saliency shape: {all_saliency.shape}")
    
    # === Step 6: Comprehensive metrics ===
    
    # Deletion curve
    audc, k_vals, confs = deletion_score(
        model, x, grad_scores, return_curve=True
    )
    print(f"Deletion AUDC: {audc:.4f}")
    
    # Insertion curve
    auic = insertion_score(model, x, grad_scores)
    print(f"Insertion AUIC: {auic:.4f}")
    
    # Bug detection
    bug_signal = bug_detection_signal(model, X_test, y_test, all_saliency)
    print(f"Bug detection ratio: {bug_signal['ratio']:.4f}")
    
    # Minimum efficacy
    efficacy = minimum_efficacy(model, X_test, all_saliency)
    print(f"Minimum efficacy: {efficacy:.4f}")
    
    return all_saliency


def example_comparison():
    """
    Example: Comparing multiple saliency methods.
    
    Demonstrates how to evaluate agreement between methods.
    """
    from qnn_saliency import (
        VQCWrapper,
        GradientSaliency,
        SmoothGrad,
        IntegratedGradients,
        Occlusion,
    )
    from qnn_saliency.metrics import (
        feature_agreement,
        saliency_entropy,
        saliency_sparseness,
    )
    
    # Setup (using mock model)
    n_features = 4
    
    def mock_predict(theta, X):
        return np.tanh(X @ theta[:X.shape[1]])
    
    theta = np.array([0.5, -0.3, 0.8, -0.1])
    model = VQCWrapper(mock_predict, theta, n_features)
    
    # Create methods
    methods = {
        'Gradient': GradientSaliency(model),
        'SmoothGrad': SmoothGrad(model, n_samples=30),
        'IntegratedGrad': IntegratedGradients(model, n_steps=25),
        'Occlusion': Occlusion(model),
    }
    
    # Compute saliency for test sample
    x = np.array([0.5, -0.2, 0.8, 0.1])
    
    results = {}
    print("\n=== Saliency Method Comparison ===\n")
    
    for name, method in methods.items():
        scores = method.attribute(x)
        entropy = saliency_entropy(scores)
        sparseness = saliency_sparseness(scores)
        
        results[name] = scores
        
        print(f"{name}:")
        print(f"  Scores: {scores.round(4)}")
        print(f"  Entropy: {entropy:.4f}, Sparseness: {sparseness:.4f}")
        print()
    
    # Compare agreement
    print("=== Feature Agreement (top-2) ===\n")
    method_names = list(methods.keys())
    for i, name1 in enumerate(method_names):
        for name2 in method_names[i+1:]:
            agreement = feature_agreement(
                results[name1], results[name2], top_k=2
            )
            print(f"{name1} vs {name2}: {agreement:.1%}")


if __name__ == "__main__":
    print("=" * 60)
    print("QNN Saliency Package - Usage Examples")
    print("=" * 60)
    
    print("\n>>> Example 1: VQC Usage")
    example_vqc_usage()
    
    print("\n>>> Example 2: Hybrid QNN Usage")
    example_hybrid_usage()
    
    print("\n>>> Example 3: Method Comparison")
    example_comparison()
