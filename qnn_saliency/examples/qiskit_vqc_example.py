# Copyright 2024-2026 QAMP Team
# Licensed under the Apache License, Version 2.0

"""
Real Qiskit VQC Integration Example
====================================

This example demonstrates how to use qnn_saliency with a real
Qiskit VQC trained on the Iris dataset. It shows the complete
workflow from training to saliency analysis.

Requirements:
    pip install qiskit qiskit-machine-learning scikit-learn

Run this script:
    python examples/qiskit_vqc_example.py
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Qiskit imports
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms import VQC
from qiskit_machine_learning.optimizers import COBYLA

# qnn_saliency imports
from qnn_saliency import (
    VQCWrapper,
    GradientSaliency,
    SmoothGrad,
    IntegratedGradients,
    Occlusion,
    FiniteDifferenceGradient,
    deletion_score,
    insertion_score,
    saliency_entropy,
    saliency_sparseness,
    hoyer_sparseness,
    gini_coefficient,
)


def load_and_preprocess_data():
    """Load Iris dataset and preprocess for binary classification."""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Binary classification: Setosa (0) vs Others (1)
    y_binary = (y > 0).astype(int)
    
    # Use only first 4 features
    X_scaled = StandardScaler().fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_binary, test_size=0.3, random_state=42
    )
    
    return X_train, X_test, y_train, y_test


def create_vqc():
    """Create a VQC with ZZFeatureMap and RealAmplitudes ansatz."""
    n_features = 4
    n_qubits = 4
    
    # Feature map: encodes classical data
    feature_map = ZZFeatureMap(
        feature_dimension=n_features,
        reps=2,
        entanglement='linear'
    )
    
    # Ansatz: variational circuit
    ansatz = RealAmplitudes(
        num_qubits=n_qubits,
        reps=2,
        entanglement='linear'
    )
    
    # Create sampler
    sampler = StatevectorSampler()
    
    # Create VQC
    vqc = VQC(
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=COBYLA(maxiter=100),
        sampler=sampler
    )
    
    return vqc


def wrap_trained_vqc(vqc, X_train):
    """
    Wrap a trained VQC for saliency analysis.
    
    Parameters
    ----------
    vqc : VQC
        Trained Qiskit VQC classifier
    X_train : np.ndarray
        Training data (used for reference)
        
    Returns
    -------
    VQCWrapper
        Wrapped model ready for saliency analysis
    """
    n_features = X_train.shape[1]
    
    # Get trained weights from VQC
    theta_trained = vqc._fit_result.x if hasattr(vqc, '_fit_result') else None
    
    # Access the underlying neural network for continuous outputs
    neural_network = vqc._neural_network
    
    def vqc_predict_fn(theta, X):
        """Prediction function using the underlying SamplerQNN."""
        X = np.atleast_2d(X)
        
        # Use the neural network's forward pass for continuous probability outputs
        # This gives us differentiable outputs instead of discrete class labels
        probs = neural_network.forward(X, theta_trained)
        
        # probs has shape (n_samples, n_classes) - return P(class=1)
        if probs.ndim == 2 and probs.shape[1] >= 2:
            return probs[:, 1]  # Return probability of class 1
        elif probs.ndim == 2 and probs.shape[1] == 1:
            return probs.flatten()
        else:
            return probs.flatten()
    
    wrapper = VQCWrapper(
        predict_fn=vqc_predict_fn,
        theta=theta_trained if theta_trained is not None else np.zeros(10),
        n_features=n_features,
        n_classes=2
    )
    
    return wrapper


def analyze_saliency(model, X_test, y_test):
    """
    Comprehensive saliency analysis on test data.
    
    Parameters
    ----------
    model : VQCWrapper
        Wrapped VQC model
    X_test : np.ndarray
        Test samples
    y_test : np.ndarray
        True labels
    """
    print("\n" + "="*60)
    print("SALIENCY ANALYSIS")
    print("="*60)
    
    # Create gradient engine for VQC
    gradient_engine = FiniteDifferenceGradient(delta=1e-3)
    
    # === Create saliency methods ===
    print("\n[1] Creating saliency methods...")
    
    gradient_saliency = GradientSaliency(model, gradient_engine)
    smoothgrad = SmoothGrad(model, gradient_engine, n_samples=30, sigma=0.1)
    integrated_grads = IntegratedGradients(model, gradient_engine, n_steps=25)
    occlusion = Occlusion(model, baseline='zero')
    
    # === Analyze a single sample ===
    print("\n[2] Analyzing sample 0...")
    x = X_test[0]
    
    # Compute saliency with different methods
    grad_scores = gradient_saliency.attribute(x)
    smooth_scores = smoothgrad.attribute(x)
    ig_scores = integrated_grads.attribute(x)
    occ_scores = occlusion.attribute(x)
    
    print(f"\n  Gradient Saliency:      {grad_scores}")
    print(f"  SmoothGrad:             {smooth_scores}")
    print(f"  Integrated Gradients:   {ig_scores}")
    print(f"  Occlusion:              {occ_scores}")
    
    # === Distribution metrics ===
    print("\n[3] Distribution metrics (using Gradient Saliency)...")
    
    entropy = saliency_entropy(grad_scores)
    sparseness = saliency_sparseness(grad_scores)
    hoyer = hoyer_sparseness(grad_scores)
    gini = gini_coefficient(grad_scores)
    
    print(f"  Entropy:     {entropy:.4f} (max = {np.log2(len(grad_scores)):.4f})")
    print(f"  Sparseness:  {sparseness:.4f} (lower = more sparse)")
    print(f"  Hoyer:       {hoyer:.4f} (higher = more sparse)")
    print(f"  Gini:        {gini:.4f} (higher = more inequality)")
    
    # === Faithfulness metrics ===
    print("\n[4] Faithfulness metrics...")
    
    # Deletion score (lower = more faithful)
    audc, k_vals, confs = deletion_score(
        model, x, grad_scores, return_curve=True
    )
    print(f"  Deletion AUDC:  {audc:.4f} (lower = better)")
    print(f"  Confidence curve: {[f'{c:.3f}' for c in confs]}")
    
    # Insertion score (higher = more faithful)
    auic = insertion_score(model, x, grad_scores)
    print(f"  Insertion AUIC: {auic:.4f} (higher = better)")
    
    # === Feature importance ranking ===
    print("\n[5] Feature importance ranking...")
    
    feature_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    ranked_indices = np.argsort(-np.abs(grad_scores))
    
    print("  Rank | Feature        | Score")
    print("  -----|----------------|--------")
    for rank, idx in enumerate(ranked_indices, 1):
        print(f"  {rank:4d} | {feature_names[idx]:14s} | {grad_scores[idx]:+.4f}")
    
    # === Batch analysis ===
    print("\n[6] Batch saliency analysis...")
    
    n_samples = min(10, len(X_test))
    batch_saliency = gradient_saliency.attribute_batch(X_test[:n_samples])
    
    # Average importance per feature
    avg_importance = np.mean(np.abs(batch_saliency), axis=0)
    print(f"  Average feature importance: {avg_importance}")
    
    most_important = feature_names[np.argmax(avg_importance)]
    print(f"  Most important feature (on average): {most_important}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


def main():
    """Main function demonstrating complete workflow."""
    print("="*60)
    print("QNN SALIENCY - QISKIT VQC INTEGRATION EXAMPLE")
    print("="*60)
    
    # Load data
    print("\n[1] Loading Iris dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    
    # Create and train VQC
    print("\n[2] Creating VQC...")
    print("  Feature map: ZZFeatureMap (2 reps)")
    print("  Ansatz: RealAmplitudes (2 reps)")
    
    # Train actual VQC
    vqc = create_vqc()
    print("\n[3] Training VQC (this may take a few minutes)...")
    vqc.fit(X_train, y_train)
    accuracy = vqc.score(X_test, y_test)
    print(f"  Test accuracy: {accuracy:.2%}")
    model = wrap_trained_vqc(vqc, X_train)
    
    # Run saliency analysis
    analyze_saliency(model, X_test, y_test)


if __name__ == "__main__":
    main()
