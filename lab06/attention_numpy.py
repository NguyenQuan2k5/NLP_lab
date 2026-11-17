"""
LAB 06 — Attention Mechanism: Core NumPy Implementation
Author: Student / NLP Course - VNU University of Science (HUS-VNU)
Course: Natural Language Processing and Applications (Xử lý ngôn ngữ tự nhiên và ứng dụng)

This module implements scaled dot-product attention and self-attention from scratch using NumPy.
Strictly adheres to Section 16 & 17 of LAB 06 requirements:
- No high-level attention APIs (torch.nn.MultiheadAttention, keras.layers.Attention).
- Numerically stable softmax implementation.
- Returns (scores, attention_weights, context).
"""

import numpy as np
from typing import Tuple, Optional


def stable_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable softmax implementation.
    Subtracts max(x) along the given axis to prevent overflow in exp(x).
    
    Formula:
        softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
        
    Args:
        x: Input array of arbitrary shape.
        axis: Axis along which softmax is applied (default -1).
        
    Returns:
        Probability distribution summing to 1.0 along the specified axis.
    """
    # Subtract max for numerical stability (prevents overflow for large positive values)
    x_max = np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(x - x_max)
    sum_exp = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / sum_exp


def scaled_dot_product_attention(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    scale: bool = True,
    temperature: float = 1.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scaled Dot-Product Attention from scratch in NumPy.
    
    Matches Section 16 & 17 of LAB 06:
        S = (Q @ K.T) / (sqrt(d_k) * temperature)  (if scale=True)
        A = softmax(S)
        C = A @ V
        
    Supports both 2D tensors:
        Q: (seq_len_q, d_k)
        K: (seq_len_k, d_k)
        V: (seq_len_k, d_v)
    And 3D batched tensors:
        Q: (batch_size, seq_len_q, d_k)
        K: (batch_size, seq_len_k, d_k)
        V: (batch_size, seq_len_k, d_v)
        
    Args:
        Q: Query matrix/tensor.
        K: Key matrix/tensor.
        V: Value matrix/tensor.
        mask: Optional mask array where 1/True allows attention and 0/False masks it out.
        scale: Whether to scale scores by sqrt(d_k).
        temperature: Temperature factor T to control entropy (default 1.0).
        
    Returns:
        scores (S): Raw dot-product scores (possibly scaled), shape (..., seq_len_q, seq_len_k)
        attention_weights (A): Probability distribution after softmax, shape (..., seq_len_q, seq_len_k)
        context (C): Context vectors, shape (..., seq_len_q, d_v)
    """
    d_k = Q.shape[-1]
    
    # Check dimensionality match
    assert K.shape[-1] == d_k, f"Key dimension {K.shape[-1]} must match Query dimension {d_k}"
    assert K.shape[-2] == V.shape[-2], f"Key sequence length {K.shape[-2]} must match Value sequence length {V.shape[-2]}"
    
    # Compute raw dot-product scores S = Q @ K^T
    # If 2D: (T_q, d_k) @ (d_k, T_k) -> (T_q, T_k)
    # If 3D: (B, T_q, d_k) @ (B, d_k, T_k) -> (B, T_q, T_k)
    if Q.ndim == 2:
        scores = np.matmul(Q, K.T)
    elif Q.ndim == 3:
        scores = np.matmul(Q, np.swapaxes(K, -1, -2))
    else:
        scores = np.matmul(Q, np.swapaxes(K, -1, -2))
        
    # Apply scaling factor sqrt(d_k)
    if scale:
        scores = scores / np.sqrt(d_k)
        
    # Apply temperature
    if temperature != 1.0:
        assert temperature > 0, "Temperature must be strictly positive"
        scores = scores / temperature
        
    # Apply optional mask (e.g., masking out padding tokens with -1e9)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    # Compute attention weights via numerically stable softmax along the last dimension (keys dimension)
    attention_weights = stable_softmax(scores, axis=-1)
    
    # Compute context vector C = A @ V
    context = np.matmul(attention_weights, V)
    
    return scores, attention_weights, context


def self_attention(
    X: np.ndarray,
    W_Q: np.ndarray,
    W_K: np.ndarray,
    W_V: np.ndarray,
    scale: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Full Self-Attention layer from scratch (Section 20 & 21).
    Given input sequence embeddings X and projection weight matrices W_Q, W_K, W_V:
        Q = X @ W_Q
        K = X @ W_K
        V = X @ W_V
        S, A, C = scaled_dot_product_attention(Q, K, V, scale=scale)
        
    Args:
        X: Input sequence matrix of shape (seq_len, d_in)
        W_Q: Query projection matrix of shape (d_in, d_k)
        W_K: Key projection matrix of shape (d_in, d_k)
        W_V: Value projection matrix of shape (d_in, d_v)
        scale: Whether to scale by sqrt(d_k)
        
    Returns:
        Q, K, V, attention_weights, context
    """
    Q = np.matmul(X, W_Q)
    K = np.matmul(X, W_K)
    V = np.matmul(X, W_V)
    scores, attention_weights, context = scaled_dot_product_attention(Q, K, V, scale=scale)
    return Q, K, V, attention_weights, context


def compute_entropy(prob_dist: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """
    Computes Shannon Entropy of probability distributions:
        H(p) = - sum(p * log(p + eps))
    Used for Experiment 2 (Section 15) to analyze temperature influence.
    """
    return -np.sum(prob_dist * np.log(prob_dist + eps), axis=axis)


# =========================================================================
# Self-Verification Test Suite
# =========================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Running Self-Verification Suite for attention_numpy.py")
    print("=" * 60)
    
    # Test 1: Numerical Stability on large values (Section 17)
    x_test = np.array([1000.0, 1001.0, 1002.0])
    sm = stable_softmax(x_test)
    print(f"\n[Test 1] Numerical Stability on {x_test}:")
    print(f"  Softmax probabilities: {sm}")
    print(f"  Sum of probabilities: {np.sum(sm):.6f}")
    assert not np.isnan(sm).any() and not np.isinf(sm).any(), "Softmax produced NaN or Inf!"
    assert np.isclose(np.sum(sm), 1.0), "Softmax probabilities do not sum to 1.0!"
    print("  => PASSED Numerical Stability Test.")
    
    # Test 2: Section 8 Bài 1 (Dot-product scores)
    q = np.array([1.0, 2.0])
    K = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 1.0]])
    expected_scores = np.array([1.0, 2.0, 3.0])
    calc_scores = np.dot(K, q)  # q^T k_i
    print(f"\n[Test 2] Bài 1 Attention Scores:")
    print(f"  q = {q}, K =\n{K}")
    print(f"  Calculated scores s = q^T k_i: {calc_scores}")
    assert np.allclose(calc_scores, expected_scores), "Bài 1 scores calculation mismatch!"
    print("  => PASSED Bài 1 Verification.")
    
    # Test 3: Section 11 Bài 4 (Full Attention Computation)
    q4 = np.array([[1.0, 0.0]])  # shape (1, 2)
    K4 = np.array([[1.0, 0.0],
                   [0.0, 1.0],
                   [1.0, 1.0]])  # shape (3, 2)
    V4 = np.array([[1.0, 0.0],
                   [0.0, 2.0],
                   [2.0, 1.0]])  # shape (3, 2)
                   
    scores4, weights4, context4 = scaled_dot_product_attention(q4, K4, V4, scale=False)
    print(f"\n[Test 3] Bài 4 Full Attention Computation (unscaled):")
    print(f"  Scores: {scores4}")
    print(f"  Attention Weights: {weights4}")
    print(f"  Context Vector: {context4}")
    
    # Analytical expected values:
    # scores: [1, 0, 1]
    # weights: [e/(2e+1), 1/(2e+1), e/(2e+1)] = [0.4223, 0.1554, 0.4223]
    # context: [1.2669, 0.7331]
    e = np.e
    expected_w = np.array([[e / (2*e + 1), 1.0 / (2*e + 1), e / (2*e + 1)]])
    expected_c = np.matmul(expected_w, V4)
    assert np.allclose(weights4, expected_w, atol=1e-4), "Bài 4 weights mismatch!"
    assert np.allclose(context4, expected_c, atol=1e-4), "Bài 4 context mismatch!"
    print("  => PASSED Bài 4 Full Attention Verification.")
    
    # Test 4: Batched Shape Integrity
    B, T_q, T_k, d_k, d_v = 4, 8, 12, 16, 32
    Q_batch = np.random.randn(B, T_q, d_k)
    K_batch = np.random.randn(B, T_k, d_k)
    V_batch = np.random.randn(B, T_k, d_v)
    s_b, a_b, c_b = scaled_dot_product_attention(Q_batch, K_batch, V_batch, scale=True)
    print(f"\n[Test 4] Batched Input Dimension Check:")
    print(f"  Q: {Q_batch.shape}, K: {K_batch.shape}, V: {V_batch.shape}")
    print(f"  Scores: {s_b.shape}, Weights: {a_b.shape}, Context: {c_b.shape}")
    assert s_b.shape == (B, T_q, T_k)
    assert a_b.shape == (B, T_q, T_k)
    assert c_b.shape == (B, T_q, d_v)
    assert np.allclose(np.sum(a_b, axis=-1), 1.0), "Batched weights do not sum to 1!"
    print("  => PASSED Batched Verification.")
    print("\nAll self-verification checks passed successfully!")
