import numpy as np
import math

# =========================================================
# 1. ENTROPY: UNIFORM VS REAL ENGLISH
# =========================================================

def entropy(probs):
    return -sum(p * math.log2(p) for p in probs if p > 0)


def english_entropy():
    # uniform distribution over 26 letters
    uniform = [1/26] * 26
    H_uniform = entropy(uniform)

    # approximate English frequencies (simplified)
    freq = np.array([
        0.0817, 0.0149, 0.0278, 0.0425, 0.1270,
        0.0223, 0.0202, 0.0609, 0.0697, 0.0015,
        0.0077, 0.0403, 0.0241, 0.0675, 0.0751,
        0.0193, 0.0010, 0.0599, 0.0633, 0.0906,
        0.0276, 0.0098, 0.0236, 0.0015, 0.0197,
        0.0007
    ])

    freq = freq / freq.sum()
    H_real = entropy(freq)

    print("=== ENTROPY ===")
    print("Uniform entropy:", H_uniform)
    print("English entropy:", H_real)

    print("\nObservation:")
    print("- Uniform distribution has MAX entropy")
    print("- Real English is lower because letters are structured, not random\n")


# =========================================================
# 2. CROSS ENTROPY (MANUAL + ZERO LOSS CASE)
# =========================================================

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def cross_entropy(logits, true_class):
    probs = softmax(logits)
    return -math.log(probs[true_class])


def cross_entropy_experiment():
    logits = np.array([5.0, 2.0, 0.5])
    true_class = 1

    loss = cross_entropy(logits, true_class)

    print("=== CROSS ENTROPY ===")
    print("Logits:", logits)
    print("Loss:", loss)

    # PyTorch verification (optional)
    try:
        import torch
        import torch.nn as nn

        t_logits = torch.tensor(logits).unsqueeze(0)
        t_label = torch.tensor([true_class])

        loss_fn = nn.CrossEntropyLoss()
        print("PyTorch loss:", loss_fn(t_logits, t_label).item())

    except ImportError:
        print("PyTorch not installed")

    print("\nZero-loss condition:")
    print("- loss = 0 when true class probability = 1")
    print("- i.e. logits_true >> others (infinite margin)\n")


# =========================================================
# 3. KL DIVERGENCE (ASYMMETRY DEMO)
# =========================================================

def kl_divergence(P, Q):
    return sum(p * math.log(p / q) for p, q in zip(P, Q) if p > 0)


def kl_experiment():
    P = [0.7, 0.2, 0.1]
    Q = [0.1, 0.2, 0.7]

    kl_pq = kl_divergence(P, Q)
    kl_qp = kl_divergence(Q, P)

    print("=== KL DIVERGENCE ===")
    print("D_KL(P || Q):", kl_pq)
    print("D_KL(Q || P):", kl_qp)

    print("\nObservation:")
    print("- KL divergence is NOT symmetric")
    print("- Swapping P and Q gives different values")
    print("- Because expectation is taken under different distributions\n")


# =========================================================
# 4. PERPLEXITY FUNCTION
# =========================================================

def perplexity(sequence):
    """
    sequence = list of (true_index, logits)
    """
    total_log_prob = 0
    n = len(sequence)

    for true_idx, logits in sequence:
        probs = softmax(logits)
        total_log_prob += math.log(probs[true_idx])

    return math.exp(-total_log_prob / n)


def perplexity_test():
    print("=== PERPLEXITY ===")

    # simulate 50 words, each word probability = 0.01
    vocab_size = 10
    logit = np.log(np.ones(vocab_size) / vocab_size)

    sequence = [(i % vocab_size, logit) for i in range(50)]

    ppl = perplexity(sequence)

    print("Perplexity:", ppl)

    print("\nInterpretation:")
    print("- lower perplexity = better model")
    print("- perplexity ≈ vocabulary size for uniform model\n")


if __name__ == "__main__":
    english_entropy()
    cross_entropy_experiment()
    kl_experiment()
    perplexity_test()