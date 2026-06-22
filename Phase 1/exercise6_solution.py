import numpy as np
import math

# =========================================================
# 1. INVERSE TRANSFORM SAMPLING (EXPONENTIAL)
# =========================================================

def sample_exponential(lam, n=10000):
    # Inverse CDF: x = -ln(1-u)/λ
    u = np.random.uniform(0, 1, n)
    samples = -np.log(1 - u) / lam
    return samples


def exponential_pdf(x, lam):
    return lam * np.exp(-lam * x)


def test_exponential():
    lam = 1.5
    samples = sample_exponential(lam)

    print("=== Exponential Sampling ===")
    print("Mean (empirical):", np.mean(samples))
    print("Mean (theoretical):", 1/lam)

    # histogram check (coarse numeric comparison)
    hist_mean = np.mean(np.histogram(samples, bins=50, density=True)[0])
    print("Histogram density avg:", hist_mean)
    print()


# =========================================================
# 2. JOINT DISTRIBUTION OF LOADED DICE
# =========================================================

def build_joint_distribution():
    # biased dice probabilities
    die1 = np.array([0.1, 0.1, 0.2, 0.3, 0.2, 0.1])
    die2 = np.array([0.2, 0.2, 0.1, 0.1, 0.2, 0.2])

    joint = np.outer(die1, die2)

    marginal1 = np.sum(joint, axis=1)
    marginal2 = np.sum(joint, axis=0)

    independent_check = np.allclose(joint, np.outer(marginal1, marginal2))

    print("=== Joint Distribution ===")
    print("Joint sum:", joint.sum())
    print("Marginal1:", marginal1)
    print("Marginal2:", marginal2)
    print("Independent?", independent_check)
    print()

    return joint, marginal1, marginal2


# =========================================================
# 3. CROSS-ENTROPY LOSS (MANUAL + PYTORCH)
# =========================================================

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)


def cross_entropy_manual(logits, target):
    probs = softmax(logits)
    return -np.log(probs[target])


def test_cross_entropy():
    logits = np.array([2.0, 0.5, -1.0, 3.0, 0.1])
    target = 3

    loss = cross_entropy_manual(logits, target)

    print("=== Cross Entropy ===")
    print("Manual loss:", loss)

    try:
        import torch
        import torch.nn as nn

        logits_t = torch.tensor(logits, dtype=torch.float32)
        target_t = torch.tensor([target])

        loss_fn = nn.CrossEntropyLoss()
        pytorch_loss = loss_fn(logits_t.unsqueeze(0), target_t)

        print("PyTorch loss:", pytorch_loss.item())

    except ImportError:
        print("PyTorch not installed")

    print()


# =========================================================
# 4. LOG PROBABILITY SEQUENCE DECODING
# =========================================================

def analyze_log_probs(log_probs):
    """
    log_probs: list of log probabilities per word
    """

    total_log_prob = sum(log_probs)
    raw_prob = math.exp(total_log_prob)

    most_likely_sequence = ["word"] * len(log_probs)

    print("=== Log Probability Sequence ===")
    print("Sequence length:", len(log_probs))
    print("Total log probability:", total_log_prob)
    print("Raw probability:", raw_prob)
    print()

    return most_likely_sequence, total_log_prob, raw_prob


def test_sequence():
    # 50 words, each probability = 0.01
    prob = 0.01
    log_prob = math.log(prob)

    log_probs = [log_prob] * 50

    analyze_log_probs(log_probs)


# =========================================================
# RUN ALL TESTS
# =========================================================

if __name__ == "__main__":
    test_exponential()
    build_joint_distribution()
    test_cross_entropy()
    test_sequence()