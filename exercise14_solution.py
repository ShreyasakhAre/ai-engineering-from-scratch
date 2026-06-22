import numpy as np
from itertools import combinations

# =========================================================
# 1. L1, L2, L∞ DISTANCES + INEQUALITY
# =========================================================

def distances(a, b):
    a = np.array(a)
    b = np.array(b)
    diff = a - b

    l1 = np.sum(np.abs(diff))
    l2 = np.sqrt(np.sum(diff**2))
    linf = np.max(np.abs(diff))

    return l1, l2, linf


def distance_experiment():
    print("=== Distance Computation ===")

    a = [1, 2, 3]
    b = [4, 0, 6]

    l1, l2, linf = distances(a, b)

    print("L1:", l1)
    print("L2:", l2)
    print("L∞:", linf)

    print("\nInequality check:")
    print("L∞ <= L2 <= L1 ?", linf <= l2 <= l1)

    print("\nWhy it holds:")
    print("- L∞ takes max coordinate deviation")
    print("- L2 aggregates squared deviations (always ≤ L1 in finite dim)")
    print("- L1 sums all absolute deviations (largest)\n")


# =========================================================
# 2. COSINE vs L2 CONTRADICTIONS
# =========================================================

def cosine(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def cosine_l2_examples():
    print("=== Cosine vs L2 Geometry ===")

    # High cosine (>0.9) but large L2 (>10)
    a = np.array([100, 100, 100])
    b = np.array([90, 90, 90])

    print("\nHigh cosine, large L2:")
    print("cosine:", cosine(a, b))
    print("L2:", np.linalg.norm(a - b))

    # Low cosine (<0.3) but small L2 (<0.5)
    a = np.array([1.0, 0.0])
    b = np.array([0.8, 0.1])

    print("\nLow cosine, small L2:")
    print("cosine:", cosine(a, b))
    print("L2:", np.linalg.norm(a - b))

    print("\nInsight:")
    print("- cosine measures angle (direction)")
    print("- L2 measures magnitude + direction difference\n")


# =========================================================
# 3. NEAREST NEIGHBOR UNDER DIFFERENT METRICS
# =========================================================

def l1(a, b): return np.sum(np.abs(a - b))
def l2(a, b): return np.linalg.norm(a - b)

def cosine_dist(a, b):
    return 1 - cosine(a, b)

def mahalanobis(a, b, cov_inv):
    d = a - b
    return np.sqrt(d.T @ cov_inv @ d)


def nearest_neighbors():
    print("=== Nearest Neighbor Disagreement ===")

    np.random.seed(0)

    data = np.random.randn(5, 3)
    query = np.array([0.2, -0.1, 0.3])

    cov = np.cov(data.T) + 1e-6 * np.eye(3)
    cov_inv = np.linalg.inv(cov)

    metrics = {
        "L1": lambda x, y: l1(x, y),
        "L2": lambda x, y: l2(x, y),
        "Cosine": lambda x, y: cosine_dist(x, y),
        "Mahalanobis": lambda x, y: mahalanobis(x, y, cov_inv),
    }

    results = {}

    for name, metric in metrics.items():
        dists = [metric(query, p) for p in data]
        results[name] = np.argmin(dists)

    print("Nearest indices:")
    for k, v in results.items():
        print(k, "→", v)

    print("\nObservation:")
    print("- different geometry ⇒ different nearest neighbors\n")


# =========================================================
# 4. WASSERSTEIN DISTANCE (CDF METHOD)
# =========================================================

def wasserstein(p, q):
    p = np.array(p)
    q = np.array(q)

    return np.sum(np.abs(np.cumsum(p) - np.cumsum(q)))


def wasserstein_experiment():
    print("=== Wasserstein Distance ===")

    p1 = [0.5, 0.5, 0, 0]
    q1 = [0, 0, 0.5, 0.5]

    p2 = [0.25, 0.25, 0.25, 0.25]
    q2 = [0, 0, 0.5, 0.5]

    d1 = wasserstein(p1, q1)
    d2 = wasserstein(p2, q2)

    print("W(p1, q1):", d1)
    print("W(p2, q2):", d2)

    print("\nObservation:")
    print("- second pair has higher spread mismatch")
    print("- Wasserstein captures 'mass transport distance'\n")


# =========================================================
# 5. MINHASH VS JACCARD SIMILARITY
# =========================================================

def jaccard(a, b):
    return len(a & b) / len(a | b)


def minhash_signature(sets, num_hashes):
    max_val = 1000
    hashes = np.random.randint(1, max_val, size=(num_hashes, 2))

    def hash_fn(x, a, b):
        return (a * x + b) % max_val

    signatures = []

    for s in sets:
        sig = []
        for a, b in hashes:
            min_hash = min(hash_fn(x, a, b) for x in s)
            sig.append(min_hash)
        signatures.append(sig)

    return np.array(signatures)


def minhash_similarity(sig1, sig2):
    return np.mean(sig1 == sig2)


def minhash_experiment():
    print("=== MinHash Experiment ===")

    np.random.seed(0)

    sets = []
    for _ in range(100):
        sets.append(set(np.random.randint(0, 200, 20)))

    exact_errors = []

    for k in [50, 100, 200]:
        sigs = minhash_signature(sets, k)

        errors = []

        for i, j in combinations(range(10), 2):
            exact = jaccard(sets[i], sets[j])
            approx = minhash_similarity(sigs[i], sigs[j])
            errors.append(abs(exact - approx))

        print(f"\nHash functions = {k}")
        print("Avg error:", np.mean(errors))

    print("\nObservation:")
    print("- more hashes → lower variance")
    print("- MinHash approximates Jaccard efficiently\n")


if __name__ == "__main__":
    distance_experiment()
    cosine_l2_examples()
    nearest_neighbors()
    wasserstein_experiment()
    minhash_experiment()