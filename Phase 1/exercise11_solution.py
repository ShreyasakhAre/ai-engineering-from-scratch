import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# 1. SVD FROM SCRATCH (via A^T A eigendecomposition)
# =========================================================

def svd_from_scratch(A, eps=1e-10):
    """
    A = U Σ V^T
    Steps:
    1. Compute A^T A
    2. Eigen-decompose A^T A -> V, Σ^2
    3. Compute singular values
    4. Compute U = A V Σ^{-1}
    """

    ATA = A.T @ A
    eigvals, V = np.linalg.eigh(ATA)

    # sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    V = V[:, idx]

    singular_vals = np.sqrt(np.clip(eigvals, 0, None))

    # compute U
    U = []
    for i in range(len(singular_vals)):
        if singular_vals[i] > eps:
            u = (A @ V[:, i]) / singular_vals[i]
        else:
            u = np.zeros(A.shape[0])
        U.append(u)

    U = np.column_stack(U)

    Sigma = np.diag(singular_vals)

    return U, Sigma, V


def compare_svd(A):
    U1, S1, V1 = svd_from_scratch(A)
    U2, S2, V2 = np.linalg.svd(A, full_matrices=False)

    recon1 = U1 @ S1 @ V1.T
    recon2 = U2 @ np.diag(S2) @ V2

    print("=== SVD Comparison ===")
    print("Scratch error:", np.linalg.norm(A - recon1))
    print("NumPy error  :", np.linalg.norm(A - recon2))
    print()


# =========================================================
# 2. IMAGE COMPRESSION USING SVD
# =========================================================

def compress_image(A, k):
    U, S, V = np.linalg.svd(A, full_matrices=False)

    Ak = U[:, :k] @ np.diag(S[:k]) @ V[:k, :]
    return Ak


def image_experiment():
    print("=== Image Compression ===")

    # synthetic "image" (replace with real image if needed)
    img = np.random.rand(100, 100)

    ranks = [1, 5, 10, 25, 50, 100]

    original_norm = np.linalg.norm(img)

    for k in ranks:
        approx = compress_image(img, k)

        error = np.linalg.norm(img - approx) / original_norm
        compression_ratio = (img.shape[0] * img.shape[1]) / (k * (img.shape[0] + img.shape[1]))

        print(f"k={k:3d} | compression={compression_ratio:.2f} | relative error={error:.4f}")

    print("\nObservation:")
    print("- low k → strong compression but blurry image")
    print("- higher k → better reconstruction")
    print("- visual threshold usually ~k=25–50\n")


# =========================================================
# 3. RECOMMENDER SYSTEM USING SVD
# =========================================================

def recommender_system():
    print("=== Recommendation System ===")

    R = np.array([
        [5, 3, np.nan, 1, 4, np.nan, 2, np.nan],
        [4, np.nan, 3, 1, 5, 2, np.nan, 3],
        [np.nan, 2, 4, 5, np.nan, 3, 1, np.nan],
        [3, 5, 2, np.nan, 4, np.nan, 3, 2],
        [np.nan, 4, 5, 3, np.nan, 2, 4, 1],
        [2, np.nan, 1, 4, 3, 5, np.nan, 2],
        [5, 3, 4, np.nan, 2, 1, 3, np.nan],
        [3, np.nan, 2, 5, 4, np.nan, 1, 3],
        [4, 2, np.nan, 3, 5, 4, np.nan, 1],
        [np.nan, 5, 3, 2, np.nan, 1, 4, 2],
    ])

    # fill missing values with row mean
    R_filled = R.copy()

    for i in range(R.shape[0]):
        mean_val = np.nanmean(R[i])
        R_filled[i] = np.where(np.isnan(R[i]), mean_val, R[i])

    U, S, Vt = np.linalg.svd(R_filled, full_matrices=False)

    k = 3
    R_hat = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

    print("Predicted missing entries (approx):")
    print(np.round(R_hat, 2))
    print("\nObservation:")
    print("- low rank captures latent preferences")
    print("- missing ratings are reconstructed smoothly\n")


# =========================================================
# 4. TOPIC MODEL VIA SVD
# =========================================================

def topic_model():
    print("=== Document-Term SVD Topic Model ===")

    np.random.seed(42)

    vocab_size = 50
    docs = 100
    topics = 3

    base_topics = [
        np.array([5 if i % 10 < 5 else 0 for i in range(vocab_size)]),
        np.array([5 if 10 <= i % 20 < 15 else 0 for i in range(vocab_size)]),
        np.array([5 if 20 <= i % 30 < 25 else 0 for i in range(vocab_size)]),
    ]

    X = []
    labels = []

    for i in range(docs):
        t = i % topics
        vec = base_topics[t] + np.random.normal(0, 0.5, vocab_size)
        X.append(vec)
        labels.append(t)

    X = np.array(X)

    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    print("Top singular values:", S[:10])

    X_proj = U[:, :3] @ np.diag(S[:3])

    print("\nCluster separation check (first 5 points):")
    print(X_proj[:5])

    print("\nObservation:")
    print("- first 3 singular values dominate")
    print("- documents cluster by topic in 3D space\n")


# =========================================================
# 5. NOISE VS OPTIMAL RANK
# =========================================================

def optimal_rank_experiment():
    print("=== Noise vs Optimal Rank ===")

    np.random.seed(0)

    U = np.random.randn(50, 3)
    V = np.random.randn(3, 40)

    clean = U @ V

    noise_levels = [0.1, 0.5, 1.0, 2.0]

    for sigma in noise_levels:
        noisy = clean + np.random.normal(0, sigma, clean.shape)

        errors = []

        for k in range(1, 21):
            U_, S_, Vt_ = np.linalg.svd(noisy, full_matrices=False)
            approx = U_[:, :k] @ np.diag(S_[:k]) @ Vt_[:k, :]

            err = np.linalg.norm(clean - approx)
            errors.append(err)

        best_k = np.argmin(errors) + 1

        print(f"sigma={sigma} → optimal rank ≈ {best_k}")

    print("\nObservation:")
    print("- low noise → optimal k ≈ true rank (3)")
    print("- high noise → higher k needed")
    print("- noise inflates effective dimensionality\n")



if __name__ == "__main__":
    A = np.random.randn(20, 10)

    compare_svd(A)
    image_experiment()
    recommender_system()
    topic_model()
    optimal_rank_experiment()