import numpy as np

# =========================================================
# 1. CATASTROPHIC CANCELLATION (VARIANCE)
# =========================================================

def naive_variance(x):
    x = np.array(x, dtype=np.float32)
    return np.mean(x**2) - np.mean(x)**2


def welford_variance(x):
    mean = 0.0
    M2 = 0.0
    n = 0

    for xi in x:
        n += 1
        delta = xi - mean
        mean += delta / n
        delta2 = xi - mean
        M2 += delta * delta2

    return M2 / n


def variance_experiment():
    x = [1000000.0, 1000001.0, 1000002.0]

    true_var = np.var(np.array(x, dtype=np.float64))

    naive = naive_variance(x)
    stable = welford_variance(x)

    print("=== Variance Stability ===")
    print("True variance:", true_var)
    print("Naive variance (float32):", naive)
    print("Welford variance:", stable)

    print("\nObservation:")
    print("- naive suffers catastrophic cancellation")
    print("- Welford remains numerically stable\n")


# =========================================================
# 2. FLOAT32 MACHINE EPSILON
# =========================================================

def epsilon_experiment():
    x = np.float32(1.0)
    eps = np.float32(1.0)

    while np.float32(1.0 + eps) != np.float32(1.0):
        eps /= np.float32(2.0)

    print("=== Machine Epsilon ===")
    print("Computed epsilon:", eps)
    print("NumPy epsilon:", np.finfo(np.float32).eps)

    print("\nObservation:")
    print("- smallest x where 1 + x == 1 in float32\n")


# =========================================================
# 3. LOG-SUM-EXP STABILITY
# =========================================================

def logsumexp_naive(x):
    return np.log(np.sum(np.exp(x)))


def logsumexp_stable(x):
    x = np.array(x)
    m = np.max(x)
    return m + np.log(np.sum(np.exp(x - m)))


def logsumexp_experiment():
    print("=== Log-Sum-Exp Stability ===")

    cases = {
        "equal": [1.0, 1.0, 1.0],
        "one large": [1000, 1, 2],
        "very negative": [-1000, -1000, -1000],
    }

    for name, x in cases.items():
        print(f"\nCase: {name}")

        try:
            print("Naive:", logsumexp_naive(x))
        except Exception as e:
            print("Naive failed:", e)

        print("Stable:", logsumexp_stable(x))

    print("\nObservation:")
    print("- naive overflows with large values")
    print("- stable form always subtracts max\n")


# =========================================================
# 4. GRADIENT CHECKING (LINEAR LAYER)
# =========================================================

def forward(W, x, b):
    return W @ x + b


def loss(y):
    return np.sum(y**2)


def analytical_grad(W, x, b):
    y = forward(W, x, b)
    dL_dy = 2 * y

    dW = np.outer(dL_dy, x)
    dx = W.T @ dL_dy
    db = dL_dy

    return dW, dx, db


def numerical_grad(W, x, b, eps=1e-5):
    dW = np.zeros_like(W)

    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            Wp = W.copy()
            Wm = W.copy()

            Wp[i, j] += eps
            Wm[i, j] -= eps

            lp = loss(forward(Wp, x, b))
            lm = loss(forward(Wm, x, b))

            dW[i, j] = (lp - lm) / (2 * eps)

    return dW


def gradient_check():
    print("=== Gradient Check ===")

    np.random.seed(0)

    W = np.random.randn(3, 2)
    x = np.random.randn(2)
    b = np.random.randn(3)

    dW_anal, dx, db = analytical_grad(W, x, b)
    dW_num = numerical_grad(W, x, b)

    diff = np.linalg.norm(dW_anal - dW_num)

    print("Gradient difference (W):", diff)

    print("\nObservation:")
    print("- small error confirms correct backprop implementation\n")


# =========================================================
# 5. FLOAT16 LOSS SCALING EXPERIMENT
# =========================================================

def loss_scaling_experiment():
    print("=== Loss Scaling (FP16) ===")

    grads = np.random.uniform(1e-9, 1e-3, 10000)

    fp16 = grads.astype(np.float16)
    zero_fraction = np.mean(fp16 == 0)

    scaled = grads * 1024
    fp16_scaled = scaled.astype(np.float16)
    unscaled = fp16_scaled / 1024

    zero_fraction_scaled = np.mean(fp16_scaled == 0)

    print("Zero fraction (no scaling):", zero_fraction)
    print("Zero fraction (scaled):", zero_fraction_scaled)

    print("\nObservation:")
    print("- small gradients underflow to zero in FP16")
    print("- loss scaling preserves small gradients\n")


# =========================================================
# RUN ALL
# =========================================================

if __name__ == "__main__":
    variance_experiment()
    epsilon_experiment()
    logsumexp_experiment()
    gradient_check()
    loss_scaling_experiment()