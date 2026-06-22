import numpy as np
import time
import matplotlib.pyplot as plt

# ============================================================
# 1. GAUSSIAN ELIMINATION
# ============================================================

def gaussian_elimination(A, b):
    A = A.astype(float).copy()
    b = b.astype(float).copy()

    n = len(b)

    # Forward elimination
    for k in range(n):

        pivot = np.argmax(np.abs(A[k:, k])) + k

        A[[k, pivot]] = A[[pivot, k]]
        b[[k, pivot]] = b[[pivot, k]]

        for i in range(k + 1, n):
            factor = A[i, k] / A[k, k]

            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (
            b[i] -
            np.dot(A[i, i + 1:], x[i + 1:])
        ) / A[i, i]

    return x


# ============================================================
# 2. LU DECOMPOSITION
# ============================================================

def lu_decomposition(A):
    A = A.astype(float)

    n = A.shape[0]

    L = np.eye(n)
    U = A.copy()

    for k in range(n - 1):

        for i in range(k + 1, n):

            factor = U[i, k] / U[k, k]

            L[i, k] = factor

            U[i, k:] -= factor * U[k, k:]

    return L, U


def lu_solve(A, b):

    L, U = lu_decomposition(A)

    n = len(b)

    y = np.zeros(n)

    # Forward solve Ly=b
    for i in range(n):
        y[i] = b[i] - np.dot(L[i, :i], y[:i])

    x = np.zeros(n)

    # Back solve Ux=y
    for i in reversed(range(n)):
        x[i] = (
            y[i] -
            np.dot(U[i, i+1:], x[i+1:])
        ) / U[i, i]

    return x


# ============================================================
# 3. CONJUGATE GRADIENT
# ============================================================

def conjugate_gradient(A, b,
                       tol=1e-8,
                       max_iter=None):

    n = len(b)

    if max_iter is None:
        max_iter = n

    x = np.zeros(n)

    r = b - A @ x
    p = r.copy()

    rs_old = r @ r

    for it in range(max_iter):

        Ap = A @ p

        alpha = rs_old / (p @ Ap)

        x = x + alpha * p

        r = r - alpha * Ap

        rs_new = r @ r

        if np.sqrt(rs_new) < tol:
            return x, it + 1

        p = r + (rs_new / rs_old) * p

        rs_old = rs_new

    return x, max_iter


# ============================================================
# 4. CHOLESKY SOLVER
# ============================================================

def cholesky_solve(A, b):

    L = np.linalg.cholesky(A)

    y = np.linalg.solve(L, b)

    x = np.linalg.solve(L.T, y)

    return x


# ============================================================
# EXERCISE 1
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 1")
print("=" * 60)

A = np.array([
    [1,2,3],
    [4,5,6],
    [7,8,10]
], dtype=float)

b = np.array([6,15,27], dtype=float)

x_gauss = gaussian_elimination(A,b)
x_lu = lu_solve(A,b)
x_np = np.linalg.solve(A,b)

print("Gaussian:", x_gauss)
print("LU      :", x_lu)
print("NumPy   :", x_np)

print("All close:",
      np.allclose(x_gauss,x_np) and
      np.allclose(x_lu,x_np))


# ============================================================
# EXERCISE 2
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 2")
print("=" * 60)

np.random.seed(42)

X = np.random.randn(50,5)

w_true = np.array([1,2,3,4,5])

noise = 0.1*np.random.randn(50)

y = X @ w_true + noise

# Normal equations
w_normal = np.linalg.solve(
    X.T @ X,
    X.T @ y
)

# QR
Q,R = np.linalg.qr(X)

w_qr = np.linalg.solve(
    R,
    Q.T @ y
)

# SVD
U,S,Vt = np.linalg.svd(X,full_matrices=False)

w_svd = (
    Vt.T
    @ np.diag(1/S)
    @ U.T
    @ y
)

# lstsq
w_lstsq = np.linalg.lstsq(
    X,y,rcond=None
)[0]

print("Normal :", w_normal)
print("QR     :", w_qr)
print("SVD    :", w_svd)
print("Lstsq  :", w_lstsq)

cond = np.linalg.cond(X.T @ X)

print("\nCondition number(X^T X):", cond)

print("""
Explanation:
- Normal equations square condition number
- QR more stable
- SVD most stable
- lstsq internally uses QR/SVD
""")


# ============================================================
# EXERCISE 3
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 3")
print("=" * 60)

n = 10

A = np.random.randn(n,n)

A[:,1] = A[:,0] + 1e-10*np.random.randn(n)

b = np.random.randn(n)

cond = np.linalg.cond(A)

print("Condition number:", cond)

x_bad = np.linalg.solve(A,b)

lam = 0.01

A_reg = A + lam*np.eye(n)

x_reg = np.linalg.solve(A_reg,b)

res_bad = np.linalg.norm(A@x_bad - b)
res_reg = np.linalg.norm(A@x_reg - b)

print("Residual (original):", res_bad)
print("Residual (regularized):", res_reg)

print("""
Why regularization helps:
- increases smallest singular value
- improves conditioning
- reduces instability
""")


# ============================================================
# EXERCISE 4
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 4")
print("=" * 60)

n = 100

M = np.random.randn(n,n)

A = M.T @ M + n*np.eye(n)

b = np.random.randn(n)

x_cg, iterations = conjugate_gradient(
    A,b,tol=1e-8
)

x_true = np.linalg.solve(A,b)

print("Iterations:", iterations)

print("Theoretical max:", n)

print("Error:",
      np.linalg.norm(
          x_cg-x_true
      ))


# ============================================================
# EXERCISE 5
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 5")
print("=" * 60)

sizes = [10,50,200,500]

chol_times = []
lu_times = []
np_times = []

for n in sizes:

    M = np.random.randn(n,n)

    A = M.T @ M + n*np.eye(n)

    b = np.random.randn(n)

    # Cholesky
    start = time.perf_counter()

    cholesky_solve(A,b)

    chol_times.append(
        time.perf_counter()-start
    )

    # LU
    if n <= 200:

        start = time.perf_counter()

        lu_solve(A,b)

        lu_times.append(
            time.perf_counter()-start
        )

    else:
        lu_times.append(np.nan)

    # NumPy
    start = time.perf_counter()

    np.linalg.solve(A,b)

    np_times.append(
        time.perf_counter()-start
    )

print("\nSizes:", sizes)
print("Cholesky:", chol_times)
print("LU:", lu_times)
print("NumPy:", np_times)

plt.figure(figsize=(8,5))

plt.plot(
    sizes,
    chol_times,
    marker="o",
    label="Cholesky"
)

plt.plot(
    sizes,
    lu_times,
    marker="o",
    label="LU"
)

plt.plot(
    sizes,
    np_times,
    marker="o",
    label="np.linalg.solve"
)

plt.xlabel("Matrix Size")
plt.ylabel("Seconds")
plt.title("Solver Timing Comparison")
plt.legend()
plt.grid(True)
plt.show()

print("""
Expected result:

For SPD matrices:

Cholesky:
~ n^3 / 3 flops

LU:
~ 2 n^3 / 3 flops

Thus Cholesky should be
roughly 2x faster than LU.
""")