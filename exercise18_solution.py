import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXERCISE 1
# CONVEXITY GALLERY
# ============================================================

def convexity_checker_1d(f,
                         xmin=-5,
                         xmax=5,
                         trials=5000,
                         tol=1e-8):

    for _ in range(trials):

        x = np.random.uniform(xmin, xmax)
        y = np.random.uniform(xmin, xmax)

        t = np.random.rand()

        lhs = f(t*x + (1-t)*y)

        rhs = (
            t*f(x)
            + (1-t)*f(y)
        )

        if lhs > rhs + tol:
            return False

    return True


def convexity_checker_2d(f,
                         xmin=-5,
                         xmax=5,
                         trials=5000,
                         tol=1e-8):

    for _ in range(trials):

        x = np.random.uniform(xmin, xmax, 2)
        y = np.random.uniform(xmin, xmax, 2)

        t = np.random.rand()

        lhs = f(
            t*x + (1-t)*y
        )

        rhs = (
            t*f(x)
            + (1-t)*f(y)
        )

        if lhs > rhs + tol:
            return False

    return True


print("\n" + "="*60)
print("EXERCISE 1: CONVEXITY")
print("="*60)

f1 = lambda x: x**4
f2 = lambda x: np.sin(x)

f3 = lambda v: v[0]**2 + v[1]**2
f4 = lambda v: v[0]*v[1]

f5 = lambda x: max(x,0)

print("x^4:", convexity_checker_1d(f1))
print("sin(x):", convexity_checker_1d(f2))
print("x²+y²:", convexity_checker_2d(f3))
print("x*y:", convexity_checker_2d(f4))
print("ReLU:", convexity_checker_1d(f5))

print("""
Expected:
x^4        -> convex
sin(x)     -> not convex
x²+y²      -> convex
x*y        -> not convex
max(x,0)   -> convex
""")


# ============================================================
# EXERCISE 2
# NEWTON VS GRADIENT DESCENT
# ============================================================

def f(v):
    x,y = v
    return 50*x*x + y*y


def grad(v):
    x,y = v

    return np.array([
        100*x,
        2*y
    ])


def hessian(v):

    return np.array([
        [100,0],
        [0,2]
    ])


def gradient_descent(
        start,
        lr=0.01,
        tol=1e-10):

    x = start.copy()

    steps = 0

    while f(x) > tol:

        x = x - lr*grad(x)

        steps += 1

        if steps > 100000:
            break

    return steps


def newton_method(
        start,
        tol=1e-10):

    x = start.copy()

    Hinv = np.linalg.inv(
        hessian(x)
    )

    steps = 0

    while f(x) > tol:

        x = x - Hinv @ grad(x)

        steps += 1

        if steps > 100:
            break

    return steps


print("\n" + "="*60)
print("EXERCISE 2: GD VS NEWTON")
print("="*60)

start = np.array([10.0,10.0])

gd_steps = gradient_descent(
    start,
    lr=0.01
)

newton_steps = newton_method(
    start
)

print("GD steps:", gd_steps)
print("Newton steps:", newton_steps)

print("""
Condition number:
kappa = 100 / 2 = 50

As condition number increases:
- valley becomes narrower
- GD zig-zags
- Newton rescales using Hessian
""")


# ============================================================
# EXERCISE 3
# LAGRANGE MULTIPLIERS
# ============================================================

print("\n" + "="*60)
print("EXERCISE 3: LAGRANGE")
print("="*60)

# Analytical solution

# Constraint:
# x + 2y = 4

# grad f = λ grad g

# [2(x-3),2(y-3)] = λ[1,2]

A = np.array([
    [2,0,-1],
    [0,2,-2],
    [1,2,0]
])

b = np.array([
    6,
    6,
    4
])

sol = np.linalg.solve(A,b)

x,y,lam = sol

print("Solution:")
print("x =", x)
print("y =", y)
print("lambda =", lam)

grad_f = np.array([
    2*(x-3),
    2*(y-3)
])

grad_g = np.array([
    1,
    2
])

print("\nGradient f:", grad_f)
print("Gradient g:", grad_g)

ratio = grad_f[0]/grad_g[0]

print("Parallel check:",
      np.allclose(
          grad_f,
          ratio*grad_g
      ))


# ============================================================
# EXERCISE 4
# L1 CONSTRAINED OPTIMIZATION
# ============================================================

print("\n" + "="*60)
print("EXERCISE 4: L1 CONSTRAINT")
print("="*60)

best_loss = float("inf")
best_point = None

grid = np.linspace(-1,1,2000)

for x in grid:

    ymax = 1 - abs(x)

    ys = np.linspace(
        -ymax,
        ymax,
        200
    )

    for y in ys:

        loss = (
            (x-3)**2
            + (y-2)**2
        )

        if loss < best_loss:

            best_loss = loss
            best_point = (x,y)

print("Best point:", best_point)
print("Loss:", best_loss)

print("""
L1 ball is a diamond.

Optimal solution occurs
at a corner/edge.

This causes sparsity.
One coordinate often becomes 0.
""")


# ============================================================
# EXERCISE 5
# ROSENBROCK HESSIAN
# ============================================================

print("\n" + "="*60)
print("EXERCISE 5: ROSENBROCK HESSIAN")
print("="*60)

def rosenbrock_hessian(x,y):

    h11 = (
        1200*x*x
        - 400*y
        + 2
    )

    h12 = -400*x

    h22 = 200

    return np.array([
        [h11,h12],
        [h12,h22]
    ])


points = [
    (1,1),
    (-1,1)
]

for p in points:

    H = rosenbrock_hessian(*p)

    eigvals = np.linalg.eigvals(H)

    print("\nPoint:", p)
    print("Hessian:")
    print(H)

    print("Eigenvalues:")
    print(eigvals)

print("""
Interpretation:

(1,1):
- global minimum
- both eigenvalues positive
- strong local curvature

(-1,1):
- far from optimum
- curvature highly anisotropic
- one direction steep
- one direction relatively flat

Large eigenvalue ratio
=> optimization difficulty.
""")