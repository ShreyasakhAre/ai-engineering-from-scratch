import numpy as np
import math

# =========================================================
# 1. ROSENBROCK FUNCTION
# =========================================================

def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2


def grad_rosenbrock(x, y, a=1, b=100):
    dx = -2 * (a - x) - 4 * b * x * (y - x**2)
    dy = 2 * b * (y - x**2)
    return dx, dy


# =========================================================
# 2. SAFE VANILLA GRADIENT DESCENT
# =========================================================

def gd_rosenbrock(lr, steps=5000):
    x, y = -1.2, 1.0

    try:
        for _ in range(steps):
            dx, dy = grad_rosenbrock(x, y)

            x -= lr * dx
            y -= lr * dy

            # Divergence check
            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or abs(x) > 1e6
                or abs(y) > 1e6
            ):
                return float("inf")

        return rosenbrock(x, y)

    except OverflowError:
        return float("inf")


def learning_rate_sweep():
    lrs = [0.0001, 0.0005, 0.001, 0.005, 0.01]

    print("=== Learning Rate Sweep (Rosenbrock) ===")

    for lr in lrs:
        loss = gd_rosenbrock(lr)

        if math.isinf(loss):
            print(f"lr={lr}: DIVERGED")
        else:
            print(f"lr={lr}: final loss = {loss:.8f}")

    print("\nObservation:")
    print("- very small lr → slow convergence")
    print("- moderate lr → best convergence")
    print("- large lr → divergence")
    print()


# =========================================================
# 3. MOMENTUM
# =========================================================

def momentum_rosenbrock(lr=0.0005, beta=0.9, steps=2000):
    x, y = -1.2, 1.0
    vx, vy = 0.0, 0.0

    losses = []

    try:
        for _ in range(steps):
            dx, dy = grad_rosenbrock(x, y)

            vx = beta * vx - lr * dx
            vy = beta * vy - lr * dy

            x += vx
            y += vy

            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or abs(x) > 1e6
                or abs(y) > 1e6
            ):
                losses.append(float("inf"))
                break

            losses.append(rosenbrock(x, y))

    except OverflowError:
        losses.append(float("inf"))

    return losses


def momentum_experiment():
    betas = [0.0, 0.5, 0.9, 0.99]

    print("=== Momentum Comparison ===")

    for beta in betas:
        losses = momentum_rosenbrock(beta=beta)

        final_loss = losses[-1]

        if math.isinf(final_loss):
            print(f"beta={beta}: DIVERGED")
        else:
            print(f"beta={beta}: final loss = {final_loss:.8f}")

    print("\nObservation:")
    print("- beta=0.0 → vanilla SGD")
    print("- beta=0.5 → smoother updates")
    print("- beta=0.9 → usually best")
    print("- beta=0.99 → may overshoot")
    print()


# =========================================================
# 4. SADDLE POINT
# =========================================================

def saddle_grad(x, y):
    return 2*x, -2*y


def gd_saddle():
    x, y = 0.01, 0.01

    for _ in range(100):
        dx, dy = saddle_grad(x, y)

        x -= 0.01 * dx
        y -= 0.01 * dy

    return x, y


def momentum_saddle():
    x, y = 0.01, 0.01
    vx, vy = 0.0, 0.0

    beta = 0.9

    for _ in range(100):
        dx, dy = saddle_grad(x, y)

        vx = beta * vx - 0.01 * dx
        vy = beta * vy - 0.01 * dy

        x += vx
        y += vy

    return x, y


def adam_saddle():
    x, y = 0.01, 0.01

    mx = my = 0.0
    vx = vy = 0.0

    beta1 = 0.9
    beta2 = 0.999

    lr = 0.01
    eps = 1e-8

    for t in range(1, 101):
        dx, dy = saddle_grad(x, y)

        mx = beta1 * mx + (1 - beta1) * dx
        my = beta1 * my + (1 - beta1) * dy

        vx = beta2 * vx + (1 - beta2) * dx**2
        vy = beta2 * vy + (1 - beta2) * dy**2

        mx_hat = mx / (1 - beta1**t)
        my_hat = my / (1 - beta1**t)

        vx_hat = vx / (1 - beta2**t)
        vy_hat = vy / (1 - beta2**t)

        x -= lr * mx_hat / (np.sqrt(vx_hat) + eps)
        y -= lr * my_hat / (np.sqrt(vy_hat) + eps)

    return x, y


def saddle_experiment():
    print("=== Saddle Point Escape ===")

    print("Vanilla GD :", gd_saddle())
    print("Momentum   :", momentum_saddle())
    print("Adam       :", adam_saddle())

    print("\nObservation:")
    print("- Momentum escapes faster")
    print("- Adam escapes fastest")
    print()


# =========================================================
# 5. LEARNING RATE DECAY
# =========================================================

def gd_with_decay(lr0=0.01, steps=5000):
    x, y = -1.2, 1.0

    try:
        for t in range(steps):

            lr = lr0 * (0.999 ** t)

            dx, dy = grad_rosenbrock(x, y)

            x -= lr * dx
            y -= lr * dy

            if (
                not math.isfinite(x)
                or not math.isfinite(y)
                or abs(x) > 1e6
                or abs(y) > 1e6
            ):
                return float("inf")

        return rosenbrock(x, y)

    except OverflowError:
        return float("inf")


def decay_experiment():
    print("=== Learning Rate Decay ===")

    fixed_loss = gd_rosenbrock(0.001)
    decay_loss = gd_with_decay(0.01)

    print("Fixed LR loss :", fixed_loss)
    print("Decay LR loss :", decay_loss)

    print("\nObservation:")
    print("- decay stabilizes large initial steps")
    print("- helps fine-tune near optimum")
    print()


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    learning_rate_sweep()
    momentum_experiment()
    saddle_experiment()
    decay_experiment()