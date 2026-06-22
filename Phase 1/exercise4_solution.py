# =========================
# 1. NUMERICAL DERIVATIVES
# =========================

def numerical_derivative(f, x, h=1e-5):
    return (f(x + h) - f(x - h)) / (2 * h)


def numerical_second_derivative(f, x, h=1e-5):
    def first_derivative(x_):
        return numerical_derivative(f, x_, h)
    
    return numerical_derivative(first_derivative, x, h)


# =========================
# TEST: SECOND DERIVATIVE
# f(x) = x^3 at x = 2
# =========================

f = lambda x: x**3

second_deriv = numerical_second_derivative(f, 2)
print("===================================")
print("Second derivative of x^3 at x=2")
print("Expected: 12")
print("Computed:", second_deriv)
print("===================================\n")


# =========================
# 2. GRADIENT DESCENT (2D)
# f(x,y) = (x-3)^2 + (y+1)^2
# =========================

def grad_f(x, y):
    dx = 2 * (x - 3)
    dy = 2 * (y + 1)
    return dx, dy


def gradient_descent_2d(lr=0.1, steps=50):
    x, y = 0.0, 0.0  # start

    print("=== Gradient Descent (No Momentum) ===")

    for i in range(steps):
        dx, dy = grad_f(x, y)

        x -= lr * dx
        y -= lr * dy

        if i % 10 == 0:
            print(f"Step {i}: x={x:.4f}, y={y:.4f}")

    return x, y


final_x, final_y = gradient_descent_2d()

print("\nFinal result (should be close to (3, -1)):")
print(final_x, final_y)
print("===================================\n")


# =========================
# 3. GRADIENT DESCENT 1D
# WITH & WITHOUT MOMENTUM
# f(x) = x^4 - 3x^2
# =========================

def f1(x):
    return x**4 - 3*x**2


def df1(x):
    return 4*x**3 - 6*x


# -------------------------
# Without Momentum
# -------------------------
def gd_no_momentum(lr=0.01, steps=100):
    x = 2.0

    print("=== GD WITHOUT MOMENTUM ===")

    for i in range(steps):
        grad = df1(x)
        x -= lr * grad

        if i % 10 == 0:
            print(f"Step {i}: x={x:.5f}")

    return x


# -------------------------
# With Momentum
# -------------------------
def gd_momentum(lr=0.01, beta=0.9, steps=100):
    x = 2.0
    v = 0.0

    print("\n=== GD WITH MOMENTUM ===")

    for i in range(steps):
        grad = df1(x)

        v = beta * v - lr * grad
        x += v

        if i % 10 == 0:
            print(f"Step {i}: x={x:.5f}")

    return x


x_no_momentum = gd_no_momentum()
x_momentum = gd_momentum()

print("\n===================================")
print("Final comparison:")
print("Without momentum:", x_no_momentum)
print("With momentum   :", x_momentum)
print("===================================")