import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# EXERCISE 1
# COMPLEX ARITHMETIC BY HAND
# ============================================================

print("=" * 60)
print("EXERCISE 1: COMPLEX ARITHMETIC")
print("=" * 60)

# (2+3i)(4-i)
z1 = 2 + 3j
z2 = 4 - 1j

result1 = z1 * z2

# By hand:
# (2+3i)(4-i)
# = 8 -2i +12i -3i²
# = 8 +10i +3
# = 11 +10i

print("(2+3i)(4-i) =", result1)
print("Expected     =", 11 + 10j)

# --------------------------------

# (5+2i)/(1-3i)

z3 = 5 + 2j
z4 = 1 - 3j

result2 = z3 / z4

# By hand:
#
# multiply top/bottom by (1+3i)
#
# numerator:
# (5+2i)(1+3i)
# =5+15i+2i+6i²
# =-1+17i
#
# denominator:
# 1+9=10
#
# = -0.1 + 1.7i

print("\n(5+2i)/(1-3i) =", result2)
print("Expected       =", -0.1 + 1.7j)

# Plot
plt.figure(figsize=(6, 6))

points = [
    ("Product", result1),
    ("Division", result2)
]

for label, z in points:
    plt.scatter(z.real, z.imag)
    plt.text(z.real, z.imag, label)

plt.axhline(0)
plt.axvline(0)

plt.title("Complex Plane")
plt.grid(True)
plt.show()

print("""
Multiplication in the complex plane:
- scales by |z|
- rotates by arg(z)
""")


# ============================================================
# EXERCISE 2
# ROTATION SEQUENCE
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 2: ROTATION SEQUENCE")
print("=" * 60)

primitive = np.exp(1j * np.pi / 6)

z = 1 + 0j

points = []

for k in range(13):

    points.append(z)

    print(
        f"{k:2d}: "
        f"({z.real:.6f}, {z.imag:.6f})"
    )

    z *= primitive

print("\nAfter 12 rotations:")
print(points[-1])

# Plot polygon
plt.figure(figsize=(6, 6))

xs = [p.real for p in points]
ys = [p.imag for p in points]

plt.plot(xs, ys, marker="o")

plt.axis("equal")
plt.grid(True)
plt.title("Regular 12-gon from Rotations")
plt.show()


# ============================================================
# EXERCISE 3
# DFT OF KNOWN SIGNAL
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 3: DFT")
print("=" * 60)

N = 32

t = np.arange(N) / N

signal = (
    np.sin(2*np.pi*3*t)
    +
    0.5*np.sin(2*np.pi*7*t)
)

# DFT from scratch
X = []

for k in range(N):

    s = 0j

    for n in range(N):

        s += (
            signal[n]
            *
            np.exp(-2j*np.pi*k*n/N)
        )

    X.append(s)

X = np.array(X)

magnitude = np.abs(X)

print("Largest frequency bins:")

top = np.argsort(magnitude)[-10:]

for idx in reversed(top):
    print(
        f"freq={idx:2d} "
        f"mag={magnitude[idx]:.3f}"
    )

plt.figure(figsize=(8, 4))
plt.stem(magnitude)
plt.title("DFT Magnitude Spectrum")
plt.xlabel("Frequency Bin")
plt.ylabel("Magnitude")
plt.show()

print("""
Expected peaks:
frequency 3
frequency 7

Peak at 7 should be roughly
half of peak at 3.
""")


# ============================================================
# EXERCISE 4
# ROOTS OF UNITY
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 4: ROOTS OF UNITY")
print("=" * 60)

n = 8

roots = []

for k in range(n):

    root = np.exp(
        2j*np.pi*k/n
    )

    roots.append(root)

print("Roots:")

for r in roots:
    print(r)

root_sum = sum(roots)

print("\nSum of roots:")
print(root_sum)

primitive = np.exp(
    2j*np.pi/n
)

print("\nVerify successor property:")

for k in range(n):

    next_root = roots[k] * primitive

    expected = roots[(k+1) % n]

    print(
        np.allclose(
            next_root,
            expected
        )
    )

plt.figure(figsize=(6, 6))

for r in roots:
    plt.scatter(r.real, r.imag)

plt.axis("equal")
plt.grid(True)
plt.title("8th Roots of Unity")
plt.show()


# ============================================================
# EXERCISE 5
# ROTATION MATRIX EQUIVALENCE
# ============================================================

print("\n" + "=" * 60)
print("EXERCISE 5")
print("ROTATION MATRIX EQUIVALENCE")
print("=" * 60)

np.random.seed(42)

max_error = 0

for _ in range(10):

    theta = np.random.uniform(
        -2*np.pi,
        2*np.pi
    )

    point = np.random.randn(2)

    # -----------------
    # Complex rotation
    # -----------------

    z = point[0] + 1j*point[1]

    z_rot = z * np.exp(1j*theta)

    complex_result = np.array([
        z_rot.real,
        z_rot.imag
    ])

    # -----------------
    # Matrix rotation
    # -----------------

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])

    matrix_result = R @ point

    err = np.linalg.norm(
        complex_result
        - matrix_result
    )

    max_error = max(
        max_error,
        err
    )

print(
    "Maximum numerical difference:",
    max_error
)

print("""
Complex multiplication:

z -> z * e^(iθ)

is exactly equivalent to

[cosθ -sinθ]
[sinθ  cosθ]

acting on a vector.
""")