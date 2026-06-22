"""
Exercise 3
1. Apply rotation, scaling, and shearing to a unit square and verify rotation preserves distances.
2. Find eigenvalues of [[4,2],[1,3]] by hand, verify with from-scratch and NumPy.
3. Compose transformations and apply to 8 circle points. Verify determinant rule.
"""

import math
import numpy as np


def matmul(A, B):
    rows = len(A)
    cols = len(B[0])
    inner = len(B)

    result = [[0 for _ in range(cols)] for _ in range(rows)]

    for i in range(rows):
        for j in range(cols):
            for k in range(inner):
                result[i][j] += A[i][k] * B[k][j]

    return result


def apply_transform(M, point):
    x, y = point
    return [
        M[0][0] * x + M[0][1] * y,
        M[1][0] * x + M[1][1] * y
    ]


def determinant_2x2(M):
    return M[0][0] * M[1][1] - M[0][1] * M[1][0]


print("=" * 60)
print("QUESTION 1")
print("=" * 60)

square = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]

theta = math.radians(45)

rotation = [
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
]

scaling = [
    [2, 0],
    [0, 3]
]

shearing = [
    [1, 0.5],
    [0, 1]
]

print("\nOriginal Square")
for p in square:
    print(p)

print("\nRotated Square")
rotated = [apply_transform(rotation, p) for p in square]
for p in rotated:
    print(p)

print("\nScaled Square")
scaled = [apply_transform(scaling, p) for p in square]
for p in scaled:
    print(p)

print("\nSheared Square")
sheared = [apply_transform(shearing, p) for p in square]
for p in sheared:
    print(p)

print("\nVerify Rotation Preserves Distances")

def distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

orig = distance(square[0], square[1])
rot = distance(rotated[0], rotated[1])

print("Original edge length:", orig)
print("Rotated edge length :", rot)


print("\n" + "=" * 60)
print("QUESTION 2")
print("=" * 60)

print("""
Matrix:
[4 2]
[1 3]

Characteristic equation:

|A - λI| = 0

(4-λ)(3-λ) - 2 = 0

12 - 7λ + λ² - 2 = 0

λ² - 7λ + 10 = 0

(λ-5)(λ-2)=0

Eigenvalues = 5 and 2
""")


def eigenvalues_2x2(M):
    a, b = M[0]
    c, d = M[1]

    trace = a + d
    det = a * d - b * c

    disc = trace**2 - 4 * det
    root = math.sqrt(disc)

    return (
        (trace + root) / 2,
        (trace - root) / 2
    )


A = [[4, 2], [1, 3]]

scratch_vals = eigenvalues_2x2(A)
numpy_vals = np.linalg.eigvals(np.array(A))

print("From Scratch:", scratch_vals)
print("NumPy       :", numpy_vals)


print("\n" + "=" * 60)
print("QUESTION 3")
print("=" * 60)

theta = math.radians(30)

R = [
    [math.cos(theta), -math.sin(theta)],
    [math.sin(theta), math.cos(theta)]
]

S = [
    [1.5, 0],
    [0, 0.8]
]

H = [
    [1, 0.3],
    [0, 1]
]

composed = matmul(H, matmul(S, R))

circle_points = []

for i in range(8):
    angle = 2 * math.pi * i / 8
    circle_points.append([
        round(math.cos(angle), 6),
        round(math.sin(angle), 6)
    ])

print("\nBefore -> After")

for p in circle_points:
    transformed = apply_transform(composed, p)
    print(f"{p} -> {[round(x,6) for x in transformed]}")

det_R = determinant_2x2(R)
det_S = determinant_2x2(S)
det_H = determinant_2x2(H)

det_composed = determinant_2x2(composed)
det_product = det_R * det_S * det_H

print("\nDet(R) =", det_R)
print("Det(S) =", det_S)
print("Det(H) =", det_H)

print("\nDet(Composed Matrix) =", det_composed)
print("Product of Individual Determinants =", det_product)
print("Difference =", abs(det_composed - det_product))
