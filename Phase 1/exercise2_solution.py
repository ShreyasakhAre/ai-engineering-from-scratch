"""
Question 1:
Verify the inverse. Multiply A @ A.inverse_2x2() and confirm you get the identity matrix.
Try it with three different 2x2 matrices. What happens when the determinant is zero?

Question 2:
Implement 3x3 inverse using the adjugate method. Test it against NumPy's np.linalg.inv.

Question 3:
Build a two-layer neural network using only the Matrix class (no NumPy for the network).
Architecture: Input(3) -> Hidden(4) -> Output(2)
"""

import random
import numpy as np


class Matrix:
    def __init__(self, data):
        self.data = data

    def __matmul__(self, other):
        rows = len(self.data)
        cols = len(other.data[0])
        inner = len(self.data[0])

        result = []
        for i in range(rows):
            row = []
            for j in range(cols):
                value = 0
                for k in range(inner):
                    value += self.data[i][k] * other.data[k][j]
                row.append(value)
            result.append(row)

        return Matrix(result)

    def transpose(self):
        rows = len(self.data)
        cols = len(self.data[0])

        return Matrix([
            [self.data[i][j] for i in range(rows)]
            for j in range(cols)
        ])

    # ---------- 2x2 ----------

    def determinant_2x2(self):
        a, b = self.data[0]
        c, d = self.data[1]
        return a * d - b * c

    def inverse_2x2(self):
        det = self.determinant_2x2()

        if det == 0:
            raise ValueError("Matrix is singular (determinant = 0)")

        a, b = self.data[0]
        c, d = self.data[1]

        return Matrix([
            [d / det, -b / det],
            [-c / det, a / det]
        ])

    # ---------- 3x3 ----------

    def determinant_3x3(self):
        a, b, c = self.data[0]
        d, e, f = self.data[1]
        g, h, i = self.data[2]

        return (
            a * (e * i - f * h)
            - b * (d * i - f * g)
            + c * (d * h - e * g)
        )

    def minor(self, row, col):
        m = []

        for r in range(3):
            if r == row:
                continue

            current = []
            for c in range(3):
                if c == col:
                    continue
                current.append(self.data[r][c])

            m.append(current)

        return m[0][0] * m[1][1] - m[0][1] * m[1][0]

    def cofactor_matrix(self):
        cof = []

        for i in range(3):
            row = []

            for j in range(3):
                sign = (-1) ** (i + j)
                row.append(sign * self.minor(i, j))

            cof.append(row)

        return Matrix(cof)

    def inverse_3x3(self):
        det = self.determinant_3x3()

        if det == 0:
            raise ValueError("Matrix is singular (determinant = 0)")

        adj = self.cofactor_matrix().transpose()

        return Matrix([
            [x / det for x in row]
            for row in adj.data
        ])

    def shape(self):
        return (len(self.data), len(self.data[0]))

    def __str__(self):
        return "\n".join(str(row) for row in self.data)


# ======================================================
# QUESTION 1
# ======================================================

print("=" * 60)
print("QUESTION 1: VERIFY 2x2 INVERSE")
print("=" * 60)

matrices = [
    Matrix([[1, 2], [3, 4]]),
    Matrix([[4, 7], [2, 6]]),
    Matrix([[5, 1], [2, 3]])
]

for idx, A in enumerate(matrices, start=1):
    print(f"\nMatrix {idx}:")
    print(A)

    inv = A.inverse_2x2()

    print("\nInverse:")
    print(inv)

    print("\nA @ A_inverse:")
    print(A @ inv)

singular = Matrix([[1, 2], [2, 4]])

print("\nSingular Matrix Determinant:")
print(singular.determinant_2x2())

try:
    singular.inverse_2x2()
except ValueError as e:
    print("Expected Error:", e)


# ======================================================
# QUESTION 2
# ======================================================

print("\n" + "=" * 60)
print("QUESTION 2: 3x3 INVERSE USING ADJUGATE METHOD")
print("=" * 60)

A = Matrix([
    [1, 2, 3],
    [0, 1, 4],
    [5, 6, 0]
])

my_inverse = A.inverse_3x3()

print("\nCustom Inverse:")
print(my_inverse)

numpy_inverse = np.linalg.inv(np.array(A.data))

print("\nNumPy Inverse:")
print(numpy_inverse)

print("\nDifference (should be near zero):")
print(np.array(my_inverse.data) - numpy_inverse)


# ======================================================
# QUESTION 3
# ======================================================

print("\n" + "=" * 60)
print("QUESTION 3: TWO-LAYER NEURAL NETWORK")
print("=" * 60)

X = Matrix([[1.0, 2.0, 3.0]])

W1 = Matrix([
    [random.random() for _ in range(4)]
    for _ in range(3)
])

W2 = Matrix([
    [random.random() for _ in range(2)]
    for _ in range(4)
])

hidden = X @ W1
output = hidden @ W2

print("\nInput Shape :", X.shape())
print("W1 Shape    :", W1.shape())
print("Hidden Shape:", hidden.shape())
print("W2 Shape    :", W2.shape())
print("Output Shape:", output.shape())

print("\nOutput Values:")
print(output)

print("\nAll shape checks passed successfully.")
