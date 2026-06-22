import math

# =========================================================
# 1. REVERSE MODE AUTODIFF (Value class)
# =========================================================

class Value:
    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_children)
        self._op = _op
        self._backward = lambda: None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    # -------------------------
    # REQUIRED: POWER OPERATOR
    # -------------------------
    def __pow__(self, power):
        out = Value(self.data ** power, (self,), f'**{power}')

        def _backward():
            self.grad += power * (self.data ** (power - 1)) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')

        def _backward():
            self.grad += (1 if self.data > 0 else 0) * out.grad
        out._backward = _backward
        return out

    # -------------------------
    # TANH ACTIVATION
    # -------------------------
    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# =========================================================
# 2. VERIFY: d/dx x^3 at x=2 = 12
# =========================================================

x = Value(2.0)
y = x ** 3
y.backward()

print("=== Power Test ===")
print("x^3 derivative at x=2:")
print("Expected: 12")
print("Computed:", x.grad)
print()


# =========================================================
# 3. TANH TESTS
# =========================================================

def tanh_test(x_val):
    x = Value(x_val)
    y = x.tanh()
    y.backward()
    return x.grad

print("=== TANH TEST ===")
print("tanh'(0) =", tanh_test(0.0))   # should be ~1
print("tanh'(2) =", tanh_test(2.0))   # should be ~0.0707
print()


# =========================================================
# 4. SINGLE NEURON GRAPH
# y = relu(w1*x1 + w2*x2 + b)
# =========================================================

def neuron_forward_backward():
    # inputs
    x1 = Value(1.0)
    x2 = Value(-2.0)

    # weights
    w1 = Value(0.5)
    w2 = Value(-1.5)

    # bias
    b = Value(0.1)

    # forward
    z = x1*w1 + x2*w2 + b
    y = z.relu()

    y.backward()

    print("=== NEURON GRADIENTS ===")
    print("dy/dw1:", w1.grad)
    print("dy/dw2:", w2.grad)
    print("dy/dx1:", x1.grad)
    print("dy/dx2:", x2.grad)
    print("dy/db :", b.grad)


neuron_forward_backward()


# =========================================================
# 5. PYTORCH VERIFICATION (optional if installed)
# =========================================================

def pytorch_check():
    try:
        import torch

        x1 = torch.tensor(1.0, requires_grad=True)
        x2 = torch.tensor(-2.0, requires_grad=True)
        w1 = torch.tensor(0.5, requires_grad=True)
        w2 = torch.tensor(-1.5, requires_grad=True)
        b  = torch.tensor(0.1, requires_grad=True)

        z = x1*w1 + x2*w2 + b
        y = torch.relu(z)

        y.backward()

        print("\n=== PYTORCH CHECK ===")
        print("dw1:", w1.grad.item())
        print("dw2:", w2.grad.item())
        print("dx1:", x1.grad.item())
        print("dx2:", x2.grad.item())
        print("db :", b.grad.item())

    except ImportError:
        print("\nPyTorch not installed, skipping check.")

pytorch_check()


# =========================================================
# 6. FORWARD MODE AUTODIFF (DUAL NUMBERS)
# =========================================================

class Dual:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0)
        return Dual(self.real + other.real, self.dual + other.dual)

    def __mul__(self, other):
        other = other if isinstance(other, Dual) else Dual(other, 0)
        return Dual(
            self.real * other.real,
            self.real * other.dual + self.dual * other.real
        )

    def __pow__(self, power):
        return Dual(
            self.real ** power,
            power * (self.real ** (power - 1)) * self.dual
        )

    def tanh(self):
        t = math.tanh(self.real)
        return Dual(t, (1 - t**2) * self.dual)


# =========================================================
# 7. VERIFY FORWARD VS REVERSE MODE
# =========================================================

print("\n=== FORWARD MODE CHECK ===")

# Forward Mode
x_forward = Dual(2.0, 1.0)
y_forward = x_forward ** 3

print("Forward mode d/dx x^3 at 2:", y_forward.dual)

# Reverse Mode
x_reverse = Value(2.0)
y_reverse = x_reverse ** 3
y_reverse.backward()

print("Reverse mode:", x_reverse.grad)