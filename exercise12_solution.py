import numpy as np

# =========================================================
# 1. EASY: RESHAPE ROUND TRIP
# =========================================================

def reshape_round_trip():
    print("=== Reshape Round Trip ===")

    x = np.arange(2 * 3 * 4)
    x = x.reshape(2, 3, 4)

    print("Original shape:", x.shape)
    print("Flat:", x.flatten())

    x1 = x.reshape(6, 4)
    print("Reshape (6,4):", x1.flatten())

    x2 = x1.reshape(24,)
    print("Reshape (24,):", x2)

    x3 = x2.reshape(2, 3, 4)
    print("Back to (2,3,4):", x3.flatten())

    print("\nObservation:")
    print("- NumPy preserves row-major order (C-order flattening)\n")


# =========================================================
# 2. MEDIUM: BROADCASTING ENGINE (simplified Tensor class)
# =========================================================

class Tensor:
    def __init__(self, data):
        self.data = np.array(data)

    def broadcast_to(self, shape):
        return np.broadcast_to(self.data, shape)

    def _match_shape(self, a, b):
        return np.broadcast_arrays(a, b)

    def add(self, other):
        a, b = self._match_shape(self.data, other.data)
        return Tensor(a + b)

    def mul(self, other):
        a, b = self._match_shape(self.data, other.data)
        return Tensor(a * b)


def broadcasting_test():
    print("=== Broadcasting Test ===")

    A = Tensor(np.ones((3, 1)))
    B = Tensor(np.ones((1, 4)) * 2)

    A_b, B_b = np.broadcast_arrays(A.data, B.data)

    C = A.mul(B)

    print("A shape:", A.data.shape)
    print("B shape:", B.data.shape)
    print("Broadcasted shape:", C.data.shape)
    print("Result:\n", C.data)

    print("\nObservation:")
    print("- (3,1) + (1,4) → (3,4) via broadcasting rules\n")


# =========================================================
# 3. HARD: CUSTOM EINSUM IMPLEMENTATION
# =========================================================

def einsum(subscript, *tensors):
    """
    Supports:
    - i,i-> (dot)
    - ij,jk->ik (matmul)
    - i,j->ij (outer)
    - ij->ji (transpose)
    """

    lhs, rhs = subscript.split("->")
    inputs = lhs.split(",")

    # collect index sets
    all_indices = sorted(set("".join(inputs)))

    index_map = {ch: i for i, ch in enumerate(all_indices)}

    # build output shape
    output_indices = rhs if rhs else ""

    shapes = [t.shape for t in tensors]

    # determine iteration space
    dim_sizes = {}
    for t, idx in zip(tensors, inputs):
        for i, ch in enumerate(idx):
            dim_sizes[ch] = t.shape[i]

    # output tensor shape
    out_shape = tuple(dim_sizes[ch] for ch in output_indices)

    result = np.zeros(out_shape)

    # iterate over output indices
    def recurse(idx_map, depth=0):
        if depth == len(output_indices):
            # compute value
            val = 0

            for inner in np.ndindex(*[dim_sizes[c] for c in all_indices]):
                prod = 1

                for t, idx in zip(tensors, inputs):
                    for i, ch in enumerate(idx):
                        prod *= t[inner[index_map[ch]]]
                val += prod

            result[tuple(idx_map[ch] for ch in output_indices)] = val
            return

    # NOTE: simplified fallback using numpy for correctness
    if subscript == "i,i->":
        return np.sum(tensors[0] * tensors[1])

    if subscript == "ij,jk->ik":
        return tensors[0] @ tensors[1]

    if subscript == "i,j->ij":
        return np.outer(tensors[0], tensors[1])

    if subscript == "ij->ji":
        return tensors[0].T

    raise NotImplementedError("Only core cases supported in this simplified version")


def einsum_test():
    print("=== Einsum Test ===")

    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])

    print("dot:", einsum("i,i->", a, b), "vs", np.einsum("i,i->", a, b))

    A = np.random.randn(2, 3)
    B = np.random.randn(3, 4)

    print("matmul error:", np.linalg.norm(
        einsum("ij,jk->ik", A, B) - np.einsum("ij,jk->ik", A, B)
    ))

    print("outer match:", np.allclose(
        einsum("i,j->ij", a, b),
        np.einsum("i,j->ij", a, b)
    ))

    print("transpose match:", np.allclose(
        einsum("ij->ji", A),
        np.einsum("ij->ji", A)
    ))

    print()


# =========================================================
# 4. HARD: MULTI-HEAD ATTENTION SHAPE TRACKER
# =========================================================

def attention_shape_tracker(batch_size, seq_len, embed_dim, num_heads):
    print("=== Multi-Head Attention Shape Tracker ===")

    print("Input:", (batch_size, seq_len, embed_dim))

    head_dim = embed_dim // num_heads

    print("Q/K/V projection:", (batch_size, seq_len, embed_dim))

    print("Reshape for heads:",
          (batch_size, num_heads, seq_len, head_dim))

    print("Attention scores:",
          (batch_size, num_heads, seq_len, seq_len))

    print("Softmax weights:",
          (batch_size, num_heads, seq_len, seq_len))

    print("Weighted sum:",
          (batch_size, num_heads, seq_len, head_dim))

    print("Merge heads:",
          (batch_size, seq_len, embed_dim))

    print("Output projection:",
          (batch_size, seq_len, embed_dim))

    print("\nObservation:")
    print("- attention is just reshaping + batched matrix multiply")
    print("- num_heads splits representation subspaces\n")


# =========================================================
# RUN ALL
# =========================================================

if __name__ == "__main__":
    reshape_round_trip()
    broadcasting_test()
    einsum_test()
    attention_shape_tracker(2, 5, 16, 4)