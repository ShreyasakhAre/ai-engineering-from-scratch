import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================================
# DFT FROM SCRATCH
# ==========================================================

def dft(x):
    x = np.asarray(x, dtype=complex)
    N = len(x)

    X = np.zeros(N, dtype=complex)

    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j*np.pi*k*n/N)

    return X


# ==========================================================
# EXERCISE 1
# PURE TONE IDENTIFICATION
# ==========================================================

print("\n" + "="*60)
print("EXERCISE 1")
print("="*60)

fs = 128
N = 128

freq_true = np.random.randint(1, 51)

t = np.arange(N) / fs

signal = np.sin(2*np.pi*freq_true*t)

X = dft(signal)

magnitude = np.abs(X[:N//2])

freq_est = np.argmax(magnitude)

print("True frequency :", freq_true)
print("Detected        :", freq_est)

# ------------------------------------
# Add noise
# ------------------------------------

noise = np.random.normal(0, 0.5, N)

signal_noisy = signal + noise

X_noise = dft(signal_noisy)

mag_noise = np.abs(X_noise[:N//2])

freq_est_noise = np.argmax(mag_noise)

print("\nWith Gaussian noise σ=0.5")

print("True frequency :", freq_true)
print("Detected        :", freq_est_noise)

plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.plot(magnitude)
plt.title("Clean Spectrum")

plt.subplot(1,2,2)
plt.plot(mag_noise)
plt.title("Noisy Spectrum")

plt.tight_layout()
plt.show()

print("""
Noise raises the spectral floor.
The frequency peak remains visible
but becomes less sharp.
""")


# ==========================================================
# EXERCISE 2
# FFT VS DFT
# ==========================================================

print("\n" + "="*60)
print("EXERCISE 2")
print("="*60)

x = np.random.randn(64)

X_dft = dft(x)

X_fft = np.fft.fft(x)

max_diff = np.max(np.abs(X_dft - X_fft))

print("Maximum coefficient difference:")
print(max_diff)

sizes = [256,512,1024,2048]

dft_times = []
fft_times = []

for N in sizes:

    x = np.random.randn(N)

    start = time.perf_counter()
    dft(x)
    dft_t = time.perf_counter() - start

    start = time.perf_counter()
    np.fft.fft(x)
    fft_t = time.perf_counter() - start

    dft_times.append(dft_t)
    fft_times.append(fft_t)

    print(
        f"N={N:4d} "
        f"DFT={dft_t:.4f}s "
        f"FFT={fft_t:.6f}s "
        f"Ratio={dft_t/fft_t:.1f}"
    )

ratios = np.array(dft_times)/np.array(fft_times)

plt.figure(figsize=(8,4))
plt.plot(sizes, ratios, marker="o")
plt.title("DFT Time / FFT Time")
plt.xlabel("Signal Length")
plt.ylabel("Speedup")
plt.grid(True)
plt.show()

print("""
DFT complexity: O(N²)
FFT complexity: O(N log N)

The ratio grows rapidly with N.
""")


# ==========================================================
# EXERCISE 3
# CONVOLUTION THEOREM
# ==========================================================

print("\n" + "="*60)
print("EXERCISE 3")
print("="*60)

x = np.array([1,2,3,4,0,0,0,0], dtype=float)
h = np.array([1,1,1,0,0,0,0,0], dtype=float)

N = len(x)

# ----------------------------------
# Circular convolution
# ----------------------------------

circ = np.zeros(N)

for n in range(N):

    s = 0

    for k in range(N):
        s += x[k] * h[(n-k)%N]

    circ[n] = s

# FFT version

circ_fft = np.real(
    np.fft.ifft(
        np.fft.fft(x)
        *
        np.fft.fft(h)
    )
)

print("Circular convolution matches:",
      np.allclose(circ,circ_fft))

print(circ)

# ----------------------------------
# Linear convolution
# ----------------------------------

linear_direct = np.convolve(x,h)

padN = len(x)+len(h)-1

linear_fft = np.real(
    np.fft.ifft(
        np.fft.fft(x,padN)
        *
        np.fft.fft(h,padN)
    )
)

print("Linear convolution matches:",
      np.allclose(linear_direct,
                  linear_fft))

print(linear_direct)


# ==========================================================
# EXERCISE 4
# WINDOWING EFFECTS
# ==========================================================

print("\n" + "="*60)
print("EXERCISE 4")
print("="*60)

fs = 128

t = np.arange(fs)/fs

signal = (
    np.sin(2*np.pi*10*t)
    +
    np.sin(2*np.pi*12*t)
)

windows = {
    "Rectangular": np.ones(fs),
    "Hann": np.hanning(fs),
    "Hamming": np.hamming(fs)
}

plt.figure(figsize=(12,4))

for i,(name,w) in enumerate(windows.items()):

    X = np.fft.fft(signal*w)

    power = np.abs(X[:fs//2])**2

    plt.subplot(1,3,i+1)

    plt.plot(power)

    plt.title(name)

plt.tight_layout()
plt.show()

print("""
Rectangular:
- narrow peak
- large side lobes

Hann:
- wider peak
- lower leakage

Hamming:
- often best compromise

The two frequencies become easier
to distinguish because leakage
is reduced.
""")


# ==========================================================
# EXERCISE 5
# POSITIONAL ENCODING ANALYSIS
# ==========================================================

print("\n" + "="*60)
print("EXERCISE 5")
print("="*60)

d_model = 128
max_pos = 512

PE = np.zeros((max_pos,d_model))

for pos in range(max_pos):

    for i in range(0,d_model,2):

        div = 10000 ** (i/d_model)

        PE[pos,i] = np.sin(pos/div)

        PE[pos,i+1] = np.cos(pos/div)

# -------------------------------------
# Dot products
# -------------------------------------

pairs = [
    (10,20),
    (20,30),
    (100,110),
    (200,210)
]

print("Same distance comparisons:\n")

for p1,p2 in pairs:

    dp = PE[p1] @ PE[p2]

    print(
        f"{p1}-{p2}"
        f" distance={abs(p1-p2)} "
        f" dot={dp:.4f}"
    )

# -------------------------------------
# Dot product vs distance
# -------------------------------------

distances = np.arange(0,200)

dots = []

base = 0

for d in distances:

    dots.append(
        PE[base] @ PE[d]
    )

plt.figure(figsize=(8,4))

plt.plot(distances,dots)

plt.xlabel("Distance")

plt.ylabel("Dot Product")

plt.title(
    "Positional Encoding Similarity"
)

plt.grid(True)
plt.show()

print("""
The dot product depends primarily
on relative distance.

Positions close together have
high similarity.

As distance increases, similarity
oscillates and generally decreases.

This is why sinusoidal encodings
naturally encode relative position.
""")