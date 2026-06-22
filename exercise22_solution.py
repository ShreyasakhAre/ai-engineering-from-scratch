import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, Counter
import random

# ============================================================
# QUESTION 1
# RANDOM WALK CENTRAL LIMIT THEOREM
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 1: RANDOM WALK")
print("=" * 60)

num_walks = 1000
num_steps = 10000

final_positions = []

for _ in range(num_walks):

    steps = np.random.choice(
        [-1, 1],
        size=num_steps
    )

    final_positions.append(np.sum(steps))

final_positions = np.array(final_positions)

mean = np.mean(final_positions)
std = np.std(final_positions)

print(f"Mean: {mean:.4f}")
print(f"Std : {std:.4f}")
print(f"Expected Std ≈ {np.sqrt(num_steps):.4f}")

plt.figure(figsize=(8,5))

plt.hist(
    final_positions,
    bins=40,
    density=True,
    alpha=0.7
)

x = np.linspace(
    final_positions.min(),
    final_positions.max(),
    500
)

gaussian = (
    1/(std*np.sqrt(2*np.pi))
    *
    np.exp(-(x-mean)**2/(2*std**2))
)

plt.plot(x, gaussian, linewidth=3)

plt.title("Distribution of Final Positions")
plt.show()

print("""
Central Limit Theorem:
Sum of many independent ±1 steps
approaches a Gaussian distribution.
""")


# ============================================================
# QUESTION 2
# MARKOV CHAIN TEXT GENERATOR
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 2: MARKOV TEXT GENERATOR")
print("=" * 60)

corpus = """
the cat sat on the mat
the dog sat on the rug
the cat chased the mouse
the dog chased the cat
the mouse ran away
""".lower()

words = corpus.split()

transitions = defaultdict(Counter)

for i in range(len(words)-1):
    current_word = words[i]
    next_word = words[i+1]

    transitions[current_word][next_word] += 1

print("Transition Matrix\n")

for word, counts in transitions.items():
    print(word, "->", dict(counts))

def generate_sentence(start_word,
                      max_len=15):

    sentence = [start_word]

    current = start_word

    for _ in range(max_len):

        if current not in transitions:
            break

        next_words = list(
            transitions[current].keys()
        )

        weights = list(
            transitions[current].values()
        )

        current = random.choices(
            next_words,
            weights=weights
        )[0]

        sentence.append(current)

    return " ".join(sentence)

print("\nGenerated Sentences\n")

for _ in range(5):
    print(
        generate_sentence("the")
    )


# ============================================================
# QUESTION 3
# SIMULATED ANNEALING
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 3: SIMULATED ANNEALING")
print("=" * 60)

def objective(x):

    return (
        x**2
        + 10*np.sin(5*x)
    )

x = np.random.uniform(-5,5)

best_x = x
best_f = objective(x)

T0 = 10
cooling = 0.995

trajectory = []

for step in range(5000):

    T = T0 * (cooling ** step)

    proposal = (
        x
        + np.random.normal(0,1)
    )

    delta = (
        objective(proposal)
        - objective(x)
    )

    if (
        delta < 0
        or
        np.random.rand()
        <
        np.exp(-delta/T)
    ):
        x = proposal

    if objective(x) < best_f:
        best_f = objective(x)
        best_x = x

    trajectory.append(x)

print("Best x:", best_x)
print("Best value:", best_f)

xs = np.linspace(-5,5,1000)

plt.figure(figsize=(8,5))
plt.plot(xs, objective(xs))
plt.scatter(
    [best_x],
    [best_f],
    s=100
)
plt.title("Simulated Annealing Result")
plt.show()

print("""
High temperature:
accept almost everything

Low temperature:
accept mostly improvements
""")


# ============================================================
# QUESTION 4
# LANGEVIN DYNAMICS
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 4: LANGEVIN DYNAMICS")
print("=" * 60)

def U(x):
    return (x*x - 1)**2

def grad_U(x):
    return 4*x*(x*x - 1)

def langevin(
        T,
        n_steps=30000,
        dt=0.01):

    x = 0.5

    samples = []

    for _ in range(n_steps):

        noise = (
            np.sqrt(2*T*dt)
            *
            np.random.randn()
        )

        x = (
            x
            - grad_U(x)*dt
            + noise
        )

        samples.append(x)

    return np.array(samples)

temperatures = [
    0.05,
    0.2,
    0.5,
    1.0
]

plt.figure(figsize=(12,8))

for i,T in enumerate(temperatures):

    samples = langevin(T)

    plt.subplot(
        2,
        2,
        i+1
    )

    plt.hist(
        samples,
        bins=60,
        density=True
    )

    plt.title(
        f"T={T}"
    )

plt.tight_layout()
plt.show()

print("""
Low temperature:
stuck in one well.

High temperature:
moves between wells.

Critical mixing temperature
typically around T≈0.2–0.5
for this potential.
""")


# ============================================================
# QUESTION 5
# DIFFUSION PROCESS
# ============================================================

print("\n" + "=" * 60)
print("QUESTION 5: DIFFUSION")
print("=" * 60)

N = 512

x = np.linspace(
    0,
    4*np.pi,
    N
)

signal = np.sin(x)

steps = 100

betas = np.linspace(
    0.0001,
    0.02,
    steps
)

current = signal.copy()

saved = []

for t in range(steps):

    noise = np.random.randn(N)

    beta = betas[t]

    current = (
        np.sqrt(1-beta)*current
        +
        np.sqrt(beta)*noise
    )

    if t in [0,25,50,75,99]:
        saved.append(
            current.copy()
        )

plt.figure(figsize=(12,8))

for i,s in enumerate(saved):

    plt.subplot(
        len(saved),
        1,
        i+1
    )

    plt.plot(s)

    plt.title(
        f"Step { [0,25,50,75,99][i] }"
    )

plt.tight_layout()
plt.show()

print(
    "\nSignal progressively becomes noise."
)

# ------------------------------------------------
# Simple Reverse Process
# ------------------------------------------------

print(
    "\nRunning naive denoiser..."
)

noisy_signal = current.copy()

denoised = noisy_signal.copy()

for _ in range(20):

    smoothed = denoised.copy()

    smoothed[1:-1] = (
        denoised[:-2]
        +
        denoised[1:-1]
        +
        denoised[2:]
    ) / 3

    denoised = smoothed

plt.figure(figsize=(10,6))

plt.plot(
    signal,
    label="Original"
)

plt.plot(
    noisy_signal,
    alpha=0.6,
    label="Noisy"
)

plt.plot(
    denoised,
    linewidth=3,
    label="Denoised"
)

plt.legend()

plt.title(
    "Simple Diffusion Reverse Process"
)

plt.show()

print("""
Modern diffusion models learn
the noise prediction network:

εθ(x,t)

instead of using this naive
smoothing denoiser.

The reverse diffusion process
iteratively removes predicted noise
until a clean sample emerges.
""")
