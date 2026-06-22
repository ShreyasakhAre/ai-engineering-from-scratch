import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# 1. CAUCHY DISTRIBUTION (INVERSE CDF SAMPLING)
# ============================================================

def sample_cauchy(n):
    u = np.random.rand(n)
    return np.tan(np.pi * (u - 0.5))

def cauchy_pdf(x):
    return 1 / (np.pi * (1 + x**2))

def cauchy_demo():
    samples = sample_cauchy(10000)

    x = np.linspace(-20, 20, 1000)

    plt.figure(figsize=(8,4))
    plt.hist(samples,
             bins=200,
             density=True,
             range=(-20,20),
             alpha=0.6,
             label="Samples")

    plt.plot(x, cauchy_pdf(x),
             linewidth=2,
             label="True PDF")

    plt.title("Cauchy Distribution")
    plt.legend()
    plt.show()

    print("Min sample:", np.min(samples))
    print("Max sample:", np.max(samples))
    print("Notice extreme values due to heavy tails.")


# ============================================================
# 2. REJECTION SAMPLING (BETA(2,5))
# ============================================================

def beta25_pdf(x):
    return 30 * x * (1 - x)**4

def rejection_beta25(n):
    samples = []

    # max of Beta(2,5)
    x_mode = 1/5
    M = beta25_pdf(x_mode)

    trials = 0

    while len(samples) < n:
        trials += 1

        x = np.random.rand()
        y = np.random.rand() * M

        if y <= beta25_pdf(x):
            samples.append(x)

    return np.array(samples), n/trials

def beta_demo():
    samples, acc_rate = rejection_beta25(10000)

    x = np.linspace(0,1,500)

    plt.figure(figsize=(8,4))
    plt.hist(samples,
             bins=50,
             density=True,
             alpha=0.6)

    plt.plot(x,
             beta25_pdf(x),
             linewidth=2)

    plt.title("Beta(2,5) via Rejection Sampling")
    plt.show()

    print("Empirical acceptance rate:", acc_rate)

    x_mode = 1/5
    M = beta25_pdf(x_mode)

    theoretical = 1/M
    print("Theoretical acceptance rate:", theoretical)


# ============================================================
# 3. MONTE CARLO INTEGRATION
# ============================================================

def monte_carlo_integral(N):
    x = np.random.uniform(0, np.pi, N)

    estimate = np.pi * np.mean(np.sin(x))

    return estimate

def mc_demo():

    true_value = 2

    print("\nMonte Carlo Integral")

    for N in [1000,10000,100000]:

        est = monte_carlo_integral(N)

        err = abs(est - true_value)

        print(
            f"N={N:6d}  estimate={est:.6f} "
            f"error={err:.6f} "
            f"error*sqrt(N)={err*np.sqrt(N):.4f}"
        )

    print("\nerror*sqrt(N) roughly constant => O(1/sqrt(N))")


# ============================================================
# 4. METROPOLIS HASTINGS
# ============================================================

def log_target(x,y):
    return -(
        x*x*y*y +
        x*x +
        y*y -
        8*x -
        8*y
    ) / 2

def metropolis_hastings(
        n=20000,
        proposal_std=1.0):

    x = 0
    y = 0

    samples = []

    for _ in range(n):

        xp = np.random.normal(x, proposal_std)
        yp = np.random.normal(y, proposal_std)

        log_ratio = (
            log_target(xp,yp)
            - log_target(x,y)
        )

        if np.log(np.random.rand()) < log_ratio:
            x,y = xp,yp

        samples.append([x,y])

    return np.array(samples)

def mh_demo():

    samples = metropolis_hastings(
        proposal_std=1.0
    )

    plt.figure(figsize=(8,6))

    plt.scatter(
        samples[:,0],
        samples[:,1],
        s=1,
        alpha=0.4
    )

    plt.title("MH Samples")
    plt.show()

    plt.figure(figsize=(8,6))

    plt.plot(
        samples[:500,0],
        samples[:500,1]
    )

    plt.title("Chain Trajectory")
    plt.show()


# ============================================================
# 5. TEXT GENERATION DEMO
# ============================================================

VOCAB = [
    "the","cat","dog","runs","sleeps",
    "fast","slow","on","under","grass"
]

LOGITS = np.array([
    4.0,3.2,3.0,2.5,2.4,
    1.5,1.4,1.2,1.1,0.9
])

def softmax(logits):
    e = np.exp(logits - np.max(logits))
    return e / np.sum(e)

def sample_temperature(logits, T):

    probs = softmax(logits / T)

    return np.random.choice(
        len(probs),
        p=probs
    )

def sample_top_k(logits, k):

    idx = np.argsort(logits)[-k:]

    probs = softmax(logits[idx])

    return np.random.choice(idx,p=probs)

def sample_top_p(logits,p=0.9):

    probs = softmax(logits)

    order = np.argsort(probs)[::-1]

    sorted_probs = probs[order]

    cumsum = np.cumsum(sorted_probs)

    cutoff = np.searchsorted(cumsum,p)

    keep = order[:cutoff+1]

    keep_probs = probs[keep]
    keep_probs /= keep_probs.sum()

    return np.random.choice(
        keep,
        p=keep_probs
    )

def generate(strategy):

    tokens = []

    for _ in range(20):

        if strategy=="greedy":
            idx = np.argmax(LOGITS)

        elif strategy=="temp":
            idx = sample_temperature(LOGITS,0.7)

        elif strategy=="topk":
            idx = sample_top_k(LOGITS,3)

        elif strategy=="topp":
            idx = sample_top_p(LOGITS,0.9)

        tokens.append(VOCAB[idx])

    return " ".join(tokens)

def text_demo():

    strategies = [
        "greedy",
        "temp",
        "topk",
        "topp"
    ]

    for strat in strategies:

        print("\n====================")
        print(strat.upper())
        print("====================")

        for i in range(5):
            print(generate(strat))


# ============================================================
# RUN EVERYTHING
# ============================================================

if __name__ == "__main__":

    cauchy_demo()

    beta_demo()

    mc_demo()

    mh_demo()

    text_demo()