import math
import random

# ============================================================
# DESCRIPTIVE STATISTICS
# ============================================================

def mean(x):
    return sum(x) / len(x)


def median(x):
    x = sorted(x)
    n = len(x)

    if n % 2 == 1:
        return x[n // 2]

    return (x[n // 2 - 1] + x[n // 2]) / 2


def mode(x):
    counts = {}

    for v in x:
        counts[v] = counts.get(v, 0) + 1

    max_count = max(counts.values())

    return [k for k, v in counts.items() if v == max_count]


def variance(x, sample=True):
    m = mean(x)

    ss = 0

    for v in x:
        ss += (v - m) ** 2

    denom = len(x) - 1 if sample else len(x)

    return ss / denom


def std(x, sample=True):
    return math.sqrt(variance(x, sample))


def percentile(x, p):
    x = sorted(x)

    idx = p * (len(x) - 1)

    lower = int(idx)
    upper = min(lower + 1, len(x) - 1)

    frac = idx - lower

    return x[lower] * (1 - frac) + x[upper] * frac


def iqr(x):
    return percentile(x, 0.75) - percentile(x, 0.25)


# ============================================================
# COVARIANCE + CORRELATION
# ============================================================

def covariance(x, y):
    mx = mean(x)
    my = mean(y)

    total = 0

    for a, b in zip(x, y):
        total += (a - mx) * (b - my)

    return total / (len(x) - 1)


def pearson(x, y):
    return covariance(x, y) / (std(x) * std(y))


def rank_data(x):
    sorted_idx = sorted(range(len(x)), key=lambda i: x[i])

    ranks = [0] * len(x)

    for rank, idx in enumerate(sorted_idx):
        ranks[idx] = rank + 1

    return ranks


def spearman(x, y):
    return pearson(rank_data(x), rank_data(y))


def covariance_matrix(data):
    k = len(data)

    matrix = []

    for i in range(k):
        row = []

        for j in range(k):
            row.append(covariance(data[i], data[j]))

        matrix.append(row)

    return matrix


# ============================================================
# T TESTS
# ============================================================

def one_sample_ttest(sample, mu0):

    m = mean(sample)
    s = std(sample)

    return (m - mu0) / (s / math.sqrt(len(sample)))


def two_sample_ttest(x, y):

    mx = mean(x)
    my = mean(y)

    vx = variance(x)
    vy = variance(y)

    nx = len(x)
    ny = len(y)

    pooled = ((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2)

    se = math.sqrt(pooled * (1/nx + 1/ny))

    return (mx - my) / se


# ============================================================
# CHI SQUARED TEST
# ============================================================

def chi_squared_test(observed):

    rows = len(observed)
    cols = len(observed[0])

    row_totals = [sum(r) for r in observed]

    col_totals = []

    for c in range(cols):
        col_totals.append(sum(observed[r][c] for r in range(rows)))

    grand_total = sum(row_totals)

    chi2 = 0

    for r in range(rows):
        for c in range(cols):

            expected = (
                row_totals[r] *
                col_totals[c] /
                grand_total
            )

            chi2 += (
                (observed[r][c] - expected) ** 2
            ) / expected

    return chi2


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================

def bootstrap_ci(data,
                 statistic=mean,
                 num_samples=5000,
                 confidence=0.95):

    estimates = []

    n = len(data)

    for _ in range(num_samples):

        sample = [
            random.choice(data)
            for _ in range(n)
        ]

        estimates.append(statistic(sample))

    estimates.sort()

    alpha = 1 - confidence

    lower_idx = int(alpha/2 * num_samples)
    upper_idx = int((1-alpha/2) * num_samples)

    return (
        estimates[lower_idx],
        estimates[upper_idx]
    )


# ============================================================
# A/B TEST SIMULATOR
# ============================================================

def generate_group(n, mean_value, sigma):

    return [
        random.gauss(mean_value, sigma)
        for _ in range(n)
    ]


def run_ab_test(
        n=100,
        control_mean=100,
        treatment_mean=100,
        sigma=10):

    control = generate_group(
        n,
        control_mean,
        sigma
    )

    treatment = generate_group(
        n,
        treatment_mean,
        sigma
    )

    t = two_sample_ttest(
        treatment,
        control
    )

    return t


def type1_error_simulation(
        experiments=1000):

    false_positive = 0

    for _ in range(experiments):

        t = run_ab_test(
            treatment_mean=100,
            control_mean=100
        )

        if abs(t) > 1.96:
            false_positive += 1

    return false_positive / experiments


def type2_error_simulation(
        experiments=1000):

    missed = 0

    for _ in range(experiments):

        t = run_ab_test(
            treatment_mean=101,
            control_mean=100
        )

        if abs(t) <= 1.96:
            missed += 1

    return missed / experiments


# ============================================================
# STATISTICAL VS PRACTICAL SIGNIFICANCE
# ============================================================

def significance_demo():

    print("\nSmall Sample")

    a = generate_group(
        50,
        100,
        10
    )

    b = generate_group(
        50,
        100.5,
        10
    )

    print(
        "t-stat:",
        two_sample_ttest(a, b)
    )

    print("\nHuge Sample")

    a = generate_group(
        50000,
        100,
        10
    )

    b = generate_group(
        50000,
        100.5,
        10
    )

    print(
        "t-stat:",
        two_sample_ttest(a, b)
    )

    print(
        "\nSame tiny effect size, "
        "but massive n makes it "
        "look extremely significant."
    )


if __name__ == "__main__":

    data = [1,2,3,4,5,6,7,8,9,10]

    print("Mean:", mean(data))
    print("Median:", median(data))
    print("Mode:", mode(data))
    print("Std:", std(data))

    print("25th percentile:",
          percentile(data, 0.25))

    print("75th percentile:",
          percentile(data, 0.75))

    print("IQR:", iqr(data))

    x = [1,2,3,4,5]
    y = [2,4,6,8,10]

    print("\nPearson:",
          pearson(x, y))

    print("Spearman:",
          spearman(x, y))

    print("\nCovariance Matrix:")
    print(covariance_matrix([x, y]))

    sample = [10,12,9,11,13,8,10]

    print("\nOne Sample t:",
          one_sample_ttest(sample, 10))

    g1 = [10,11,12,9,10]
    g2 = [15,14,16,15,14]

    print("Two Sample t:",
          two_sample_ttest(g1, g2))

    table = [
        [20,30],
        [30,20]
    ]

    print("\nChi Squared:",
          chi_squared_test(table))

    ci = bootstrap_ci(
        sample,
        statistic=mean
    )

    print("\nBootstrap CI:", ci)

    print(
        "\nType I Error:",
        type1_error_simulation()
    )

    print(
        "Type II Error:",
        type2_error_simulation()
    )

    significance_demo()