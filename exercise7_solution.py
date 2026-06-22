import numpy as np
from collections import defaultdict
import math

# =========================================================
# 1. BAYESIAN UPDATE: TWO INDEPENDENT TESTS
# =========================================================

def bayes_update(prior, sensitivity, specificity):
    """
    Returns posterior P(sick | test+)
    """
    p_sick = prior
    p_healthy = 1 - prior

    p_pos_given_sick = sensitivity
    p_pos_given_healthy = 1 - specificity

    numerator = p_pos_given_sick * p_sick
    denominator = numerator + p_pos_given_healthy * p_healthy

    return numerator / denominator


def two_tests():
    prior = 1 / 10000
    sensitivity = 0.99
    specificity = 0.99

    # First test
    post1 = bayes_update(prior, sensitivity, specificity)

    # Second test uses posterior as prior
    post2 = bayes_update(post1, sensitivity, specificity)

    print("=== Two Independent Tests ===")
    print("Prior:", prior)
    print("After 1st test:", post1)
    print("After 2nd test:", post2)
    print()


# =========================================================
# 2. NAIVE BAYES + SMOOTHING EFFECT
# =========================================================

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.word_counts = {"spam": defaultdict(int), "ham": defaultdict(int)}
        self.class_counts = {"spam": 0, "ham": 0}
        self.vocab = set()

    def fit(self, data):
        for label, words in data:
            self.class_counts[label] += 1
            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)

    def word_prob(self, word, label):
        alpha = self.alpha
        total = sum(self.word_counts[label].values())
        return (self.word_counts[label][word] + alpha) / (total + alpha * len(self.vocab))

    def predict_log_score(self, words):
        scores = {}
        for c in ["spam", "ham"]:
            log_prob = math.log(self.class_counts[c] / sum(self.class_counts.values()))
            for w in words:
                log_prob += math.log(self.word_prob(w, c))
            scores[c] = log_prob
        return scores


def smoothing_experiment():
    data = [
        ("spam", ["buy", "cheap", "offer"]),
        ("spam", ["cheap", "discount", "buy"]),
        ("ham", ["meeting", "schedule"]),
        ("ham", ["project", "meeting"]),
    ]

    test_word = ["unknown_word"]

    print("=== Smoothing Impact ===")

    for alpha in [0.0, 0.01, 0.1, 1.0, 10.0]:
        model = NaiveBayes(alpha=alpha)
        model.fit(data)

        try:
            scores = model.predict_log_score(test_word)
            print(f"\nAlpha={alpha}")
            print("Spam score:", scores["spam"])
            print("Ham score:", scores["ham"])
        except:
            print(f"\nAlpha={alpha}")
            print("Error (likely division by zero when alpha=0)")

    print("\nObservation:")
    print("- alpha=0 causes zero probability for unseen words")
    print("- higher alpha flattens distribution")
    print("- very large alpha makes model almost uniform\n")


# =========================================================
# 3. ADD FEATURE: MESSAGE LENGTH
# =========================================================

class NaiveBayesWithLength(NaiveBayes):
    def __init__(self, alpha=1.0):
        super().__init__(alpha)
        self.length_counts = {"spam": {"short": 0, "long": 0},
                              "ham": {"short": 0, "long": 0}}

    def fit(self, data):
        for label, words in data:
            super().fit([(label, words)])

            length = "short" if len(words) < 3 else "long"
            self.length_counts[label][length] += 1

    def length_prob(self, length, label):
        total = sum(self.length_counts[label].values())
        return self.length_counts[label][length] / total

    def predict(self, words):
        length = "short" if len(words) < 3 else "long"

        scores = {}
        for c in ["spam", "ham"]:
            score = math.log(self.class_counts[c] / sum(self.class_counts.values()))
            score += math.log(self.length_prob(length, c))

            for w in words:
                score += math.log(self.word_prob(w, c))

            scores[c] = score

        return scores


def test_length_feature():
    data = [
        ("spam", ["buy", "cheap", "offer"]),
        ("spam", ["discount"]),
        ("ham", ["meeting", "schedule"]),
        ("ham", ["project", "discussion"]),
    ]

    model = NaiveBayesWithLength(alpha=1.0)
    model.fit(data)

    test = ["buy", "cheap"]
    scores = model.predict(test)

    print("=== Length Feature Model ===")
    print("Scores:", scores)
    print()


# =========================================================
# 4. MAP VS MLE (BETA PRIOR)
# =========================================================

def beta_map_mle():
    heads = 7
    flips = 10

    # MLE
    mle = heads / flips

    # Beta prior Beta(2,2)
    a, b = 2, 2

    # MAP estimate
    map_est = (heads + a - 1) / (flips + a + b - 2)

    print("=== MAP vs MLE ===")
    print("MLE:", mle)
    print("MAP (Beta(2,2)):", map_est)
    print()


# =========================================================
# RUN ALL
# =========================================================

if __name__ == "__main__":
    two_tests()
    smoothing_experiment()
    test_length_feature()
    beta_map_mle()