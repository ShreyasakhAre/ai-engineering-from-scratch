import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import KMeans
from collections import deque
import heapq

# ==========================================================
# QUESTION 1
# PAGERANK FROM SCRATCH
# ==========================================================

def pagerank(graph,
             d=0.85,
             tol=1e-6,
             max_iter=100):

    nodes = list(graph.keys())
    n = len(nodes)

    node_to_idx = {
        node: idx
        for idx, node in enumerate(nodes)
    }

    scores = np.ones(n) / n

    for iteration in range(max_iter):

        new_scores = np.ones(n) * ((1 - d) / n)

        for u in nodes:

            u_idx = node_to_idx[u]
            out_links = graph[u]

            if len(out_links) == 0:

                for v in nodes:
                    new_scores[node_to_idx[v]] += (
                        d * scores[u_idx] / n
                    )

                continue

            contribution = (
                d * scores[u_idx]
                / len(out_links)
            )

            for v in out_links:
                v_idx = node_to_idx[v]
                new_scores[v_idx] += contribution

        change = np.sum(
            np.abs(new_scores - scores)
        )

        scores = new_scores

        if change < tol:
            print(
                f"PageRank converged after {iteration+1} iterations"
            )
            break

    return {
        node: scores[node_to_idx[node]]
        for node in nodes
    }


print("\n" + "=" * 60)
print("QUESTION 1: PAGERANK")
print("=" * 60)

web_graph = {
    "A": ["B", "C"],
    "B": ["C"],
    "C": ["A"],
    "D": ["C"],
    "E": ["C", "D"]
}

ranks = pagerank(web_graph)

print("\nFinal PageRank Scores")

for node, score in sorted(
        ranks.items(),
        key=lambda x: x[1],
        reverse=True):

    print(
        f"{node}: {score:.6f}"
    )


# ==========================================================
# QUESTION 2
# SPECTRAL CLUSTERING
# ==========================================================

print("\n" + "=" * 60)
print("QUESTION 2: SPECTRAL CLUSTERING")
print("=" * 60)

n1 = 6
n2 = 6

N = n1 + n2

A = np.zeros((N, N))

# Clique 1
for i in range(n1):
    for j in range(n1):
        if i != j:
            A[i, j] = 1

# Clique 2
for i in range(n1, N):
    for j in range(n1, N):
        if i != j:
            A[i, j] = 1

# Single bridge edge
A[5, 6] = 1
A[6, 5] = 1

D = np.diag(A.sum(axis=1))
L = D - A

eigvals, eigvecs = np.linalg.eigh(L)

fiedler_vector = eigvecs[:, 1]

labels = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
).fit_predict(
    fiedler_vector.reshape(-1, 1)
)

print("Cluster labels:")
print(labels)

plt.figure(figsize=(8, 4))

colors = [
    "red" if x == 0 else "blue"
    for x in labels
]

plt.scatter(
    range(N),
    np.zeros(N),
    c=colors,
    s=200
)

plt.title(
    "Spectral Clustering Result"
)

plt.show()

print(
    "\nTry adding more bridge edges.\n"
    "Eventually the clusters become harder to separate."
)


# ==========================================================
# QUESTION 3
# DIJKSTRA VS BFS
# ==========================================================

print("\n" + "=" * 60)
print("QUESTION 3: DIJKSTRA VS BFS")
print("=" * 60)

weighted_graph = {
    0: [(1, 4), (2, 1)],
    1: [(3, 1)],
    2: [(1, 2), (3, 5)],
    3: []
}


def dijkstra(graph, start):

    dist = {
        node: float("inf")
        for node in graph
    }

    dist[start] = 0

    pq = [(0, start)]

    while pq:

        cur_dist, node = heapq.heappop(pq)

        if cur_dist > dist[node]:
            continue

        for nbr, w in graph[node]:

            nd = cur_dist + w

            if nd < dist[nbr]:

                dist[nbr] = nd

                heapq.heappush(
                    pq,
                    (nd, nbr)
                )

    return dist


dijkstra_result = dijkstra(
    weighted_graph,
    0
)

print("Dijkstra Distances")
print(dijkstra_result)

# ----------------------------
# BFS
# ----------------------------

unweighted_graph = {
    0: [1, 2],
    1: [3],
    2: [1, 3],
    3: []
}


def bfs(graph, start):

    dist = {
        node: -1
        for node in graph
    }

    dist[start] = 0

    q = deque([start])

    while q:

        node = q.popleft()

        for nbr in graph[node]:

            if dist[nbr] == -1:

                dist[nbr] = (
                    dist[node] + 1
                )

                q.append(nbr)

    return dist


bfs_result = bfs(
    unweighted_graph,
    0
)

print("\nBFS Distances")
print(bfs_result)

print(
    "\nBFS assumes every edge weight = 1.\n"
    "Dijkstra handles arbitrary positive weights."
)


# ==========================================================
# QUESTION 4
# TWO-LAYER MESSAGE PASSING NETWORK
# ==========================================================

print("\n" + "=" * 60)
print("QUESTION 4: MESSAGE PASSING")
print("=" * 60)

# Chain:
# 0 -- 1 -- 2 -- 3

A = np.array([
    [0, 1, 0, 0],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=float)

X = np.array([
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
], dtype=float)

W1 = np.array([
    [1, 2],
    [0, 1]
])

W2 = np.array([
    [2, 0],
    [1, 1]
])

# Layer 1
H1 = A @ X @ W1

# Layer 2
H2 = A @ H1 @ W2

print("Initial Node Features")
print(X)

print("\nAfter Layer 1")
print(H1)

print("\nAfter Layer 2")
print(H2)

print(
    "\nAfter 2 layers each node has\n"
    "information from nodes up to\n"
    "2 hops away."
)


# ==========================================================
# QUESTION 5
# KARATE CLUB GRAPH
# ==========================================================

print("\n" + "=" * 60)
print("QUESTION 5: KARATE CLUB ANALYSIS")
print("=" * 60)

G = nx.karate_club_graph()

print("Nodes:", G.number_of_nodes())
print("Edges:", G.number_of_edges())

# --------------------------------
# Degree Distribution
# --------------------------------

degrees = [
    deg
    for _, deg in G.degree()
]

print(
    "\nAverage Degree:",
    np.mean(degrees)
)

plt.figure(figsize=(6, 4))

plt.hist(
    degrees,
    bins=10
)

plt.title(
    "Karate Club Degree Distribution"
)

plt.xlabel("Degree")
plt.ylabel("Count")

plt.show()

# --------------------------------
# Laplacian
# --------------------------------

L = nx.laplacian_matrix(
    G
).toarray()

eigvals, eigvecs = np.linalg.eigh(L)

print("\nSmallest Eigenvalues")

print(
    np.round(
        eigvals[:10],
        6
    )
)

# --------------------------------
# Spectral Clustering
# --------------------------------

fiedler_vector = eigvecs[:, 1]

predicted = KMeans(
    n_clusters=2,
    random_state=42,
    n_init=10
).fit_predict(
    fiedler_vector.reshape(-1, 1)
)

truth = []

for node in G.nodes():

    if G.nodes[node]["club"] == "Mr. Hi":
        truth.append(0)
    else:
        truth.append(1)

truth = np.array(truth)

accuracy = max(
    np.mean(predicted == truth),
    np.mean(1 - predicted == truth)
)

print(
    "\nSpectral Clustering Accuracy:"
)

print(
    f"{accuracy * 100:.2f}%"
)

plt.figure(figsize=(8, 6))

pos = nx.spring_layout(
    G,
    seed=42
)

nx.draw_networkx(
    G,
    pos,
    node_color=predicted,
    cmap="coolwarm",
    with_labels=True
)

plt.title(
    "Karate Club Spectral Clustering"
)

plt.show()

print(
    "\nThe Fiedler vector (2nd smallest Laplacian eigenvector)\n"
    "captures the natural split in the network."
)

print("\nAll Questions Completed Successfully!")