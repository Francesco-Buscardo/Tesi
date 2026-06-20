import numpy as np
from sklearn.cluster import AgglomerativeClustering


def hamming_distance(a, b):
    bits = np.zeros(len(a), dtype=int)
    counter = 0

    for i in range(len(a)):
        if a[i] != b[i]:
            bits[i] = 1
            counter += 1 
    
    return counter, bits

def build_pairwise_hamming_matrix(Z):
    n = len(Z)
    D = np.zeros((n, n), dtype=int)

    for i in range(n):
        for j in range(n):
            d, _ = hamming_distance(Z[i], Z[j])
            D[i][j] = d

    return D

def compute_threshold(n, alpha=0.40):
    return max(1, int(alpha * n))

def cluster_qals_solutions(D, n):
    th = compute_threshold(n=n)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=th
    )

    labels = model.fit_predict(D)

    cluster_sizes = {}

    for l in labels:
        if l not in cluster_sizes:
            cluster_sizes[l] = 0

        cluster_sizes[l] += 1

    return labels, cluster_sizes

def find_cluster_medoid(cluster):
    medoid  = None
    avg_min = float("inf")

    for i, z in enumerate(cluster):
        distances = []

        for j, zi in enumerate(cluster):
            if i != j:
                d, _ = hamming_distance(z, zi)
                distances.append(d)

        avg_d = sum(distances) / len(distances) if distances else 0

        if avg_d < avg_min:
            avg_min = avg_d
            medoid  = z
            
    return medoid