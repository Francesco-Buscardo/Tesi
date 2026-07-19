import numpy as np

import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

from QA4QUBO import solver


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
  
def plot_dendogram(model, **kwargs):

    # ogni pos in counts avrà il numero di soluz originali presenti nel cluster
    counts = np.zeros(model.children_.shape[0])

    # num soluz originali del model
    n_original_sol = len(model.labels_)

    for i, merge in enumerate(model.children_):
        # tiene il conteggio di quante soluz originali sono presenti nel nuovo cluster
        current_count = 0

        for child in merge:
            if child < n_original_sol:
                current_count += 1
            else:
                current_count += counts[child - n_original_sol]

        counts[i] = current_count

    """
    Ogni riga della linkage_matrix deve avere questa forma:

    [cluster_sinistro, cluster_destro, distanza_fusione, numero_elementi_originali]
    """
    linkage_matrix = np.column_stack([
        model.children_,
        model.distances_,
        counts
    ]).astype(float)

    fig, _ = plt.subplots(figsize=(14, 7))

    dendrogram(linkage_matrix, **kwargs)

    fig.tight_layout()

    return fig


def cluster_qals_solutions(D, n, Q, Z, f_best):
    th = compute_threshold(n=n)

    model = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=th,
        compute_full_tree=True,
        compute_distances=True
    )

    labels = model.fit_predict(D)

    fz_array  = [solver.function_f(Q, i) for i in Z]
    min_index = np.argmin(fz_array)
    
    cluster_sizes = {}

    for l in labels:
        if l not in cluster_sizes:
            cluster_sizes[l] = 0

        cluster_sizes[l] += 1
    
    cluster_best_z = labels[min_index]

    dendogram = plot_dendogram(
        model,
        color_threshold=th,
        show_contracted=True
    )

    return labels, cluster_sizes, dendogram, cluster_best_z

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