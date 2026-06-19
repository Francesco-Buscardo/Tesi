import numpy as np 
import ksp_config as ksp_config

# items[n_items][2]: 
#    | w |  p 
# ---|---|----
# 0  |   |    
# ---|---|----
# 1  |   |    
# ---|---|----
# 2  |   |    
# ---|---|----
# .  |   |    
# .  |   |    
# .  |   |    
# ---|---|----
# n-1|   |    
def build_knapsack(ksp_file):
    n_items = 0
    C       = 0
    items   = []

    with open(ksp_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

        n_items = int(lines[0])
        C       = int(lines[1])

        for i in range(2, 2 + n_items):
            w, p = map(int, lines[i].split())
            items.append((w, p))

    return n_items, C, items

def compute_penalty(C, items):
    # Compute the penalty for the knapsack problem
    # A = (sum of profits) / 3

    # Per quanto riguarda il valore di λ, se vuoi garantire che il 
    # vincolo sia “hard”, deve essere sufficientemente grande da 
    # penalizzare qualsiasi violazione. In pratica, può essere definito:
    #               (sum of profits) / min(|C - Ci|)
    # come il rapporto tra la somma dei profitti e la distanza minima 
    # tra C e il valore raggiungibile più vicino senza eguagliarlo.

    # Ad esempio, se C=101, si considerano tutte le combinazioni possibili 
    # dei pesi S1,S2,S3,… e si individua il valore Si più vicino a 101. 
    # Nel nostro caso sembra essere 102, quindi la distanza è ∣101−102∣=1. 

    # ll punto critico però è che con questo valore sei sicuro di ottenere 
    # un vincolo “hard”, ma non hai la garanzia che riducendo leggermente λ 
    # il vincolo diventi “soft” in modo controllato. 
    # Per questo motivo ti direi di provare con entrambi i valori 
    # (sum pesi) / 3 e quello che già usavi.

    if ksp_config.LAMBDA_VALUE == "lambda_div_3":
        A = sum(p for _, p in items) / 3
        return A
    elif ksp_config.LAMBDA_VALUE == "lambda_650_dot_C":
        A = sum(p for _, p in items) / (650 * C)
        return A 
    elif ksp_config.LAMBDA_VALUE == "lambda_6500_dot_C":
        A = sum(p for _, p in items) / (6500 * C)
        return A
    elif ksp_config.LAMBDA_VALUE == "lambda_div_C":
        A = sum(p for _, p in items) / C
        return A
    else:
        return 0

def generate_QUBO_knapsack(n_items, C, items):
    # Generate the matrix -Q for the knapsack problem
    
    Q = np.zeros((n_items, n_items))
    A = compute_penalty(C, items)

    for i in range(n_items):
            w_i, p_i = items[i]
            Q[i][i] = (-p_i + A * (w_i ** 2) - 2 * A * C * w_i)

            for j in range(i + 1, n_items):
                w_j, p_j = items[j]
                Q[i][j] = (A * w_i * w_j)
                Q[j][i] = Q[i][j]

    return Q

def ksp_dp(n, p, w, C):
    DP = np.zeros((n+1, C+1), dtype=int)

    for i in range(1, n + 1):
        for j in range(C + 1):
            if w[i-1] <= j:
                DP[i][j] = max(
                    DP[i-1][j],
                    DP[i-1][j-w[i -1]] + p[i-1]
                )
            else:
                DP[i][j] = DP[i-1][j]

    sol = np.zeros(n, dtype=int)
    i = n
    j = C

    while i > 0 and j >= 0:
        if w[i-1] <= j:
            without_item = DP[i-1][j]
            with_item = DP[i-1][j-w[i-1]] + p[i-1]

            if with_item > without_item:
                sol[i-1] = 1
                j -= w[i-1]
        i -= 1

    best_weight = sum(w[i] * sol[i] for i in range(n))

    return DP[n][C], best_weight, sol