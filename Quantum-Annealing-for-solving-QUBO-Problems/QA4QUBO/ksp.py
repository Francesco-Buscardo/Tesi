import numpy as np

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

    # ? OPT 1 - mediamente meglio
    A = sum(p for _, p in items)
    return  A / 3

    # ? OPT 2
    # A = sum(p for _, p in items)
    # return A / C

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

def ksp_dp(W, wt, val, n):
    if n == 0 or W == 0:
        return 0, 0 
    
    if wt[n - 1] > W:
        return ksp_dp(W, wt, val, n - 1)
    else:
        profit_take, weight_take = ksp_dp(W - wt[n - 1], wt, val, n - 1)
        profit_take += val[n - 1]
        weight_take += wt[n - 1]

        profit_leave, weight_leave = ksp_dp(W, wt, val, n - 1)

        if profit_take > profit_leave:
            return profit_take, weight_take
        else:
            return profit_leave, weight_leave

def bf(n, index, items_b, sols):
    if index == n:
        sols.append(np.copy(items_b))
        return

    for i in (0, 1):
        items_b[index] = i
        bf(n, index + 1, items_b, sols)
    
def ksp_bf(n_items, C, items):

    profits = [p for _, p in items]
    weights = [w for w, _ in items]

    best_profit_so_far = 0
    items_so_far       = np.zeros(n_items, dtype = bool)
    items_b            = np.zeros(n_items, dtype = bool)
    sols               = []

    bf(n_items, 0, items_b, sols)

    for i in sols:

        w_i = sum(weights[j] for j in range(n_items) if i[j] == 1)
        if w_i <= C:
            sum_i = sum(profits[j] for j in range(n_items) if i[j] == 1)
            if sum_i > best_profit_so_far:
                best_profit_so_far = sum_i
                items_so_far = np.copy(i)

    return best_profit_so_far, items_so_far
