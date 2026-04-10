import numpy as np


def build_knapsack(ksp_file):
    n_items = 0
    C = 0
    items = [] # (weight, profit)

    with open(ksp_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

        n_items = int(lines[0])
        C = int(lines[1])

        for i in range(2, 2 + n_items):
            weight, value = map(int, lines[i].split())
            items.append((weight, value))

    return n_items, C, items

def compute_penalty(C, items):
    # Compute the penalty for the knapsack problem
    # A = (sum of profits) / Capacity
    
    A = sum(p for _, p in items)
    
    return  2 * A

def generate_QUBO_knapsack(n_items, C, items):
    # Generate the matrix -Q for the knapsack problem
    
    Q = np.zeros((n_items, n_items))
    A = compute_penalty(C, items)

    for i in range(n_items):
            w_i, p_i = items[i]
            Q[i][i] = (-p_i + A * (w_i ** 2) - 2 * A * C * w_i)

            for j in range(i + 1, n_items):
                w_j, p_j = items[j]
                Q[i][j] = (2 * A * w_i * w_j)
                Q[j][i] = Q[i][j]

    return Q
    
def ksp_solve(n_items, C, items):
    weights = [w for w, _ in items]
    profits = [p for _, p in items]
    dp = [[0] * (C + 1) for _ in range(n_items + 1)]

    for i in range(1, n_items + 1):
        w = weights[i - 1]
        p = profits[i - 1]
        for c in range(C + 1):
            dp[i][c] = dp[i - 1][c]
            if w <= c:
                dp[i][c] = max(dp[i][c], dp[i - 1][c - w] + p)

    x = [0] * n_items
    c = C
    for i in range(n_items, 0, -1):
        if dp[i][c] != dp[i - 1][c]:
            x[i - 1] = 1
            c -= weights[i - 1]

    best_profit = dp[n_items][C]
    total_weight = sum(w for w, take in zip(weights, x) if take)

    return best_profit, total_weight