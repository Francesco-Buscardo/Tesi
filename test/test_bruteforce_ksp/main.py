import numpy as np
from os import listdir
from os.path import isfile, join

class colors:
    BOLD = '\033[1m'
    HEADER = '\033[95m'
    ENDC = '\033[0m'

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
def build_ksp(ksp_file):
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

def log_write(label, value):
    return f"{label}: {value}\n"

def main():

    string = str()
    
    n_items, capacity, items = build_ksp("ksp_1.txt")

    weights, profits = [w for w, _ in items], [p for _, p in items]

    best_profit1, choosen1 = ksp_bf(n_items, capacity, items)
    w1 = sum(weights[j] for j in range(n_items) if choosen1[j] == 1)

    string += colors.BOLD + colors.HEADER + "\nBrute Force Knapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", best_profit1)
    string += log_write("Weight", w1)
    string += log_write("Choosen items", choosen1)

    best_profit2, w2 = ksp_dp(capacity, weights, profits, n_items)

    string += colors.BOLD + colors.HEADER + "\nDynamic Programming Knapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", best_profit2)
    string += log_write("Weight", w2)

    print(string)

if __name__ == "__main__":
    main()