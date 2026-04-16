import numpy as np

import io
import re
from pymhlib.demos.mkp import MKPInstance, MKPSolution
from pymhlib.demos.common import run_optimization
from contextlib import redirect_stdout

# items[n_items][D + 1]: 
#    | p | w1 | w2 | ... | wD
# ---|---|----|----|-----|----
# 0  |   |    |    |     |
# ---|---|----|----|-----|----
# 1  |   |    |    |     |
# ---|---|----|----|-----|----
# 2  |   |    |    |     |
# ---|---|----|----|-----|----
# .  |   |    |    |     |
# .  |   |    |    |     |
# .  |   |    |    |     |
# ---|---|----|----|-----|----
# n-1|   |    |    |     |

def build_mknapsack(mksp_file):
    n_items = 0
    C_d     = []
    items   = []

    with open(mksp_file, "r") as file:
        lines = [line.strip() for line in file if line.strip()]

        n_items = int(lines[0])
        C_d = list(map(int, lines[1].split()))

        for i in range(2, 2 + n_items):
            items.append(list(map(int, lines[i].split())))

    return n_items, C_d, items

def mksp_solve():
    buffer = io.StringIO()

    with redirect_stdout(buffer):
        run_optimization("MKP", MKPInstance, MKPSolution, "file.txt")

    output = buffer.getvalue()

    best_solution = None
    best_obj = None

    m1 = re.search(r"T best solution:\s*\[([^\]]*)\]", output)
    if m1:
        text = m1.group(1).strip()
        best_solution = [] if not text else [int(x) for x in text.split()]

    m2 = re.search(r"T best obj:\s*([0-9.+-]+)", output)
    if m2:
        best_obj = float(m2.group(1))

    print("Items used:", best_solution)
    print("Best value:", best_obj)

def compute_mpenalty(C_d, profits):
    # Compute the penalties for the mkps
    # A_i = (sum of profits_i) / Cd[i]

    D = len(C_d)
    A_d = np.zeros(D)

    for i in range(D):
        A_d[i] = sum(p for _, p in profits[i]) / C_d[i]

    return A_d

def generate_QUBO_mksp(n_items, C_d, items):
    # Generate the matrix _Q for the m-dim knapsack problem

    _Q = np.zeros((n_items, n_items))
    
    profits = [item[0] for item in items[:n_items]]

    A_d  = compute_mpenalty(C_d, profits)

    for i in range(n_items):
        p_id = items[i][0]
        _sum = 0

        for d in range(len(C_d)):
            w_id = items[i][d + 1]
            _sum += (A_d[d] * (w_id ** 2)) - (2 * A_d[d] * C_d[d] * w_id)
            
        _Q[i][i] = (-p_id + _sum)

        for j in range(i + 1, n_items):
            _sum = 0

            for d in range(len(C_d)):
                w_id = items[i][d + 1]
                w_jd = items[j][d + 1]
                _sum += (A_d[d] * w_id * w_jd)
            
            _Q[i][j] = _sum
            _Q[j][i] = _sum
        
    return _Q