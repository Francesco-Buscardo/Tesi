from os import system, name, listdir, path, makedirs
import re

from gurobipy import Model, GRB, quicksum # type: ignore

from QA4QUBO import ksp, app
from QA4QUBO.colors import colors

def generate_file_ksp(n_items, C):
    nok = True
    i = 0
    _dir = "KSP_" + str(n_items)+ "_" + str(C)

    while(nok):
        try:
            with open("outputs/" + _dir.replace("KSP", "KSP_LOG") + ".csv", "r") as file:
                pass
            i += 1
            _dir = "KSP_" + str(n_items) + "_" + str(C) + "_" + str(i)
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/" + _dir

    return DIR

def test_gurobi_optimizer(n_items, capacity, items):
    # create model
    knapsack_model = Model('knapsack')

    # add decision variables to model
    x = knapsack_model.addVars(n_items, vtype = GRB.BINARY, name = "x")

    #define objective function Q(x) = x^T Q x = ∑​_i(∑​_j(Qij ​xi ​xj​))
    Q = ksp.generate_QUBO_knapsack(n_items, capacity, items)
    obj_fun = quicksum(Q[i,j] * x[i] * x[j] for i in range(n_items) for j in range(n_items))
    knapsack_model.setObjective(obj_fun, GRB.MINIMIZE)

    # run
    knapsack_model.setParam('OutputFlag', False) 
    knapsack_model.optimize()

    print("Optimization is done:", round(knapsack_model.ObjVal, 2))
    sol = []
    for i in range(n_items):
        val = int(round(x[i].X))
        sol.append(val)
        # print(f"x[{i}]: {val}")

    total_weight = sum(items[i][0] for i in range(n_items) if sol[i] == 1)
    total_profit = sum(items[i][1] for i in range(n_items) if sol[i] == 1)

    print("Total profit: ", total_profit)
    print("Total weight: ", total_weight)

    return total_profit, total_weight, sol, round(knapsack_model.ObjVal, 2)

def remove_ansi(text):
    return re.sub(r'\x1b\[[0-9;]*m', '', text)

def generate_match_k_TIMES():
    print("\t\t" + colors.BOLD + colors.OKGREEN + "   GENERATING MATCH" + colors.ENDC + "\t\t")
    
    # ! IDEA: lasciare TIMES = 10, variare tanto k = {10, 100, 1000, 2000, 5000}
    # k = quante volte risolvo il problema QUBO
    match_k_t = [ 
        (1000, 10),
        (2000, 10),
        (3000, 10),
        (4000, 10),
        (5000, 10)
    ]
    
    folder = ""
    makedirs(folder, exist_ok=True)

    for k, t in match_k_t:
        filename = f"file_{k}_{t}.txt"
        filepath = path.join(folder, filename)

        with open(filepath, "w") as f:
            f.write(f"k     = {k}\n")
            f.write(f"TIMES = {t}\n")

        print(f"Create: {filepath}")
    
    print("\t\t" + colors.BOLD + colors.OKGREEN + "   END GENERATING MATCH" + colors.ENDC + "\n\n\t\t")
    return match_k_t, folder

def run_match_k_TIMES(_QALS, n, capacity, items, _Q, log_DIR):
    match_k_t, folder = generate_match_k_TIMES()

    # for (k, TIMES) in match_k_t:
    #     filepath = path.join(folder, f"file_{k}_{TIMES}.txt")
        
    #     with open(filepath, "a") as f:
    #         if _QALS:
    #             f.write("QALS\n\n")
    #             f.write(remove_ansi(app.app1(TIMES, k, n, _Q, log_DIR, capacity, items)))
    #         else:
    #             f.write("NO QALS\n\n")
    #             f.write(remove_ansi(app.app2(TIMES, k, _Q, n, capacity, items)))

    for (k, TIMES) in match_k_t:
        filepath = path.join(folder, f"file_{k}_{TIMES}.txt")
        
        with open(filepath, "a") as f:
                f.write("\nQALS\n")
                f.write(remove_ansi(app.app1(TIMES, k, n, _Q, log_DIR, capacity, items)))

                f.write("\nNO QALS\n\n")
                f.write(remove_ansi(app.app2(TIMES, k, _Q, n, capacity, items)))

def main():
    # =========================
    # COSTRUZIONE MATRICE Q
    # =========================
    n, capacity, items = ksp.build_knapsack("QA4QUBO/ksp/ksp_1.txt")
       
    _DIR = generate_file_ksp(n, capacity)
    log_DIR = _DIR.replace("KSP","KSP_LOG") + ".csv"

    _Q = ksp.generate_QUBO_knapsack(n, capacity, items)
    print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")

    # =========================
    # ESECUZIONE ALG
    # =========================
    run_match_k_TIMES(_QALS=0, n=n, capacity=capacity, items=items, _Q=_Q, log_DIR=log_DIR)


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')

    main()