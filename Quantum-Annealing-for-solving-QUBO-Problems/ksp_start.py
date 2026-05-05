from os import system, name, listdir, path, makedirs
import re
from pathlib import Path

from gurobipy import Model, GRB, quicksum # type: ignore

from QA4QUBO import ksp, app
from QA4QUBO.colors import colors
import ksp_config as ksp_config

def test_gurobi_optimizer(n_items, capacity, items):
    # create model
    knapsack_model = Model('knapsack')

    # add decision variables to model
    x = knapsack_model.addVars(n_items, vtype = GRB.BINARY, name = "x")

    #define objective function Q(x) = x^T Q x = ∑​_i(∑​_j(Qij ​xi ​xj​))
    Q       = ksp.generate_QUBO_knapsack(n_items, capacity, items)
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

def generate_folder_match_k_TIMES(file):
    if ksp_config.LAMBDA_VALUE == "lambda_div_3":
        lambda_folder = "lambda_div_3"
    elif ksp_config.LAMBDA_VALUE == "lambda_650_dot_C":
        lambda_folder = "lambda_650_dot_C"
    elif ksp_config.LAMBDA_VALUE == "lambda_div_C":
        lambda_folder = "lambda_div_C"
    else:
        lambda_folder = ""

    ksp_name = Path(file).stem
    folder   = f"test/test_i_max/{ksp_name}/{lambda_folder}/"

    makedirs(folder, exist_ok=True)

    for k, t in ksp_config.MATCH_K_T:
        filename = f"file_{k}_{t}.txt"
        filepath = path.join(folder, filename)

        with open(filepath, "w") as f:
            f.write(f"k     = {k}\n")
            f.write(f"TIMES = {t}\n")

        print(f"Create: {filepath}")
    
    return folder

def run_match_k_TIMES(file, n, capacity, items, _Q):
    folder = generate_folder_match_k_TIMES(file)

    for (k, TIMES) in ksp_config.MATCH_K_T:
        filepath = path.join(folder, f"file_{k}_{TIMES}.txt")

        with open(filepath, "a") as f:
            f.write("\nQALS\n")
            f.write(remove_ansi(app.app1(TIMES, k, _Q, n, capacity, items)))

            f.write("\nNO QALS\n\n")
            f.write(remove_ansi(app.app2(TIMES, k, _Q, n, capacity, items)))

def main():
    
    for file in ksp_config.KSP_EXAMPLES:
        print("\t\t" + colors.BOLD + colors.OKGREEN + f"{file}" + colors.ENDC + "\n\n\t\t")
        # =========================
        # COSTRUZIONE MATRICE Q
        # =========================
        n, capacity, items = ksp.build_knapsack(file)

        _Q = ksp.generate_QUBO_knapsack(n, capacity, items)
        print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")

        # =========================
        # ESECUZIONE ALGORITMO
        # =========================
        run_match_k_TIMES(file=file, n=n, capacity=capacity, items=items, _Q=_Q)


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')
    main()