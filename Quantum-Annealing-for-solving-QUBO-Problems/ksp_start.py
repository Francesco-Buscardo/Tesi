from os import system, name
from os.path import isfile, join
from os import listdir

from gurobipy import Model, GRB, quicksum # type: ignore

from QA4QUBO import ksp, app
from QA4QUBO.colors import colors

ksp_files = sorted(
    f for f in listdir("QA4QUBO/ksp/")
    if isfile(join("QA4QUBO/ksp/", f)) and f.startswith("ksp_") and f.endswith(".txt")
)

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

def main():
    # =========================
    # COSTRUZIONE MATRICE Q
    # =========================
    capacity = 0
    items    = []

    # while nn <= 0 or nn > 10:
    #     nn = int(input("[" + colors.FAIL + colors.BOLD + "Invalid n" + colors.ENDC + "] Insert n: "))
    
    # ksp_file = join("QA4QUBO/ksp/", ksp_files[nn])

    nn, capacity, items = ksp.build_knapsack("QA4QUBO/ksp/ksp_2.txt")

    gurobi_profit, gurobi_weight, gurobi_x, fQ = test_gurobi_optimizer(nn, capacity, items)

    gurobi_sol = {
        "profit": gurobi_profit,
        "weight": gurobi_weight,
        "x":      gurobi_x,
        "fQ":     fQ
    }

    TIMES       = 10
    QALS        = 1

    _DIR = generate_file_ksp(nn, capacity)
    log_DIR = _DIR.replace("KSP","KSP_LOG") + ".csv"

    _Q = ksp.generate_QUBO_knapsack(nn, capacity, items)

    print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")

    # =========================
    # ESECUZIONE DI QALS
    # =========================
    fz = 0
    if QALS:
        fz = app.app1(TIMES, nn, _Q, log_DIR, capacity, items)
    else:
        k = 1
        app.app2(TIMES, k, _Q, nn, capacity, items)
    
    print(colors.BOLD + colors.HEADER + "\nGUROBI Solution" + colors.ENDC)
    print("profit:              ", gurobi_sol["profit"])
    print("weight:              ", gurobi_sol["weight"])
    # for i in range(nn):
    #     print(f"x[{i}]: {gurobi_x[i]}")
    print("fQ:                  ", gurobi_sol["fQ"])
    print(f"fQ_QALS - fQ_GUROBI:  {abs(fz) - abs(gurobi_sol["fQ"])}")


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')

    main()