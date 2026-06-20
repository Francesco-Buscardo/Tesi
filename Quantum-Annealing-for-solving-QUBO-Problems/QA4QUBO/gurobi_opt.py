from gurobipy import Model, GRB, quicksum 
from QA4QUBO import ksp

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
