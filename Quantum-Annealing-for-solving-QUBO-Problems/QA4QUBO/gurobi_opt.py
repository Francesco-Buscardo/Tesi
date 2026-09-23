import time

from gurobipy import Model, GRB, quicksum, GurobiError #type: ignore

def test_gurobi_optimizer(n_items, Q, items):

    knapsack_model = Model('knapsack')
    x = knapsack_model.addVars(n_items, vtype = GRB.BINARY, name = "x")

    #define objective function Q(x) = x^T Q x = ∑​_i(∑​_j(Qij ​xi ​xj​))
    obj_fun = quicksum(Q[i,j] * x[i] * x[j] for i in range(n_items) for j in range(n_items))
    knapsack_model.setObjective(obj_fun, GRB.MINIMIZE)

    start = time.perf_counter()

    knapsack_model.setParam('OutputFlag', False) 
    knapsack_model.optimize()

    end = time.perf_counter()

    return end - start

    # sol = []
    # for i in range(n_items):
    #     val = int(round(x[i].X))
    #     sol.append(val)

    # total_weight = sum(items[i][0] for i in range(n_items) if sol[i] == 1)
    # total_profit = sum(items[i][1] for i in range(n_items) if sol[i] == 1)

    # return total_profit, total_weight, sol, round(knapsack_model.ObjVal, 2), (end - start)