import time
import datetime

# - minimizzre formulazione qubo in un altro modo
# - calcolare tutto in funzione di energia Q(x)  
# - minimizzare classicamente la funzione non tramite l'annealer -> GUROBI
# - ore 11 venerdì prossimo 

# - correzione metodi classici 

import neal # type: ignore

from QA4QUBO.colors import colors
from QA4QUBO import ksp, solver
from QA4QUBO.script import annealer

def log_write(tpe, var):
    return "["+colors.BOLD+str(tpe)+colors.ENDC+"]\t"+str(var)+"\n"

def percentage_gap(best, my):
    if best == 0:
        return 0
    return ((best - my) / best) * 100

def app1(TIMES, nn, _Q, log_DIR, capacity, items):
    zz      = []
    r_times = [] 
    mins_z  = [] 

    string = str()

    start = time.time()

    for i in range(TIMES):
        z, r_time = solver.solve(
            d_min = 70,
            eta = 0.01,
            i_max = 10,
            k = 1,
            lambda_zero = 3/2,
            n = nn,
            N = 10,
            N_max = 100,
            p_delta = 0.1,
            q = 0.2,
            topology = 'pegasus',
            Q = _Q,
            log_DIR = log_DIR,
            sim = True
        )

        zz.append(z)
        r_times.append(r_time)

        fz = solver.function_f(_Q, z).item()
        mins_z.append(fz)

        string += log_write("Z", z)
        string += log_write("fQ", round(fz, 2))

    print("\t\t\t" + colors.BOLD + colors.OKGREEN + "RESULTS" + colors.ENDC + "\n")

    conv = datetime.timedelta(seconds = int(time.time() - start))
    
    # =========================
    # POST-PROCESSING
    # =========================
    string += log_write("Status", "In " + str(TIMES) + " runs")
    
    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], nn)
    # ksp_bf_profit, choosen       = ksp.ksp_bf(nn, capacity, items)

    # ksp_bf_weight = sum(items[j][0] for j in range(nn) if choosen[j] == 1)

    # if ksp_bf_profit == ksp_dp_profit and ksp_bf_weight == ksp_dp_weight:
    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
    # string += log_write("Choosen items", choosen)

    p_gap = []
    w_gap = []
    
    for t in range(TIMES):
        sol_w = sum(items[j][0] for j in range(nn) if zz[t][j] == 1)
        sol_p = sum(items[j][1] for j in range(nn) if zz[t][j] == 1)
        
        w_gap.append(ksp_dp_weight - sol_w)
        p_gap.append(percentage_gap(ksp_dp_profit, sol_p))
    
    avg_w_gap = round(sum(w_gap) / len(w_gap), 1)
    avg_p_gap = round(sum(p_gap) / len(p_gap), 1)
    
    string += log_write("", "")

    string += log_write("Avg Weight GAP", avg_w_gap)
    string += log_write("Avg Profit GAP", avg_p_gap)
    
    print(string)

def app2(TIMES, k, _Q, nn, capacity, items):
    string = str()

    sampler = neal.SimulatedAnnealingSampler()

    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], nn)
    # ksp_bf_profit, choosen       = ksp.ksp_bf(nn, capacity, items)
   
    # ksp_bf_weight = sum(items[j][0] for j in range(nn) if choosen[j] == 1)

    # if ksp_bf_profit == ksp_dp_profit and ksp_dp_weight == ksp_dp_weight:
    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
    # string += log_write("Choosen items", choosen)
    
    print(string)
    string = ""
    
    zz     = []
    mins_z = []

    w_gap = []
    p_gap = []
    
    for t in range(TIMES):
        z = annealer(_Q, sampler, k)
        zz.append(z)

        fz = solver.function_f(_Q, z).item()
        mins_z.append(fz)

        # string += log_write("Z", z)
        # string += log_write("fQ", round(fz, 2))
        
        sol_w = sum(items[j][0] for j in range(nn) if z[j])
        sol_p = sum(items[j][1] for j in range(nn) if z[j])

        # string += log_write("Sol Weight:", sol_w)
        # string += log_write("Sol Profit:", sol_p)

        w_gap.append(ksp_dp_weight - sol_w)
        p_gap.append(percentage_gap(ksp_dp_profit, sol_p))

        with open("./solux.txt", "a") as soluz_file:
            line = f'{" ".join(map(str, z))}\n'
            soluz_file.write(line)

    avg_p_gap = round((sum(p_gap)) / len(p_gap), 1)
    avg_w_gap = round((sum(w_gap)) / len(w_gap), 1)

    string += log_write("Avg Weight GAP", avg_w_gap)
    string += log_write("Avg Profit GAP", avg_p_gap)
    
    print(string)