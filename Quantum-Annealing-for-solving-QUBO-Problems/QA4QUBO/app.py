import time
import datetime
import math

import neal # type: ignore

from QA4QUBO.colors import colors
from QA4QUBO import ksp, solver
from QA4QUBO.script import annealer

def log_write(tpe, var):
    return "["+colors.BOLD+str(tpe)+colors.ENDC+"]\t"+str(var)+"\n"

def app1(TIMES, k, nn, _Q, log_DIR, capacity, items):
    zz      = []
    r_times = [] 
    mins_z  = []

    string = str()

    start = time.time()

    for t in range(TIMES):
        z, r_time = solver.solve(
            d_min = 70,
            eta = 0.01,
            i_max = 10,
            k = k,
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
    
    fz_min_found = min(mins_z)
    cntr_fz_min  = sum(1 for fz in mins_z if math.isclose(fz, fz_min_found))
    sol_min      = [(t, round(fz, 2)) for t, fz in enumerate(mins_z) if math.isclose(fz, fz_min_found)]

    print("\t\t\t" + colors.BOLD + colors.OKGREEN + "RESULTS" + colors.ENDC + "\n")

    conv = datetime.timedelta(seconds = int(time.time() - start))
    
    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], nn)
    
    # ksp_bf_profit, choosen       = ksp.ksp_bf(nn, capacity, items)
    # ksp_bf_weight = sum(items[j][0] for j in range(nn) if choosen[j] == 1)

    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
   
    string += colors.BOLD + colors.HEADER + "\nQALS Solution" + colors.ENDC + "\n"

    p_gap  = []
    w_gap  = []
    
    for t in range(TIMES):
        string += log_write(f"{t}", zz[t])
        string += log_write("fQ", round(mins_z[t], 2))

        sol_p = sum(items[j][1] for j in range(nn) if zz[t][j] == 1)
        sol_w = sum(items[j][0] for j in range(nn) if zz[t][j] == 1)

        string += log_write("P", sol_p)
        string += log_write("W", sol_w)
        
        p_gap.append(ksp_dp_profit - sol_p)
        w_gap.append(ksp_dp_weight - sol_w)
    
    avg_p_gap = round(sum(p_gap) / len(p_gap), 1)
    avg_w_gap = round(sum(w_gap) / len(w_gap), 1)
    avg_fz    = round((sum(mins_z[i] for i in range(TIMES)) / len(mins_z)), 2)
    
    string += log_write("Avg Profit GAP   ", avg_p_gap)
    string += log_write("Avg Weight GAP   ", avg_w_gap)
    string += log_write("Avg fQ           ", avg_fz)
    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("n of fQ Min Found", cntr_fz_min)
    for i in range(len(sol_min)):
        string += log_write(f"RUN", sol_min[i][0])
    if len(sol_min) != 0:
        string += log_write("Items", [int(x) for x in zz[sol_min[0][0]]])

    print(string)

def app2(TIMES, k, _Q, nn, capacity, items):
    string = str()

    sampler = neal.SimulatedAnnealingSampler()

    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], nn)
    
    # ksp_bf_profit, choosen       = ksp.ksp_bf(nn, capacity, items)
    # ksp_bf_weight = sum(items[j][0] for j in range(nn) if choosen[j] == 1)

    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
    
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

        string += log_write(f"{t}", [int(x) for x in z])
        string += log_write("fQ", round(fz, 2))
        
        sol_w = sum(items[j][0] for j in range(nn) if z[j])
        sol_p = sum(items[j][1] for j in range(nn) if z[j])

        string += log_write("P", sol_p)
        string += log_write("W", sol_w)

        w_gap.append(ksp_dp_weight - sol_w)
        p_gap.append(ksp_dp_profit - sol_p)

        # with open("./solux.txt", "a") as soluz_file:
        #     line = f'{" ".join(map(str, z))}\n'
        #     soluz_file.write(line)

    fz_min_found = min(mins_z)
    cntr_fz_min  = sum(1 for fz in mins_z if math.isclose(fz, fz_min_found))
    sol_min      = [(t, round(fz, 2)) for t, fz in enumerate(mins_z) if math.isclose(fz, fz_min_found)]
    
    avg_p_gap = round((sum(p_gap)) / len(p_gap), 1)
    avg_w_gap = round((sum(w_gap)) / len(w_gap), 1)
    avg_fz    = round((sum(mins_z[i] for i in range(TIMES)) / len(mins_z)), 2)

    string += log_write("Avg Weight GAP   ", avg_w_gap)
    string += log_write("Avg Profit GAP   ", avg_p_gap)
    string += log_write("Avg fQ           ", avg_fz)
    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("n of fQ Min Found", cntr_fz_min)
    for i in range(len(sol_min)):
        string += log_write(f"RUN", sol_min[i][0])
    if len(sol_min) != 0:
        string += log_write("Items", [int(x) for x in zz[sol_min[0][0]]])

    print(string)