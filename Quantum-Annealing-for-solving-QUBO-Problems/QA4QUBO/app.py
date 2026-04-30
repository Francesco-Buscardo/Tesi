import time
import datetime
import math
import numpy as np # type: ignore

import neal # type: ignore

from QA4QUBO.colors import colors
from QA4QUBO import ksp, solver
from QA4QUBO.script import annealer

def log_write(tpe, var):
    return "["+colors.BOLD+str(tpe)+colors.ENDC+"]\t"+str(var)+"\n"

def app1(TIMES, k, n, _Q, log_DIR, capacity, items):
    zz      = []
    r_times = [] 
    mins_z  = []

    string = str()

    start = time.time()

    for t in range(TIMES):
        # Params:
        # ! i_max:       numero massimo di iterazioni di QALS, è il limite superiore di ricerca, quanto si lascia 
        # !              che QALS cerchi soluzioni
        """
            IMPLICAZIONI SU TIMES:
                Spulciando meglio i parametri di QALS mi sono accorto che era già impostato un parametro realtivo alle iterazioni di
                QALS: i_max. Il parmetro in questione rappresenta il numero massimo di iterazioni di QALS, è il limite superiore di 
                ricerca quindi rappresenta quanto si lascia che QALS cerchi soluzioni:
                  - i_max -> basso = QALS è molto più veloce (a causa delle meno iterazioni) ma le soluzioni che si vanno a trovare 
                                     sono peggiori inquanto QALS non aggiorna abbastanza la tabu matrix e non riesce ad esplorare bene
                                     lo spazio delle soluzioni.
                  - i_max -> alto = QALS è chiaramente più lento ma a discapito della velocità riesce ad esplorare meglio lo spazio delle
                                    soluzioni e ad aggiornare in modo efficace la tabu matrix, qundi le soluzioni dovrebbero risultare migliori.
                "i_max" infatti è uno dei parametri che definisce la condizione stop di QALS:
                        if ((i == i_max) or ((e + d >= N_max) and (d < d_min)))

                Questa condizione permette di verificare se QALS ha raggiunto il numero massimo di iterazioni oppure 
                se è entrato in una condizione di convergenza o stallo.
                
                Quindi quello che sugerisco è anzichè testare con un dominio di TIMES più alto lo abbasserei (per esempio io solo {10}),
                d'altro canto aumenterei invece il dominio di i_max a {10, 50, 100, 250, 500, 1000}.

                Gli altri parametri di QALS che definiscono la condizione di stop sono:
                    - N_max: numero massimo di iterazioni se l'alg non migliora.
                             Quindi imposterei un dominio, rispettivamente con i_max, di questo genere {5, 25, 50, 125, 250, 500}.
                    - d_min: conta quante volte trovi una soluzione diversa ma peggiore della migliore corrente.
                             Qui farei giusto per provare d_min = 0.7 * N_max: {4, 18, 35, 88, 175, 350}.

                I parametri fino ad ora erano settati così:
                    - i_max = 10
                    - N_max = 50
                    - d_min = 70
        """
        # - d_min:       conta quante volte trovi una soluzione diversa ma peggiore della migliore corrente
        # - p_delta:     prob modifica permutazione
        # - eta:         controlla quanto velocemente decresce p_delta
        # - q:           prob di perturbazione della soluz candidata  
        # - N:           numero di iterazioni per cui p rimane costante 
        # - N_max:       numero massimo di iterazioni se l'alg non migliora
        # - lambda_zero: fattore di penalita iniziale della tabu matrix
        # - n:           è la dimensione del problema
        # - k:           numero di soluzioni candidate generate ad ogni iterazione all'annealing
        # - topology:    topologia hardware
        # - sim:         False indica che non sta usando la modalità simulata del solver QALS
        z, r_time = solver.solve(
            d_min = 70,
            eta = 0.10,
            i_max = 100, # before: 10
            k = k,
            lambda_zero = 1.5,
            n = n,
            N = 10,
            N_max = 50, # before: 100
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
    
    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], n)

    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
   
    string += colors.BOLD + colors.HEADER + "\nQALS Solution" + colors.ENDC + "\n"

    p_gap  = []
    w_gap  = []
    
    for t in range(TIMES):
        # string += log_write(f"{t}", zz[t])
        # string += log_write("fQ", round(mins_z[t], 2))

        sol_p = sum(items[j][1] for j in range(n) if zz[t][j] == 1)
        sol_w = sum(items[j][0] for j in range(n) if zz[t][j] == 1)

        # string += log_write("P", sol_p)
        # string += log_write("W", sol_w)
        
        p_gap.append(ksp_dp_profit - sol_p)
        w_gap.append(ksp_dp_weight - sol_w)
    
    avg_p_gap = round(sum(p_gap) / len(p_gap), 1)
    avg_w_gap = round(sum(w_gap) / len(w_gap), 1)
    avg_fz    = round((sum(mins_z[i] for i in range(TIMES)) / len(mins_z)), 2)
    
    itms = []
    if len(sol_min) != 0:
        itms.extend(int(x) for x in zz[sol_min[0][0]])
    
    p_best_found = sum(items[i][1] * itms[i] for i in range(len(itms)))
    w_best_found = sum(items[i][0] * itms[i] for i in range(len(itms)))


    string += log_write("Avg Profit GAP   ", avg_p_gap)
    string += log_write("Avg Weight GAP   ", avg_w_gap)
    string += log_write("Avg fQ           ", avg_fz)
    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("Profit Found     ", p_best_found)
    string += log_write("Weight Found     ", w_best_found)
    string += log_write("Profit GAP       ", ksp_dp_profit - p_best_found)
    string += log_write("Weight GAP       ", ksp_dp_weight - w_best_found)
    string += log_write("n of fQ Min Found", cntr_fz_min)
    # for i in range(len(sol_min)):
    #     string += log_write(f"RUN              ", sol_min[i][0])
    string += log_write("Items            ", itms)

    return string

def app2(TIMES, k, _Q, n, capacity, items):
    string = str()

    sampler = neal.SimulatedAnnealingSampler()

    ksp_dp_profit, ksp_dp_weight = ksp.ksp_dp(capacity, [item[0] for item in items], [item[1] for item in items], n)

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

        # string += log_write(f"{t}", [int(x) for x in z])
        # string += log_write("fQ", round(fz, 2))
        
        sol_w = sum(items[j][0] for j in range(n) if z[j])
        sol_p = sum(items[j][1] for j in range(n) if z[j])

        # string += log_write("P", sol_p)
        # string += log_write("W", sol_w)

        w_gap.append(ksp_dp_weight - sol_w)
        p_gap.append(ksp_dp_profit - sol_p)

    fz_min_found = min(mins_z)
    cntr_fz_min  = sum(1 for fz in mins_z if math.isclose(fz, fz_min_found))
    sol_min      = [(t, round(fz, 2)) for t, fz in enumerate(mins_z) if math.isclose(fz, fz_min_found)]
    
    avg_p_gap = round((sum(p_gap)) / len(p_gap), 1)
    avg_w_gap = round((sum(w_gap)) / len(w_gap), 1)
    avg_fz    = round((sum(mins_z[i] for i in range(TIMES)) / len(mins_z)), 2)

    itms = []
    if len(sol_min) != 0:
        itms.extend(int(x) for x in zz[sol_min[0][0]])
    
    p_best_found = sum(items[i][1] * itms[i] for i in range(len(itms)))
    w_best_found = sum(items[i][0] * itms[i] for i in range(len(itms)))

    string += log_write("Avg Profit GAP   ", avg_p_gap)
    string += log_write("Avg Weight GAP   ", avg_w_gap)
    string += log_write("Avg fQ           ", avg_fz)
    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("Profit Found     ", p_best_found)
    string += log_write("Weight Found     ", w_best_found)
    string += log_write("Profit GAP       ", ksp_dp_profit - p_best_found)
    string += log_write("Weight GAP       ", ksp_dp_weight - w_best_found)
    string += log_write("n of fQ Min Found", cntr_fz_min)
    # for i in range(len(sol_min)):
    #     string += log_write(f"RUN              ", sol_min[i][0])
    string += log_write("Items            ", itms)

    return string