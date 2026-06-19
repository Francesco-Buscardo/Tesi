import time
import csv
import math
import os
import numpy as np
import matplotlib.pyplot as plt

import neal

from QA4QUBO.colors import colors
from QA4QUBO import ksp, solver
from QA4QUBO.script import ksp_annealer
import ksp_config as ksp_config


def vector_to_string(v):
    return "".join(str(int(x)) for x in v)

def save_single_hamming_plot(matrix, title, y_label, plot_path):
    ham_distances = [item[2] for item in matrix]
    t_values = list(range(len(ham_distances)))

    plt.figure(figsize=(10, 5))
    plt.plot(t_values, ham_distances)

    plt.xlabel("t")
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

def write_single_matrix_to_csv(writer, label, i_max, n_max, d_min, matrix):
    writer.writerow([])
    writer.writerow([label])
    writer.writerow([i_max, n_max, d_min])

    for item in matrix:
        z        = item[0]
        fQ       = item[1]
        hamm_d   = item[2]
        hamm_vec = item[3]

        changed_items = np.where(hamm_vec == 1)[0].tolist()

        writer.writerow([
            vector_to_string(z),
            fQ,
            hamm_d,
            changed_items
        ])

def save_diagnostics_to_csv( i_max, n_max, d_min, solutions_matrix_star, solutions_matrix_best, solutions_matrix_opt):
    file_exists = os.path.exists(ksp_config.PATH_CSV)

    with open(ksp_config.PATH_CSV, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow([
                "i_max",
                "N_max",
                "d_min",
                "z",
                "fQ",
                "hamming_distance",
                "changed_items"
            ])

        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_star",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_star
        )
        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_best",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_best
        )
        write_single_matrix_to_csv(
            writer=writer,
            label="Hamming rispetto a z_opt",
            i_max=i_max,
            n_max=n_max,
            d_min=d_min,
            matrix=solutions_matrix_opt
        )

    save_single_hamming_plot(
        matrix=solutions_matrix_star,
        title=r"Andamento $d_H(z_t, z_{star})$",
        y_label="Hamming distance",
        plot_path=f"{ksp_config.TEST_FOLDER}/diagnostics_Hamming_star_i{i_max}_N{n_max}_d{d_min}.png"
    )
    save_single_hamming_plot(
        matrix=solutions_matrix_best,
        title=r"Andamento $d_H(z_t, z_{best})$",
        y_label="Hamming distance",
        plot_path=f"{ksp_config.TEST_FOLDER}/diagnostics_Hamming_best_i{i_max}_N{n_max}_d{d_min}.png"
    )
    save_single_hamming_plot(
        matrix=solutions_matrix_opt,
        title=r"Andamento $d_H(z_t, z_{opt})$",
        y_label="Hamming distance",
        plot_path=f"{ksp_config.TEST_FOLDER}/diagnostics_Hamming_opt_i{i_max}_N{n_max}_d{d_min}.png"
    )

def log_write(tpe, var):
    return "[" + colors.BOLD + str(tpe) + colors.ENDC + "]\t" + str(var) + "\n"

# QALS
def app1(TIMES, k, _Q, n, capacity, items):
    zz      = []
    r_times = [] 
    mins_z  = []

    string = str()

    start = time.time()

    I_max = ksp_config.QALS_PARAMS.i_max
    N_max = ksp_config.QALS_PARAMS.N_max
    D_min = ksp_config.QALS_PARAMS.d_min

    ksp_dp_profit, ksp_dp_weight, z_opt = ksp.ksp_dp(n=n, p=[item[1] for item in items], w=[item[0] for item in items], C=capacity)
    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)

    for i in range(I_max.__len__()):
        for _ in range(TIMES):
            """
                Params:
                - d_min:       conta quante volte trovi una soluzione diversa ma peggiore della migliore corrente
                - p_delta:     prob modifica permutazione
                - eta:         controlla quanto velocemente decresce p_delta
                - q:           prob di perturbazione della soluz candidata  
                - N:           numero di iterazioni per cui p rimane costante 
                - N_max:       numero massimo di iterazioni se l'alg non migliora
                - lambda_zero: fattore di penalita iniziale della tabu matrix
                - n:           è la dimensione del problema
                - k:           numero di soluzioni candidate generate ad ogni iterazione all'annealing
                - topology:    topologia hardware
                - sim:         False indica che non sta usando la modalità simulata del solver QALS
            """ 
            z, r_time, solutions_matrix_star, solutions_matrix_best, solutions_matrix_opt = solver.solve(
                d_min = D_min[i],
                eta = 0.10,
                i_max = I_max[i],
                k = k,
                lambda_zero = 1.5,
                n = n,
                N = 10,
                N_max = N_max[i],
                p_delta = 0.1,
                q = 0.2,
                topology = 'pegasus',
                Q = _Q,
                sim = True,
                z_opt = z_opt
            )

            save_diagnostics_to_csv(
                i_max=I_max[i],
                n_max=N_max[i],
                d_min=D_min[i],
                solutions_matrix_star=solutions_matrix_star,
                solutions_matrix_best=solutions_matrix_best,
                solutions_matrix_opt=solutions_matrix_opt
            )
          
            zz.append(z)
            r_times.append(r_time)

            fz = solver.function_f(_Q, z).item()
            mins_z.append(fz)
        
        fz_min_found   = min(mins_z)
        counter_fz_min = sum(1 for fz in mins_z if math.isclose(fz, fz_min_found))
        sol_min        = [(t, round(fz, 2)) for t, fz in enumerate(mins_z) if math.isclose(fz, fz_min_found)]

        print("\t\t\t" + colors.BOLD + colors.OKGREEN + "RESULTS" + colors.ENDC + "\n")

        # conv = datetime.timedelta(seconds = int(time.time() - start))
    
        # avg_fz = round((sum(mins_z[i] for i in range(TIMES)) / len(mins_z)), 2)
        
        itms = []
        if len(sol_min) != 0:
            itms.extend(int(x) for x in zz[sol_min[0][0]])
        
        p_best_found = sum(items[i][1] * itms[i] for i in range(len(itms)))
        w_best_found = sum(items[i][0] * itms[i] for i in range(len(itms)))

        string += log_write("i_max, N_max, d_min", f"{I_max[i]}, {N_max[i]}, {D_min[i]}")
        # string += log_write("Avg fQ             ", avg_fz)
        string += log_write("fQ Min Found       ", round(fz_min_found, 2))
        string += log_write("Profit Found       ", p_best_found)
        string += log_write("Weight Found       ", w_best_found)
        # string += log_write("Profit GAP         ", ksp_dp_profit - p_best_found)
        # string += log_write("Weight GAP         ", ksp_dp_weight - w_best_found)
        # string += log_write("n of fQ Min Found  ", counter_fz_min)
        # string += log_write("Items              ", itms)

    return string

# NO QALS
def app2(TIMES, k, _Q, n, capacity, items):
    string = str()

    sampler = neal.SimulatedAnnealingSampler()

    ksp_dp_profit, ksp_dp_weight, _ = ksp.ksp_dp(n=n, p=[item[1] for item in items], w=[item[0] for item in items], C=capacity)

    string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    string += log_write("Best profit", ksp_dp_profit)
    string += log_write("Weight", ksp_dp_weight)
    
    print(string)
    string = ""
    
    zz     = []
    mins_z = []

    for t in range(TIMES):
        z, num_occ_z = ksp_annealer(_Q, sampler, k)
        zz.append(z)

        fz = solver.function_f(_Q, z).item()
        mins_z.append((fz, num_occ_z))

    fz_min_found, num_occ = min(mins_z, key=lambda x: x[0])
    cntr_fz_min           = sum(1 for fz, _ in mins_z if math.isclose(fz, fz_min_found))
    sol_min = [
        (t, round(fz, 2), num_occ_z)
        for t, (fz, num_occ_z) in enumerate(mins_z)
        if math.isclose(fz, fz_min_found)
    ]    

    # avg_fz = round(sum(fz for fz, _ in mins_z) / len(mins_z), 2)
    
    itms = []
    if len(sol_min) != 0:
        itms.extend(int(x) for x in zz[sol_min[0][0]])
    
    p_best_found = sum(items[i][1] * itms[i] for i in range(len(itms)))
    w_best_found = sum(items[i][0] * itms[i] for i in range(len(itms)))

    # string += log_write("Avg fQ           ", avg_fz)
    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("Profit Found     ", p_best_found)
    string += log_write("Weight Found     ", w_best_found)
    # string += log_write("Profit GAP       ", ksp_dp_profit - p_best_found)
    # string += log_write("Weight GAP       ", ksp_dp_weight - w_best_found)
    string += log_write("n of fQ Min Found", cntr_fz_min)
    # string += log_write("Items            ", itms)

    return string