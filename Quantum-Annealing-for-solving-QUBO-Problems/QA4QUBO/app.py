import datetime
from os import path
import time
import math

import neal

import QA4QUBO.hamming as hamming
import QA4QUBO.gen_test as gen_test
import ksp_config as ksp_config
from QA4QUBO.colors import colors
from QA4QUBO import ksp, solver
from QA4QUBO.script import ksp_annealer


def log_write(tpe, var):
    return "[" + colors.BOLD + str(tpe) + colors.ENDC + "]\t" + str(var) + "\n"

# QALS
def app1(folder, TIMES, k, _Q, n, capacity, items):
    zz      = []
    fz      = []
    r_times = [] 
    string  = str()

    start = time.time()

    I_max = ksp_config.QALS_PARAMS.i_max
    N_max = ksp_config.QALS_PARAMS.N_max
    D_min = ksp_config.QALS_PARAMS.d_min

    # string += colors.BOLD + colors.HEADER + "\nKnapsack Solution" + colors.ENDC + "\n"
    ksp_dp_profit, ksp_dp_weight, z_opt = ksp.ksp_dp(n=n, p=[item[1] for item in items], w=[item[0] for item in items], C=capacity)
    # string += log_write("Profit", ksp_dp_profit)
    # string += log_write("Weight", ksp_dp_weight)

    for i in range(I_max.__len__()):
        string += log_write("i_max, N_max, d_min", f"{I_max[i]}, {N_max[i]}, {D_min[i]}")
        
        for _ in range(TIMES):
            """
                Params:
                - n:           è la dimensione del problema
                - topology:    topologia hardware
                - sim:         False indica che non sta usando la modalità simulata del solver QALS
                - k:           numero di soluzioni candidate generate ad ogni iterazione all'annealing
                - I_max:       iterazioni massime
                - N_max:       numero massimo di iterazioni se l'alg non migliora
                - d_min:       conta quante volte trovi una soluzione diversa ma peggiore della migliore corrente
                - eta:         controlla quanto velocemente decresce p_delta
                - N:           numero di iterazioni per cui p rimane costante 
                - lambda_zero: fattore di penalita iniziale della tabu matrix
                - p_delta:     prob modifica permutazione
                - q:           prob di perturbazione della soluz candidata  
            """ 
            z_best, f_best, r_time, Z, solutions_matrix_zacc, solutions_matrix_star, solutions_matrix_zopt_zprop, solutions_matrix_zopt_zstar = solver.solve(
                z_opt       = z_opt,
                n           = n,
                Q           = _Q,
                topology    = 'pegasus',
                sim         = True,
                k           = k,
                i_max       = I_max[i],
                N_max       = N_max[i],
                d_min       = D_min[i],
                eta         = 0.10,
                N           = 10,
                lambda_zero = 1.5,
                p_delta     = 0.1,
                q           = 0.2,
            )

            D = hamming.build_pairwise_hamming_matrix(Z)

            labels, cluster_sizes, dendogram, cluster_best_z = hamming.cluster_qals_solutions(D=D, n=n, Q=_Q, Z=Z, f_best=f_best)
           
            dendogram.savefig(path.join(folder, f"dendrogram_{k}_{TIMES}.png"), dpi=300, bbox_inches="tight")

            cluster_max, cluster_max_size = max(cluster_sizes.items(), key=lambda kv: kv[1])
            # indici degli elementi che appartengono al cluster più grande
            cluster_max_sol_indices       = [idx for idx, label in enumerate(labels) if label == cluster_max]
            cluster_max_solutions         = [Z[i] for i in  cluster_max_sol_indices]     
            cluster_max_medoid            = hamming.find_cluster_medoid(cluster=cluster_max_solutions)
            if cluster_max_medoid is None:
                cluster_max_medoid = [0] * n
            
            # tutti i cluster
            cluster_string = ", ".join(
                f"cluster {cluster_id}: {size}"
                for cluster_id, size in sorted(cluster_sizes.items())
            )

            medoid_fQ = round(solver.function_f(Q=_Q, x=cluster_max_medoid).item(), 2)
            medoid_profit = sum(
                items[j][1] * cluster_max_medoid[j]
                for j in range(len(cluster_max_medoid))
            )
            medoid_weight = sum(
                items[j][0] * cluster_max_medoid[j]
                for j in range(len(cluster_max_medoid))
            )

            single_clusters = [c for c, s in cluster_sizes.items() if s == 1]

            string += log_write("n clusters         ", len(cluster_sizes))
            # string += log_write("clusters found     ", cluster_string)
            string += log_write("max cluster size   ", cluster_max_size)
            string += log_write("cluster best z     ", cluster_best_z)
            string += log_write("single cluster     ", len(single_clusters))
            string += log_write("medoid max cluster ", cluster_max_medoid)
            string += log_write("medoid fQ          ", medoid_fQ)
            string += log_write("Profit             ", medoid_profit)
            string += log_write("Weight             ", medoid_weight)

            gen_test.save_plot_solution_matrix(
                matrix=solutions_matrix_zacc,
                title_dh=f"{k}_{TIMES}_HD_z_star_old_z_star_new",
                title_fq=f"{k}_{TIMES}_FQ_z_star_old_z_star_new",
                folder=folder
            )
            gen_test.save_plot_solution_matrix(
                matrix=solutions_matrix_star,
                title_dh=f"{k}_{TIMES}_HD_z_t_z_star",
                title_fq=f"{k}_{TIMES}_FQ_z_t_z_star",
                folder=folder
            )
            gen_test.save_plot_solution_matrix(
                matrix=solutions_matrix_zopt_zstar,
                title_dh=f"{k}_{TIMES}_HD_z_opt_z_star",
                title_fq=f"{k}_{TIMES}_FQ_z_opt_z_star",
                folder=folder
            )
            gen_test.save_plot_solution_matrix(
                matrix=solutions_matrix_zopt_zprop,
                title_dh=f"{k}_{TIMES}_HD_z_opt_z_prop",
                title_fq=f"{k}_{TIMES}_FQ_z_opt_z_prop",
                folder=folder
            )
          
            zz.append(z_best)
            fz.append(f_best)
            r_times.append(r_time)
        
        fz_min_found = float("inf")
        z_min_found  = []
        for z, f in zip(zz, fz):
            if (f < fz_min_found):
                fz_min_found = f
                z_min_found  = z 

        # counter_fz_min = sum(1 for f in fz if math.isclose(f, fz_min_found))

        print("\t\t\t" + colors.BOLD + colors.OKGREEN + "RESULTS" + colors.ENDC + "\n")

        conv = datetime.timedelta(seconds = int(time.time() - start))

        p_best_found = sum(items[i][1] * z_min_found[i] for i in range(len(z_min_found)))
        w_best_found = sum(items[i][0] * z_min_found[i] for i in range(len(z_min_found)))

        # string += log_write("i_max, N_max, d_min ", f"{I_max[i]}, {N_max[i]}, {D_min[i]}")
        string += log_write("fQ Min Found       ", round(fz_min_found, 2))
        string += log_write("Profit Found       ", p_best_found)
        string += log_write("Weight Found       ", w_best_found)
        # string += log_write("n of fQ Min Found   ", counter_fz_min)
        # string += log_write("Items               ", itms)

    return string

# NO QALS
def app2(TIMES, k, _Q, items):
    string = str()

    sampler = neal.SimulatedAnnealingSampler()
    
    print(string)
    string = ""
    
    zz     = []
    mins_z = []

    for _ in range(TIMES):
        z, num_occ_z = ksp_annealer(_Q, sampler, k)
        zz.append(z)

        fz = solver.function_f(_Q, z).item()
        mins_z.append((fz, num_occ_z))

    fz_min_found, _ = min(mins_z, key=lambda x: x[0])
    sol_min = [
        (t, round(fz, 2), num_occ_z)
        for t, (fz, num_occ_z) in enumerate(mins_z)
        if math.isclose(fz, fz_min_found)
    ]    

    itms = []
    if len(sol_min) != 0:
        itms.extend(int(x) for x in zz[sol_min[0][0]])
    
    p_best_found = sum(items[i][1] * itms[i] for i in range(len(itms)))
    w_best_found = sum(items[i][0] * itms[i] for i in range(len(itms)))

    string += log_write("fQ Min Found     ", round(fz_min_found, 2))
    string += log_write("Profit Found     ", p_best_found)
    string += log_write("Weight Found     ", w_best_found)

    return string