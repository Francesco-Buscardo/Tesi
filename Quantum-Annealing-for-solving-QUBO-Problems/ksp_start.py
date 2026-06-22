from os import system, name
import numpy as np  
from os import path

from QA4QUBO import ksp, app
from QA4QUBO.colors import colors
import QA4QUBO.gen_test as gen_test
import ksp_config as ksp_config

def run_match_k_TIMES(file, n, capacity, items, _Q):
    folder = gen_test.generate_folder_match_k_TIMES(file)

    for (k, TIMES) in ksp_config.MATCH_K_T:
        filepath = path.join(folder, f"file_{k}_{TIMES}.txt")

        with open(filepath, "a") as f:
            f.write("\nQALS\n")
            f.write(gen_test.remove_ansi(app.app1(file, TIMES, k, _Q, n, capacity, items)))

            f.write("\nNO QALS\n")
            f.write(gen_test.remove_ansi(app.app2(TIMES, k, _Q, items)))

def main():
    for file in ksp_config.KSP_EXAMPLES:
        print("\t\t" + colors.BOLD + colors.OKGREEN + f"{file}" + colors.ENDC + "\n\n\t\t")
        # =========================
        # COSTRUZIONE MATRICE Q
        # =========================
        n, capacity, items = ksp.build_knapsack(file)

        _Q = ksp.generate_QUBO_knapsack(n, capacity, items)
        _Q = np.array(_Q)
        
        Q_scale =  _Q / ksp_config.SCALE_QUBO
        print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")

        # =========================
        # ESECUZIONE ALGORITMO
        # =========================
        run_match_k_TIMES(file=file, n=n, capacity=capacity, items=items, _Q=Q_scale)


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')
    main()