from os import system, name
import numpy as np  

from QA4QUBO import ksp
from QA4QUBO.colors import colors
import QA4QUBO.gen_test as gen_test
import ksp_config as ksp_config

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
        gen_test.run_match_k_TIMES(file=file, n=n, capacity=capacity, items=items, _Q=Q_scale)


if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')
    main()