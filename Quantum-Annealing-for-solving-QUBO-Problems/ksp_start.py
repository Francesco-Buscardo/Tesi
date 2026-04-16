from os import system, name
from os.path import isfile, join
from os import listdir

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

def main():
    # =========================
    # COSTRUZIONE MATRICE Q
    # =========================
    capacity = 0
    items    = []

    # while nn <= 0 or nn > 10:
    #     nn = int(input("[" + colors.FAIL + colors.BOLD + "Invalid n" + colors.ENDC + "] Insert n: "))
    
    # ksp_file = join("QA4QUBO/ksp/", ksp_files[nn])

    nn, capacity, items = ksp.build_knapsack("QA4QUBO/ksp/ksp_1.txt")

    _DIR = generate_file_ksp(nn, capacity)
    log_DIR = _DIR.replace("KSP","KSP_LOG") + ".csv"

    _Q = ksp.generate_QUBO_knapsack(nn, capacity, items)

    print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")
    
    # =========================
    # ESECUZIONE DI QALS
    # =========================
    TIMES = 1000
    QALS  = 1

    if QALS:
        app.app1(TIMES, nn, _Q, log_DIR, capacity, items)
    else:
        k = 1
        app.app2(TIMES, k, _Q, nn, capacity, items)

if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')

    main()