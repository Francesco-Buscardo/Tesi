#!/usr/local/bin/python3
import pandas as pd
import numpy as np
import datetime
import time
import csv
import sys
from os import listdir, mkdir, system, name
from os.path import isfile, join, exists

from QA4QUBO import matrix, vector, solver, tsp
from QA4QUBO.colors import colors

# qap: lista dei problemi
qap = [f for f in listdir("QA4QUBO/qap/") if isfile(join("QA4QUBO/qap/", f))]

np.set_printoptions(threshold = sys.maxsize)


def log_write(tpe, var):
    return "["+colors.BOLD+str(tpe)+colors.ENDC+"]\t"+str(var)+"\n"

def getproblem():
    elements = list()
    i = 0
    for element in qap:
        elements.append(element)
        element = element[:-4]
        print(f"Write {i} for the problem {element}")
        i += 1
    
    problem = int(input("Which problem do you want to select? "))
    DIR = "QA4QUBO/qap/" + qap[problem]
    return DIR, qap[problem]

def write(dir, string):
    file = open(dir, 'a')
    file.write(string+'\n')
    file.close()

def csv_write(DIR, l):
    with open(DIR, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(l)


def generate_file_npp(_n:int):
    nok = True
    i = 0
    max_range = 100000
    _dir = "NPP_"+str(_n)+"_"+ str(max_range)
    while(nok):
        try:
            with open("outputs/"+_dir.replace("NPP","NPP_LOG")+".csv", "r") as file:
                pass
            max_range = int(max_range/10)
            if(max_range < 10):
                exit("File output terminati")
            _dir = "NPP_"+str(_n)+"_"+ str(max_range)
            i += 1
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR, max_range

def generate_file_tsp(n:int):
    nok = True
    i = 1
    _dir = "TSP_"+str(n)+"_"+str(i)
    while(nok):
        try:
            with open("outputs/"+_dir.replace("TSP","TSP_LOG")+".csv", "r") as file:
                pass
            i += 1
            _dir = "TSP_"+str(n)+"_"+str(i)
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR

def generate_file_qap(name):
    nok = True
    i = 0
    _dir = "QAP_"+str(name)
    while(nok):
        try:
            with open("outputs/"+_dir+".csv", "r") as file:
                pass
            i += 1
            _dir = "QAP_"+str(name)+"_"+ str(i)
        except FileNotFoundError:
            nok = False
        
    DIR = "outputs/"+_dir

    return DIR

def convert_qubo_to_Q(qubo, n):
    # ? Converte il dizionario qubo nella matrice Q di f(z) = z^T Q z

    Q = np.zeros((n,n))

    for x,y in qubo.keys():
        Q[x][y] = qubo[x,y]

    return Q

def main(nn):    

    print("\t\t" + colors.BOLD + colors.WARNING + "  BUILDING PROBLEM..." + colors.ENDC)
    pr = input(colors.OKCYAN + "Which problem would you like to run? (NPP, QAP, TSP)  " + colors.ENDC)

    if pr == "NPP":
        NPP = True
        QAP = False
        TSP = False
    elif pr == "QAP":
        NPP = False
        QAP = True
        TSP = False
    elif pr == "TSP":
        NPP = False
        QAP = False
        TSP = True
    else:
        print("[" + colors.FAIL + "ERROR" + colors.ENDC + "] string " + colors.BOLD + pr + colors.ENDC + " is not valid, exiting...")
        exit(2)

    c = 0
    penalty = 0
    name = ""
    y = 0
    S = []
    tsp_matrix = []
    df = pd.DataFrame()

    # =========================
    # COSTRUZIONE DEL PROBLEMA (Q): 
    #   - QAP 
    #   - NPP
    #   - TLS
    # =========================
    if QAP:
        # fa scegliere un file di istanza
        _dir, name = getproblem()
        
        # genera la matrice QUBO Q associata partendo dal file
        _Q, penalty, nn, y = matrix.generate_QAP_problem(_dir)
        
        name = name.replace(".txt", "")
        _DIR = generate_file_qap(name)
       
        log_DIR = _DIR.replace("QAP", "QAP_LOG") + ".csv"
    elif NPP:
        while nn <= 0:
            nn = int(input("[" + colors.FAIL + colors.BOLD + "Invalid n" + colors.ENDC + "] Insert n: "))
        
        # genera un vettore casuale S
        _DIR, max_range = generate_file_npp(nn)
        S = vector.generate_S(nn, max_range)
        print("[" + colors.BOLD + colors.OKCYAN + "S" + colors.ENDC + f"] {S}")

        # costruisce il QUBO del problema :
        # Q di f(z) = z^T Q z
        # c
        _Q, c = matrix.generate_QUBO_problem(S)
         
        log_DIR = _DIR.replace("NPP","NPP_LOG") + ".csv"
    else:
        while nn <= 0 or nn > 12:
            nn = int(input("[" + colors.FAIL + colors.BOLD + "Invalid n" + colors.ENDC + "] Insert n: "))

        _DIR = generate_file_tsp(nn)
        log_DIR = _DIR.replace("TSP","TSP_LOG") + ".csv"
        csv_write(DIR = log_DIR, l = ["i", "f'", "f*", "p", "e", "d", "lambda", "z'", "z*"])
        
        # DataFrame dove salvare i risultati
        df = pd.DataFrame(
            columns = ["Solution", "Cost", "Fixed solution", "Fixed cost", "Response time", "Total time", "Response"],
            index = ['Bruteforce', 'D-Wave', 'Hybrid', 'QALS']
        )
        
        # crea la fomrmulazione QUBO/Hamiltoniana del TSP 
        tsp_matrix, qubo = tsp.tsp(nn, _DIR + "_solution.csv" , _DIR[:-1] + "DATA.csv", df) 
        
        _Q = convert_qubo_to_Q(qubo, nn ** 2)
    
    print("\t\t" + colors.BOLD + colors.OKGREEN + "   PROBLEM BUILDED" + colors.ENDC + "\n\n\t\t" + colors.BOLD + colors.OKGREEN + "   START ALGORITHM" + colors.ENDC + "\n")
    
    # =========================
    # ESECUZIONE DI QALS
    # =========================
    start = time.time()

    # Chiama il solver QALS
    # Parametri principali:
    # - d_min, eta, i_max, k, lambda_zero, ...
    # - n è la dimensione del problema
    #     nn^2 per TSP, perché il TSP viene codificato con n^2 variabili binarie
    # - topology = topologia hardware/logica usata
    # - sim = False indica che non sta usando la modalità simulata del solver QALS
    z, r_time = solver.solve(
        d_min = 70,
        eta = 0.01,
        i_max = 10,
        k = 1,
        lambda_zero = 3/2,
        n = nn if NPP or QAP else nn ** 2,
        N = 10,
        N_max = 100,
        p_delta = 0.1,
        q = 0.2,
        topology = 'pegasus',
        Q = _Q,
        log_DIR = log_DIR,
        sim = True
    )

    conv = datetime.timedelta(seconds = int(time.time() - start))

    # Calcola il valore della funzione obiettivo
    min_z = solver.function_f(_Q, z).item()

    print("\t\t\t" + colors.BOLD + colors.OKGREEN + "RESULTS" + colors.ENDC + "\n")

    # stringa con i risultati finali
    string = str()

    # Se il problema è piccolo, stampa la soluzione z
    # altrimenti rimanda al file csv
    if nn < 16:
        string += log_write("Z", z)
    else:
        string += log_write("Z", "Too big to print, see " + _DIR + "_solution.csv for the complete result")

    # Aggiunge il valore minimo trovato della funzione QUBO
    string += log_write("fQ", round(min_z, 2))

    # =========================
    # POST-PROCESSING
    # =========================
    if NPP:
        # calcolata la differenza tra le due partizioni
        diff2 = (c ** 2 + 4 * min_z)

        string += log_write("c", c) + log_write("C", c ** 2) + log_write("DIFF", round(diff2, 2)) + log_write("diff", np.sqrt(diff2))

        csv_write(DIR = _DIR + "_solution.csv", l = [c, c ** 2, diff2, np.sqrt(diff2), S, z, _Q  if nn < 5 else "too big"])   
        csv_write(DIR = _DIR + "_solution.csv", l = ["c", "c ** 2", "diff ** 2", "diff", "S", "z", "Q"])
    elif QAP:
        # y: termine costante / offset della formulazione
        # penalty: penalità usata nella costruzione del QUBO
        # y + min_z: costo complessivo ricostruito della soluzione
        string += log_write("y", y) + log_write("Penalty", penalty) + log_write("Difference", round(y + min_z, 2)) 

        csv_write(DIR = _DIR + "_solution.csv", l = ["problem", "y", "penalty", "difference (y+minimum)", "z", "Q" ])
        csv_write(DIR = _DIR + "_solution.csv", l = [name, y, penalty, y + min_z,np.atleast_2d(z).T, _Q])
    else:
        # Crea un dizionario che rappresenta il risultato di QALS per il TSP
        DW = dict()
        DW['type'] = 'QALS'
        DW['response'] = z

        # Divide il vettore binario z in nn blocchi, ciascuno di lunghezza nn
        # Ogni blocco rappresenta una "riga" della matrice binaria del TSP
        res = np.split(z,nn)

        # Controlla se la soluzione è valida:
        # ogni riga deve avere esattamente un 1
        # e nessuna colonna/posizione deve essere ripetuta
        valid = True
        fix_sol = list()

        for split in res:
            if np.count_nonzero(split == 1) != 1:
                valid = False

            where = str(np.where(split == 1))

            if str(np.where(split == 1)) in fix_sol:
                valid = False
            else:
                fix_sol.append(where)

        # Se la soluzione non è valida, chiama una funzione di correzione
        if (not valid):
            string += "[" + colors.BOLD + colors.FAIL + "ERROR" + colors.ENDC + "] Result is not valid.\n"

            DW['fixsol'] = list(tsp.fix_solution(z, True))
            
            string += "[" + colors.BOLD + colors.WARNING + "VALID" + colors.ENDC + "] Validation occurred \n"
        else:
            DW['fixsol'] = []

        # Calcola il costo della soluzione corretta (se presente)
        DW['fixcost'] = round(float(tsp.calculate_cost(tsp_matrix, DW['fixsol'])), 2)

        # Traduce il vettore binario z nell'ordine dei punti visitati nel tour
        DW['sol'] = tsp.binary_state_to_points_order(z)

        # Calcola il costo del tour originale trovato da QALS
        DW['cost'] = tsp.calculate_cost(tsp_matrix, DW['sol'])

        # Salva tempo risposta del solver e tempo totale
        DW['rtime'] = r_time
        DW['ttime'] = conv

        tsp.write_TSP_csv(df, DW)

        df.to_csv(_DIR + "_solution.csv")
        
    print(string)

if __name__ == '__main__':
    system('cls' if name == 'nt' else 'clear')

    if not exists('outputs'):
        mkdir('outputs')

    try:
        n = int(sys.argv[1])
    except IndexError:
        n = 0
    main(n)