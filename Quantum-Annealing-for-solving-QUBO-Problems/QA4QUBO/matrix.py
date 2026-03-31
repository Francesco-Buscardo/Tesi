import numpy as np
from random import SystemRandom
import networkx as nx

from dwave_networkx.algorithms.tsp import traveling_salesperson_qubo
import dwave_networkx as dnx

random = SystemRandom()

# ! NON USATA
# def tsp(n):
#     G = nx.Graph()
#     G = nx.complete_graph(n)
#     for (u, v) in G.edges():
#         G.edges[u,v]['weight'] = round(random.random()*100,2)
    
#     #G.add_edge(node1,node2,weight=round(random.random()*100, 2))
    
#     d = traveling_salesperson_qubo(G) 
    
#     indexes = dict()
#     it = 0
#     for i in range(n):
#         for j in range(n):
#             indexes[(i,j)] = it
#             it += 1

#     matrix = np.zeros((n**2, n**2), dtype = np.float64)

#     for key_1, key_2 in d:
#         matrix[indexes[key_1],indexes[key_2]] = d[key_1,key_2]

#     return G, matrix

def generate_QUBO_problem(S):
    # ? Generate a QUBO problem from a vector S

    n = len(S)
    
    # somma di S
    c = 0
    for i in range(n):
        c += S[i]

    col_max = 0
    col = 0
    QUBO = np.zeros((n, n))

    for row in range(n):
        col_max += 1
        while col < col_max:
            if row == col:
                QUBO[row][col] = S[row] * (S[row] - c)
            else:
                QUBO[row][col] = S[row] * S[col]
                QUBO[col][row] = QUBO[row][col]
            col += 1
        col = 0

    return QUBO, c

def read_integers(filename:str):
    with open(filename) as f:
        return [int(elem) for elem in f.read().split()]

def generate_QAP_problem(file):
    # il file contiene:
    # - n
    # - P
    # - L
    file_it = iter(read_integers(file))

    # n: dimensione del problema
    n = next(file_it)

    # P: matrice dei flussi/interazioni tra facility
    P = [[next(file_it) for j in range(n)] for i in range(n)]

    # L: matrice delle distanze/costi tra location
    L = [[next(file_it) for j in range(n)] for i in range(n)]
    
    # prodotto di Kronecker tra P ed L
    # P ⊗ L: https://en.wikipedia.org/wiki/Kronecker_product
    Q = np.kron(P, L)
    
    # penalità
    pen = (Q.max() * 2.25)

    # Costruisce la matrice QUBO
    matrix = qubo_qap(flow = np.array(P), distance = np.array(L), penalty = pen)

    # Calcola il termine costante y introdotto dalla formulazione con penalità
    y = pen * (len(P) + len(L))

    return matrix, pen, len(matrix), y
    
def qubo_qap(flow: np.ndarray, distance: np.ndarray, penalty):
    # ? Quadratic Assignment Problem (QAP)
    
    n = len(flow)
    q = np.einsum("ij,kl->ikjl", flow, distance).astype(float)

    i = range(len(q))

    q[i, :, i, :] += penalty
    q[:, i, :, i] += penalty
    q[i, i, i, i] -= 4 * penalty
    
    return q.reshape(n ** 2, n ** 2)


# Generano tologia Chimera/Pegasus per A    
def generate_chimera(n):
    G = dnx.chimera_graph(16)

    tmp = nx.to_dict_of_lists(G)

    rows = []
    cols = []
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in tmp[i]:
            if(j < n):
                rows.append(i)
                cols.append(j)

    return list(zip(rows, cols))

def generate_pegasus(n):
    G = dnx.pegasus_graph(16)

    tmp = nx.to_numpy_array(G)
    
    rows = []
    cols = []
           
    for i in range(n):
        rows.append(i)
        cols.append(i)
        for j in range(n):
            if(tmp.item(i,j)):
                rows.append(i)
                cols.append(j)
      
    return list(zip(rows, cols))
    