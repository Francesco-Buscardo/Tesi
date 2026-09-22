from random import SystemRandom
import networkx as nx

import dwave_networkx as dnx

random = SystemRandom()

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
    