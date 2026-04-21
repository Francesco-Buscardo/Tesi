import time as tm

def annealer(theta, sampler, k, time=False):
    start = 0

    if time:
        start = tm.time()
    
    if sampler.__class__.__name__ == "ExactSolver":
        response = sampler.sample_qubo(theta)
    else:
        response = sampler.sample_qubo(theta, num_reads = k)   
    
    if time:
        print(f"Time: {tm.time() - start}")
    
    return list(response.first.sample.values())

def hybrid(theta, sampler):
    response = sampler.sample_qubo(theta)

    return list(response.first.sample.values())
    