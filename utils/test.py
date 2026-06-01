# %%
import random
import numpy as np

from utils.fitness_functions import fitness
from utils.graph_functions import create_graph

# %%INIT
def init_individual(paths_SH, paths_HD,N,init):
    route_SH_gene = []
    route_HD_gene = []

    for i in range(N):
        s = init[i][0]
        h = init[i][1]
        d = init[i][2]

        route_SH_gene.append(random.randint(0, len(paths_SH[(s,h)])-1))
        route_HD_gene.append(random.randint(0, len(paths_HD[(h,d)])-1))

        prior_S = np.random.permutation(N).tolist()
        prior_H = np.random.permutation(N).tolist()
    return {
        "route_SH": route_SH_gene,
        "route_HD": route_HD_gene,
        "wait_S": [random.randint(0,6)for i in range(N)],
        "wait_H": [random.randint(0,6)for i in range(N)],
        "prior_S": prior_S,
        "prior_H": prior_H
    }

def first_pop(N, ARRIVAL_TIMES, init):
    POP_SIZE = 100
    
    TIME_WINDOW = (0, 48)
    ARRIVAL_TIMES = np.random.randint(0, 24, N)
    
    S = [0,1]
    H = [2,3]
    D = [4,5]
    
    init_start = [random.randint(0,1) for _ in range(N)]   # chọn S1 hoặc S2
    init_hub  = [random.randint(2,3) for _ in range(N)]   # chọn H1 hoặc H2
    init_dest  = [random.randint(4,5) for _ in range(N)]   # chọn D1 hoặc D2
    init = list(zip(init_start, init_hub, init_dest))
    
    del init_start, init_hub, init_dest
    
    # %%MAIN
    G, paths_SH, paths_HD = create_graph()
    pop = [init_individual(paths_SH, paths_HD,N,init) for _ in range(POP_SIZE)]
    best = min(pop, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G))
    best_fit = fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
    
    return best_fit