try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import networkx as nx
import random
import numpy as np

from utils.GA_functions import crossover, mutate, selection_etilist
from utils.fitness_functions import fitness

# ======================
# %%PARAMETERS
# ======================

N = 100
POP_SIZE = 1000
NUM_GEN = 250

TIME_WINDOW = (0, 24)
ARRIVAL_TIMES = np.random.randint(0, 12, N)

S = [0,1]
H = [2,3]
D = [4,5]

init_start = [random.randint(0,1) for _ in range(N)]   # chọn S1 hoặc S2
init_hub  = [random.randint(2,3) for _ in range(N)]   # chọn H1 hoặc H2
init_dest  = [random.randint(4,5) for _ in range(N)]   # chọn D1 hoặc D2
init = list(zip(init_start, init_hub, init_dest))

del init_start, init_hub, init_dest
# ======================
# %%GRAPH
# ======================
def create_graph():
    G = nx.Graph()

    # node layout
    # 0,1: start
    # 2,3: hub
    # 4,5: destination
    # 6-11: intermediate
    G.add_nodes_from(range(22))

    # paths S -> H
    paths_SH = {
        (0,2): [[0,6,2], [0,7,2]],
        (0,3): [[0,10,3], [0,11,3]],
        (1,2): [[1,18,2], [1,19,2]],
        (1,3): [[1,14,3], [1,15,3]]
    }

    # paths H -> D
    paths_HD = {
        (2,4): [[2,8,4], [2,9,4]],
        (2,5): [[2,12,5], [2,13,5]],
        (3,4): [[3,20,4], [3,21,4]],
        (3,5): [[3,16,5], [3,17,5]]
    }

    def add_paths_with_fixed_weights():
        edge_weights = {
            # S -> H
            (0,6): 1.0, (6,2): 1.0,
            (0,7): 2.0, (7,2): 1.0,
            (0,10): 2.0, (10,3): 1.0,
            (0,11): 2.0, (11,3): 2.0,
    
            (1,18): 2.0, (18,2): 1.0,
            (1,19): 3.0, (19,2): 1.0,
            (1,14): 1.0, (14,3): 1.0,
            (1,15): 2.0, (15,3): 1.0,
    
            # H -> D
            (2,8): 2.0, (8,4): 1.0,
            (2,9): 2.0, (9,4): 2.0,
            (2,12): 2.0, (12,5): 2.0,
            (2,13): 2.0, (13,5): 3.0,
    
            (3,20): 2.0, (20,4): 2.0,
            (3,21): 3.0, (21,4): 2.0,
            (3,16): 2.0, (16,5): 1.0,
            (3,17): 2.0, (17,5): 2.0
        }
    
        for (u, v), w in edge_weights.items():
            G.add_edge(u, v, weight=w)
    add_paths_with_fixed_weights()
    return G, S, H, D, paths_SH, paths_HD
# ======================
# %%INIT
# ======================
def init_individual(paths_SH, paths_HD):

    route_SH_gene = []
    route_HD_gene = []

    for i in range(N):
        s = init[i][0]
        h = init[i][1]
        d = init[i][2]

        route_SH_gene.append(random.randint(0, len(paths_SH[(s,h)])-1))
        route_HD_gene.append(random.randint(0, len(paths_HD[(h,d)])-1))

    return {
        "route_SH": route_SH_gene,
        "route_HD": route_HD_gene,
        "wait_S": [random.randint(0,3)for i in range(N)],
        "wait_H": [random.randint(0,3)for i in range(N)]
    }
# ======================
# %%MAIN (GA LOOP HERE)
# ======================

G, S, H, D, paths_SH, paths_HD = create_graph()

# ---- init pop ----
pop = [init_individual(paths_SH, paths_HD) for _ in range(POP_SIZE)]

best = None

for gen in range(NUM_GEN):

    # ---- selection ----
    

    # ---- crossover + mutation ----
    next_pop = []
    for i in range(0, POP_SIZE):
        k = np.random.randint(0,N-1)
        p1, p2 = pop[i], pop[k]

        c1, c2 = crossover(p1, p2)
        next_pop.append(mutate(c1, init, paths_SH, paths_HD))
        next_pop.append(mutate(c2, init, paths_SH, paths_HD))

    pop = next_pop
    pop = selection_etilist(pop, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
    
    # ---- evaluation ----
    current_best = min(pop, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G))

    if best is None or fitness(current_best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G) < fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
        best = current_best
    del current_best, next_pop, c1, c2, p1, p2, k, i
    print(f"Gen {gen}: {fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):.3f}")

# print("Best individual:", best)
# print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))