try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import random
import numpy as np
import time

from utils.GA_functions import crossover, mutate, selection_etilist
from utils.fitness_functions import fitness
from utils.graph_functions import create_graph
from utils.workspace_functions import save_mat

# %%INIT
def init_individual(paths_SH, paths_HD):
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
        "wait_S": [random.randint(0,max_wait)for i in range(N)],
        "wait_H": [random.randint(0,max_wait)for i in range(N)],
        "prior_S": prior_S,
        "prior_H": prior_H
    }

# %%PARAMETERS
N_set = [60, 80, 100]
for N in N_set:
    for trial in range(50):
        # N = 100
        POP_SIZE = 200
        MaxIt= 250
        
        TIME_WINDOW = (0, 48)
        ARRIVAL_TIMES = np.random.randint(0, 24, N)
        max_wait = 6
        
        S = [0,1]
        H = [2,3]
        D = [4,5]
        
        init_start = [random.randint(0,1) for _ in range(N)]   # chọn S1 hoặc S2
        init_hub  = [random.randint(2,3) for _ in range(N)]   # chọn H1 hoặc H2
        init_dest  = [random.randint(4,5) for _ in range(N)]   # chọn D1 hoặc D2
        init = list(zip(init_start, init_hub, init_dest))
        
        BestCostIt = np.zeros(MaxIt)
        
        del init_start, init_hub, init_dest
        
        # %%MAIN (GA LOOP HERE)
        G, paths_SH, paths_HD = create_graph()
        
        # ---- init pop ----
        pop = [init_individual(paths_SH, paths_HD) for _ in range(POP_SIZE)]
        
        best = None
        
        start_loop = time.time()
        for it in range(MaxIt):
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
            BestCostIt[it] = fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
            
            del current_best, next_pop, c1, c2, p1, p2, k, i
            print(f"Case {N}N, Trial {trial}, Iter {it}: {fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):.3f}")
        total_time = (time.time() - start_loop)/60
        
        folder_name = f'data/case_{N}/GA'
        file_name = f'GA_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        del D, H, S, G, it, MaxIt, paths_HD, paths_SH, POP_SIZE, start_loop, TIME_WINDOW
# print("Best individual:", best)
# print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))