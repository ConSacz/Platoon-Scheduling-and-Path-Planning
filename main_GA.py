# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import numpy as np
import time
import pickle

from map_generation.locations import locations
from utils.algorithm_functions import crossover, mutate, selection_etilist, init_individual
from utils.fitness_functions import weighted_fitness
from utils.workspace_functions import save_mat

# FITNESS FUNCTION
def fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary):
    return weighted_fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)

# %% MAP GENERATION

# N_set = [600, 800, 1000]
#N_set = [60, 80, 100]
N_set = [100]
Trial = 1

with open("map_generation/route_library.pkl","rb") as f: 
    RouteLibrary = pickle.load(f)

N_trans = 3
route_options = len(next(iter(RouteLibrary.values())))

# %% MAIN
for N in N_set:
    for trial in range(Trial):
        np.random.seed(trial)
        
        POP_SIZE = 200
        MaxFC = 50000
        
        max_wait = 4
        
        ORIGINS = [
            name for name, info in locations.items()
            if info["type"] == "start"
        ]
        
        DESTINATIONS = [
            name for name, info in locations.items()
            if info["type"] == "destination"
        ]
        
        origin_idx = np.random.randint(0, len(ORIGINS), N)
        destination_idx = np.random.randint(0, len(DESTINATIONS), N)
        ARRIVAL_TIMES = np.random.randint(0, round(N/(len(ORIGINS)*len(DESTINATIONS))/2), N)
        
        init = [
            (ORIGINS[o], DESTINATIONS[d])
            for o, d in zip(origin_idx, destination_idx)
        ]
        
        BestCostIt = np.zeros(np.round(MaxFC/POP_SIZE).astype(int)+2)
        
        del origin_idx, destination_idx
        # ---- init pop ----
        pop = [
            init_individual(N, route_options, N_trans, max_wait)
            for _ in range(POP_SIZE)
        ]
        
        best = min(
            pop,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        )
        
        best_fit = fitness(best, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        BestCostIt[0] = best_fit
        print(f"Case {N}N, Trial {trial}, Iter 0: {best_fit:.4f}")
        # %% GA LOOP
        start_time = time.time()
        FC=0
        it=0
        while FC < MaxFC:
            it+=1
            # ---- crossover + mutation ----
            next_pop = []
            for i in range(0, POP_SIZE):
                k = np.random.randint(0,POP_SIZE-1)
                p1, p2 = pop[i], pop[k]
                c1, c2 = crossover(p1, p2)
                next_pop.append(mutate(c1, route_options, N_trans, max_wait))
                next_pop.append(mutate(c2, route_options, N_trans, max_wait))
        
            pop = next_pop
            pop = selection_etilist(pop, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            FC+=POP_SIZE
            # ---- evaluation ----
            current_best = min(pop, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary))
            current_fit = fitness(current_best, init, ARRIVAL_TIMES,  N_trans, RouteLibrary)
            if current_fit < best_fit:
                best = current_best
                best_fit = current_fit
                
            BestCostIt[it] = best_fit

            print(f"Case {N}N, Trial {trial}, Iter {it}: {fitness(best, init, ARRIVAL_TIMES, N_trans, RouteLibrary):.3f}")
        total_time = (time.time() - start_time)/60
        print (f"runtime: {total_time:.4f}min")
        
        folder_name = f'data/case_{N}/GA'
        file_name = f'GA_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        # del it, MaxIt, N_trans, RouteLibrary, POP_SIZE, start_loop, TIME_WINDOW
# print("Best individual:", best)
# print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))