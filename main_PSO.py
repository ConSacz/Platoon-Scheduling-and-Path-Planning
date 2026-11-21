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
from utils.fitness_functions import weighted_fitness
from utils.workspace_functions import save_mat
from utils.algorithm_functions import PSO_init_individual, update_particle

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
        W  = 0.8
        C1 = 1.5
        C2 = 1.5
        
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
            PSO_init_individual(N, route_options, N_trans, max_wait)
            for _ in range(POP_SIZE)
        ]
        
        # pbest
        pbest = []
        
        for ind in pop:
            pbest.append({
                "route": ind["route"].copy(),
                "wait": ind["wait"].copy(),
                "prior": ind["prior"].copy()
            })
        
        # gbest
        gbest = min(
            pbest,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        )
        best_fit = fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        BestCostIt[0] = best_fit
        print(f"Case {N}N, Trial {trial}, Iter 0: {best_fit:.4f}")
        # %% PSO LOOP
        start_time = time.time()
        FC=0
        it=0
        while FC < MaxFC:
            it+=1
        
            for idx, ind in enumerate(pop):
                ind = update_particle(ind, pbest[idx], gbest, W, C1, C2, route_options, max_wait)
                
                current_fit = fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                pbest_fit = fitness(pbest[idx], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                # update pbest
                if current_fit < pbest_fit:
                    pbest[idx] = {
                        "route": ind["route"].copy(),
                        "wait": ind["wait"].copy(),
                        "prior": ind["prior"].copy()
                    }
                FC+=1
            # update gbest
            candidate = min(
                pbest,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
            can_fit = fitness(candidate, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
            if can_fit < best_fit:
                gbest = candidate.copy()
                best_fit = can_fit
            
            BestCostIt[it] = best_fit = can_fit
            
            print(f"Case {N}N, Trial {trial}, Iter {it}: {fitness(gbest, init, ARRIVAL_TIMES, N_trans, RouteLibrary):.3f}")
        total_time = (time.time() - start_time)/60
        print (f"runtime: {total_time:.4f}min")
        
        folder_name = f'data/case_{N}/PSO'
        file_name = f'PSO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, gbest, total_time)
        
        # del C1, C2, candidate, current_fit, idx, it, MaxIt, ind, pbest, pbest_fit
        # del start_loop, POP_SIZE, TIME_WINDOW, W
