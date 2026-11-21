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
from utils.algorithm_functions import init_individual, copy_ind, population_mean, teaching_phase, learner_phase2

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

        POP_SIZE = 100
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
        
        BestCostIt = np.zeros(np.round(MaxFC/POP_SIZE/2).astype(int)+2)
        
        del origin_idx, destination_idx
        
        # %%MAIN

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
        BestCostIt[0]=best_fit
        print(f"Case {N}N, Trial {trial}, Iter 0: {best_fit:.4f}")
        # %% TLBO LOOP
        start_time = time.time()
        FC=0
        it=0
        while FC < MaxFC:
            it+=1

            for i in range(POP_SIZE):
            # TEACHER PHASE
                teacher = copy_ind(best)
                mean_ind = population_mean(pop, route_options, max_wait)
                
                new_ind = teaching_phase(pop[i], teacher, mean_ind, route_options, max_wait)
                fit_old = fitness(pop[i], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                if fit_new < fit_old:
                    pop[i] = new_ind
                    fit_old = fit_new
                FC+=1
            # LEARNER PHASE
                j = np.random.randint(0, POP_SIZE-1)
                while j == i:
                    j = np.random.randint(0, POP_SIZE-1)
        
                fit_j = fitness(pop[j], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                fit_old = fitness(pop[i], init, ARRIVAL_TIMES, N_trans, RouteLibrary)
                new_ind = learner_phase2(pop[i], pop[j], fit_old, fit_j, route_options, max_wait)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
                if fit_new < fit_old:
                    pop[i] = new_ind
                    fit_old = fit_new
                FC+=1
            # GLOBAL BEST
            current_best = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
            )
        
            current_fit = fitness(current_best, init, ARRIVAL_TIMES, N_trans, RouteLibrary)
        
            if current_fit < best_fit:
                best = copy_ind(current_best)
                best_fit = current_fit
            BestCostIt[it] = best_fit
            print(f"Case {N}N, Trial {trial}, Iter {it}: {best_fit:.4f}")
        total_time = (time.time() - start_time)/60
        print (f"runtime: {total_time:.4f}min")
        
        folder_name = f'data/case_{N}/TLBO'
        file_name = f'TLBO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        # del current_best, current_fit, it, fit_i, fit_j, fit_new, fit_old, i, j, new_ind
        # del MaxIt, mean_ind, POP_SIZE, start_time, teacher
