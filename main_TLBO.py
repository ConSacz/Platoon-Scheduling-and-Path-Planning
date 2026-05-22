try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import random
import numpy as np
import time

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

# COPY
def copy_ind(ind):
    return {
        "route_SH": ind["route_SH"].copy(),
        "route_HD": ind["route_HD"].copy(),
        "wait_S": ind["wait_S"].copy(),
        "wait_H": ind["wait_H"].copy(),
        "prior_S": ind["prior_S"].copy(),
        "prior_H": ind["prior_H"].copy()
    }

# CLAMP
def clamp_individual(ind, paths_SH, paths_HD):
    for i in range(N):
        s,h,d = init[i]
        max_SH = len(paths_SH[(s,h)]) - 1
        max_HD = len(paths_HD[(h,d)]) - 1

        ind["route_SH"][i] = int(round(ind["route_SH"][i]))
        ind["route_HD"][i] = int(round(ind["route_HD"][i]))

        ind["wait_S"][i] = int(round(ind["wait_S"][i]))
        ind["wait_H"][i] = int(round(ind["wait_H"][i]))

        ind["route_SH"][i] = max(0, min(max_SH, ind["route_SH"][i]))
        ind["route_HD"][i] = max(0, min(max_HD, ind["route_HD"][i]))
        ind["wait_S"][i] = max(0, min(3, ind["wait_S"][i]))
        ind["wait_H"][i] = max(0, min(3, ind["wait_H"][i]))

# TEACHING PHASE
def teaching_phase(student, teacher, mean_ind, paths_SH, paths_HD):
    TF = random.randint(1,2)
    new_ind = copy_ind(student)
    for i in range(N):
        r = random.random()
        # route SH
        new_ind["route_SH"][i] += (r * (teacher["route_SH"][i] - TF * mean_ind["route_SH"][i]))
        # route HD
        new_ind["route_HD"][i] += (r * (teacher["route_HD"][i] - TF * mean_ind["route_HD"][i]))
        # wait S
        new_ind["wait_S"][i] += (r * (teacher["wait_S"][i] - TF * mean_ind["wait_S"][i]))
        # wait H
        new_ind["wait_H"][i] += (r * (teacher["wait_H"][i] - TF * mean_ind["wait_H"][i]))
    clamp_individual(new_ind, paths_SH, paths_HD)
    return new_ind

# LEARNER PHASE
def learner_phase(ind1, ind2, fit1, fit2, paths_SH, paths_HD):
    new_ind = copy_ind(ind1)
    for i in range(N):
        r = random.random()
        if fit1 < fit2:
            new_ind["route_SH"][i] += (r * ( ind1["route_SH"][i] - ind2["route_SH"][i]))
            new_ind["route_HD"][i] += (r * (ind1["route_HD"][i] - ind2["route_HD"][i]))
            new_ind["wait_S"][i] += (r * (ind1["wait_S"][i] - ind2["wait_S"][i]))
            new_ind["wait_H"][i] += (r * (ind1["wait_H"][i] - ind2["wait_H"][i]))
        else:
            new_ind["route_SH"][i] += (r * ( ind2["route_SH"][i] -ind1["route_SH"][i]))
            new_ind["route_HD"][i] += (r * (ind2["route_HD"][i] - ind1["route_HD"][i]))
            new_ind["wait_S"][i] += (r * (ind2["wait_S"][i] - ind1["wait_S"][i]))
            new_ind["wait_H"][i] += (r * (ind2["wait_H"][i] - ind1["wait_H"][i]))
    clamp_individual(new_ind, paths_SH, paths_HD)
    return new_ind

# MEAN
def population_mean(pop):
    mean_ind = {
        "route_SH": [],
        "route_HD": [],
        "wait_S": [],
        "wait_H": []
    }
    for key in mean_ind.keys():
        arr = np.array([p[key] for p in pop])
        mean_ind[key] = np.mean(arr, axis=0)
    return mean_ind

# =========================================================
# %%PARAMETERS
N_set = [60, 80, 100]
for N in N_set:
    for trial in range(50):
        # N = 100
        POP_SIZE = 100
        MaxIt = 250
        
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
        
        # %%MAIN
        G, paths_SH, paths_HD = create_graph()
        
        pop = [
            init_individual(paths_SH, paths_HD)
            for _ in range(POP_SIZE)
        ]
        
        best = min(
            pop,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES,
                    paths_SH, paths_HD, G)
        )
        
        best_fit = fitness(best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
        # %% TLBO LOOP
        start_time = time.time()
        for it in range(MaxIt):
            # TEACHER
            teacher = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES,
                        paths_SH, paths_HD, G)
            )
        
            mean_ind = population_mean(pop)
        
            # TEACHING PHASE
            for i in range(POP_SIZE):
        
                new_ind = teaching_phase(pop[i], teacher, mean_ind, paths_SH, paths_HD)
        
                fit_old = fitness(pop[i], init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
                if fit_new < fit_old:
                    pop[i] = new_ind
        
            # LEARNER PHASE
            for i in range(POP_SIZE):
        
                j = random.randint(0, POP_SIZE-1)
        
                while j == i:
                    j = random.randint(0, POP_SIZE-1)
        
                fit_i = fitness(pop[i], init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
                fit_j = fitness(pop[j], init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
                new_ind = learner_phase(pop[i], pop[j], fit_i, fit_j, paths_SH, paths_HD)
                fit_new = fitness(new_ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
                if fit_new < fit_i:
                    pop[i] = new_ind
            
            # GLOBAL BEST
            current_best = min(
                pop,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES,
                        paths_SH, paths_HD, G)
            )
        
            current_fit = fitness(current_best, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
            if current_fit < best_fit:
                best = copy_ind(current_best)
                best_fit = current_fit
            BestCostIt[it] = best_fit
            print(f"Case {N}N, Trial {trial}, Iter {it}: {best_fit:.4f}")
        total_time = (time.time() - start_time)/60
        
        folder_name = f'data/case_{N}/TLBO'
        file_name = f'TLBO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, best, total_time)
        
        del current_best, current_fit, D, H, S, it, fit_i, fit_j, fit_new, fit_old, G, i, j, new_ind
        del MaxIt, mean_ind, paths_HD, paths_SH, POP_SIZE, start_time, teacher, TIME_WINDOW
