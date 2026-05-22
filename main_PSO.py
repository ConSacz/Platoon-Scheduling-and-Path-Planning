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
    prior_S = np.random.permutation(N).tolist()
    prior_H = np.random.permutation(N).tolist()
    particle = {
        "route_SH": [],
        "route_HD": [],
        "wait_S": [],
        "wait_H": [],

        "v_route_SH": [],
        "v_route_HD": [],
        "v_wait_S": [],
        "v_wait_H": [],
        "prior_S": prior_S,
        "prior_H": prior_H
    }

    for i in range(N):
        s,h,d = init[i]
    
        particle["route_SH"].append(random.randint(0, len(paths_SH[(s,h)])-1))
        particle["route_HD"].append(random.randint(0, len(paths_HD[(h,d)])-1))

        particle["wait_S"].append(random.randint(0,max_wait))
        particle["wait_H"].append(random.randint(0,max_wait))

        particle["v_route_SH"].append(random.uniform(-1,1))
        particle["v_route_HD"].append(random.uniform(-1,1))
        particle["v_wait_S"].append(random.uniform(-1,1))
        particle["v_wait_H"].append(random.uniform(-1,1))
    return particle

# UPDATE PARTICLE
def update_particle(p, pbest, gbest, paths_SH, paths_HD):
    for i in range(N):
        r1 = random.random()
        r2 = random.random()
        
        # route SH
        v = (
            W * p["v_route_SH"][i]
            + C1 * r1 * (pbest["route_SH"][i] - p["route_SH"][i])
            + C2 * r2 * (gbest["route_SH"][i] - p["route_SH"][i])
        )
        p["v_route_SH"][i] = v
        x = p["route_SH"][i] + v
        s,h,d = init[i]
        max_idx = len(paths_SH[(s,h)]) - 1
        x = int(round(x))
        x = max(0, min(max_idx, x))
        p["route_SH"][i] = x
        
        # route HD
        v = (
            W * p["v_route_HD"][i]
            + C1 * r1 * (pbest["route_HD"][i] - p["route_HD"][i])
            + C2 * r2 * (gbest["route_HD"][i] - p["route_HD"][i])
        )
        p["v_route_HD"][i] = v
        x = p["route_HD"][i] + v
        max_idx = len(paths_HD[(h,d)]) - 1
        x = int(round(x))
        x = max(0, min(max_idx, x))
        p["route_HD"][i] = x
        
        # wait S
        v = (
            W * p["v_wait_S"][i]
            + C1 * r1 * (pbest["wait_S"][i] - p["wait_S"][i])
            + C2 * r2 * (gbest["wait_S"][i] - p["wait_S"][i])
        )
        p["v_wait_S"][i] = v
        x = int(round(p["wait_S"][i] + v))
        x = max(0, min(3, x))
        p["wait_S"][i] = x

        # wait H
        v = (
            W * p["v_wait_H"][i]
            + C1 * r1 * (pbest["wait_H"][i] - p["wait_H"][i])
            + C2 * r2 * (gbest["wait_H"][i] - p["wait_H"][i])
        )
        p["v_wait_H"][i] = v
        x = int(round(p["wait_H"][i] + v))
        x = max(0, min(3, x))
        p["wait_H"][i] = x
        
# %%PARAMETERS
N_set = [60, 80, 100]
for N in N_set:
    for trial in range(50):
        # N = 100
        SWARM_SIZE = 200
        MaxIt = 250
        
        W  = 0.7
        C1 = 1.5
        C2 = 1.5
        
        TIME_WINDOW = (0,48)
        ARRIVAL_TIMES = np.random.randint(0,24,N)
        max_wait = 6
        
        # %%INIT
        S = [0,1]
        H = [2,3]
        D = [4,5]
        
        init_start = [random.randint(0,1) for _ in range(N)]   # chọn S1 hoặc S2
        init_hub  = [random.randint(2,3) for _ in range(N)]   # chọn H1 hoặc H2
        init_dest  = [random.randint(4,5) for _ in range(N)]   # chọn D1 hoặc D2
        init = list(zip(init_start, init_hub, init_dest))
        
        del init_start, init_hub, init_dest
        
        BestCostIt = np.zeros(MaxIt)
        
        G, paths_SH, paths_HD = create_graph()
        
        pop = [
            init_individual(paths_SH, paths_HD)
            for _ in range(SWARM_SIZE)
        ]
        
        # pbest
        pbest = []
        
        for p in pop:
            pbest.append({
                "route_SH": p["route_SH"].copy(),
                "route_HD": p["route_HD"].copy(),
                "wait_S": p["wait_S"].copy(),
                "wait_H": p["wait_H"].copy(),
                "prior_S": p["prior_S"].copy(),
                "prior_H": p["prior_H"].copy()
            })
        
        # gbest
        gbest = min(
            pbest,
            key=lambda ind:
            fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        )
        
        # %% PSO LOOP
        start_loop = time.time()
        for it in range(MaxIt):
        
            for idx, p in enumerate(pop):
                update_particle(p, pbest[idx], gbest, paths_SH, paths_HD)
                current_fit = fitness( p, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
                pbest_fit = fitness( pbest[idx], init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
        
                # update pbest
                if current_fit < pbest_fit:
                    pbest[idx] = {
                        "route_SH": p["route_SH"].copy(),
                        "route_HD": p["route_HD"].copy(),
                        "wait_S": p["wait_S"].copy(),
                        "wait_H": p["wait_H"].copy(),
                        "prior_S": p["prior_S"].copy(),
                        "prior_H": p["prior_H"].copy()
                    }
        
            # update gbest
            candidate = min(
                pbest,
                key=lambda ind:
                fitness(ind, init, ARRIVAL_TIMES,
                        paths_SH, paths_HD, G)
            )
        
            if (fitness(candidate, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
                <
                fitness(gbest, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)):
                gbest = candidate.copy()
            
            BestCostIt[it] = fitness(gbest, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
            
            print(f"Case {N}N, Trial {trial}, Iter {it}: fitness(gbest, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):.3f")
        total_time = (time.time() - start_loop)/60
        
        folder_name = f'data/case_{N}/PSO'
        file_name = f'PSO_{trial}.mat'
        save_mat(folder_name, file_name, ARRIVAL_TIMES, init, pop, BestCostIt, gbest, total_time)
        
        del C1, C2, candidate, current_fit, D, H, S, G, idx, it, MaxIt, p, paths_HD, paths_SH, pbest, pbest_fit
        del start_loop, SWARM_SIZE, TIME_WINDOW, W
