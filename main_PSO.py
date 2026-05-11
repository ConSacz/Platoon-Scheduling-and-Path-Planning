try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import networkx as nx
import random
import numpy as np
from utils.fitness_functions import fitness

# ==================================================
# %%PARAMETERS
# ==================================================

N = 100
SWARM_SIZE = 200
NUM_ITER = 250

W  = 0.7
C1 = 1.5
C2 = 1.5

TIME_WINDOW = (0,24)

ARRIVAL_TIMES = np.random.randint(0,12,N)

# ==================================================
# %%INIT
# ==================================================

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
        
        particle["route_SH"].append(
            random.randint(0, len(paths_SH[(s,h)])-1)
        )

        particle["route_HD"].append(
            random.randint(0, len(paths_HD[(h,d)])-1)
        )

        particle["wait_S"].append(random.randint(0,3))
        particle["wait_H"].append(random.randint(0,3))

        particle["v_route_SH"].append(random.uniform(-1,1))
        particle["v_route_HD"].append(random.uniform(-1,1))
        particle["v_wait_S"].append(random.uniform(-1,1))
        particle["v_wait_H"].append(random.uniform(-1,1))

    return particle

# ==================================================
# UPDATE PARTICLE
# ==================================================

def update_particle(p, pbest, gbest, paths_SH, paths_HD):

    for i in range(N):

        r1 = random.random()
        r2 = random.random()

        # ==========================================
        # route SH
        # ==========================================
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
        
        # ==========================================
        # route HD
        # ==========================================
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

        # ==========================================
        # wait S
        # ==========================================
        v = (
            W * p["v_wait_S"][i]
            + C1 * r1 * (pbest["wait_S"][i] - p["wait_S"][i])
            + C2 * r2 * (gbest["wait_S"][i] - p["wait_S"][i])
        )
        p["v_wait_S"][i] = v
        x = int(round(p["wait_S"][i] + v))
        x = max(0, min(3, x))
        p["wait_S"][i] = x
        
        # ==========================================
        # wait H
        # ==========================================
        v = (
            W * p["v_wait_H"][i]
            + C1 * r1 * (pbest["wait_H"][i] - p["wait_H"][i])
            + C2 * r2 * (gbest["wait_H"][i] - p["wait_H"][i])
        )
        p["v_wait_H"][i] = v
        x = int(round(p["wait_H"][i] + v))
        x = max(0, min(3, x))
        p["wait_H"][i] = x
# ==================================================
# %%MAIN
# ==================================================

G, S, H, D, paths_SH, paths_HD = create_graph()

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

# ==================================================
# LOOP
# ==================================================

for it in range(NUM_ITER):

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

    if (
        fitness(candidate, init, ARRIVAL_TIMES,
                paths_SH, paths_HD, G)
        <
        fitness(gbest, init, ARRIVAL_TIMES,
                paths_SH, paths_HD, G)
    ):
        gbest = candidate.copy()

    print(
        f"Iter {it}: "
        f"{fitness(gbest, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):.3f}"
    )

print("Done")