try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import networkx as nx
import random
import numpy as np

# ======================
# %%PARAMETERS
# ======================

N = 50
POP_SIZE = 1000
NUM_GEN = 500
CROSS_RATE = 0.8
MUT_RATE = 0.2

TIME_WINDOW = (0, 24)
ARRIVAL_TIMES = np.random.randint(0, 12, N)

S = [0,1]
H = [2,3]
D = [4,5]
init_start = [random.randint(0,1) for _ in range(N)]   # chọn S1 hoặc S2
init_hub  = [random.randint(2,3) for _ in range(N)]   # chọn H1 hoặc H2
init_dest  = [random.randint(4,5) for _ in range(N)]   # chọn D1 hoặc D2

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
        s = init_start[i]
        h = init_hub[i]
        d = init_dest[i]

        route_SH_gene.append(random.randint(0, len(paths_SH[(s,h)])-1))
        route_HD_gene.append(random.randint(0, len(paths_HD[(h,d)])-1))

    return {
        "route_SH": route_SH_gene,
        "route_HD": route_HD_gene,
        "wait_S": [random.randint(0,3)for i in range(N)],
        "wait_H": [random.randint(0,3)for i in range(N)]
    }

def decode(ind, paths_SH, paths_HD, G):
    routes = []
    times_SH = []
    times_HD = []

    for i in range(N):
        s = init_start[i]
        h = init_hub[i]
        d = init_dest[i]

        path1 = paths_SH[(s,h)][ind["route_SH"][i]]
        path2 = paths_HD[(h,d)][ind["route_HD"][i]]

        # tính time đoạn S -> H
        t1 = 0
        for u, v in zip(path1[:-1], path1[1:]):
            t1 += G[u][v]["weight"]

        # tính time đoạn H -> D
        t2 = 0
        for u, v in zip(path2[:-1], path2[1:]):
            t2 += G[u][v]["weight"]

        full_path = path1 + path2[1:]

        routes.append(full_path)
        times_SH.append(t1)
        times_HD.append(t2)

    return routes, times_SH, times_HD

def get_platoons(routes, S_depart_times, H_depart_times):
    platoons = []

    # ---- START platoon ----
    start_dict = {}
    for i in range(N):
        key = (routes[i][0], routes[i][len(routes[i])//2], S_depart_times[i])
        start_dict.setdefault(key, []).append(i)

    platoons.extend(start_dict.values())

    # ---- HUB platoon ----
    hub_dict = {}
    for i in range(N):
        route = routes[i]

        # hub = node giữa (vì luôn S->H->D)
        hub = route[len(route)//2]
        des = route[-1]

        arrival = H_depart_times[i]

        # làm tròn để gom nhóm (quan trọng!)
        key = (hub, des, arrival)

        hub_dict.setdefault(key, []).append(i)

    platoons.extend(hub_dict.values())

    return start_dict, hub_dict
# ======================
# FITNESS
# ======================
def fitness(ind, paths_SH, paths_HD, G):
    routes, times_SH, times_HD = decode(ind, paths_SH, paths_HD, G)
    S_depart_times = ind["wait_S"] + ARRIVAL_TIMES
    H_depart_times = S_depart_times + times_SH + ind["wait_H"]
    start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times)

    total_fuel = 0
    total_wait = 0

    # platoon
    platoon_size_S = {}
    platoon_size_H = {}

    # START platoon
    for group in start_dict.values():
        size = len(group)
        for i in group:
            platoon_size_S[i] = size

    # HUB platoon
    for group in hub_dict.values():
        size = len(group)
        for i in group:
            platoon_size_H[i] = size

    for i in range(N):
        route = routes[i]

        size_S = platoon_size_S.get(i, 1)
        size_H = platoon_size_H.get(i, 1)

        if size_S <= 4:
            reduction_S = 1 - 0.1 * (size_S - 1)
        else:
            reduction_S = 1
            
        if size_H <= 4:    
            reduction_H = 1 - 0.1 * (size_H - 1)
        else:
            reduction_H = 1
            
        # tìm vị trí hub
        # vì route = S -> ... -> H -> ... -> D
        # hub là node chung của 2 path
        # đơn giản: node tại index chuyển tiếp
        split_idx = len(route) // 2   # ok vì bạn build dạng SH + HD

        # ---- đoạn S -> H ----
        for u, v in zip(route[:split_idx], route[1:split_idx+1]):
            w = G[u][v]["weight"]
            total_fuel += w * reduction_S

        # ---- đoạn H -> D ----
        for u, v in zip(route[split_idx:], route[split_idx+1:]):
            w = G[u][v]["weight"]
            total_fuel += w * reduction_H

        # ======================
        # waiting time
        # ======================
        total_wait += max(0, S_depart_times[i] - ARRIVAL_TIMES[i])
        total_wait += max(0, H_depart_times[i] - (S_depart_times[i] + times_SH[i]))

    return 0.7 * total_fuel + 0.3 * total_wait

# ======================
# GA OPERATORS
# ======================

def selection_etilist(pop, paths_SH, paths_HD, G):
    # sort population theo fitness tăng dần (min tốt hơn)
    sorted_pop = sorted(
        pop,
        key=lambda ind: fitness(ind, paths_SH, paths_HD, G)
    )
    return sorted_pop[:POP_SIZE]

def selection_tournament(pop, paths_SH, paths_HD, G):
    k = 3
    new_pop = []
    for _ in range(len(pop)):
        candidates = random.sample(pop, k)
        best = min(candidates, key=lambda ind: fitness(ind, paths_SH, paths_HD, G))
        new_pop.append(best)
    return new_pop

def crossover(p1, p2):
    point = random.randint(0, N-1)

    c1 = {
        "route_SH": p1["route_SH"][:point] + p2["route_SH"][point:],
        "route_HD": p1["route_HD"][:point] + p2["route_HD"][point:],
        "wait_S": p1["wait_S"][:point] + p2["wait_S"][point:],
        "wait_H": p1["wait_H"][:point] + p2["wait_H"][point:],
    }

    c2 = {
        "route_SH": p2["route_SH"][:point] + p1["route_SH"][point:],
        "route_HD": p2["route_HD"][:point] + p1["route_HD"][point:],
        "wait_S": p2["wait_S"][:point] + p1["wait_S"][point:],
        "wait_H": p2["wait_H"][:point] + p1["wait_H"][point:],
    }

    return c1, c2

def mutate(ind, paths_SH, paths_HD):
    for i in range(N):
        s = init_start[i]
        h = init_hub[i]
        d = init_dest[i]
        if random.random() < MUT_RATE:
            ind["route_SH"][i] = random.randint(0, len(paths_SH[(s,h)])-1)
            ind["route_HD"][i] = random.randint(0, len(paths_HD[(h,d)])-1)
            
        if random.random() < MUT_RATE:
            ind["wait_S"][i] = random.randint(0,3)
            ind["wait_H"][i] = random.randint(0,3)

    return ind

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
        next_pop.append(mutate(c1, paths_SH, paths_HD))
        next_pop.append(mutate(c2, paths_SH, paths_HD))

    pop = next_pop
    pop = selection_etilist(pop, paths_SH, paths_HD, G)
    
    # ---- evaluation ----
    current_best = min(pop, key=lambda ind: fitness(ind, paths_SH, paths_HD, G))

    if best is None or fitness(current_best, paths_SH, paths_HD, G) < fitness(best, paths_SH, paths_HD, G):
        best = current_best
    del current_best, next_pop, c1, c2, p1, p2, k, i
    print(f"Gen {gen}: {fitness(best, paths_SH, paths_HD, G):.3f}")

# print("Best individual:", best)
# print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))