# try:
#     from IPython import get_ipython
#     get_ipython().run_line_magic('reset', '-f')
# except:
#     pass
# %%
import random
import numpy as np

# ======================
# %%PARAMETERS
# ======================

N = 20
POP_SIZE = 1000
NUM_GEN = 250
CROSS_RATE = 0.8
MUT_RATE = 0.2

TIME_WINDOW = (0, 24)
ARRIVAL_TIMES = np.random.randint(0, 18, N)

# ======================
# %%GRAPH
# ======================

def create_graph():
    A = 0
    B = 8
    C = 9

    # 5 paths to B
    paths_to_B = [
        [0,1,8],
        [0,2,8],
        [0,3,8],
    ]
    # 5 paths to C
    paths_to_C = [
        [0,4,9],
        [0,5,9],
        [0,6,9],
    ]

    return A, B, C, paths_to_B, paths_to_C
    
# ======================
# %%INIT
# ======================

def init_individual(paths_to_B, paths_to_C, DESTINATIONS):
    route_gene = []

    for i in range(N):
        if DESTINATIONS[i] == 'B':
            route_gene.append(random.randint(0, len(paths_to_B)-1))
        else:
            route_gene.append(random.randint(0, len(paths_to_C)-1))

    return {
        "route_gene": route_gene,
        "depart_gene": [
            random.randint(max(ARRIVAL_TIMES[i], TIME_WINDOW[0]), TIME_WINDOW[1])
            for i in range(N)
        ]
    }

def init_population(paths_to_B, paths_to_C, DESTINATIONS):
    return [init_individual(paths_to_B, paths_to_C, DESTINATIONS) for _ in range(POP_SIZE)]

# ======================
# DECODE
# ======================

def decode(individual, paths_to_B, paths_to_C, DESTINATIONS):
    routes = []
    for i in range(N):
        idx = individual["route_gene"][i]
        if DESTINATIONS[i] == 'B':
            routes.append(paths_to_B[idx])
        elif DESTINATIONS[i] == 'C':
            routes.append(paths_to_C[idx])
    return routes

# ======================
# PLATOON DETECTION
# ======================

def get_platoons(routes, depart_times):
    platoon_dict = {}

    for i in range(N):
        key = (tuple(routes[i]), depart_times[i])

        if key not in platoon_dict:
            platoon_dict[key] = []

        platoon_dict[key].append(i)

    return platoon_dict

# ======================
# %%FITNESS
# ======================

def compute_fuel(route, platoon_size):
    dist = len(route)
    reduction = 1 - 0.1 * (platoon_size - 1)
    return dist * reduction

def fitness(individual, paths_to_B, paths_to_C, DESTINATIONS):
    routes = decode(individual, paths_to_B, paths_to_C, DESTINATIONS)
    depart_times = individual["depart_gene"]
    platoons = get_platoons(routes, depart_times)

    total_fuel = 0
    total_wait = 0
    platoon_size = {}
    for group in platoons.values():
        size = len(group)
        for i in group:
            platoon_size[i] = size

    for i in range(N):
        route = routes[i]
        depart = depart_times[i]
        size = platoon_size.get(i, 1)

        total_fuel += compute_fuel(route, size)
        total_wait += max(0, depart - ARRIVAL_TIMES[i])

    return 0.7 * total_fuel + 0.3 * total_wait

# ======================
# %%GA OPERATORS
# ======================

def selection(pop, paths_to_B, paths_to_C, DESTINATIONS):
    # sort population theo fitness tăng dần (min tốt hơn)
    sorted_pop = sorted(
        pop,
        key=lambda ind: fitness(ind, paths_to_B, paths_to_C, DESTINATIONS)
    )
    return sorted_pop[:POP_SIZE]

def crossover(p1, p2):
    point = random.randint(0, N-1)

    c1 = {
        "route_gene": p1["route_gene"][:point] + p2["route_gene"][point:],
        "depart_gene": p1["depart_gene"][:point] + p2["depart_gene"][point:]
    }

    c2 = {
        "route_gene": p2["route_gene"][:point] + p1["route_gene"][point:],
        "depart_gene": p2["depart_gene"][:point] + p1["depart_gene"][point:]
    }

    return c1, c2

def mutate(ind, paths_to_B, paths_to_C, DESTINATIONS):
    for i in range(N):
        if random.random() < MUT_RATE:
            if DESTINATIONS[i] == 'B':
                ind["route_gene"][i] = random.randint(0, len(paths_to_B)-1)
            else:
                ind["route_gene"][i] = random.randint(0, len(paths_to_C)-1)

        if random.random() < MUT_RATE:
            ind["depart_gene"][i] = random.randint(
                max(ARRIVAL_TIMES[i], TIME_WINDOW[0]),
                TIME_WINDOW[1]
            )

    return ind

# ======================
# %%MAIN (GA LOOP HERE)
# ======================

if __name__ == "__main__":
    A, B, C, paths_to_B, paths_to_C = create_graph()

    DESTINATIONS = np.random.choice(['B', 'C'], size=N)

    # ---- init pop ----
    pop = [init_individual(paths_to_B, paths_to_C, DESTINATIONS) for _ in range(POP_SIZE)]

    best = None

    for gen in range(NUM_GEN):

        # ---- selection ----
        pop = selection(pop, paths_to_B, paths_to_C, DESTINATIONS)

        # ---- crossover + mutation ----
        next_pop = []
        for i in range(0, POP_SIZE):
            k = np.random.randint(0,N-1)
            p1, p2 = pop[i], pop[k]

            c1, c2 = crossover(p1, p2)
            next_pop.append(mutate(c1, paths_to_B, paths_to_C, DESTINATIONS))
            next_pop.append(mutate(c2, paths_to_B, paths_to_C, DESTINATIONS))

        pop = next_pop

        # ---- evaluation ----
        current_best = min(pop, key=lambda ind: fitness(ind, paths_to_B, paths_to_C, DESTINATIONS))

        if best is None or fitness(current_best, paths_to_B, paths_to_C, DESTINATIONS) < fitness(best, paths_to_B, paths_to_C, DESTINATIONS):
            best = current_best

        print(f"Gen {gen}: {fitness(best, paths_to_B, paths_to_C, DESTINATIONS):.3f}")

    print("Destinations:", DESTINATIONS)
    print("Best individual:", best)
    # print("Decoded routes:", decode(best, paths_to_B, paths_to_C, DESTINATIONS))