from utils.fitness_functions import fitness
import random
# ======================
# GA OPERATORS
# ======================

def selection_etilist(pop, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    POP_SIZE = len(pop)
    # sort population theo fitness tăng dần (min tốt hơn)
    sorted_pop = sorted(
        pop,
        key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
    )
    return sorted_pop[:POP_SIZE]

def selection_tournament(pop, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    k = 3
    new_pop = []
    for _ in range(len(pop)):
        candidates = random.sample(pop, k)
        best = min(candidates, key=lambda ind: fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G))
        new_pop.append(best)
    return new_pop

def crossover(p1, p2):
    N = len(p1['route_HD'])
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

def mutate(ind, init, paths_SH, paths_HD):
    N = len(ind['route_HD'])
    MUT_RATE = 0.2
    
    for i in range(N):
        s = init[i][0]
        h = init[i][1]
        d = init[i][2]
        if random.random() < MUT_RATE:
            ind["route_SH"][i] = random.randint(0, len(paths_SH[(s,h)])-1)
            ind["route_HD"][i] = random.randint(0, len(paths_HD[(h,d)])-1)
            
        if random.random() < MUT_RATE:
            ind["wait_S"][i] = random.randint(0,3)
            ind["wait_H"][i] = random.randint(0,3)

    return ind