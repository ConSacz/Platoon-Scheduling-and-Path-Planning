from utils.platoon_functions import decode, get_platoons
import numpy as np

# ======================
# FITNESS
# ======================

FCF = np.array([240.525, 223.207, 214.926, 209.575, 206.678]) # fuel cost factors (g/km)
vel = 60/2 # (km/h)

gas_price = 1.1144/750 #($/L : g/L)
wait_price = 2  # $/hour

def sep_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    # %%
    N = len(ind['route_HD'])
    
    routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
    S_depart_times = ind["wait_S"] + ARRIVAL_TIMES
    H_depart_times = S_depart_times + times_SH + ind["wait_H"]
    start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times, ind["prior_S"], ind["prior_H"])
    
    # %% POSITION INFO
    platoon_info_S = {}
    platoon_info_H = {}
    
    for group in start_dict.values():
        size = len(group)
        for pos, veh in enumerate(group):
            platoon_info_S[veh] = (size, pos)
    
    for group in hub_dict.values():
        size = len(group)
        for pos, veh in enumerate(group):
            platoon_info_H[veh] = (size, pos)

    # %% COST   
    fuel_cost = 0
    wait_cost = 0
    
    for i in range(N):
        route = routes[i]
    
        size_S, pos_S = platoon_info_S.get(i, (1,0))
        size_H, pos_H = platoon_info_H.get(i, (1,0))
        
        if pos_S<=4:
            pos_S = pos_S
        else:
            pos_S = 0
        if pos_H<=4:
            pos_H = pos_H
        else:
            pos_H = 0
    
        FCF_S = vel * FCF[pos_S]
        FCF_H = vel * FCF[pos_H]
    
        split_idx = len(route)//2

        # S -> H
        for u,v in zip(route[:split_idx], route[1:split_idx+1]):
            fuel_cost += (G[u][v]["weight"] * FCF_S) * gas_price

        # H -> D
        for u,v in zip(route[split_idx:], route[split_idx+1:]):
            fuel_cost += (G[u][v]["weight"] * FCF_H) * gas_price

        # waiting
        # ------------------------------------------
    
        wait_cost += max(0, S_depart_times[i] - ARRIVAL_TIMES[i]) * wait_price
    
        wait_cost += max(0, H_depart_times[i] - ( S_depart_times[i] + times_SH[i])) * wait_price
    # return 1 * fuel_cost + 1 * wait_cost
    return fuel_cost, wait_cost

def ori_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    # %%
    N = len(ind['route_HD'])
    
    routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
    S_depart_times = ind["wait_S"] + ARRIVAL_TIMES
    H_depart_times = S_depart_times + times_SH + ind["wait_H"]
    start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times, ind["prior_S"], ind["prior_H"])

    # %% COST   
    fuel_cost = 0
    
    for i in range(N):
        route = routes[i]
    
        FCF_S = vel * FCF[0]
        FCF_H = vel * FCF[0]
    
        split_idx = len(route)//2

        # S -> H
        for u,v in zip(route[:split_idx], route[1:split_idx+1]):
            fuel_cost += (G[u][v]["weight"] * FCF_S) * gas_price

        # H -> D
        for u,v in zip(route[split_idx:], route[split_idx+1:]):
            fuel_cost += (G[u][v]["weight"] * FCF_H) * gas_price
    # return 1 * fuel_cost + 1 * wait_cost
    
    return fuel_cost

def fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    fuel_cost, wait_cost = sep_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
    return 1 * fuel_cost + 1 * wait_cost

