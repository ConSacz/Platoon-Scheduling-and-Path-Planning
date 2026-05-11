from utils.platoon_functions import decode, get_platoons
import numpy as np

# ======================
# FITNESS
# ======================

FCF = np.array([240.525, 223.207, 214.926, 209.575, 206.678]) # fuel cost factors
vel = 60

gas_price = 1.1144/0.832
wait_price = 25  # $/hour

def fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
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
    
        pos_S = min(pos_S, 4)
        pos_H = min(pos_H, 4)
    
        FCF_S = vel * FCF[pos_S]
        FCF_H = vel * FCF[pos_H]
    
        split_idx = len(route)//2

        # S -> H
        for u,v in zip(route[:split_idx], route[1:split_idx+1]):
            fuel_cost += (G[u][v]["weight"] * FCF_S)

        # H -> D
        for u,v in zip(route[split_idx:], route[split_idx+1:]):
            fuel_cost += (G[u][v]["weight"] * FCF_H)

        # waiting
        # ------------------------------------------
    
        wait_cost += max(0, S_depart_times[i] - ARRIVAL_TIMES[i]) * wait_price
    
        wait_cost += max(0, H_depart_times[i] - ( S_depart_times[i] + times_SH[i])) * wait_price
# %%
    return 1 * fuel_cost + 1 * wait_cost






