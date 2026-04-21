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
    start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times)
    
    # %%
    fuel_g = 0
    total_wait = 0

    # platoon size
    platoon_size_S = {}
    platoon_size_H = {}

    for group in start_dict.values():
        for i in group:
            platoon_size_S[i] = len(group)

    for group in hub_dict.values():
        for i in group:
            platoon_size_H[i] = len(group)

    for i in range(N):
        route = routes[i]

        size_S = platoon_size_S.get(i, 1)
        size_H = platoon_size_H.get(i, 1)

        FCF_S = vel * FCF[size_S] if size_S <= 5 else vel * FCF[0]
        FCF_H = vel * FCF[size_H] if size_H <= 5 else vel * FCF[0]

        split_idx = len(route) // 2

        # ---- fuel (g) ----
        for u, v in zip(route[:split_idx], route[1:split_idx+1]):
            fuel_g += G[u][v]["weight"] * FCF_S

        for u, v in zip(route[split_idx:], route[split_idx+1:]):
            fuel_g += G[u][v]["weight"] * FCF_H

        # ---- waiting time (seconds) ----
        total_wait += max(0, S_depart_times[i] - ARRIVAL_TIMES[i])
        total_wait += max(0, H_depart_times[i] - (S_depart_times[i] + times_SH[i]))

    # ======================
    # convert to $
    # ======================

    # fuel → $
    
    fuel_cost = fuel_g / 1000  * gas_price

    # wait → $
    wait_cost = total_wait * wait_price
# %%
    return 0.7 * fuel_cost + 0.3 * wait_cost