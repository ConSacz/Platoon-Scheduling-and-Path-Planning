from utils.platoon_functions import decode, get_platoons

# ======================
# FITNESS
# ======================
def fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G):
    N = len(ind['route_HD'])
    
    routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
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