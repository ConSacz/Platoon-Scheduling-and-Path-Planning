def decode(ind, init, paths_SH, paths_HD, G):
    N = len(ind['route_HD'])
    routes = []
    times_SH = []
    times_HD = []

    for i in range(N):
        s = init[i][0]
        h = init[i][1]
        d = init[i][2]

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
    N = len(routes)
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