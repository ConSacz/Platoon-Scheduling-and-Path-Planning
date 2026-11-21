"""
Route Manager

Manage all feasible routes between predefined
origin-destination pairs.

Author: ...
"""

import pickle
# import networkx as nx

# ==========================================================
# GLOBAL
# ==========================================================

RouteLibrary = {}

LogisticsGraph = None

# ==========================================================
# LOAD GRAPH
# ==========================================================

def load_graph(filename="logistics_graph.pkl"):

    global LogisticsGraph

    with open(filename,"rb") as f:

        LogisticsGraph = pickle.load(f)

# ==========================================================
# LOAD ROUTES
# ==========================================================

def load_routes(filename="route_library.pkl"):

    global RouteLibrary

    with open(filename,"rb") as f:

        RouteLibrary = pickle.load(f)

# ==========================================================
# SAVE ROUTES
# ==========================================================

def save_routes(filename="route_library.pkl"):

    with open(filename,"wb") as f:

        pickle.dump(RouteLibrary,f)

# ==========================================================
# REGISTER
# ==========================================================

def register(origin,
             destination,
             route_id,
             path):
    from locations import locations
    
    if (origin, destination) not in RouteLibrary:
        RouteLibrary[(origin, destination)] = {}

    # ==========================================================
    # Keep only start, hub and destination nodes
    # ==========================================================

    valid_types = {"start", "hub", "destination"}

    filtered_path = [
        node for node in path
        if locations[node]["type"] in valid_types
    ]

    # ==========================================================
    # Build segments
    # ==========================================================

    segments = list(zip(filtered_path[:-1], filtered_path[1:]))

    segment_distance = []

    # Original path still contains inter nodes
    for u, v in segments:

        # Find positions of u and v in original path
        i = path.index(u)
        j = path.index(v)

        # Sum distance of all edges between u and v
        dist = sum(
            LogisticsGraph[path[k]][path[k+1]]["distance"]
            for k in range(i, j)
        )

        segment_distance.append(dist)

    distance = sum(segment_distance)

    # ==========================================================
    # Store route
    # ==========================================================

    RouteLibrary[(origin, destination)][route_id] = {

        "route_id": route_id,

        "path": filtered_path,

        "segments": segments,

        "distance": distance,

        "segment_distance": segment_distance,
    }

# ==========================================================
# EXIST
# ==========================================================

def exist(origin,
          destination,
          route_id):

    return route_id in RouteLibrary[(origin,destination)]

# ==========================================================
# GET ROUTE
# ==========================================================

def get(origin,
        destination,
        route_id):

    return RouteLibrary[(origin,destination)][route_id]

# ==========================================================
# GET PATH
# ==========================================================

def get_path(origin,
             destination,
             route_id):

    return get(origin,destination,route_id)["path"]

# ==========================================================
# NUMBER OF ROUTES
# ==========================================================

def num_routes(origin,
               destination):

    return len(RouteLibrary[(origin,destination)])

# ==========================================================
# COMPUTE DISTANCE
# ==========================================================

def compute_distance(origin,
                     destination,
                     route_id):

    path = get_path(origin,destination,route_id)

    total = 0

    for u,v in zip(path[:-1],path[1:]):

        total += LogisticsGraph[u][v]["distance"]

    return total

# ==========================================================
# COMPUTE TRAVEL TIME
# ==========================================================

def compute_travel_time(origin,
                        destination,
                        route_id):

    path = get_path(origin,destination,route_id)

    total = 0

    for u,v in zip(path[:-1],path[1:]):

        total += LogisticsGraph[u][v]["travel_time"]

    return total

# ==========================================================
# GET GEOMETRY
# ==========================================================

def get_geometry(origin,
                 destination,
                 route_id):

    return get(origin,destination,route_id)["geometry"]

# ==========================================================
# SUMMARY
# ==========================================================

def summary():

    print()

    print("="*60)

    print("Route Library")

    print("="*60)

    for od,routes in RouteLibrary.items():

        print()

        print(od[0],"->",od[1])

        print("Routes :",len(routes))

        for rid,data in routes.items():

            d = compute_distance(od[0],od[1],rid)

            t = compute_travel_time(od[0],od[1],rid)

            print(

                f" Route {rid:02d}",

                f"{d:.1f} km",

                f"{t:.2f} h",

                data["path"]

            )

# ==========================================================
# VALIDATE
# ==========================================================

def validate():

    print()

    print("Checking Route Library ...")

    for od,routes in RouteLibrary.items():

        for rid,data in routes.items():

            path = data["path"]

            ok = True

            for u,v in zip(path[:-1],path[1:]):

                if not LogisticsGraph.has_edge(u,v):

                    ok = False

                    break

            if ok:

                print(f"{od} Route {rid:02d} : OK")

            else:

                print(f"{od} Route {rid:02d} : INVALID")