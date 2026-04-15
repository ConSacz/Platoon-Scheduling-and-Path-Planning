Path-planning and Scheduling optimization using GA version 19

-Map: 1 source (A) 0 hub 2 destination (B, C) 

-Problem: 
+ N trucks arrives at A in random time windows, assigned to travel to B or C.
+ Trucks can decide to depart right after arrive, or wait for other to form platoon

-Optimization variable: (truck routes, truck depart time)

-Fitness functions: 
+ total fuel: depends on platoon and distance travel
+ total wait time: wait time at A