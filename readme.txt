Path-planning and Scheduling optimization using GA version 3

-Map: 2 source (0, 1); 2 hub (2, 3); 2 destination (4, 5) 

-Problem: 
+ N trucks (15 tons, 60km/h average velocity) arrives at 0 or 1 in random time windows, assigned to travel to 4 or 5.
+ Trucks can decide to depart right after arrive, or wait for other to form platoon
+ When arrive hubs trucks can wait for other to form platoon

-Optimization variable: (truck routes, truck depart time)

-Fitness functions: 
+ total fuel: depends on platoon and distance travel, update to dollar
+ total wait time: wait time at 0, 1, 2, 3, update to dollar
