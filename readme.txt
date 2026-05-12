Path-planning and Scheduling optimization using GA version 3

-Map: 2 sources (0, 1); 2 hubs (2, 3); 2 destinations (4, 5) 

-Problem: 
+ N trucks (15 tons, 60km/h average velocity) arrives at 0 or 1 in random time windows, assigned to travel to 4 or 5.
+ Trucks can decide to depart right after arrive, or wait for other to form platoon, platoon order depends on priority index
+ When arrive hubs trucks can wait for other to form platoon, platoon order depends on priority index

-Optimization variable: (truck routes, truck depart time)

-Fitness functions: 
+ total fuel cost(dollar): depends on distance travel, forming platoon and order in platoon
+ total wait time(dollar): wait time at start or hub
