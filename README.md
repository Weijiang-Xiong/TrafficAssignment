## Traffic Assignment using Game Theory

Project based on UXSim, see the original readme at [HERE](README_UXSIM.md)

In our project, we have designed a toll mechanism based on marginal social cost to reduce traffic congestion over a realistic transportation network.

### Installation 

```
python -m pip install -e .
```

### Run scripts
Scripts are placed under folder `project`. We use the Sioux Falls (SF) Network https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls. The scripts can be run like 

```
python project/<script_name>.py
```

We experimented different cases:

`sf_duo.py` dynamic user optimal (DUO), where everybody takes the shortest path and updates the paths on-the-fly.

`sf_due.py` dynamic user equilibrium (DUE), where part of the drivers will update their shortest paths based on the actual traffic of a previous simulation run (a previous day).

`sf_duo_ga.py` an attempt to find the system optimal (minimize total travel time) using genetic algorithm. It is unsuccessful because the algorithm finds a case that leads to a gridlock. The total travel time is minimized because everyone is blocked and no one can travel.

`sf_duo_msc.py` a solver inspired by the concept of Marginal Social Cost (MSC), where we add tolls to the roads with higher traffic volume and prevents people from crowding to a geographically shortest path. Our strategy have significantly decreased the average delay, compared to the DUO and DUE case. More details in the project report and the docs in the script.

Main results are as follows

| **Metric**             | **DUO**    | **DUE**    | **MSC**    |
|------------------------|------------|------------|------------|
| Total Trips            | 34,690     | 34,690     | 34,690     |
| Completed Trips        | 33,350     | 34,115     | 34,185     |
| Total Travel Time      | 5.87e7     | 5.62e7     | 5.56e7     |
| Average Travel Time    | 1,760.14   | 1,646.82   | 1,626.78   |
| Total Delay            | 1.31e7     | 8.86e6     | 8.09e6     |
| Average Delay          | 391.44     | 259.60     | 236.77     |

