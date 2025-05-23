""" 
This is copied from the example notebook in the follwing URL:
https://toruseo.jp/UXsim/docs/notebooks/demo_notebook_07en_optimal_routing.html#Optimal-routing-in-a-larger-network

This example demonstrates how to use genetic algorithm, we can keep the routine, but need to change the network.
Since genetic algorithm is very slow, we also need to parallelize a lot of things, such as finding the best possible
routes, and running the simulation of the population.
"""
import random
from deap import base, creator, tools, algorithms
import pylab
from uxsim import Utilities, World

##############################################################
# Define UXsim World
W = World(
    name="simple_ga",
    deltan=5,
    tmax=1200,
    print_mode=0, save_mode=1, show_mode=1, #print is off, otherwise it will be very verbose during the genetic algorithm
    random_seed=None,
    duo_update_time=60
)

## generate grid network
Utilities.generate_grid_network(W, 3, 3, length=1000)
W.show_network()

## set demand
od_pairs = [
    ("n(0, 0)", "n(2, 2)"),
    ("n(2, 0)", "n(0, 2)"),
    ("n(0, 2)", "n(2, 0)"),
    ("n(2, 2)", "n(0, 0)"),
]
for od_pair in od_pairs:
    W.adddemand(od_pair[0], od_pair[1], 0, 500, 1)

W_orig = W.copy()

##############################################################
# Compute DUO as a reference
print("Deriving DUO")
W.exec_simulation()
print(W.analyzer.basic_to_pandas())

W_duo: World= W.copy()

##############################################################
# enumerate some routes between each OD pair
routes = {}
n_routes = 6
for od_pair in od_pairs:
    routes[od_pair] = Utilities.enumerate_k_shortest_routes(W, od_pair[0], od_pair[1], n_routes)

print("available routes for each OD pair")
for key in routes:
    for route in routes[key]:
        print(key, route)

##############################################################
# Prepare genetic algorithm using DEAP
# evaluate fitness by total travel time
def evaluate_by_total_travel_time(W):
    W.exec_simulation()
    print(W.analyzer.total_travel_time, end=" ")
    return - W.analyzer.total_travel_time,

# specify routing based on genome
def specify_routes(W, genome):
    veh_list = list(W.VEHICLES.values())
    for i, value in enumerate(genome):
        veh = veh_list[i]
        veh.set_links_prefer(routes[(veh.orig.name, veh.dest.name)][value])

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Initialize the individual
n_gene = len(W.VEHICLES)
toolbox.register("attr_gene", random.randint, 0, 5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_gene, n=n_gene)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Define the evaluation, crossover, and mutation functions
toolbox.register("evaluate", evaluate_by_total_travel_time)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)


##############################################################
# Execute genetic algorithm
print("Deriving DSO using genetic algorithm")
NPOP = 30
CXPB, MUTPB = 0.5, 0.2
NGEN = 30

# Initial population
pop = toolbox.population(n=NPOP)
for ind in pop:
    W = W_orig.copy()
    specify_routes(W, ind)
    ind.W = W
fitnesses = list(map(toolbox.evaluate, [ind.W for ind in pop]))
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

for g in range(NGEN):
    print(f"-- Generation {g} --")
    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    # Crossover and mutation
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < CXPB:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    for mutant in offspring:
        if random.random() < MUTPB:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    for ind in invalid_ind:
        W = W_orig.copy()
        specify_routes(W, ind)
        ind.W = W
    fitnesses = map(toolbox.evaluate, [ind.W for ind in invalid_ind])
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # Print the best individual
    best_ind = tools.selBest(pop, 1)[0]
    print("")
    print("Best individual: ", best_ind)
    print("Fitness: ", best_ind.fitness.values[0])
    print(best_ind.W.analyzer.basic_to_pandas())

    # Update the population
    pop[:] = offspring

W_dso = best_ind.W.copy()

##############################################################
# Compare DUO and near-DSO

print("DUO")
W_duo.analyzer.macroscopic_fundamental_diagram(fname="duo")
W_duo.analyzer.network_anim(file_name="outsimple_ga/grid_duo.gif", detailed=1, network_font_size=0, figsize=(6,6))
print("File saved as outsimple_ga/grid_duo.gif")

print("near-DSO")
W_dso.analyzer.macroscopic_fundamental_diagram(fname="dso")
W_dso.analyzer.network_anim(file_name="outsimple_ga/grid_dso.gif", detailed=1, network_font_size=0, figsize=(6,6))
print("File saved as outsimple_ga/grid_dso.gif")

print("Vehicle comparison")
pylab.figure()
pylab.subplot(111, aspect="equal")
pylab.hist2d(
    [veh.travel_time for veh in W_duo.VEHICLES.values()],
    [veh.travel_time for veh in W_dso.VEHICLES.values()],
    bins=20, range=[[0,1000],[0,1000]], cmap="Blues", cmin=1
)
pylab.colorbar().set_label("number of vehicles")
pylab.plot([0,1000], [0,1000], "k--")
pylab.xlabel("travel time of each vehicle in DUO (s)")
pylab.ylabel("travel time of each vehicle in DSO (s)")
pylab.show()
pylab.savefig("outsimple_ga/vehicle_comparison.png", dpi=300)