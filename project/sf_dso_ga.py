import pylab
import pickle
import random
from tqdm import tqdm  # for progress tracking
from copy import deepcopy
from concurrent.futures import ProcessPoolExecutor, as_completed

from deap import base, creator, tools, algorithms
from uxsim import World, Utilities, Vehicle, Route, Link, Node


##############################################################
# Define UXsim World
def create_world(name="", deltan=5, tmax=7200):
    # Simulation main
    W = World(
        name=name,
        deltan=deltan, # platoon size
        tmax=tmax,
        print_mode=0, 
        save_mode=0, 
        show_mode=0,
        random_seed=0
    )

    # Scenario definition
    W.load_scenario("dat/sfnetwork.uxsim_scenario")
    
    return W

def find_routes_parallel(od_pairs, world, n_routes=6, verbose=False):
    """
    Find routes for each OD pair in parallel using process-based parallelism
    
    Parameters:
        od_pairs: List of (origin, destination) pairs
        world: Reference World object to get parameters from
        n_routes: Number of routes to find for each OD pair
        verbose: If True, print progress information
        
    Returns:
        routes: Dictionary mapping each OD pair to its list of routes
    """
    def find_routes_for_od(od_pair, n_routes):
        # Create a new World instance for each process to avoid sharing issues
        temp_world = create_world(name="")
        
        return (od_pair, Utilities.enumerate_k_shortest_routes(temp_world, od_pair[0], od_pair[1], n_routes))
    
    routes = {}

    # Use ProcessPoolExecutor to parallelize the route enumeration
    with ProcessPoolExecutor() as executor:
        # Prepare tasks with explicit arguments
        tasks = [(od_pair, world.name, world.DELTAN, world.TMAX, n_routes) for od_pair in od_pairs]
        
        # Submit all tasks and track with progress bar
        futures_list = []
        for args in tasks:
            futures_list.append(executor.submit(find_routes_for_od, *args))
        
        # Process results as they complete with progress bar
        for future in tqdm(as_completed(futures_list), total=len(od_pairs), desc="Finding routes"):
            od_pair, od_routes = future.result()
            routes[od_pair] = od_routes
            if verbose:
                print(f"Routes for {od_pair} found")
                
    if verbose:
        print("available routes for each OD pair")
        for key in routes:
            for route in routes[key]:
                print(key, route)
    
    return routes

W = create_world()
W_orig = deepcopy(W)
W.show_network(show_link_name=False)

od_pairs = [(d['orig'], d['dest']) for d in W.demand_info['adddemand']]

try:
    print("Trying to loading routes for each OD pair")
    # Check if the routes have already been computed
    with open("out/sf_routes.pkl", "rb") as f:
        routes = pickle.load(f)
except FileNotFoundError:
    # If not, compute the routes and save them
    print("Computing routes for each OD pair")
    routes = find_routes_parallel(od_pairs, W)
    with open("out/sf_routes.pkl", "wb") as f:
        pickle.dump(routes, f)

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
        veh:Vehicle = veh_list[i]
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
    W = deepcopy(W_orig)
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
        W = deepcopy(W_orig)
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

W_dso = deepcopy(best_ind.W)

##############################################################
# Compare DUO and near-DSO

print("Deriving DUO")
W_duo = deepcopy(W_orig)
W_duo.exec_simulation()
print(W_duo.analyzer.basic_to_pandas())

print("DUO")
W_duo.analyzer.macroscopic_fundamental_diagram()
W_duo.analyzer.network_anim(file_name="out/grid_duo.gif", detailed=1, network_font_size=0, figsize=(6,6))
print("DUO animation saved as out/grid_duo.gif")

print("near-DSO")
W_dso.analyzer.macroscopic_fundamental_diagram()
W_dso.analyzer.network_anim(file_name="out/grid_dso.gif", detailed=1, network_font_size=0, figsize=(6,6))
print("DSO animation saved as out/grid_dso.gif")

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