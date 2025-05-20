import os
import pickle
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt  # Replace pylab with matplotlib.pyplot

from deap import base, creator, tools, algorithms
from uxsim import World, Utilities, Vehicle, Route, Link, Node
from typing import List, Dict, Tuple

##############################################################
# Create world and find possible routes (genes for Genetic Algorithm)
##############################################################

def create_world(name="", deltan=5, tmax=7200) -> World:
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

def find_routes_for_od(od_pair, name, deltan, tmax, n_routes):
    # Create a new World instance for each process to avoid sharing issues
    temp_world = create_world(name=name, deltan=deltan, tmax=tmax)
    top_k_routes: List = Utilities.enumerate_k_shortest_routes(temp_world, od_pair[0], od_pair[1], n_routes)
    
    return (od_pair, top_k_routes)

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

##############################################################
# Genetic Algorithm Components
##############################################################

# Evaluate fitness by total travel time
def eval_simulation_fitness(W:World):
    W.exec_simulation()
    return - W.analyzer.average_delay,

# Specify routing based on genome
def specify_routes(W, genome, routes):
    veh_list = list(W.VEHICLES.values())
    for i, value in enumerate(genome):
        veh:Vehicle = veh_list[i]
        veh.set_links_prefer(routes[(veh.orig.name, veh.dest.name)][value])

# Initialize genetic algorithm toolbox
def setup_ga_toolbox(W:World, routes, n_routes=6):
    """Setup the genetic algorithm toolbox with operators and parameters"""
    # Create fitness and individual types if not already created
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    # Initialize the individual
    n_gene = len(W.VEHICLES)
    toolbox.register("attr_gene", random.randint, 0, n_routes - 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.attr_gene, n=n_gene)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Define the evaluation function with routes parameter
    def evaluate_individual(individual):
        world = W.copy()
        specify_routes(world, individual, routes)
        return eval_simulation_fitness(world)
    
    # Define the evaluation, crossover, and mutation functions
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
    toolbox.register("select", tools.selTournament, tournsize=3)
    
    return toolbox

##############################################################
# Parallel fitness evaluation
##############################################################

def get_world_from_individual(individual, W_orig, routes) -> World:
    """Evaluate an individual by creating a copy of the world and applying the routes"""
    # Create a fresh copy of the original world
    world = W_orig.copy()
    # Apply the individual's route specifications
    specify_routes(world, individual, routes)
    return world

def compute_fitness(individual_with_index, W_orig:World, routes):
    """Helper function for parallel fitness evaluation"""
    index, individual = individual_with_index
    # Create a fresh copy of the original world
    world = W_orig.copy()
    # Apply the individual's route specifications
    specify_routes(world, individual, routes)
    # Evaluate fitness
    fitness_value = eval_simulation_fitness(world)
    # Return index with the result to maintain original order
    return index, fitness_value

def evaluate_population_parallel(population, W_orig, routes):
    """
    Evaluate fitness of a population in parallel
    
    Args:
        population: List of individuals to evaluate
        W_orig: Original world object to clone
        routes: Dictionary of available routes
        
    Returns:
        List of fitness values in the same order as population
    """
    # Prepare inputs with indices to track ordering
    inputs = [(i, ind) for i, ind in enumerate(population)]
    
    results = []
    # Use ProcessPoolExecutor for parallel evaluation
    with ProcessPoolExecutor(max_workers= min(len(population), os.cpu_count())) as executor:
        futures = []
        for inp in inputs:
            futures.append(executor.submit(compute_fitness, inp, W_orig, routes))
        
        # Process results as they complete
        for future in tqdm(as_completed(futures), total=len(population), desc="Evaluating fitness"):
            results.append(future.result())
    
    # Sort results by index to maintain original order
    results.sort(key=lambda x: x[0])
    
    # Extract and return fitness values
    fitness_values = [res[1] for res in results]

    return fitness_values

def pickle_copy(obj):
    """Create a deep copy of an object using pickle serialization
    to avoid maximum recursion depth errors"""
    return pickle.loads(pickle.dumps(obj))

##############################################################
# Main genetic algorithm function
##############################################################

def run_genetic_algorithm(W_orig, routes, NGEN=30, NPOP=30, CXPB=0.5, MUTPB=0.2, n_routes=6):
    """
    Execute the genetic algorithm to find the near-DSO solution
    
    Args:
        W_orig: Original world to clone for evaluations
        routes: Dictionary of available routes for each OD pair
        NGEN: Number of generations
        NPOP: Population size
        CXPB: Crossover probability
        MUTPB: Mutation probability
        n_routes: Number of routes per OD pair
        
    Returns:
        The best individual (routing solution)
    """
    print("Deriving DSO using genetic algorithm")
    
    # Setup GA toolbox
    toolbox = setup_ga_toolbox(W_orig, routes, n_routes)
    
    # Initial population
    pop = toolbox.population(n=NPOP)
    
    # Use parallel fitness evaluation
    fitnesses = evaluate_population_parallel(pop, W_orig, routes)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    for g in range(NGEN):
        print(f"-- Generation {g} --")
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(pickle_copy, offspring))
        
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
        
        # Use parallel fitness evaluation for invalid individuals
        if invalid_ind:
            fitnesses = evaluate_population_parallel(invalid_ind, W_orig, routes)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
        # Print the best individual
        best_ind = tools.selBest(pop, 1)[0]
        print("")
        print("Fitness: ", best_ind.fitness.values[0])
        best_ind_world = get_world_from_individual(best_ind, W_orig, routes)
        best_ind_world.exec_simulation()
        print(best_ind_world.analyzer.basic_to_pandas())
        
        # Update the population
        pop[:] = offspring
    
    # Return best individual from final population
    best_ind = tools.selBest(pop, 1)[0]
    
    return best_ind

##############################################################
# Comparison and visualization functions
##############################################################

def compare_duo_dso(W_duo:World, W_dso:World):
    """Compare DUO and near-DSO solutions with visualizations"""

    W_duo.save_mode = 1
    W_dso.save_mode = 1
    
    print("Deriving DUO")
    W_duo.exec_simulation()
    print(W_duo.analyzer.basic_to_pandas())
    
    print("DUO")
    W_duo.analyzer.macroscopic_fundamental_diagram(fname="duo")
    W_duo.analyzer.network_anim(file_name=f"out{W_duo.name}/grid_duo.gif", detailed=1, network_font_size=0, figsize=(6,6))
    print("DUO animation saved as out/grid_duo.gif")
    
    print("near-DSO")
    W_dso.exec_simulation()
    W_dso.analyzer.macroscopic_fundamental_diagram(fname="dso")
    W_dso.analyzer.network_anim(file_name=f"out{W_dso.name}/grid_dso.gif", detailed=1, network_font_size=0, figsize=(6,6))
    print("DSO animation saved as out/grid_dso.gif")
    
    print("Vehicle comparison")
    plt.figure()
    plt.subplot(111, aspect="equal")
    plt.hist2d(
        [veh.travel_time for veh in W_duo.VEHICLES.values()],
        [veh.travel_time for veh in W_dso.VEHICLES.values()],
        bins=20, range=[[0,1000],[0,1000]], cmap="Blues", cmin=1
    )
    plt.colorbar().set_label("number of vehicles")
    plt.plot([0,1000], [0,1000], "k--")
    plt.xlabel("travel time of each vehicle in DUO (s)")
    plt.ylabel("travel time of each vehicle in DSO (s)")
    plt.savefig(f"out{W_dso.name}/grid_dso.png", dpi=300)

# Execute main function if script is run directly
if __name__ == "__main__":
    
    W = create_world(name="sf_dso_ga")
    W.show_network(show_link_name=False)
    od_pairs = [(d['orig'], d['dest']) for d in W.demand_info['adddemand']]
    
    try:
        print("Trying to loading routes for each OD pair")
        # Check if the routes have already been computed
        with open("dat/sf_routes.pkl", "rb") as f:
            routes = pickle.load(f)
    except FileNotFoundError:
        # If not, compute the routes and save them
        print("Computing routes for each OD pair")
        routes = find_routes_parallel(od_pairs, W)
        with open("dat/sf_routes.pkl", "wb") as f:
            pickle.dump(routes, f)


    # Run genetic algorithm
    best_ind = run_genetic_algorithm(W.copy(), routes, NGEN=30, NPOP=30)
    
    # Compare results
    compare_duo_dso(W.copy(), get_world_from_individual(best_ind, W, routes))