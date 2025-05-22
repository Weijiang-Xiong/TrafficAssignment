"""
Dynamic User Equilibrium (DUE) simulation of San Francisco network

The users will choose their routes based on the travel time of each link in the previous simulation round.
"""

from uxsim import World
from uxsim.DTAsolvers import SolverDUE

def create_world():
    # Simulation main
    W = World(
        name="sf_due",
        deltan=5,
        tmax=7200,
        print_mode=1, save_mode=1, show_mode=0,
        random_seed=0
    )

    # Scenario definition
    W.load_scenario("dat/sfnetwork.uxsim_scenario")
    
    return W

# DUE
solver_DUE = SolverDUE(create_world)
solver_DUE.solve(max_iter=5, print_progress=True)
W_DUE = solver_DUE.W_sol

# Results analysis
W_DUE.analyzer.print_simple_stats()

print(f"Simulation finished. Results saved in the 'out{W_DUE.name}' folder.")
# Save results to CSV
df = W_DUE.analyzer.basic_to_pandas()
df.to_csv(f"out{W_DUE.name}/basics.csv", index=False)

# save simulation statistics 
W_DUE.analyzer.output_data()
# Time-averaged link state over the entire simulation period
W_DUE.analyzer.network_average()
W_DUE.analyzer.macroscopic_fundamental_diagram()
W_DUE.analyzer.network_anim(animation_speed_inverse=15, timestep_skip=8, detailed=0, network_font_size=0)


