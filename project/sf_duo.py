"""
Simply run the simulation with the Sioux Falls Network
https://github.com/bstabler/TransportationNetworks/tree/master/SiouxFalls


"""

from uxsim import World

def create_world():
    # Simulation main
    W = World(
        name="sf_duo",
        deltan=5,
        tmax=7200,
        print_mode=1, save_mode=1, show_mode=0,
        random_seed=0
    )

    # Scenario definition
    W.load_scenario("dat/sfnetwork.uxsim_scenario")
    
    return W

# Simulation execution
W = create_world()

W.show_network(show_id=False, show_link_name=False)

W.exec_simulation()

# Results analysis
W.analyzer.print_simple_stats()

print(f"Simulation finished. Results saved in the 'out{W.name}' folder.")
# Save results to CSV
df = W.analyzer.basic_to_pandas()
df.to_csv(f"out{W.name}/basics.csv", index=False)

# save simulation statistics 
W.analyzer.output_data()
# Time-averaged link state over the entire simulation period
W.analyzer.network_average()
W.analyzer.macroscopic_fundamental_diagram()
W.analyzer.network_anim(animation_speed_inverse=15, timestep_skip=8, detailed=0, network_font_size=0)


