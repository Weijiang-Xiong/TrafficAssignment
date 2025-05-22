import os
import logging
import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import partial
from collections import defaultdict

from uxsim import World, Link
from uxsim.DTAsolvers import SolverDUE
from sf_dso_ga import create_world


def setup_logger(name, log_file:str = "log.log", level=logging.INFO):
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(level)
    plainer_formatter = logging.Formatter(
        "[%(asctime)s %(name)s]: %(message)s", datefmt="%m/%d %H:%M:%S"
    )
    fh.setFormatter(plainer_formatter)
    logger.addHandler(fh)
        
    # console handler or stream handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = plainer_formatter
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

def print_route_choice_penalty(world: World, logger=None):
    
    print_fn = print if logger is None else logger.info
    print_fn("Checking route choice penalty for all links")
    
    has_penalty = False
    for link in world.LINKS:
        link: Link
        if link.route_choice_penalty != 0:
            has_penalty = True
            print_fn(f"link {link.name} has route choice penalty {link.route_choice_penalty}")

    if not has_penalty:
        print_fn("No route choice penalty set for any link")

def create_world_and_add_toll(create_world_func=create_world, toll_settings=None, **kwargs):
    world: World= create_world_func(**kwargs)
    
    if toll_settings is not None:
        for link in world.LINKS:
            link: Link
            link.route_choice_penalty = toll_settings.get(link.name, 0)
    
    return world

class SolverMarginalSocialCost:
    
    
    def __init__(self, quantile_ratio=0.5, dso_iter=3, due_iter=3):
        
        self.quantile_ratio = quantile_ratio
        self.dso_iter = dso_iter
        self.due_iter = due_iter
        self.metrics = defaultdict(list)
        self.save_path = "outsf_dso_msc"
        
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        self.logger = setup_logger(name="SolverMSC", log_file=f"{self.save_path}/log.log", level=logging.INFO)


    def solve(self):
        
        initial_world = create_world(name="sf_dso_msc")
        initial_world.exec_simulation()
        initial_world.analyzer.basic_to_pandas().to_csv(f"{self.save_path}/basics_init.csv", index=False)
        toll_settings = self.compute_marginal_social_cost(initial_world)
        
        # self.logger.info(f"initial toll settings: {self.collect_non_zero_tolls(toll_settings)}")
        
        for i in range(self.dso_iter):
            self.logger.info("=====================================")
            self.logger.info(f"DSO iteration {i+1}/{self.dso_iter}")
            self.logger.info("=====================================")
            
            # DUE
            partial_create_world_with_toll = partial(
                create_world_and_add_toll,
                toll_settings=toll_settings,
                name = "sf_dso_msc",
            )
            
            signature = inspect.signature(partial_create_world_with_toll)
            nzt = self.collect_non_zero_tolls(signature.parameters['toll_settings'].default)
            self.logger.info(f"Using these toll settings for the simulations {nzt}")
            
            solver_DUE = SolverDUE(partial_create_world_with_toll)
            solver_DUE.solve(max_iter=self.due_iter, print_progress=False)
            
            W_DUE:World = solver_DUE.W_sol
            print_route_choice_penalty(W_DUE, self.logger)
            W_DUE.analyzer.print_simple_stats(force_print=True)
            df_DUE = W_DUE.analyzer.basic_to_pandas().iloc[0, :] # it's a DF with one row, so ... 
            for key, value in df_DUE.items():
                self.metrics[key].append(value)
            
            toll_settings.update(self.compute_marginal_social_cost(W_DUE))
        
        self.final_analysis(W_DUE)
        self.plot_metrics()

    
    def compute_marginal_social_cost(self, world: World):
        
        assert hasattr(world, "analyzer"), "please first run world.exec_simulation()"
        
        link_groups = world.analyzer.link_traffic_state_to_pandas().groupby("link")

        # if the link is experiencing congestion (1 - QUANTILE_RATIO) of the time, then we set the route choice penalty
        px_density = link_groups["k"].quantile(self.quantile_ratio)
        px_flow = link_groups["q"].quantile(self.quantile_ratio)

        toll_settings = dict()
        for idx, link in enumerate(world.LINKS):
            link: Link
            if px_density[link.name] > link.k_star:
                toll_settings[link.name] = - link.length / (link.tau * link.w * px_flow[link.name])

        # normalize and apply sigmoid function to toll values
        toll_values = np.array([v for k,v in toll_settings.items()])
        if toll_values.size == 1: 
            # for 1 element case, we have zero std, and normalization won't work. just set it to 1
            toll_values = np.array([1.0])
        else:
            mean = np.mean(toll_values)
            std = np.std(toll_values)
            toll_values = (toll_values - mean) / std
            toll_values = 1 / (1 + np.exp(-toll_values))
        
        # now set the non zero tolls to the new values
        for idx, key in enumerate(toll_settings.keys()):
            toll_settings[key] = toll_values[idx] * world.LINKS_NAME_DICT[key].traveltime_actual.mean()
        
        return toll_settings

        
    @staticmethod
    def collect_non_zero_tolls(toll_settings):
        non_zero_tolls = {link: round(float(toll), 1) for link, toll in toll_settings.items() if toll != 0}
        return non_zero_tolls

        
    def plot_metrics(self):
        for key, value in self.metrics.items():
            fig, ax = plt.subplots()
            ax.plot(value)
            ax.set_title(key)
            ax.set_xlabel("iteration")
            ax.set_ylabel(key)
            fig.savefig(f"{self.save_path}/{key}.png")
    
    
    def final_analysis(self, world):
        
        self.logger.info(f"Simulation finished. Saving results into 'out{self.save_path}' folder.")
        
        tolls = {link.name: link.route_choice_penalty for link in world.LINKS if link.route_choice_penalty != 0}
        tolls_pd = pd.DataFrame.from_dict(tolls, orient="index", columns=["toll"])
        tolls_pd.to_csv(f"{self.save_path}/tolls.csv", index=True)
        self.logger.info(f"tolls: {tolls_pd}")
        
        # save the tolls to a csv file
        df = world.analyzer.basic_to_pandas()
        df.to_csv(f"{self.save_path}/basics.csv", index=False)
        self.logger.info(f"Simulation statistics: \n {df}")
        
        # save simulation statistics 
        world.analyzer.output_data()
        # Time-averaged link state over the entire simulation period
        world.analyzer.network_average()
        world.analyzer.macroscopic_fundamental_diagram()
        world.analyzer.network_anim(animation_speed_inverse=15, timestep_skip=8, detailed=0, network_font_size=0)
    
    
if __name__ == "__main__":
    SolverMarginalSocialCost(dso_iter=50, due_iter=20).solve() 
