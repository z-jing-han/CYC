import numpy as np
import matplotlib.pyplot as plt
from config import Config
from marl_env import CloudEdgeEnvironment
from dwpa_solver import DWPASolver
from marl_agent import RandomAgent, MARLController

def run_simulation(algorithm='DWPA', steps=50):
    print(f"Starting Simulation with Algorithm: {algorithm}")
    
    env = CloudEdgeEnvironment()
    state = env.reset()
    
    # Record data
    history_carbon = []
    history_q_edge = []
    
    # Init Controller
    solver = None
    marl_controller = None
    
    if algorithm == 'DWPA':
        solver = DWPASolver(env)
    else:
        agents = [RandomAgent(i, Config.NUM_EDGE_SERVERS) for i in range(Config.NUM_EDGE_SERVERS)]
        marl_controller = MARLController(env, agents)
    
    for t in range(steps):
        # 1. Get Decisions
        if algorithm == 'DWPA':
            decisions = solver.solve(state)
        else:
            decisions = marl_controller.get_decisions(state)
            
        # 2. Environment Step
        next_state, carbon, info = env.step(decisions)
        
        # 3. Log
        history_carbon.append(carbon)
        history_q_edge.append(info['q_avg'])
        
        if t % 10 == 0:
            print(f"Step {t}: Carbon={carbon:.4f}, Avg Q={info['q_avg']:.2f}")
            
        state = next_state
        
    return history_carbon, history_q_edge

if __name__ == "__main__":
    # Exec DWPA
    dwpa_carbon, dwpa_q = run_simulation('DWPA', steps=100)
    
    # Exec Random MARL
    marl_carbon, marl_q = run_simulation('MARL_RANDOM', steps=100)
    
    print("\nSimulation Complete.")
    print(f"DWPA Avg Carbon: {np.mean(dwpa_carbon):.2f}")
    print(f"MARL Avg Carbon: {np.mean(marl_carbon):.2f}")
    
    # (Optional) Plotting code would go here