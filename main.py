import numpy as np
import argparse
import os
import sys

# Project File
from config import Config
from data_loader import DataLoader
from marl_env import CloudEdgeEnvironment
from logger_utils import SimulationLogger

# DWPA Solver File
from dwpa_solver.dwpa import DWPASolver
from dwpa_solver.dwpa_local import DWPALFSolver
from dwpa_solver.dwpa_vertical import DWPAVOSolver
from dwpa_solver.dwpa_horizontal import DWPAHFSolver

# Other Competitors
from dwpa_competitor.dola22_solver import DOLA22Solver
from dwpa_competitor.icsoc19_solver import ICSOC19Solver
from dwpa_competitor.ycl24_solver import YCL24Solver

# RL
from marl_agent import RandomAgent, QLearningAgent, MARLController

def parse_arguments():
    parser = argparse.ArgumentParser(description="Green MEC Simulation Runner")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Path to the input directory containing config.json, carbon_intensity.csv, and data_arrival.csv")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Path to the output directory where logs and csv folders will be created")
    
    args = parser.parse_args()
    return args

def validate_input_files(input_dir):
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Error: Input directory '{input_dir}' does not exist.")

    required_files = {
        'config': 'config.json',
        'carbon': 'carbon_intensity.csv',
        'task': 'data_arrival.csv'
    }

    found_paths = {}

    for key, filename in required_files.items():
        file_path = os.path.join(input_dir, filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Error: Missing required file '{filename}' in '{input_dir}'")
        found_paths[key] = file_path
    
    return found_paths

def run_simulation(algorithm='DWPA', output_dir='Base_Output'):
    # print(f"\n=== Starting Simulation with Algorithm: {algorithm} ===")
    
    logger = SimulationLogger(algorithm, output_dir)
    data_loader = DataLoader()
    env = CloudEdgeEnvironment(data_loader, logger=logger)
    total_steps = env.max_time_steps
    
    solver, marl_controller = None, None

    solver_classes = {
        'DWPA': DWPASolver,
        'DWPALF': DWPALFSolver,
        'DWPAVO': DWPAVOSolver,
        'DWPAHF': DWPAHFSolver,
        'DOLA22': DOLA22Solver,
        'ICSOC19': ICSOC19Solver,
        'YCL24': YCL24Solver
    }

    if algorithm in solver_classes:
        solver = solver_classes[algorithm](env)
    elif algorithm == 'MARL': 
        agents = [RandomAgent(i, Config.NUM_EDGE_SERVERS) for i in range(Config.NUM_EDGE_SERVERS)]
        marl_controller = MARLController(env, agents)
    elif algorithm == 'QLearning':
        agents = [QLearningAgent(i, Config.NUM_EDGE_SERVERS) for i in range(Config.NUM_EDGE_SERVERS)]
        marl_controller = MARLController(env, agents)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Execution Loop
    state = env.reset()
    history_carbon, history_q = [], []
    done = False
    step_count = 0
    TO_MB = 1.0 / Config.MB_TO_BITS
    
    while not done:
        # Get Action
        if solver is not None:
            decisions = solver.solve(state)
        else: # MARL or QLearning
            decisions = marl_controller.get_decisions(state)
            
        # Environment Step (Logging happens inside here now)
        next_state, carbon, done, info = env.step(decisions)
        
        # Training Step
        if algorithm == 'QLearning':
            q_values = next_state['Q_edge']
            rewards = []
            for i in range(Config.NUM_EDGE_SERVERS):
                r = - (0.1 * carbon / Config.NUM_EDGE_SERVERS + 1e-6 * q_values[i])
                rewards.append(r)
            
            marl_controller.update_agents(state, decisions, rewards, next_state)
        
        history_carbon.append(carbon)
        history_q.append(info['q_avg_total']) 
        
        # Terminal Output (with units)
        # if step_count % 50 == 0 or step_count == total_steps - 1:
        #      print(f"Step {step_count:04d}: C={carbon:.4f} g, Q_sys={info['q_avg_total']*TO_MB:.2f} MB "
        #            f"(Loc:{info['processed_local']*TO_MB:.2f} MB, Cld:{info['processed_cloud']*TO_MB:.2f} MB, OffC:{info['offloaded_cloud']*TO_MB:.2f} MB)")
        
        state = next_state
        step_count += 1
        
    total_carbon = sum(history_carbon)
    avg_q = np.mean(history_q)
    
    logger.close()
    
    # print(f"\n>>> Simulation Finished ({algorithm}) <<<")
    # print(f"Total Carbon: {total_carbon:.4f} g")
    # print(f"Avg System Queue: {avg_q*TO_MB:.2f} MB")
    
    return total_carbon, avg_q

if __name__ == "__main__":
    # 1. Parse Arguments
    args = parse_arguments()
    
    try:
        # 2. Validate Inputs
        files = validate_input_files(args.input_dir)
        
        # 3. Setup Global Config (This affects DataLoader)
        Config.CONFIG_JSON = files['config']
        Config.CARBON_FILE = files['carbon']
        Config.TASK_FILE = files['task']
        
        # 4. Setup Output Directory
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # 5. Run Simulations
        algorithms_to_run = [
            'DWPA', 'DWPALF', 'DWPAVO', 'DWPAHF',
            'DOLA22', 'ICSOC19', 'YCL24'
            # 'MARL', 'QLearning'
        ]
        
        results = {}
        for algo in algorithms_to_run:
            c, q = run_simulation(algo, args.output_dir)
            results[algo] = {'carbon': c, 'queue': q}
        
        TO_MB = 1.0 / Config.MB_TO_BITS
        
        # print("=== FINAL COMPARISON (Units: g, MB) ===")
        print("="*63)
        for algo in algorithms_to_run:
            print(f"{algo:<10}| Carbon: {results[algo]['carbon']:10.4f} g | Avg Queue: {results[algo]['queue'] * TO_MB:10.2f} MB")
        print("="*63)
        
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        sys.exit(1)