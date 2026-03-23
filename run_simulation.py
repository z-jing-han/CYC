import numpy as np
import argparse
import os
import sys
import json

# Project File
from config import Config
from data_loader import DataLoader
from env_simulator import CloudEdgeEnvironment
from logger_utils import SimulationLogger

# DWPA Solver File
from dwpa_solver.dwpa import DWPASolver
from dwpa_solver.fixtimedwpa import FIXTIMEDWPASolver
from dwpa_solver.AO import AOSolver
from dwpa_solver.gurobi import GurobiSolver

# Other Competitors
from dwpa_competitor.dola22_solver import DOLA22Solver
from dwpa_competitor.icsoc19_solver import ICSOC19Solver
from dwpa_competitor.ycl24_solver import YCL24Solver

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
    logger = SimulationLogger(algorithm, output_dir)
    data_loader = DataLoader()
    warning_file_path = os.path.join(output_dir, f"logs/{algorithm}_constraint_warnings.log")
    env = CloudEdgeEnvironment(data_loader, warning_log_file=warning_file_path, logger=logger)
    total_steps = env.max_time_steps
    
    solver = None

    solver_classes = {
        'DWPA': DWPASolver,
        'DWPALF': lambda env: DWPASolver(env, 'LF'),
        'DWPAVO': lambda env: DWPASolver(env, 'VO'),
        'DWPAHF': lambda env: DWPASolver(env, 'HF'),
        'FIXTIMEDWPA': FIXTIMEDWPASolver,
        'AODWPA': AOSolver,
        'GUROBI': GurobiSolver,
        'DOLA22': DOLA22Solver,
        'ICSOC19': ICSOC19Solver,
        'YCL24': YCL24Solver
    }

    if algorithm in solver_classes:
        solver = solver_classes[algorithm](env)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # Execution Loop
    state = env.reset()
    history_carbon, history_q = [], []
    done = False
    step_count = 0
    
    while not done:
        # Get Action
        if solver is not None:
            decisions = solver.solve(state)
        else:
            raise ValueError(f"Solver is None, Unkown Solver, algorithm is: {algorithm}")
            
        # Environment Step
        next_state, carbon, done, info = env.step(decisions)
        
        history_carbon.append(carbon)
        history_q.append(info['q_avg_total']) 
               
        state = next_state
        step_count += 1
        
    total_carbon = sum(history_carbon)
    avg_q = np.mean(history_q)
    
    logger.close()
    
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
        
        with open(files['config'], 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        algorithms_to_run = config_data.get('algorithms', {}).get('run_list', [])
        
        if not algorithms_to_run:
             print("[Warning] No algorithms specified in config.json under 'algorithms.run_list'.")

        print("="*63)
        for algo in algorithms_to_run:
            c, q = run_simulation(algo, args.output_dir)
            print(f"{algo:<11}| Carbon: {c:10.4f} g | Avg Queue: {q / Config.MB_TO_BITS:10.2f} MB")
        print("="*63)
        
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        sys.exit(1)