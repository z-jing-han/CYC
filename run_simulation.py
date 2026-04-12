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

# Other Competitors
from dwpa_competitor.dola22_solver import DOLA22Solver
from dwpa_competitor.icsoc19_solver import ICSOC19Solver
from dwpa_competitor.ycl24_solver import YCL24Solver

# DWPA Optimization Version
from dwpa_opt.fixtime import FIXTIMESolver
from dwpa_opt.AO import AOSolver
from dwpa_opt.gurobi import GurobiSolver

# MARL Method
from marl_solver.action_decoder import XPDecoder, XTDecoder, XTRDecoder
from marl_solver.maddpg_solver import MADDPGSolver, run_marl_training
from marl_solver.mappo_solver import MAPPOSolver, run_mappo_training

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


def setup_marl_solver(algorithm_config_str, env, output_dir):
    parts = algorithm_config_str.split('_')
    algo_name = parts[0]
    decoder_name = parts[1] if len(parts) > 1 else 'XT'
    use_ctde = True if len(parts) > 2 and parts[2] == 'CTDE' else False

    available_decoders = {
        'XP': XPDecoder(),
        'XT': XTDecoder(),
        'XTR': XTRDecoder()
    }

    available_marl_solvers = {
        'MADDPG': MADDPGSolver,
        'MAPPO': MAPPOSolver
    }

    if algo_name not in available_marl_solvers:
        raise ValueError(f"Unknown marl algorithm: {algo_name}")
    if decoder_name not in available_decoders:
        raise ValueError(f"Unknown decoder: {decoder_name}")

    decoder_instance = available_decoders[decoder_name]
    SolverClass = available_marl_solvers[algo_name]

    solver = SolverClass(env=env, decoder=decoder_instance, use_ctde=use_ctde)

    weights_path = os.path.join(output_dir, solver.weight_filename)
    if os.path.exists(weights_path):
        solver.load_weights(output_dir)
        solver.is_training = False 
    else:
        raise FileNotFoundError(f"Can't find the weight {weights_path}")
        
    return solver


def check_and_train_marl(algorithms_to_run, output_dir):
    for algo_config_str in algorithms_to_run:
        if algo_config_str.startswith("MA"):
            parts = algo_config_str.split('_')
            algo_name = parts[0]
            decoder_name = parts[1] if len(parts) > 1 else 'XT'
            use_ctde = True if len(parts) > 2 and parts[2] == 'CTDE' else False
            
            train_env = CloudEdgeEnvironment(DataLoader())
            
            if decoder_name == 'XP': train_decoder = XPDecoder()
            elif decoder_name == 'XT': train_decoder = XTDecoder()
            elif decoder_name == 'XTR': train_decoder = XTRDecoder()
            else: train_decoder = XTDecoder()
            
            if algo_name == 'MADDPG':
                train_solver = MADDPGSolver(train_env, train_decoder, use_ctde)
                training_func = run_marl_training
            elif algo_name == 'MAPPO':
                train_solver = MAPPOSolver(train_env, train_decoder, use_ctde)
                training_func = run_mappo_training
            else:
                print(f"[Warning] Unknown algorithm name {algo_name}")
                continue
            
            expected_weight_path = os.path.join(output_dir, train_solver.weight_filename)
            
            if not os.path.exists(expected_weight_path):
                print(f"[Training] Start training for {algo_config_str}...")
                training_func(train_env, train_solver, output_dir)
            
def run_simulation(algorithm_config_str, output_dir='Base_Output'):
    logger = SimulationLogger(algorithm_config_str, output_dir)
    data_loader = DataLoader()
    warning_file_path = os.path.join(output_dir, f"logs/{algorithm_config_str}_constraint_warnings.log")
    env = CloudEdgeEnvironment(data_loader, warning_log_file=warning_file_path, logger=logger)
    
    solver = None

    traditional_solver_classes = {
        'DWPA': DWPASolver,
        'DWPALF': lambda env: DWPASolver(env, 'LF'),
        'DWPAVO': lambda env: DWPASolver(env, 'VO'),
        'DWPAHF': lambda env: DWPASolver(env, 'HF'),
        'FIXTIME': FIXTIMESolver,
        'AO': AOSolver,
        'GUROBI': GurobiSolver,
        'DOLA22': DOLA22Solver,
        'ICSOC19': ICSOC19Solver,
        'YCL24': YCL24Solver
    }

    if algorithm_config_str.startswith("MA"):
        solver = setup_marl_solver(algorithm_config_str, env, output_dir)
    else:
        if algorithm_config_str in traditional_solver_classes:
            solver = traditional_solver_classes[algorithm_config_str](env)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_config_str}")
    
    state = env.reset()
    history_carbon, history_q = [], []
    done = False
    
    while not done:
        if solver is not None:
            decisions = solver.solve(state)
        else:
            raise ValueError(f"Solver is None, Unknown Solver: {algorithm_config_str}")
        
        next_state, carbon, done, info = env.step(decisions)
        
        history_carbon.append(carbon)
        history_q.append(info['q_avg_total']) 
               
        state = next_state
        
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
        
        # 5. traing marl model
        check_and_train_marl(algorithms_to_run, args.output_dir)

        # 6. Run Simulations
        print("="*63)
        for algo in algorithms_to_run:
            c, q = run_simulation(algo, args.output_dir)
            print(f"{algo:<20}| Carbon: {c:10.4f} g | Avg Queue: {q / Config.MB_TO_BITS:10.2f} MB")
        print("="*63)
        
    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[UNEXPECTED ERROR] {e}")
        sys.exit(1)