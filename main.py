import numpy as np
from config import Config
from data_loader import DataLoader
from marl_env import CloudEdgeEnvironment
from dwpa_solver import DWPASolver
from marl_agent import RandomAgent, QLearningAgent, MARLController
from logger_utils import SimulationLogger

def run_simulation(algorithm='DWPA'):
    print(f"\n=== Starting Simulation with Algorithm: {algorithm} ===")
    
    # 1. Initialize Logger
    logger = SimulationLogger(algorithm, Config.CONFIG_FILE)
    
    # 2. Initialize Data and Environment
    data_loader = DataLoader()
    env = CloudEdgeEnvironment(data_loader, logger=logger, enable_dvfs=True)
    
    total_steps = env.max_time_steps
    print(f"Total Time Slots for this round: {total_steps}")
    
    # 3. Initialize Controller / Solver
    solver = None
    marl_controller = None
    
    if algorithm == 'DWPA':
        solver = DWPASolver(env)
    elif algorithm == 'MARL': 
        agents = [RandomAgent(i, Config.NUM_EDGE_SERVERS) for i in range(Config.NUM_EDGE_SERVERS)]
        marl_controller = MARLController(env, agents)
    elif algorithm == 'QLearning':
        agents = [QLearningAgent(i, Config.NUM_EDGE_SERVERS) for i in range(Config.NUM_EDGE_SERVERS)]
        marl_controller = MARLController(env, agents)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    # 4. Execution Loop
    state = env.reset()
    
    history_carbon = []
    history_q = []
    
    done = False
    step_count = 0
    
    TO_MB = 1.0 / Config.MB_TO_BITS
    
    while not done:
        # Get Decisions
        if algorithm == 'DWPA':
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
        if step_count % 50 == 0 or step_count == total_steps - 1:
             print(f"Step {step_count:04d}: C={carbon:.4f} g, Q_sys={info['q_avg_total']*TO_MB:.2f} MB "
                   f"(Loc:{info['processed_local']*TO_MB:.2f} MB, Cld:{info['processed_cloud']*TO_MB:.2f} MB, OffC:{info['offloaded_cloud']*TO_MB:.2f} MB)")
        
        state = next_state
        step_count += 1
        
    total_carbon = sum(history_carbon)
    avg_q = np.mean(history_q)
    
    logger.close()
    
    print(f"\n>>> Simulation Finished ({algorithm}) <<<")
    print(f"Total Carbon: {total_carbon:.4f} g")
    print(f"Avg System Queue: {avg_q*TO_MB:.2f} MB")
    
    return total_carbon, avg_q

if __name__ == "__main__":
    c_dwpa, q_dwpa = run_simulation('DWPA')
    c_marl, q_marl = run_simulation('MARL')
    c_ql, q_ql = run_simulation('QLearning')
    
    TO_MB = 1.0 / Config.MB_TO_BITS
    
    print("\n" + "="*40)
    print("=== FINAL COMPARISON (單位: g, MB) ===")
    print("="*40)
    print(f"DWPA      | Carbon: {c_dwpa:10.4f} g | Avg Queue: {q_dwpa*TO_MB:10.2f} MB")
    print(f"MARL(Rnd) | Carbon: {c_marl:10.4f} g | Avg Queue: {q_marl*TO_MB:10.2f} MB")
    print(f"QLearning | Carbon: {c_ql:10.4f} g | Avg Queue: {q_ql*TO_MB:10.2f} MB")
    print("="*40)