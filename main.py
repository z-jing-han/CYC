import numpy as np
from config import Config
from data_loader import DataLoader
from marl_env import CloudEdgeEnvironment
from dwpa_solver import DWPASolver
from marl_agent import RandomAgent, QLearningAgent, MARLController

def run_simulation(algorithm='DWPA'):
    print(f"\n=== Starting Simulation with Algorithm: {algorithm} ===")
    
    # 1. Initialize Data and Environment
    data_loader = DataLoader()
    env = CloudEdgeEnvironment(data_loader)
    
    # Get total steps
    total_steps = env.max_time_steps
    print(f"Total Time Slots for this round: {total_steps}")
    
    # 2. Initialize Controller / Solver
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
    
    # 3. Execution Loop (Single Round)
    state = env.reset()
    
    history_carbon = []
    history_q = []
    
    done = False
    step_count = 0
    
    while not done:
        # Get Decisions
        if algorithm == 'DWPA':
            decisions = solver.solve(state)
        else: # MARL or QLearning
            decisions = marl_controller.get_decisions(state)
            
        # Environment Step
        next_state, carbon, done, info = env.step(decisions)
        
        # Training Step (For RL Algorithms)
        if algorithm == 'QLearning':
            # Reward: -(Weight * Carbon + Queue)
            q_values = next_state['Q_edge']
            rewards = []
            for i in range(Config.NUM_EDGE_SERVERS):
                r = - (0.1 * carbon / Config.NUM_EDGE_SERVERS + 1e-6 * q_values[i])
                rewards.append(r)
            
            marl_controller.update_agents(state, decisions, rewards, next_state)
        
        history_carbon.append(carbon)
        history_q.append(info['q_avg_total']) 
        
        # Log
        if step_count % 50 == 0 or step_count == total_steps - 1:
             print(f"Step {step_count:04d}: C={carbon:.4f}, Q_sys={info['q_avg_total']/1e6:.2f}Mb "
                   f"(Loc:{info['processed_local']/1e6:.2f}, Cld:{info['processed_cloud']/1e6:.2f}, OffC:{info['offloaded_cloud']/1e6:.2f})")
        
        state = next_state
        step_count += 1
        
    total_carbon = sum(history_carbon)
    avg_q = np.mean(history_q)
    
    print(f"\n>>> Simulation Finished ({algorithm}) <<<")
    print(f"Total Carbon: {total_carbon:.4f}")
    print(f"Avg System Queue: {avg_q:.2f}")
    
    return total_carbon, avg_q

if __name__ == "__main__":
    c_dwpa, q_dwpa = run_simulation('DWPA')
    c_marl, q_marl = run_simulation('MARL')
    c_ql, q_ql = run_simulation('QLearning')
    
    print("\n" + "="*30)
    print("=== FINAL COMPARISON ===")
    print("="*30)
    print(f"DWPA      | Carbon: {c_dwpa:10.4f} | Avg Queue: {q_dwpa/1e6:10.2f} Mb")
    print(f"MARL(Rnd) | Carbon: {c_marl:10.4f} | Avg Queue: {q_marl/1e6:10.2f} Mb")
    print(f"QLearning | Carbon: {c_ql:10.4f} | Avg Queue: {q_ql/1e6:10.2f} Mb")
    print("="*30)