import os
import csv
import random
import torch
import numpy as np
from collections import deque

from config import Config
from env_simulator import CloudEdgeEnvironment
from data_loader import DataLoader

from .action_related.decoders import XPDecoder, XTDecoder, XTRDecoder
from .action_related.ao_decoders import AO_XPDecoder, AO_XTDecoder

from .algorithm_related.base_solver import device
from .algorithm_related.maddpg import MADDPGSolver
from .algorithm_related.mappo import MAPPOSolver
from .algorithm_related.ao_solver import MAAODDPGSolver, MAAOPPOSolver

from .utils import calculate_rewards, compute_gae

def setup_marl_solver(algorithm_config_str, env, output_dir):
    parts = algorithm_config_str.split('_')
    algo_name = parts[0]
    decoder_name = parts[1] if len(parts) > 1 else 'XT'
    use_ctde = True if len(parts) > 2 and parts[2] == 'CTDE' else False

    available_decoders = {
        'XP': XPDecoder(),
        'XT': XTDecoder(),
        'XTR': XTRDecoder(),
        'AOXP': AO_XPDecoder(),
        'AOXT': AO_XTDecoder()
    }

    available_marl_solvers = {
        'MADDPG': MADDPGSolver,
        'MAPPO': MAPPOSolver,
        'MAAODDPG': MAAODDPGSolver,
        'MAAOPPO': MAAOPPOSolver
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
            
            train_loader = DataLoader(carbon_path=Config.CARBON_TRAIN_FILE)
            train_env = CloudEdgeEnvironment(train_loader, is_training=True)
            
            if decoder_name == 'XP': train_decoder = XPDecoder()
            elif decoder_name == 'XT': train_decoder = XTDecoder()
            elif decoder_name == 'XTR': train_decoder = XTRDecoder()
            elif decoder_name == 'AOXP': train_decoder = AO_XPDecoder()
            elif decoder_name == 'AOXT': train_decoder = AO_XTDecoder()
            else: train_decoder = XTDecoder()
            
            if algo_name == 'MADDPG':
                train_solver = MADDPGSolver(train_env, train_decoder, use_ctde)
                training_func = run_ddpg_training
            elif algo_name == 'MAPPO':
                train_solver = MAPPOSolver(train_env, train_decoder, use_ctde)
                training_func = run_mappo_training
            elif algo_name == 'MAAODDPG':
                train_solver = MAAODDPGSolver(train_env, train_decoder, use_ctde) 
                training_func = run_ddpg_training                           
            elif algo_name == 'MAAOPPO':
                train_solver = MAAOPPOSolver(train_env, train_decoder, use_ctde)  
                training_func = run_mappo_training
            else:
                print(f"[Warning] Unknown algorithm name {algo_name}")
                continue
            
            expected_weight_path = os.path.join(output_dir, train_solver.weight_filename)
            
            if not os.path.exists(expected_weight_path):
                print(f"[Training] Start training for {algo_config_str}...")
                training_func(train_env, train_solver, output_dir)

def run_ddpg_training(env, solver, output_dir):
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"{solver.algo_name}_{solver.decoder.__class__.__name__}_{'CTDE' if solver.use_ctde else 'Decentralized'}_Reward.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    training_history = [] 

    episodes = getattr(Config, 'MARL_EPISODES', 500)
    batch_size = getattr(Config, 'MARL_BATCH_SIZE', 64)
    buffer_size = getattr(Config, 'MARL_BUFFER_SIZE', 10000)
    
    replay_buffer = deque(maxlen=buffer_size)
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset()
        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        
        while not done:
            decisions = solver.solve(state)
            next_state, carbon, done, info = env.step(decisions)

            epoch_queue.append(np.mean(next_state['Q_edge'])) 
            epoch_carbon += carbon

            rewards = calculate_rewards(state, next_state, info, carbon, decisions, V_param=Config.MARL_V)
            epoch_reward += sum(rewards.values())
            
            obs_dict, act_dict, nobs_dict = {}, {}, {}
            for i in range(env.num_edge):
                obs_dict[i] = solver._extract_obs(state, i).squeeze(0).cpu().numpy()
                nobs_dict[i] = solver._extract_obs(next_state, i).squeeze(0).cpu().numpy()
                act_dict[i] = decisions['raw_actions'][i]
                
            replay_buffer.append((obs_dict, act_dict, rewards, nobs_dict, float(done)))
            
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                b_done = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                
                b_obs_dict, b_act_dict, b_rew_dict, b_nobs_dict = {}, {}, {}, {}
                for i in range(env.num_edge):
                    b_obs_dict[i] = torch.tensor(np.array([x[0][i] for x in batch]), dtype=torch.float32).to(device)
                    b_act_dict[i] = torch.tensor(np.array([x[1][i] for x in batch]), dtype=torch.float32).to(device)
                    b_rew_dict[i] = torch.tensor(np.array([x[2][i] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    b_nobs_dict[i] = torch.tensor(np.array([x[3][i] for x in batch]), dtype=torch.float32).to(device)
                
                for i in range(env.num_edge):
                    solver.train(i, b_obs_dict, b_act_dict, b_rew_dict, b_nobs_dict, b_done)
            
            state = next_state
        
        avg_q = np.mean(epoch_queue)
        training_history.append([ep + 1, epoch_reward, epoch_carbon, avg_q])
        print(f"[{solver.algo_name}] Ep {ep+1:3d} | R: {epoch_reward:12.4f} | C: {epoch_carbon:10.4f} g | Avg Q: {avg_q:12.4f} bits")

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            print(f"*** New best reward {best_reward:.4f}! Saving weights... ***")
            solver.save_weights(output_dir)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Total_Carbon", "Avg_Queue"])
        writer.writerows(training_history)
    print(f"MARL training history saved to: {csv_path}")

def run_mappo_training(env, solver, output_dir):
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"{solver.algo_name}_{solver.decoder.__class__.__name__}_{'CTDE' if solver.use_ctde else 'Decentralized'}_Reward.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    training_history = []
    episodes = getattr(Config, 'MARL_EPISODES', 500)
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset()

        if Config.OBSERVATION_PREV:
            solver.reset_internal_state(state['Q_edge'])

        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        
        # Collect a full trajectory (on-policy)
        rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []} for i in range(env.num_edge)}
        
        while not done:
            # Instruct the Solver to store log_probs and values
            decisions = solver.solve(state, store_rollout=True)
            next_state, carbon, done, info = env.step(decisions)
            
            epoch_queue.append(np.mean(next_state['Q_edge'])) 
            epoch_carbon += carbon
            
            rewards = calculate_rewards(state, next_state, info, carbon, decisions, V_param=Config.MARL_V)
            epoch_reward += sum(rewards.values())
            
            global_obs = np.concatenate([solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
            
            for i in range(env.num_edge):
                rollouts[i]['obs'].append(solver._extract_obs(state, i).squeeze(0).cpu().numpy())
                rollouts[i]['global_obs'].append(global_obs)
                rollouts[i]['acts'].append(decisions['unclipped_actions'][i])
                rollouts[i]['log_probs'].append(decisions['log_probs'][i])
                rollouts[i]['rewards'].append(rewards[i])
                rollouts[i]['values'].append(decisions['values'][i])
                rollouts[i]['dones'].append(float(done))
                
            state = next_state
            
        # Episode End, compute GAE and Return
        with torch.no_grad():
            global_next_obs = np.concatenate([solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
            global_next_obs_tensor = torch.tensor(global_next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            for i in range(env.num_edge):
                next_obs_tensor = solver._extract_obs(state, i)
                if solver.use_ctde:
                    next_value = solver.agents[i]['critic'](global_next_obs_tensor).item()
                else:
                    next_value = solver.agents[i]['critic'](next_obs_tensor).item()
                    
                advs = compute_gae(rollouts[i]['rewards'], rollouts[i]['values'], next_value, rollouts[i]['dones'])
                returns = [adv + val for adv, val in zip(advs, rollouts[i]['values'])]
                
                rollouts[i]['advs'] = advs
                rollouts[i]['returns'] = returns
        
        solver.train(rollouts)

        avg_q = np.mean(epoch_queue)
        training_history.append([ep + 1, epoch_reward, epoch_carbon, avg_q])
        print(f"[{solver.algo_name}] Ep {ep+1:3d} | R: {epoch_reward:12.4f} | C: {epoch_carbon:10.4f} g | Avg Q: {avg_q:12.4f} bits")

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            print(f"*** New best reward {best_reward:.4f}! Saving weights... ***")
            solver.save_weights(output_dir)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Total_Carbon", "Avg_Queue"])
        writer.writerows(training_history)
    print(f"MARL training history saved to: {csv_path}")