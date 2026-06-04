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
from .algorithm_related.split_solver import MAFreqPPOSolver
from .algorithm_related.rmappo import MARPPOSolver, MARAOPPOSolver, MARFreqPPOSolver

from .utils import calculate_rewards, compute_gae, compute_post_comp_state

def setup_marl_solver(algorithm_config_str, env, output_dir):
    parts = algorithm_config_str.split('_')
    algo_name = parts[0]
    decoder_name = parts[1] if len(parts) > 1 else 'XT'
    use_ctde = True if len(parts) > 2 and parts[2] == 'CTDE' else False

    if algo_name in ['MAAODDPG', 'MAAOPPO', 'MATWOPPO', 'MARTWOPPO']:
        if decoder_name == 'XP':
            decoder_name = 'AOXP'
        elif decoder_name == 'XT':
            decoder_name = 'AOXT'

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
        'MAAOPPO': MAAOPPOSolver,
        'MARPPO': MARPPOSolver,
        'MARAOPPO': MARAOPPOSolver
    }

    if algo_name not in available_marl_solvers and algo_name not in ['MATWOPPO', 'MARTWOPPO']:
        raise ValueError(f"Unknown marl algorithm: {algo_name}")
    if decoder_name not in available_decoders:
        raise ValueError(f"Unknown decoder: {decoder_name}")

    decoder_instance = available_decoders[decoder_name]

    # Create a lightweight wrapper to provide a unified 'solve' interface
    class MATWOPPOWrapper:
        def __init__(self, f_solver, o_solver):
            self.freq_solver = f_solver
            self.offload_solver = o_solver
            self.algo_name = "MATWOPPO"
            
        def reset_internal_state(self, initial_Q_edge):
            if hasattr(self.freq_solver, 'reset_internal_state'):
                self.freq_solver.reset_internal_state(initial_Q_edge)
            if hasattr(self.offload_solver, 'reset_internal_state'):
                self.offload_solver.reset_internal_state(initial_Q_edge)

        def solve(self, state, **kwargs):
            # Exe two agent action
            f_dec = self.freq_solver.solve(state, store_rollout=False)
            post_state = compute_post_comp_state(state, f_dec['f_edge'], f_dec['f_cloud'])
            o_dec = self.offload_solver.solve(post_state, store_rollout=False)
            
            # Combine decision result
            combined_dec = {**o_dec, 'f_edge': f_dec['f_edge'], 'f_cloud': f_dec['f_cloud']}
            return combined_dec

    # Handle MATWOPPO special loading logic (Dual-Agent architecture)
    if algo_name == 'MATWOPPO':
        from .algorithm_related.split_solver import MAFreqPPOSolver

        # Load Freq Solver
        freq_solver = MAFreqPPOSolver(env, use_ctde)
        freq_solver.algo_name = "MATWOPPO_Freq"
        freq_solver.weight_filename = f"{freq_solver.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        
        freq_path = os.path.join(output_dir, "weight", freq_solver.weight_filename)
        if os.path.exists(freq_path):
            freq_solver.load_weights(output_dir)
            freq_solver.is_training = False
        else:
            raise FileNotFoundError(f"Can't find Freq weight: {freq_path}")

        # Load Offload Solver
        offload_solver = MAPPOSolver(env, decoder_instance, use_ctde)
        offload_solver.algo_name = "MATWOPPO_Offload"
        offload_solver.weight_filename = f"{offload_solver.algo_name}_{decoder_instance.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        
        offload_path = os.path.join(output_dir, "weight", offload_solver.weight_filename)
        if os.path.exists(offload_path):
            offload_solver.load_weights(output_dir)
            offload_solver.is_training = False
        else:
            # Compatibility: Fallback to MAAOPPO weights if no checkpoints were updated during offload training
            ao_weight_name = f"MAAOPPO_{decoder_instance.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
            ao_weight_path = os.path.join(output_dir, "weight", ao_weight_name)
            if os.path.exists(ao_weight_path):
                original_name = offload_solver.weight_filename
                offload_solver.weight_filename = ao_weight_name
                offload_solver.load_weights(output_dir)
                offload_solver.weight_filename = original_name
                offload_solver.is_training = False
            else:
                raise FileNotFoundError(f"Can't find Offload weight {offload_path} or {ao_weight_path}")

        return MATWOPPOWrapper(freq_solver, offload_solver)
    elif algo_name == 'MARTWOPPO':
        freq_solver = MARFreqPPOSolver(env, use_ctde)
        freq_solver.algo_name = "MARTWOPPO_Freq"
        freq_solver.weight_filename = f"{freq_solver.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        
        freq_path = os.path.join(output_dir, "weight", freq_solver.weight_filename)
        if os.path.exists(freq_path):
            freq_solver.load_weights(output_dir)
            freq_solver.is_training = False
        else:
            raise FileNotFoundError(f"Can't find Freq weight: {freq_path}")

        offload_solver = MARPPOSolver(env, decoder_instance, use_ctde)
        offload_solver.algo_name = "MARTWOPPO_Offload"
        offload_solver.weight_filename = f"{offload_solver.algo_name}_{decoder_instance.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        
        offload_path = os.path.join(output_dir, "weight", offload_solver.weight_filename)
        if os.path.exists(offload_path):
            offload_solver.load_weights(output_dir)
            offload_solver.is_training = False
        else:
            # Load weight from MARAOPPO
            ao_weight_name = f"MARAOPPO_{decoder_instance.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
            ao_weight_path = os.path.join(output_dir, "weight", ao_weight_name)
            if os.path.exists(ao_weight_path):
                original_name = offload_solver.weight_filename
                offload_solver.weight_filename = ao_weight_name
                offload_solver.load_weights(output_dir)
                offload_solver.weight_filename = original_name
                offload_solver.is_training = False
            else:
                raise FileNotFoundError(f"Can't find Offload weight {offload_path} or {ao_weight_path}")
        return MATWOPPOWrapper(freq_solver, offload_solver)
    else:
        # Handle algorithm loading logic for standard single-Agent
        SolverClass = available_marl_solvers[algo_name]
        solver = SolverClass(env=env, decoder=decoder_instance, use_ctde=use_ctde)

        weights_path = os.path.join(output_dir, "weight", solver.weight_filename)
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
            
            if algo_name in ['MAAODDPG', 'MAAOPPO', 'MATWOPPO', 'MARTWOPPO']:
                if decoder_name == 'XP':
                    decoder_name = 'AOXP'
                elif decoder_name == 'XT':
                    decoder_name = 'AOXT'

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
            elif algo_name == 'MARPPO':
                train_solver = MARPPOSolver(train_env, train_decoder, use_ctde)
                training_func = run_rmappo_training
            elif algo_name == 'MARAOPPO':
                train_solver = MARAOPPOSolver(train_env, train_decoder, use_ctde)
                training_func = run_rmappo_training
            elif algo_name == 'MATWOPPO':
                # Split Two Agent Training Logic

                # Freq Solver (alaways training)
                freq_solver = MAFreqPPOSolver(train_env, use_ctde)
                freq_solver.algo_name = "MATWOPPO_Freq"
                freq_solver.weight_filename = f"{freq_solver.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
                
                # Offloading Solver
                offload_solver = MAPPOSolver(train_env, train_decoder, use_ctde)
                offload_solver.algo_name = "MATWOPPO_Offload"
                offload_solver.weight_filename = f"{offload_solver.algo_name}_{train_decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"

                # Check Computing weight and offloading weight is exist or not
                freq_weight_path = os.path.join(output_dir, "weight", freq_solver.weight_filename)
                offload_weight_path = os.path.join(output_dir, "weight", offload_solver.weight_filename)
                
                if os.path.exists(freq_weight_path) and os.path.exists(offload_weight_path):
                    continue

                # Load offloading weight
                if not getattr(Config, 'MATWOPPO_TRAIN_FROM_SCRATCH', True):
                    # Find AOPPO offloading wieght
                    ao_weight_name = f"MAAOPPO_{train_decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
                    ao_weight_path = os.path.join(output_dir, "weight", ao_weight_name)
                    
                    if os.path.exists(ao_weight_path):
                        original_name = offload_solver.weight_filename
                        offload_solver.weight_filename = ao_weight_name
                        offload_solver.load_weights(output_dir)
                        offload_solver.weight_filename = original_name
                        offload_solver.is_training = False
                        print(f"[{algo_name}] Load AOPPO Weight")
                    else:
                        print(f"[{algo_name}] Train AOPPO Weight")
                        offload_solver.is_training = True
                else:
                    offload_solver.is_training = True
                    print(f"[{algo_name}] Mode: Train both Freq and Offloading")

                # Run the decouple training process
                run_decoupled_split_ppo_training(train_env, freq_solver, offload_solver, output_dir)

                continue
            elif algo_name == 'MARTWOPPO':
                freq_solver = MARFreqPPOSolver(train_env, use_ctde)
                freq_solver.algo_name = "MARTWOPPO_Freq"
                freq_solver.weight_filename = f"{freq_solver.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
                
                offload_solver = MARPPOSolver(train_env, train_decoder, use_ctde)
                offload_solver.algo_name = "MARTWOPPO_Offload"
                offload_solver.weight_filename = f"{offload_solver.algo_name}_{train_decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"

                freq_weight_path = os.path.join(output_dir, "weight", freq_solver.weight_filename)
                offload_weight_path = os.path.join(output_dir, "weight", offload_solver.weight_filename)
                
                if os.path.exists(freq_weight_path) and os.path.exists(offload_weight_path):
                    continue

                if not getattr(Config, 'MATWOPPO_TRAIN_FROM_SCRATCH', True):
                    # Load weight from MARAOPPO
                    ao_weight_name = f"MARAOPPO_{train_decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
                    ao_weight_path = os.path.join(output_dir, "weight", ao_weight_name)
                    
                    if os.path.exists(ao_weight_path):
                        original_name = offload_solver.weight_filename
                        offload_solver.weight_filename = ao_weight_name
                        offload_solver.load_weights(output_dir)
                        offload_solver.weight_filename = original_name
                        offload_solver.is_training = False
                        print(f"[{algo_name}] Load MARAOPPO Weight")
                    else:
                        print(f"[{algo_name}] Train MARAOPPO Weight")
                        offload_solver.is_training = True
                else:
                    offload_solver.is_training = True
                    print(f"[{algo_name}] Mode: Train both Freq and Offloading")
                
                run_decoupled_split_rmappo_training(train_env, freq_solver, offload_solver, output_dir)
                continue
            else:
                print(f"[Warning] Unknown algorithm name {algo_name}")
                continue
            
            expected_weight_path = os.path.join(output_dir, "weight", train_solver.weight_filename)
            
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

            rewards = calculate_rewards(state, next_state, info, carbon, decisions)
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
            
            rewards = calculate_rewards(state, next_state, info, carbon, decisions)
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

def run_decoupled_split_ppo_training(env, freq_solver, offload_solver, output_dir):
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    ctde_str = 'CTDE' if freq_solver.use_ctde else 'Decentralized'
    decoder_name = offload_solver.decoder.__class__.__name__
    csv_path = os.path.join(csv_dir, f"MATWOPPO_{decoder_name}_{ctde_str}_Reward.csv")
    
    training_history = []
    episodes = getattr(Config, 'MARL_EPISODES', 500)
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset()
        
        if Config.OBSERVATION_PREV:
            if freq_solver.is_training: freq_solver.reset_internal_state(state['Q_edge'])
            if offload_solver.is_training: offload_solver.reset_internal_state(state['Q_edge'])

        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        
        # Two independent Rollout Buffer for different agent
        freq_rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []} for i in range(env.num_edge)}
        offload_rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': []} for i in range(env.num_edge)}
        
        while not done:
            f_dec = freq_solver.solve(state, store_rollout=freq_solver.is_training)
            post_state = compute_post_comp_state(state, f_dec['f_edge'], f_dec['f_cloud'])
            
            o_dec = offload_solver.solve(post_state, store_rollout=offload_solver.is_training)
            
            combined_dec = {**o_dec, 'f_edge': f_dec['f_edge'], 'f_cloud': f_dec['f_cloud']}
            next_state, carbon, done, info = env.step(combined_dec)
            rewards = calculate_rewards(state, next_state, info, carbon, combined_dec)

            epoch_queue.append(np.mean(next_state['Q_edge']))
            epoch_carbon += carbon
            epoch_reward += sum(rewards.values())
            
            # Store Freq Agent Training Data
            if freq_solver.is_training:
                global_obs_f = np.concatenate([freq_solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                for i in range(env.num_edge):
                    freq_rollouts[i]['obs'].append(freq_solver._extract_obs(state, i).squeeze(0).cpu().numpy())
                    freq_rollouts[i]['global_obs'].append(global_obs_f)
                    freq_rollouts[i]['acts'].append(f_dec['unclipped_actions'][i])
                    freq_rollouts[i]['log_probs'].append(f_dec['log_probs'][i])
                    freq_rollouts[i]['rewards'].append(rewards[i])
                    freq_rollouts[i]['values'].append(f_dec['values'][i])
                    freq_rollouts[i]['dones'].append(float(done))

            # Store Offload Agent Training Data
            if offload_solver.is_training:
                global_obs_o = np.concatenate([offload_solver._extract_obs(post_state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                for i in range(env.num_edge):
                    offload_rollouts[i]['obs'].append(offload_solver._extract_obs(post_state, i).squeeze(0).cpu().numpy())
                    offload_rollouts[i]['global_obs'].append(global_obs_o)
                    offload_rollouts[i]['acts'].append(o_dec['unclipped_actions'][i])
                    offload_rollouts[i]['log_probs'].append(o_dec['log_probs'][i])
                    offload_rollouts[i]['rewards'].append(rewards[i])
                    offload_rollouts[i]['values'].append(o_dec['values'][i])
                    offload_rollouts[i]['dones'].append(float(done))
                    
            state = next_state
        
        # Compute GAE
        with torch.no_grad():
            if freq_solver.is_training:
                global_next_obs_f = np.concatenate([freq_solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                global_next_obs_tensor_f = torch.tensor(global_next_obs_f, dtype=torch.float32).unsqueeze(0).to(device)
                
                for i in range(env.num_edge):
                    next_obs_tensor = freq_solver._extract_obs(state, i)
                    if freq_solver.use_ctde:
                        next_value = freq_solver.agents[i]['critic'](global_next_obs_tensor_f).item()
                    else:
                        next_value = freq_solver.agents[i]['critic'](next_obs_tensor).item()
                        
                    advs = compute_gae(freq_rollouts[i]['rewards'], freq_rollouts[i]['values'], next_value, freq_rollouts[i]['dones'])
                    freq_rollouts[i]['advs'] = advs
                    freq_rollouts[i]['returns'] = [adv + val for adv, val in zip(advs, freq_rollouts[i]['values'])]
            
            if offload_solver.is_training:
                next_f_dec = freq_solver.solve(state, store_rollout=False)
                next_post_state = compute_post_comp_state(state, next_f_dec['f_edge'], next_f_dec['f_cloud'])
                
                global_next_obs_o = np.concatenate([offload_solver._extract_obs(next_post_state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                global_next_obs_tensor_o = torch.tensor(global_next_obs_o, dtype=torch.float32).unsqueeze(0).to(device)
                
                for i in range(env.num_edge):
                    next_obs_tensor = offload_solver._extract_obs(next_post_state, i)
                    if offload_solver.use_ctde:
                        next_value = offload_solver.agents[i]['critic'](global_next_obs_tensor_o).item()
                    else:
                        next_value = offload_solver.agents[i]['critic'](next_obs_tensor).item()
                        
                    advs = compute_gae(offload_rollouts[i]['rewards'], offload_rollouts[i]['values'], next_value, offload_rollouts[i]['dones'])
                    offload_rollouts[i]['advs'] = advs
                    offload_rollouts[i]['returns'] = [adv + val for adv, val in zip(advs, offload_rollouts[i]['values'])]

        # Update
        if freq_solver.is_training:
            freq_solver.train(freq_rollouts)
            
        if offload_solver.is_training:
            offload_solver.train(offload_rollouts)
        
        avg_q = np.mean(epoch_queue)
        training_history.append([ep + 1, epoch_reward, epoch_carbon, avg_q])
        
        mode_str = "Train Both" if offload_solver.is_training else "Train Freq Only (Offload Frozen)"
        print(f"[MATWOPPO | {mode_str}] Ep {ep+1:3d} | R: {epoch_reward:12.4f} | C: {epoch_carbon:10.4f} g | Avg Q: {avg_q:12.4f} bits")

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            print(f"*** New best reward {best_reward:.4f}! Saving weights... ***")
            if freq_solver.is_training: freq_solver.save_weights(output_dir)
            if offload_solver.is_training: offload_solver.save_weights(output_dir)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Total_Carbon", "Avg_Queue"])
        writer.writerows(training_history)
    print(f"MATWOPPO training history saved to: {csv_path}")

def run_rmappo_training(env, solver, output_dir):
    # Almost follow mappo
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_filename = f"{solver.algo_name}_{solver.decoder.__class__.__name__}_{'CTDE' if solver.use_ctde else 'Decentralized'}_Reward.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    training_history = []
    episodes = getattr(Config, 'MARL_EPISODES', 500)
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset()
        
        # RNN
        solver.reset_rnn_states() 

        if Config.OBSERVATION_PREV:
            solver.reset_internal_state(state['Q_edge'])

        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        
        # New record actor_rnn_states and critic_rnn_states
        rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'actor_rnn_states': [], 'critic_rnn_states': []} for i in range(env.num_edge)}
        
        while not done:
            decisions = solver.solve(state, store_rollout=True)
            next_state, carbon, done, info = env.step(decisions)

            epoch_queue.append(np.mean(next_state['Q_edge'])) 
            epoch_carbon += carbon
            
            rewards = calculate_rewards(state, next_state, info, carbon, decisions)
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
                rollouts[i]['actor_rnn_states'].append(decisions['actor_rnn_states'][i])
                rollouts[i]['critic_rnn_states'].append(decisions['critic_rnn_states'][i])
                
            state = next_state
            
        with torch.no_grad():
            global_next_obs = np.concatenate([solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
            global_next_obs_tensor = torch.tensor(global_next_obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            for i in range(env.num_edge):
                next_obs_tensor = solver._extract_obs(state, i)
                if solver.use_ctde:
                    next_value, _ = solver.agents[i]['critic'](global_next_obs_tensor, solver.critic_rnn_states[i])
                else:
                    next_value, _ = solver.agents[i]['critic'](next_obs_tensor, solver.critic_rnn_states[i])
                    
                next_value = next_value.item()
                advs = compute_gae(rollouts[i]['rewards'], rollouts[i]['values'], next_value, rollouts[i]['dones'])
                rollouts[i]['advs'] = advs
                rollouts[i]['returns'] = [adv + val for adv, val in zip(advs, rollouts[i]['values'])]
        
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
    print(f"MARPPO training history saved to: {csv_path}")

def run_decoupled_split_rmappo_training(env, freq_solver, offload_solver, output_dir):
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    
    ctde_str = 'CTDE' if freq_solver.use_ctde else 'Decentralized'
    decoder_name = offload_solver.decoder.__class__.__name__
    csv_path = os.path.join(csv_dir, f"MARTWOPPO_{decoder_name}_{ctde_str}_Reward.csv")
    
    training_history = []
    episodes = getattr(Config, 'MARL_EPISODES', 500)
    best_reward = -float('inf')

    for ep in range(episodes):
        state = env.reset()
        
        # Reset two rnn state
        freq_solver.reset_rnn_states()
        offload_solver.reset_rnn_states()
        
        if Config.OBSERVATION_PREV:
            if freq_solver.is_training: freq_solver.reset_internal_state(state['Q_edge'])
            if offload_solver.is_training: offload_solver.reset_internal_state(state['Q_edge'])

        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        
        freq_rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'actor_rnn_states': [], 'critic_rnn_states': []} for i in range(env.num_edge)}
        offload_rollouts = {i: {'obs': [], 'global_obs': [], 'acts': [], 'log_probs': [], 'rewards': [], 'values': [], 'dones': [], 'actor_rnn_states': [], 'critic_rnn_states': []} for i in range(env.num_edge)}
        
        while not done:
            f_dec = freq_solver.solve(state, store_rollout=freq_solver.is_training)
            post_state = compute_post_comp_state(state, f_dec['f_edge'], f_dec['f_cloud'])
            
            o_dec = offload_solver.solve(post_state, store_rollout=offload_solver.is_training)
            
            combined_dec = {**o_dec, 'f_edge': f_dec['f_edge'], 'f_cloud': f_dec['f_cloud']}
            next_state, carbon, done, info = env.step(combined_dec)
            rewards = calculate_rewards(state, next_state, info, carbon, combined_dec)

            epoch_queue.append(np.mean(next_state['Q_edge']))
            epoch_carbon += carbon
            epoch_reward += sum(rewards.values())
            
            if freq_solver.is_training:
                global_obs_f = np.concatenate([freq_solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                for i in range(env.num_edge):
                    freq_rollouts[i]['obs'].append(freq_solver._extract_obs(state, i).squeeze(0).cpu().numpy())
                    freq_rollouts[i]['global_obs'].append(global_obs_f)
                    freq_rollouts[i]['acts'].append(f_dec['unclipped_actions'][i])
                    freq_rollouts[i]['log_probs'].append(f_dec['log_probs'][i])
                    freq_rollouts[i]['rewards'].append(rewards[i])
                    freq_rollouts[i]['values'].append(f_dec['values'][i])
                    freq_rollouts[i]['dones'].append(float(done))
                    freq_rollouts[i]['actor_rnn_states'].append(f_dec['actor_rnn_states'][i])
                    freq_rollouts[i]['critic_rnn_states'].append(f_dec['critic_rnn_states'][i])
            
            if offload_solver.is_training:
                global_obs_o = np.concatenate([offload_solver._extract_obs(post_state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                for i in range(env.num_edge):
                    offload_rollouts[i]['obs'].append(offload_solver._extract_obs(post_state, i).squeeze(0).cpu().numpy())
                    offload_rollouts[i]['global_obs'].append(global_obs_o)
                    offload_rollouts[i]['acts'].append(o_dec['unclipped_actions'][i])
                    offload_rollouts[i]['log_probs'].append(o_dec['log_probs'][i])
                    offload_rollouts[i]['rewards'].append(rewards[i])
                    offload_rollouts[i]['values'].append(o_dec['values'][i])
                    offload_rollouts[i]['dones'].append(float(done))
                    offload_rollouts[i]['actor_rnn_states'].append(o_dec['actor_rnn_states'][i])
                    offload_rollouts[i]['critic_rnn_states'].append(o_dec['critic_rnn_states'][i])
                    
            state = next_state
        
        with torch.no_grad():
            if freq_solver.is_training:
                global_next_obs_f = np.concatenate([freq_solver._extract_obs(state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                global_next_obs_tensor_f = torch.tensor(global_next_obs_f, dtype=torch.float32).unsqueeze(0).to(device)
                
                for i in range(env.num_edge):
                    next_obs_tensor = freq_solver._extract_obs(state, i)
                    if freq_solver.use_ctde:
                        next_value, _ = freq_solver.agents[i]['critic'](global_next_obs_tensor_f, freq_solver.critic_rnn_states[i])
                    else:
                        next_value, _ = freq_solver.agents[i]['critic'](next_obs_tensor, freq_solver.critic_rnn_states[i])
                        
                    advs = compute_gae(freq_rollouts[i]['rewards'], freq_rollouts[i]['values'], next_value.item(), freq_rollouts[i]['dones'])
                    freq_rollouts[i]['advs'] = advs
                    freq_rollouts[i]['returns'] = [adv + val for adv, val in zip(advs, freq_rollouts[i]['values'])]
            
            if offload_solver.is_training:
                next_f_dec = freq_solver.solve(state, store_rollout=False)
                next_post_state = compute_post_comp_state(state, next_f_dec['f_edge'], next_f_dec['f_cloud'])
                
                global_next_obs_o = np.concatenate([offload_solver._extract_obs(next_post_state, j).squeeze(0).cpu().numpy() for j in range(env.num_edge)])
                global_next_obs_tensor_o = torch.tensor(global_next_obs_o, dtype=torch.float32).unsqueeze(0).to(device)
                
                for i in range(env.num_edge):
                    next_obs_tensor = offload_solver._extract_obs(next_post_state, i)
                    if offload_solver.use_ctde:
                        next_value, _ = offload_solver.agents[i]['critic'](global_next_obs_tensor_o, offload_solver.critic_rnn_states[i])
                    else:
                        next_value, _ = offload_solver.agents[i]['critic'](next_obs_tensor, offload_solver.critic_rnn_states[i])
                        
                    advs = compute_gae(offload_rollouts[i]['rewards'], offload_rollouts[i]['values'], next_value.item(), offload_rollouts[i]['dones'])
                    offload_rollouts[i]['advs'] = advs
                    offload_rollouts[i]['returns'] = [adv + val for adv, val in zip(advs, offload_rollouts[i]['values'])]
        
        if freq_solver.is_training:
            freq_solver.train(freq_rollouts)
            
        if offload_solver.is_training:
            offload_solver.train(offload_rollouts)
        
        avg_q = np.mean(epoch_queue)
        training_history.append([ep + 1, epoch_reward, epoch_carbon, avg_q])
        
        mode_str = "Train Both" if offload_solver.is_training else "Train Freq Only (Offload Frozen)"
        print(f"[MARTWOPPO | {mode_str}] Ep {ep+1:3d} | R: {epoch_reward:12.4f} | C: {epoch_carbon:10.4f} g | Avg Q: {avg_q:12.4f} bits")

        if epoch_reward > best_reward:
            best_reward = epoch_reward
            print(f"*** New best reward {best_reward:.4f}! Saving weights... ***")
            if freq_solver.is_training: freq_solver.save_weights(output_dir)
            if offload_solver.is_training: offload_solver.save_weights(output_dir)
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Total_Carbon", "Avg_Queue"])
        writer.writerows(training_history)
    print(f"MARTWOPPO training history saved to: {csv_path}")