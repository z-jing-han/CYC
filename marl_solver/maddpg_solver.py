import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
import csv
import random
from collections import deque
from config import Config

from .base_solver import BaseMARLSolver, RunningMeanStd, device

class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.fc(x)

class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + action_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=1)
        return self.fc(x)

class MADDPGSolver(BaseMARLSolver):
    def __init__(self, env, decoder, use_ctde=False):
        super().__init__(env, algo_name='MADDPG', decoder=decoder, use_ctde=use_ctde)
        
        self.gamma = getattr(Config, 'MARL_GAMMA', 0.99)
        lr_actor = getattr(Config, 'MARL_LR_ACTOR', 1e-4)
        lr_critic = getattr(Config, 'MARL_LR_CRITIC', 1e-3)
        
        # 1. Precompute the global dimensions required for CTDE
        self.global_obs_dim = 0
        self.global_action_dim = 0
        for i in range(self.num_edge):
            num_neighbors = len(env.neighbors_map.get(i, []))
            self.global_obs_dim += (4 + num_neighbors)
            self.global_action_dim += self.decoder.get_action_dim(num_neighbors)
        
        # Init Network
        for i in range(self.num_edge):
            neighbors = env.neighbors_map.get(i, [])
            num_neighbors = len(neighbors)
            
            obs_dim = 4 + num_neighbors
            action_dim = self.decoder.get_action_dim(num_neighbors)
            
            # Determine the Critic input dimension based on whether CTDE is enabled
            critic_obs_dim = self.global_obs_dim if self.use_ctde else obs_dim 
            critic_action_dim = self.global_action_dim if self.use_ctde else action_dim
            
            actor = ActorNetwork(obs_dim, action_dim).to(device)
            critic = CriticNetwork(critic_obs_dim, critic_action_dim).to(device)

            actor_target = copy.deepcopy(actor).to(device)
            critic_target = copy.deepcopy(critic).to(device)
            
            self.agents[i] = {
                'neighbors': neighbors,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'actor': actor,
                'critic': critic,
                'actor_target': actor_target,
                'critic_target': critic_target,
                'actor_opt': optim.Adam(actor.parameters(), lr=lr_actor),
                'critic_opt': optim.Adam(critic.parameters(), lr=lr_critic),
                'obs_normalizer': RunningMeanStd(shape=(obs_dim,)) 
            }

    # 2. Receive a full batch dictionary to support global concatenation
    def train(self, agent_id, b_obs_dict, b_act_dict, b_rew_dict, b_nobs_dict, b_done):
        agent = self.agents[agent_id]
        
        obs = b_obs_dict[agent_id]
        action = b_act_dict[agent_id]
        reward = b_rew_dict[agent_id]
        next_obs = b_nobs_dict[agent_id]

        if self.use_ctde:
            # Concatenate the states and actions of all agents
            global_obs = torch.cat([b_obs_dict[j] for j in range(self.num_edge)], dim=1)
            global_act = torch.cat([b_act_dict[j] for j in range(self.num_edge)], dim=1)
            next_global_obs = torch.cat([b_nobs_dict[j] for j in range(self.num_edge)], dim=1)
            
            with torch.no_grad():
                next_global_act_list = []
                for j in range(self.num_edge):
                    next_global_act_list.append(self.agents[j]['actor_target'](b_nobs_dict[j]))
                next_global_act = torch.cat(next_global_act_list, dim=1)
                
                target_q = reward + (1 - b_done) * self.gamma * agent['critic_target'](next_global_obs, next_global_act)
            
            current_q = agent['critic'](global_obs, global_act)
            
        else:
            # Decentralized
            with torch.no_grad():
                next_action = agent['actor_target'](next_obs)
                target_q = reward + (1 - b_done) * self.gamma * agent['critic_target'](next_obs, next_action)
            current_q = agent['critic'](obs, action)

        # Update Critic
        agent['critic_opt'].zero_grad()
        critic_loss = F.mse_loss(current_q, target_q)
        critic_loss.backward()
        agent['critic_opt'].step()
        
        # Update Actor (DPG)
        agent['actor_opt'].zero_grad()
        if self.use_ctde:
            # updating the Actor, replace its specific action in global_act while keeping other agents' actions unchanged
            curr_act_list = []
            for j in range(self.num_edge):
                if j == agent_id:
                    curr_act_list.append(agent['actor'](obs))
                else:
                    curr_act_list.append(b_act_dict[j].detach())
            new_global_act = torch.cat(curr_act_list, dim=1)
            actor_loss = -agent['critic'](global_obs, new_global_act).mean()
        else:
            actor_loss = -agent['critic'](obs, agent['actor'](obs)).mean()
            
        actor_loss.backward()
        agent['actor_opt'].step()

        # Soft Update Target Networks
        tau = 0.005
        for param, target_param in zip(agent['critic'].parameters(), agent['critic_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for param, target_param in zip(agent['actor'].parameters(), agent['actor_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()

def calculate_rewards(state, next_state, info, carbon, decisions, V_param=Config.MARL_V, penalty_weight=1e6):
    rewards = {}
    num_edge = len(state['Q_edge'])
    edge_metrics = info.get('edge_metrics', [])
    cloud_metrics = info.get('cloud_metrics', [])
    penalties = decisions.get('penalties', np.zeros(num_edge))

    for i in range(num_edge):
        q_penalty = next_state['Q_edge'][i] + next_state['Q_cloud'][i]
        carbon_penalty = 0.0
        if i < len(edge_metrics):
            carbon_penalty += edge_metrics[i]['carbon']
        if i < len(cloud_metrics):
            carbon_penalty += cloud_metrics[i]['carbon']
        power_violation_penalty = penalty_weight * penalties[i]
        
        rewards[i] = - (q_penalty + V_param * carbon_penalty + power_violation_penalty)
        
    return rewards

def run_marl_training(env, solver, output_dir):
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
                    b_rew_dict[i] = torch.tensor(np.array([x[2][i] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device) / 1e10
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