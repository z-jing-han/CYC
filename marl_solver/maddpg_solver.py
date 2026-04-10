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
        
        # Init Network
        for i in range(self.num_edge):
            neighbors = env.neighbors_map.get(i, [])
            num_neighbors = len(neighbors)
            
            obs_dim = 4 + num_neighbors
            # Ask the decoder about the output dim of NN
            action_dim = self.decoder.get_action_dim(num_neighbors)
            
            # For future CTDE expansion, the critic's obs_dim can be set to global_obs_dim
            critic_obs_dim = obs_dim 
            
            actor = ActorNetwork(obs_dim, action_dim).to(device)
            critic = CriticNetwork(critic_obs_dim, action_dim).to(device)

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

    def train(self, agent_id, obs, action, reward, next_obs, done):
        agent = self.agents[agent_id]
        
        # 1. Update Critic
        agent['critic_opt'].zero_grad()
        with torch.no_grad():
            next_action = agent['actor_target'](next_obs)
            target_q = reward + (1 - done) * self.gamma * agent['critic_target'](next_obs, next_action)
        
        current_q = agent['critic'](obs, action)
        critic_loss = F.mse_loss(current_q, target_q)
        critic_loss.backward()
        agent['critic_opt'].step()
        
        # 2. Update Actor (DPG)
        agent['actor_opt'].zero_grad()
        actor_loss = -agent['critic'](obs, agent['actor'](obs)).mean()
        actor_loss.backward()
        agent['actor_opt'].step()

        # 3. Soft Update Target Networks
        tau = 0.005
        for param, target_param in zip(agent['critic'].parameters(), agent['critic_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for param, target_param in zip(agent['actor'].parameters(), agent['actor_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()

def calculate_rewards(state, next_state, info, carbon, V_param=Config.MARL_V):
    rewards = {}
    num_edge = len(state['Q_edge'])
    edge_metrics = info.get('edge_metrics', [])
    cloud_metrics = info.get('cloud_metrics', [])

    for i in range(num_edge):
        q_penalty = next_state['Q_edge'][i] + next_state['Q_cloud'][i]
        carbon_penalty = 0.0
        if i < len(edge_metrics):
            carbon_penalty += edge_metrics[i]['carbon']
        if i < len(cloud_metrics):
            carbon_penalty += cloud_metrics[i]['carbon']
        rewards[i] = - (q_penalty + V_param * carbon_penalty)
    return rewards


def run_marl_training(env, solver, output_dir):
    csv_dir = os.path.join(output_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    csv_filename = f"{solver.algo_name}_{solver.decoder.__class__.__name__}_Reward.csv"
    csv_path = os.path.join(csv_dir, csv_filename)
    
    training_history = [] 

    episodes = getattr(Config, 'MARL_EPISODES', 500)
    batch_size = getattr(Config, 'MARL_BATCH_SIZE', 64)
    buffer_size = getattr(Config, 'MARL_BUFFER_SIZE', 10000)
    
    replay_buffer = {i: deque(maxlen=buffer_size) for i in range(env.num_edge)}
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

            rewards = calculate_rewards(state, next_state, info, carbon, V_param=Config.MARL_V)
            epoch_reward += sum(rewards.values())
            
            for i in range(env.num_edge):
                obs = solver._extract_obs(state, i).squeeze(0).cpu().numpy()
                next_obs = solver._extract_obs(next_state, i).squeeze(0).cpu().numpy()
                action = decisions['raw_actions'][i]
                
                replay_buffer[i].append((obs, action, rewards[i], next_obs, float(done)))
            
            if len(replay_buffer[0]) >= batch_size:
                for i in range(env.num_edge):
                    batch = random.sample(replay_buffer[i], batch_size)
                    
                    b_obs = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
                    b_act = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
                    b_rew = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    b_rew = b_rew / 1e10
                    b_nobs = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(device)
                    b_done = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    
                    solver.train(i, b_obs, b_act, b_rew, b_nobs, b_done)
            
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