import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import Config
import copy
import os
import csv

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

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
        # Evaluate Q-value based on State and Action
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


class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

class MARLSolver:
    def __init__(self, env, variant='default'):
        self.env = env
        self.variant = variant
        self.num_edge = env.num_edge
        
        self.gamma = getattr(Config, 'MARL_GAMMA', 0.99)
        lr_actor = getattr(Config, 'MARL_LR_ACTOR', 1e-4)
        lr_critic = getattr(Config, 'MARL_LR_CRITIC', 1e-3)
        self.noise_std = getattr(Config, 'MARL_NOISE', 0.05)
        
        self.agents = {}
        for i in range(self.num_edge):
            neighbors = env.neighbors_map.get(i, [])
            num_neighbors = len(neighbors)
            
            obs_dim = 4 + num_neighbors
            action_dim = 4 + (2 * num_neighbors)
            
            actor = ActorNetwork(obs_dim, action_dim).to(device)
            critic = CriticNetwork(obs_dim, action_dim).to(device)

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

        self.is_training = True

    def _extract_obs(self, state, agent_id):
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        CI_edge = state['CI_edge']
        CI_cloud = state['CI_cloud']
        neighbors = self.agents[agent_id]['neighbors']
        
        obs = [
            Q_edge[agent_id],
            Q_cloud[agent_id],
            CI_edge[agent_id],
            CI_cloud[agent_id]
        ]
        
        for neighbor_id in neighbors:
            obs.append(Q_edge[neighbor_id])
        
        raw_obs = np.array(obs, dtype=np.float32)
        normalizer = self.agents[agent_id]['obs_normalizer']
        if self.is_training:
            normalizer.update(np.array([raw_obs]))
        normalized_obs = (raw_obs - normalizer.mean) / (np.sqrt(normalizer.var) + 1e-8)
        
        return torch.tensor(normalized_obs, dtype=torch.float32).unsqueeze(0).to(device)
    

    def solve(self, state):
        """Inference"""
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        f_edge = np.zeros(self.num_edge)
        f_cloud = np.zeros(self.num_edge)
        x_cloud = np.zeros(self.num_edge)
        p_cloud = np.zeros(self.num_edge)
        x_peer = np.zeros((self.num_edge, self.num_edge))
        p_peer = np.zeros((self.num_edge, self.num_edge))
        
        # tore raw outputs for training purposes
        raw_actions = {} 

        for i in range(self.num_edge):
            obs_tensor = self._extract_obs(state, i)
            
            # Add exploration noise for training
            with torch.no_grad():
                action_tensor = self.agents[i]['actor'](obs_tensor)
                if getattr(self, 'is_training', False):
                    noise = torch.normal(0, self.noise_std, size=action_tensor.size(), device=device)
                    action_tensor = torch.clamp(action_tensor + noise, 0.0, 1.0)
                action = action_tensor.squeeze(0).cpu().detach().numpy()
            
            raw_actions[i] = action
            
            # Freq Max mask
            needed_f_edge = (Q_edge[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_edge = min(Config.EDGE_F_MAX, needed_f_edge)
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_cloud = min(Config.CLOUD_F_MAX, needed_f_cloud)

            f_edge[i] = action[0] * max_valid_f_edge
            f_cloud[i] = action[1] * max_valid_f_cloud
            # f_edge[i] = action[0] * Config.EDGE_F_MAX
            # f_cloud[i] = action[1] * Config.CLOUD_F_MAX
            p_cloud[i] = action[3] * Config.EDGE_P_MAX
            raw_x_cloud = action[2] * Q_edge[i]


            
            raw_x_peers = []
            idx = 4
            for neighbor_id in self.agents[i]['neighbors']:
                if Q_edge[i] > Q_edge[neighbor_id]:
                    raw_x_peers.append((neighbor_id, action[idx] * Q_edge[i], action[idx+1] * Config.EDGE_P_MAX))
                else:
                    raw_x_peers.append((neighbor_id, 0.0, 0.0))
                idx += 2

            total_x_request = raw_x_cloud + sum([x for _, x, _ in raw_x_peers])
            scale_factor = min(1.0, Q_edge[i] / (total_x_request + 1e-9))
            
            x_cloud[i] = raw_x_cloud * scale_factor
            for neighbor_id, x_p, p_p in raw_x_peers:
                x_peer[i, neighbor_id] = x_p * scale_factor
                p_peer[i, neighbor_id] = p_p

        decisions = {
            'f_edge': f_edge,
            'f_cloud': f_cloud,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'raw_actions': raw_actions
        }
        return decisions

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
        # torch.nn.utils.clip_grad_norm_(agent['critic'].parameters(), max_norm=1.0)
        agent['critic_opt'].step()

        # 2. Update Actor (Via Deterministic Policy Gradient)
        agent['actor_opt'].zero_grad()
        actor_loss = -agent['critic'](obs, agent['actor'](obs)).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent['actor'].parameters(), max_norm=1.0)
        agent['actor_opt'].step()

        # 3. Soft Update Target Networks
        tau = 0.005
        for param, target_param in zip(agent['critic'].parameters(), agent['critic_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            
        for param, target_param in zip(agent['actor'].parameters(), agent['actor_target'].parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        return critic_loss.item(), actor_loss.item()
    
    def save_weights(self, filepath):
        state_dicts = {}
        for i, agent in self.agents.items():
            state_dicts[i] = {
                'actor': agent['actor'].state_dict(),
                'critic': agent['critic'].state_dict()
            }
        torch.save(state_dicts, filepath)
        print(f"MARL Weight saved: {filepath}")

    def load_weights(self, filepath):
        state_dicts = torch.load(filepath)
        for i, agent in self.agents.items():
            agent['actor'].load_state_dict(state_dicts[i]['actor'])
            agent['critic'].load_state_dict(state_dicts[i]['critic'])
        print(f"MARL weight load from {filepath}")
    

def calculate_rewards(state, next_state, info, V_param=Config.MARL_V):
    rewards = {}
    num_edge = len(state['Q_edge'])

    edge_metrics = info.get('edge_metrics', [])
    cloud_metrics = info.get('cloud_metrics', [])

    for i in range(num_edge):
        # Queue penalty: Encourage low queue length
        q_penalty = next_state['Q_edge'][i] + next_state['Q_cloud'][i]
        
        # Carbon penalty: Total system emissions
        carbon_penalty = 0.0
        if i < len(edge_metrics):
            carbon_penalty += edge_metrics[i]['carbon']
        if i < len(cloud_metrics):
            carbon_penalty += cloud_metrics[i]['carbon']
            
        rewards[i] = - (q_penalty + V_param * carbon_penalty)

    return rewards

from collections import deque
import random

def run_marl_training(env, solver, save_path):

    # --- Record the reword information ---
    out_dir = os.path.dirname(save_path)
    csv_dir = os.path.join(out_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, "maddpgReward.csv")
    
    training_history = [] # each epoch reward

    # Config parameter
    episodes = getattr(Config, 'MARL_EPISODES', 500)
    batch_size = getattr(Config, 'MARL_BATCH_SIZE', 64)
    buffer_size = getattr(Config, 'MARL_BUFFER_SIZE', 10000)
    # Simple Replay Buffer
    replay_buffer = {i: deque(maxlen=buffer_size) for i in range(env.num_edge)}
    
    for ep in range(episodes):
        state = env.reset()
        done = False
        epoch_carbon = 0.0
        epoch_queue = []
        epoch_reward = 0.0
        step_count = 0
        
        while not done:
            decisions = solver.solve(state)
            next_state, carbon, done, info = env.step(decisions)

            epoch_queue.append(np.mean(next_state['Q_edge'])) 
            epoch_carbon += carbon

            rewards = calculate_rewards(state, next_state, info, V_param=Config.MARL_V)
            
            epoch_reward += sum(rewards.values())
            
            # Store transition in buffer
            for i in range(env.num_edge):
                obs = solver._extract_obs(state, i).squeeze(0).cpu().numpy()
                next_obs = solver._extract_obs(next_state, i).squeeze(0).cpu().numpy()
                action = decisions['raw_actions'][i]
                
                replay_buffer[i].append((obs, action, rewards[i], next_obs, float(done)))
                
            # Sample from buffer and train
            if len(replay_buffer[0]) >= batch_size:
                for i in range(env.num_edge):
                    batch = random.sample(replay_buffer[i], batch_size)
                    
                    b_obs = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
                    b_act = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(device)
                    b_rew = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    b_rew = (b_rew - b_rew.mean()) / (b_rew.std() + 1e-8)
                    # print(b_rew)
                    b_nobs = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(device)
                    b_done = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).unsqueeze(1).to(device)
                    
                    c_loss, a_loss = solver.train(i, b_obs, b_act, b_rew, b_nobs, b_done)
            
            state = next_state
            step_count += 1
        
        avg_q = np.mean(epoch_queue)
        training_history.append([ep + 1, epoch_reward, epoch_carbon, avg_q])
        
        print(f"Episode {ep+1:3d} | Reward: {epoch_reward:12.4f} | Carbon: {epoch_carbon:10.4f} g | Avg Queue: {avg_q:12.4f} MB")
        # print(f"Episode {ep} | Total Carbon: {total_carbon:.4f} | Avg Queue: {info['q_avg_total']:.4f}")
    
    solver.save_weights(save_path)

    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Episode", "Total_Reward", "Total_Carbon", "Avg_Queue"])
        writer.writerows(training_history)
    print(f"MARL training history saved to: {csv_path}")
    