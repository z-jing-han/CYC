import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import csv
from config import Config

from .base_solver import BaseMARLSolver, RunningMeanStd, device
from .maddpg_solver import calculate_rewards

class MAPPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super(MAPPOActor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, action_dim),
            nn.Sigmoid()
        )
        # Set standard deviation as a trainable independent parameter, independent of state
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def forward(self, obs, action=None):
        mu = self.net(obs)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()
            
        # Use unclipped actions to calculate log probability and entropy
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        action_clipped = torch.clamp(action, 0.0, 1.0)
        
        return action_clipped, action, log_prob, entropy

class MAPPOCritic(nn.Module):
    def __init__(self, obs_dim):
        # Critic is a state-value function V(s) and does not take actions as input
        super(MAPPOCritic, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, obs):
        return self.net(obs)

class MAPPOSolver(BaseMARLSolver):
    def __init__(self, env, decoder, use_ctde=False):
        super().__init__(env, algo_name='MAPPO', decoder=decoder, use_ctde=use_ctde)
        
        self.gamma = getattr(Config, 'MARL_GAMMA', 0.99)
        self.clip_param = getattr(Config, 'MAPPO_CLIP', 0.2)
        self.ppo_epoch = getattr(Config, 'MAPPO_EPOCH', 10)
        self.entropy_coef = getattr(Config, 'MAPPO_ENTROPY_COEF', 0.01)
        
        lr_actor = getattr(Config, 'MARL_LR_ACTOR', 3e-4)
        lr_critic = getattr(Config, 'MARL_LR_CRITIC', 1e-3)
        
        self.global_obs_dim = 0
        for i in range(self.num_edge):
            num_neighbors = len(env.neighbors_map.get(i, []))
            self.global_obs_dim += (4 + num_neighbors)
        
        for i in range(self.num_edge):
            neighbors = env.neighbors_map.get(i, [])
            num_neighbors = len(neighbors)
            
            obs_dim = 4 + num_neighbors
            action_dim = self.decoder.get_action_dim(num_neighbors)
            critic_obs_dim = self.global_obs_dim if self.use_ctde else obs_dim 
            
            actor = MAPPOActor(obs_dim, action_dim).to(device)
            critic = MAPPOCritic(critic_obs_dim).to(device)
            
            self.agents[i] = {
                'neighbors': neighbors,
                'obs_dim': obs_dim,
                'action_dim': action_dim,
                'actor': actor,
                'critic': critic,
                'actor_opt': optim.Adam(actor.parameters(), lr=lr_actor),
                'critic_opt': optim.Adam(critic.parameters(), lr=lr_critic),
                'obs_normalizer': RunningMeanStd(shape=(obs_dim,)) 
            }

    def solve(self, state, store_rollout=False):
        raw_actions = {}
        log_probs = {}
        values = {}
        unclipped_actions = {}
        
        global_obs_list = [self._extract_obs(state, j) for j in range(self.num_edge)]
        global_obs = torch.cat(global_obs_list, dim=1) if self.use_ctde else None

        for i in range(self.num_edge):
            obs_tensor = global_obs_list[i]
            
            with torch.no_grad():
                if self.is_training:
                    action_clipped, action_unclipped, log_prob, _ = self.agents[i]['actor'](obs_tensor)
                else:
                    mu = self.agents[i]['actor'].net(obs_tensor)
                    action_clipped = torch.clamp(mu, 0.0, 1.0)
                    action_unclipped = action_clipped
                    log_prob = torch.zeros(1)
                
                if self.use_ctde:
                    val = self.agents[i]['critic'](global_obs)
                else:
                    val = self.agents[i]['critic'](obs_tensor)
                    
            raw_actions[i] = action_clipped.squeeze(0).cpu().numpy()
            
            if store_rollout:
                unclipped_actions[i] = action_unclipped.squeeze(0).cpu().numpy()
                log_probs[i] = log_prob.item()
                values[i] = val.item()
        
        decisions = self.decoder.decode(state, raw_actions, self.num_edge, self.env.neighbors_map)
        decisions['raw_actions'] = raw_actions
        
        if store_rollout:
            decisions['unclipped_actions'] = unclipped_actions
            decisions['log_probs'] = log_probs
            decisions['values'] = values
            
        return decisions

    def train(self, rollouts):
        """PPO Update Logic"""
        for i in range(self.num_edge):
            agent = self.agents[i]
            
            obs = torch.tensor(np.array(rollouts[i]['obs']), dtype=torch.float32).to(device)
            acts = torch.tensor(np.array(rollouts[i]['acts']), dtype=torch.float32).to(device)
            old_log_probs = torch.tensor(np.array(rollouts[i]['log_probs']), dtype=torch.float32).unsqueeze(1).to(device)
            returns = torch.tensor(np.array(rollouts[i]['returns']), dtype=torch.float32).unsqueeze(1).to(device)
            advantages = torch.tensor(np.array(rollouts[i]['advs']), dtype=torch.float32).unsqueeze(1).to(device)
            
            if self.use_ctde:
                global_obs = torch.tensor(np.array(rollouts[i]['global_obs']), dtype=torch.float32).to(device)
                critic_input = global_obs
            else:
                critic_input = obs

            # Normalize Advantage
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Update K epoch
            for _ in range(self.ppo_epoch):
                _, _, log_probs, entropy = agent['actor'](obs, action=acts)
                values = agent['critic'](critic_input)
                
                # Compute r(theta)
                ratios = torch.exp(log_probs - old_log_probs.detach())
                
                # PPO Clipped Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                
                # Critic Loss (MSE)
                critic_loss = nn.MSELoss()(values, returns)
                
                agent['actor_opt'].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent['actor'].parameters(), 0.5)
                agent['actor_opt'].step()
                
                agent['critic_opt'].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent['critic'].parameters(), 0.5)
                agent['critic_opt'].step()

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advs = []
    gae = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        else:
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advs.insert(0, gae)
    return advs

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
                rollouts[i]['rewards'].append(rewards[i] / 1e10)
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