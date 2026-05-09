# algorithm_related/rmappo.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import os
import csv
from config import Config

from .base_solver import BaseMARLSolver, device
from ..utils import RunningMeanStd, compute_gae, compute_ao_state

class RMAPPOActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        super(RMAPPOActor, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Fully connect
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        # RNN (GRU) - batch_first=True imply input dim (batch, seq, feature)
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        
        # Output layer: Fully connect
        self.fc2 = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim) - 0.5)

    def forward(self, obs, rnn_state, action=None):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
            
        x = self.fc1(obs)
        x, new_rnn_state = self.rnn(x, rnn_state)
        x = x.squeeze(1)
        
        mu = self.fc2(x)
        std = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, std)

        if action is None:
            action = dist.sample()
            
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        action_clipped = torch.clamp(action, 0.0, 1.0)
        
        return action_clipped, action, log_prob, entropy, new_rnn_state

class RMAPPOCritic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super(RMAPPOCritic, self).__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        self.rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, obs, rnn_state):
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)
        x = self.fc1(obs)
        x, new_rnn_state = self.rnn(x, rnn_state)
        x = x.squeeze(1)
        return self.fc2(x), new_rnn_state

class MARPPOSolver(BaseMARLSolver):
    def __init__(self, env, decoder, use_ctde=False):
        super().__init__(env, algo_name='MARPPO', decoder=decoder, use_ctde=use_ctde)
        
        self.gamma = getattr(Config, 'MARL_GAMMA', 0.99)
        self.clip_param = getattr(Config, 'MAPPO_CLIP', 0.2)
        self.ppo_epoch = getattr(Config, 'MAPPO_EPOCH', 10)
        self.entropy_coef = getattr(Config, 'MAPPO_ENTROPY_COEF', 0.01)
        self.hidden_dim = getattr(Config, 'MARPPO_HIDDEN_DIM', 64)
        
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
            
            actor = RMAPPOActor(obs_dim, action_dim, self.hidden_dim).to(device)
            critic = RMAPPOCritic(critic_obs_dim, self.hidden_dim).to(device)
            
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
        
        self.reset_rnn_states()

    def reset_rnn_states(self):
        self.actor_rnn_states = {i: torch.zeros(1, 1, self.hidden_dim).to(device) for i in range(self.num_edge)}
        self.critic_rnn_states = {i: torch.zeros(1, 1, self.hidden_dim).to(device) for i in range(self.num_edge)}

    def solve(self, state, store_rollout=False):
        raw_actions, log_probs, values, unclipped_actions = {}, {}, {}, {}
        
        # Buffer for RNN
        actor_rnn_states_rollout = {i: self.actor_rnn_states[i].clone().detach() for i in range(self.num_edge)}
        critic_rnn_states_rollout = {i: self.critic_rnn_states[i].clone().detach() for i in range(self.num_edge)}
        
        global_obs_list = [self._extract_obs(state, j) for j in range(self.num_edge)]
        global_obs = torch.cat(global_obs_list, dim=1) if self.use_ctde else None

        for i in range(self.num_edge):
            obs_tensor = global_obs_list[i]
            
            with torch.no_grad():
                if self.is_training:
                    action_clipped, action_unclipped, log_prob, _, new_actor_state = self.agents[i]['actor'](
                        obs_tensor, self.actor_rnn_states[i]
                    )
                else:
                    # Inference
                    if obs_tensor.dim() == 2:
                        obs_tensor_sq = obs_tensor.unsqueeze(1)
                    else:
                        obs_tensor_sq = obs_tensor
                    x = self.agents[i]['actor'].fc1(obs_tensor_sq)
                    x, new_actor_state = self.agents[i]['actor'].rnn(x, self.actor_rnn_states[i])
                    x = x.squeeze(1)
                    mu = self.agents[i]['actor'].fc2(x)
                    action_clipped = torch.clamp(mu, 0.0, 1.0)
                    action_unclipped = action_clipped
                    log_prob = torch.zeros(1)
                
                if self.use_ctde:
                    val, new_critic_state = self.agents[i]['critic'](global_obs, self.critic_rnn_states[i])
                else:
                    val, new_critic_state = self.agents[i]['critic'](obs_tensor, self.critic_rnn_states[i])
                
                self.actor_rnn_states[i] = new_actor_state
                self.critic_rnn_states[i] = new_critic_state
                    
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
            decisions['actor_rnn_states'] = actor_rnn_states_rollout
            decisions['critic_rnn_states'] = critic_rnn_states_rollout
        
        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = np.copy(state['Q_edge'])    
        
        return decisions

    def train(self, rollouts):
        """RNN 版本的 PPO Update"""
        for i in range(self.num_edge):
            agent = self.agents[i]
            
            obs = torch.tensor(np.array(rollouts[i]['obs']), dtype=torch.float32).to(device)
            acts = torch.tensor(np.array(rollouts[i]['acts']), dtype=torch.float32).to(device)
            old_log_probs = torch.tensor(np.array(rollouts[i]['log_probs']), dtype=torch.float32).unsqueeze(1).to(device)
            returns = torch.tensor(np.array(rollouts[i]['returns']), dtype=torch.float32).unsqueeze(1).to(device)
            advantages = torch.tensor(np.array(rollouts[i]['advs']), dtype=torch.float32).unsqueeze(1).to(device)
            
            # Retrieve the initial RNN state for this batch
            # Simplified here by using a batch-processed trajectory approximation
            batch_actor_rnn = torch.cat(rollouts[i]['actor_rnn_states'], dim=1).to(device)
            batch_critic_rnn = torch.cat(rollouts[i]['critic_rnn_states'], dim=1).to(device)
            
            if self.use_ctde:
                critic_input = torch.tensor(np.array(rollouts[i]['global_obs']), dtype=torch.float32).to(device)
            else:
                critic_input = obs

            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            for _ in range(self.ppo_epoch):
                _, _, log_probs, entropy, _ = agent['actor'](obs, batch_actor_rnn, action=acts)
                values, _ = agent['critic'](critic_input, batch_critic_rnn)
                
                ratios = torch.exp(log_probs - old_log_probs.detach())
                
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy.mean()
                critic_loss = nn.MSELoss()(values, returns)
                
                agent['actor_opt'].zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(agent['actor'].parameters(), 0.5)
                agent['actor_opt'].step()
                
                agent['critic_opt'].zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(agent['critic'].parameters(), 0.5)
                agent['critic_opt'].step()

class MARAOPPOSolver(MARPPOSolver):
    def __init__(self, env, decoder, use_ctde=False):
        super().__init__(env, decoder=decoder, use_ctde=use_ctde)
        self.algo_name = "MARAOPPO"
        self.weight_filename = f"{self.algo_name}_{self.decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"

    def _extract_obs(self, state, agent_id):
        _, _, post_state = compute_ao_state(state)
        return super()._extract_obs(post_state, agent_id)

    def solve(self, state, store_rollout=False):
        f_edge, f_cloud, post_state = compute_ao_state(state)
        
        raw_actions, log_probs, values, unclipped_actions = {}, {}, {}, {}
        
        actor_rnn_states_rollout = {i: self.actor_rnn_states[i].clone().detach() for i in range(self.num_edge)}
        critic_rnn_states_rollout = {i: self.critic_rnn_states[i].clone().detach() for i in range(self.num_edge)}
        
        # post_state
        global_obs_list = [self._extract_obs(post_state, j) for j in range(self.num_edge)]
        global_obs = torch.cat(global_obs_list, dim=1) if self.use_ctde else None

        for i in range(self.num_edge):
            obs_tensor = global_obs_list[i]
            
            with torch.no_grad():
                if self.is_training:
                    action_clipped, action_unclipped, log_prob, _, new_actor_state = self.agents[i]['actor'](
                        obs_tensor, self.actor_rnn_states[i]
                    )
                else:
                    if obs_tensor.dim() == 2:
                        obs_tensor_sq = obs_tensor.unsqueeze(1)
                    else:
                        obs_tensor_sq = obs_tensor
                    x = self.agents[i]['actor'].fc1(obs_tensor_sq)
                    x, new_actor_state = self.agents[i]['actor'].rnn(x, self.actor_rnn_states[i])
                    x = x.squeeze(1)
                    mu = self.agents[i]['actor'].fc2(x)
                    action_clipped = torch.clamp(mu, 0.0, 1.0)
                    action_unclipped = action_clipped
                    log_prob = torch.zeros(1)
                
                if self.use_ctde:
                    val, new_critic_state = self.agents[i]['critic'](global_obs, self.critic_rnn_states[i])
                else:
                    val, new_critic_state = self.agents[i]['critic'](obs_tensor, self.critic_rnn_states[i])
                    
                self.actor_rnn_states[i] = new_actor_state
                self.critic_rnn_states[i] = new_critic_state
                    
            raw_actions[i] = action_clipped.squeeze(0).cpu().numpy()
            
            if store_rollout:
                unclipped_actions[i] = action_unclipped.squeeze(0).cpu().numpy()
                log_probs[i] = log_prob.item()
                values[i] = val.item()
        
        decisions = self.decoder.decode(post_state, raw_actions, self.num_edge, self.env.neighbors_map)
        
        decisions['f_edge'] = f_edge
        decisions['f_cloud'] = f_cloud
        decisions['raw_actions'] = raw_actions
        
        if store_rollout:
            decisions['unclipped_actions'] = unclipped_actions
            decisions['log_probs'] = log_probs
            decisions['values'] = values
            decisions['actor_rnn_states'] = actor_rnn_states_rollout
            decisions['critic_rnn_states'] = critic_rnn_states_rollout
        
        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = np.copy(post_state['Q_edge'])    
        
        return decisions

from ..action_related.split_decoders import FreqDecoder, QueueDecoder

class MARFreqPPOSolver(MARPPOSolver):
    def __init__(self, env, use_ctde=False):
        super().__init__(env, decoder=FreqDecoder(), use_ctde=use_ctde)
        self.algo_name = "MARFreqPPO"
        self.weight_filename = f"{self.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        self.global_obs_dim = 4 * self.num_edge
        
        for i in range(self.num_edge):
            obs_dim = 4
            action_dim = 2 
            critic_obs_dim = self.global_obs_dim if self.use_ctde else obs_dim
            
            self.agents[i]['actor'] = RMAPPOActor(obs_dim, action_dim, self.hidden_dim).to(device)
            self.agents[i]['critic'] = RMAPPOCritic(critic_obs_dim, self.hidden_dim).to(device)
            self.agents[i]['actor_opt'] = torch.optim.Adam(self.agents[i]['actor'].parameters(), lr=3e-4)
            self.agents[i]['critic_opt'] = torch.optim.Adam(self.agents[i]['critic'].parameters(), lr=1e-3)
            self.agents[i]['obs_normalizer'] = RunningMeanStd(shape=(obs_dim,))
            
        self.reset_rnn_states()

    def _extract_obs(self, state, agent_id):
        raw_obs = np.array([
            state['Q_edge'][agent_id], state['Q_cloud'][agent_id], 
            state['CI_edge'][agent_id], state['CI_cloud'][agent_id]
        ], dtype=np.float32)
        
        normalizer = self.agents[agent_id]['obs_normalizer']
        if getattr(self, 'is_training', False):
            normalizer.update(np.array([raw_obs]))
            
        normalized_obs = (raw_obs - normalizer.mean) / (np.sqrt(normalizer.var) + 1e-8)
        return torch.tensor(normalized_obs, dtype=torch.float32).unsqueeze(0).to(device)