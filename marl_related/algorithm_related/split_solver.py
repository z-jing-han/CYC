import torch
import torch.nn as nn
import numpy as np
import os
from .base_solver import BaseMARLSolver, device
from .mappo import MAPPOSolver, MAPPOActor, MAPPOCritic
from ..action_related.split_decoders import FreqDecoder, QueueDecoder
from ..utils import RunningMeanStd

class MAFreqPPOSolver(MAPPOSolver):
    def __init__(self, env, use_ctde=False):
        super().__init__(env, decoder=FreqDecoder(), use_ctde=use_ctde)
        self.algo_name = "MAFreqPPO"
        self.weight_filename = f"{self.algo_name}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"
        self.global_obs_dim = 4 * self.num_edge
        
        for i in range(self.num_edge):
            obs_dim = 4
            action_dim = 2 
            critic_obs_dim = self.global_obs_dim if self.use_ctde else obs_dim
            
            self.agents[i]['actor'] = MAPPOActor(obs_dim, action_dim).to(device)
            self.agents[i]['critic'] = MAPPOCritic(critic_obs_dim).to(device)
            self.agents[i]['actor_opt'] = torch.optim.Adam(self.agents[i]['actor'].parameters(), lr=3e-4)
            self.agents[i]['critic_opt'] = torch.optim.Adam(self.agents[i]['critic'].parameters(), lr=1e-3)
            self.agents[i]['obs_normalizer'] = RunningMeanStd(shape=(obs_dim,))

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

    def solve(self, state, store_rollout=False):
        return super().solve(state, store_rollout=store_rollout)