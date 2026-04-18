import torch
import numpy as np
import os
from config import Config

# Unified device setting
device = torch.device("cpu")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaseMARLSolver:
    def __init__(self, env, algo_name, decoder, use_ctde=False):
        self.env = env
        self.algo_name = algo_name
        self.decoder = decoder
        self.use_ctde = use_ctde
        self.num_edge = env.num_edge
        
        # Load the weight file, e.g.: MADDPG_XTDecoder_Decentralized_weights.pth
        decoder_name = self.decoder.__class__.__name__
        ctde_str = "CTDE" if self.use_ctde else "Decentralized"
        self.weight_filename = f"{self.algo_name}_{decoder_name}_{ctde_str}_weights.pth"
        
        self.noise_std = getattr(Config, 'MARL_NOISE', 0.05)
        self.is_training = True
        self.agents = {}

        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = None
    
    def reset_internal_state(self, initial_Q_edge):
        self.prev_Q_edge = np.copy(initial_Q_edge)
        
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

        if Config.OBSERVATION_PREV and self.prev_Q_edge is None:
            self.prev_Q_edge = np.copy(state['Q_edge'])
        
        for neighbor_id in neighbors:
            if Config.OBSERVATION_PREV:
                obs.append(self.prev_Q_edge[neighbor_id])
            else:
                obs.append(Q_edge[neighbor_id])
        
        raw_obs = np.array(obs, dtype=np.float32)
        normalizer = self.agents[agent_id]['obs_normalizer']
        
        if getattr(self, 'is_training', False):
            normalizer.update(np.array([raw_obs]))
            
        normalized_obs = (raw_obs - normalizer.mean) / (np.sqrt(normalizer.var) + 1e-8)
        
        return torch.tensor(normalized_obs, dtype=torch.float32).unsqueeze(0).to(device)

    def solve(self, state):
        raw_actions = {} 

        for i in range(self.num_edge):
            obs_tensor = self._extract_obs(state, i)
            
            with torch.no_grad():
                action_tensor = self.agents[i]['actor'](obs_tensor)
                
                # Inference during training with exploration noise
                if getattr(self, 'is_training', False):
                    noise = torch.normal(0, self.noise_std, size=action_tensor.size(), device=device)
                    action_tensor = torch.clamp(action_tensor + noise, 0.0, 1.0)
                
                action = action_tensor.squeeze(0).cpu().detach().numpy()
            
            raw_actions[i] = action
        
        decisions = self.decoder.decode(state, raw_actions, self.num_edge, self.env.neighbors_map)
        decisions['raw_actions'] = raw_actions

        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = np.copy(state['Q_edge'])

        return decisions

    def save_weights(self, output_dir):
        filepath = os.path.join(output_dir, self.weight_filename)
        state_dicts = {}
        for i, agent in self.agents.items():
            state_dicts[i] = {
                'actor': agent['actor'].state_dict(),
                'critic': agent['critic'].state_dict()
            }
        torch.save(state_dicts, filepath)
        print(f"[{self.algo_name}] Weights saved to: {filepath}")

    def load_weights(self, output_dir):
        filepath = os.path.join(output_dir, self.weight_filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Weight file not found: {filepath}")
            
        state_dicts = torch.load(filepath, map_location=device)
        for i, agent in self.agents.items():
            agent['actor'].load_state_dict(state_dicts[i]['actor'])
            agent['critic'].load_state_dict(state_dicts[i]['critic'])
            
        self.is_training = False
        print(f"[{self.algo_name}] Weights loaded from {filepath}")