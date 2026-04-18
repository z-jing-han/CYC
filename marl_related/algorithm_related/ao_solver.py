import torch
import numpy as np
from config import Config
from .maddpg import MADDPGSolver
from .mappo import MAPPOSolver
from ..utils import compute_ao_state

class MAAODDPGSolver(MADDPGSolver):
    def __init__(self, env, decoder, use_ctde=False):
        # Only allow AO_XPDecoder or AO_XTDecoder
        super().__init__(env, decoder=decoder, use_ctde=use_ctde)
        self.algo_name = "MAAODDPG"
        self.weight_filename = f"{self.algo_name}_{self.decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"

    def _extract_obs(self, state, agent_id):
        # Change the observation state to moment after computing via AO
        _, _, post_state = compute_ao_state(state)
        return super()._extract_obs(post_state, agent_id)

    def solve(self, state, **kwargs):
        f_edge, f_cloud, post_state = compute_ao_state(state)
        raw_actions = {}
        
        for i in range(self.num_edge):
            obs_tensor = self._extract_obs(state, i) 
            with torch.no_grad():
                action_tensor = self.agents[i]['actor'](obs_tensor)
                if getattr(self, 'is_training', False):
                    noise = torch.normal(0, self.noise_std, size=action_tensor.size(), device=action_tensor.device)
                    action_tensor = torch.clamp(action_tensor + noise, 0.0, 1.0)
                action = action_tensor.squeeze(0).cpu().detach().numpy()
            raw_actions[i] = action
        
        decisions = self.decoder.decode(post_state, raw_actions, self.num_edge, self.env.neighbors_map)
        decisions['f_edge'] = f_edge
        decisions['f_cloud'] = f_cloud
        decisions['raw_actions'] = raw_actions
        
        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = np.copy(post_state['Q_edge'])
            
        return decisions


class MAAOPPOSolver(MAPPOSolver):
    def __init__(self, env, decoder, use_ctde=False):
        super().__init__(env, decoder=decoder, use_ctde=use_ctde)
        self.algo_name = "MAAOPPO"
        self.weight_filename = f"{self.algo_name}_{self.decoder.__class__.__name__}_{'CTDE' if use_ctde else 'Decentralized'}_weights.pth"

    def _extract_obs(self, state, agent_id):
        _, _, post_state = compute_ao_state(state)
        return super()._extract_obs(post_state, agent_id)

    def solve(self, state, store_rollout=False):
        f_edge, f_cloud, post_state = compute_ao_state(state)
        
        raw_actions, log_probs, values, unclipped_actions = {}, {}, {}, {}
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
                
                val = self.agents[i]['critic'](global_obs) if self.use_ctde else self.agents[i]['critic'](obs_tensor)
                    
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
        
        if Config.OBSERVATION_PREV:
            self.prev_Q_edge = np.copy(post_state['Q_edge'])    
        
        return decisions