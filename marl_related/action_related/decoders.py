import numpy as np
from config import Config
from .base_decoder import BaseActionDecoder
from ..utils import compute_actual_x_and_p

class XPDecoder(BaseActionDecoder):
    """
    Original marl.py Logic:
    Action outputs Task Size and Power directly.
    """
    def get_action_dim(self, num_neighbors):
        # f_edge, f_cloud, x_cloud, p_cloud + (x_peer, p_peer) * N
        return 4 + (2 * num_neighbors)

    def decode(self, state, raw_actions, num_edge, neighbors_map):
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        f_edge = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))

        for i in range(num_edge):
            action = raw_actions[i]
            
            # Freq Max mask
            needed_f_edge = (Q_edge[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_edge = min(Config.EDGE_F_MAX, needed_f_edge)
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            max_valid_f_cloud = min(Config.CLOUD_F_MAX, needed_f_cloud)

            # Freq decision
            f_edge[i] = action[0] * max_valid_f_edge
            f_cloud[i] = action[1] * max_valid_f_cloud

            # Cloud Transmission
            raw_x_cloud = action[2] * Q_edge[i]
            p_cloud[i] = action[3] * Config.EDGE_P_MAX

            # Peer Transmission
            raw_x_peers = []
            idx = 4
            for neighbor_id in neighbors_map.get(i, []):
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

        return {
            'f_edge': f_edge, 'f_cloud': f_cloud, 'x_cloud': x_cloud,
            'p_cloud': p_cloud, 'x_peer': x_peer, 'p_peer': p_peer
        }


class XTDecoder(BaseActionDecoder):
    """
    MADDPG.py Logic (Lemma 1):
    Action outputs Task Size and Time Allocation (Tau).
    """
    def get_action_dim(self, num_neighbors):
        # f_edge, f_cloud, x_cloud, tau_cloud + (x_peer, tau_peer) * N
        return 4 + (2 * num_neighbors)

    def decode(self, state, raw_actions, num_edge, neighbors_map):
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        f_edge = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))

        SAFE_T_OFF = Config.TIME_SLOT_DURATION * 0.99999

        for i in range(num_edge):
            action = raw_actions[i]
            
            # 1. Local Computation
            max_ec_bits = (Config.EDGE_F_MAX * Config.TIME_SLOT_DURATION) / Config.PHI
            raw_x_comp = action[0] * min(Q_edge[i], max_ec_bits)

            # Cloud Computation is isolated
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            f_cloud[i] = action[1] * min(Config.CLOUD_F_MAX, needed_f_cloud)

            # 2. Offloading task size
            raw_x_cloud = action[2] * Q_edge[i]
            raw_tau_cloud = action[3]

            raw_x_peers = []
            idx = 4
            for neighbor_id in neighbors_map.get(i, []):
                if Q_edge[i] > Q_edge[neighbor_id]:
                    # (target node, task size, time)
                    raw_x_peers.append((neighbor_id, action[idx] * Q_edge[i], action[idx+1]))
                else:
                    raw_x_peers.append((neighbor_id, 0.0, 0.0))
                idx += 2

            # 3. Joint Data Allocation handle queue constraint
            total_x_request = raw_x_comp + raw_x_cloud + sum([x for _, x, _ in raw_x_peers])
            scale_factor = min(1.0, Q_edge[i] / (total_x_request + 1e-9))
            
            # Local frequency
            actual_x_comp = raw_x_comp * scale_factor
            f_edge[i] = (actual_x_comp * Config.PHI) / Config.TIME_SLOT_DURATION

            # 4. Lemma 1 (Time Allocation and actual power calculation)
            active_taus = [raw_tau_cloud] + [tau for _, _, tau in raw_x_peers]
            total_tau = sum(active_taus) + 1e-9
            
            t_cloud = (raw_tau_cloud / total_tau) * SAFE_T_OFF
            
            x_cloud[i], p_cloud[i] = compute_actual_x_and_p(
                x_target=raw_x_cloud * scale_factor,
                t_alloc=t_cloud,
                W=Config.BANDWIDTH,
                g=Config.G_IC,
                N0=Config.NOISE_POWER,
                p_max=Config.EDGE_P_MAX
            )
            
            for neighbor_id, x_p, tau_p in raw_x_peers:
                if x_p > 0 and tau_p > 0:
                    t_peer = (tau_p / total_tau) * SAFE_T_OFF
                    x_actual, p_actual = compute_actual_x_and_p(
                        x_target=x_p * scale_factor,
                        t_alloc=t_peer,
                        W=Config.BANDWIDTH,
                        g=Config.G_IJ,
                        N0=Config.NOISE_POWER,
                        p_max=Config.EDGE_P_MAX
                    )
                    x_peer[i, neighbor_id] = x_actual
                    p_peer[i, neighbor_id] = p_actual

        return {
            'f_edge': f_edge, 'f_cloud': f_cloud, 'x_cloud': x_cloud,
            'p_cloud': p_cloud, 'x_peer': x_peer, 'p_peer': p_peer
        }

class XTRDecoder(BaseActionDecoder):
    """
    Proportional Relaxed Decoder:
    Uses Logits and Softmax to allocate Data and Time proportionally.
    Outputs soft 'penalties' for exceeding P_MAX to allow Lagrangian Relaxation in RL.
    """
    def get_action_dim(self, num_neighbors):
        # 0: a_comp (local compute ratio)
        # 1: a_fcloud (cloud compute ratio)
        # Data Logits: [cloud, peer_1, ..., peer_N, idle] -> num_neighbors + 2
        # Time Logits: [cloud, peer_1, ..., peer_N, idle] -> num_neighbors + 2
        return 2 + (num_neighbors + 2) + (num_neighbors + 2)

    def decode(self, state, raw_actions, num_edge, neighbors_map):
        Q_edge = state['Q_edge']
        Q_cloud = state['Q_cloud']
        f_edge = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))
        
        penalties = np.zeros(num_edge) 
        
        SAFE_T_OFF = Config.TIME_SLOT_DURATION * 0.99999

        for i in range(num_edge):
            action = raw_actions[i]
            neighbors = neighbors_map.get(i, [])
            num_neighbors = len(neighbors)
            
            # --- 1. Local & Cloud Computation Allocation ---
            a_comp = action[0]
            a_fcloud = action[1]
            
            max_ec_bits = (Config.EDGE_F_MAX * Config.TIME_SLOT_DURATION) / Config.PHI
            actual_x_comp = a_comp * min(Q_edge[i], max_ec_bits)
            f_edge[i] = (actual_x_comp * Config.PHI) / Config.TIME_SLOT_DURATION
            
            needed_f_cloud = (Q_cloud[i] * Config.PHI) / Config.TIME_SLOT_DURATION
            f_cloud[i] = a_fcloud * min(Config.CLOUD_F_MAX, needed_f_cloud)
            
            # --- 2. Extract Data & Time Logits ---
            start_data = 2
            end_data = 2 + num_neighbors + 2
            data_logits = action[start_data:end_data]
            
            start_time = end_data
            end_time = start_time + num_neighbors + 2
            time_logits = action[start_time:end_time]
            
            # --- 3. Create Validity Mask (Cloud, N Peers, Idle) ---
            mask = np.ones(num_neighbors + 2, dtype=bool)
            for idx, neighbor_id in enumerate(neighbors):
                if Q_edge[i] <= Q_edge[neighbor_id]:
                    mask[idx + 1] = False
            
            data_probs = self._softmax_with_mask(data_logits, mask)
            time_probs = self._softmax_with_mask(time_logits, mask)
            
            # --- 4. Allocate Remaining Data & Time ---
            Q_rem = Q_edge[i] - actual_x_comp
            
            raw_x_cloud = data_probs[0] * Q_rem
            raw_t_cloud = time_probs[0] * SAFE_T_OFF
            
            # --- 5. Calculate Power & Penalty for Cloud ---
            p_cloud_req = self._calc_p_req(raw_x_cloud, raw_t_cloud, Config.BANDWIDTH, Config.G_IC, Config.NOISE_POWER)
            if p_cloud_req > Config.EDGE_P_MAX:
                penalties[i] += (p_cloud_req - Config.EDGE_P_MAX)
            
            x_cloud[i], p_cloud[i] = compute_actual_x_and_p(
                x_target=raw_x_cloud, t_alloc=raw_t_cloud,
                W=Config.BANDWIDTH, g=Config.G_IC,
                N0=Config.NOISE_POWER, p_max=Config.EDGE_P_MAX
            )
            
            # --- 6. Calculate Power & Penalty for Peers ---
            for idx, neighbor_id in enumerate(neighbors):
                if mask[idx + 1]:
                    raw_x_p = data_probs[idx + 1] * Q_rem
                    raw_t_p = time_probs[idx + 1] * SAFE_T_OFF
                    
                    p_peer_req = self._calc_p_req(raw_x_p, raw_t_p, Config.BANDWIDTH, Config.G_IJ, Config.NOISE_POWER)
                    if p_peer_req > Config.EDGE_P_MAX:
                        penalties[i] += (p_peer_req - Config.EDGE_P_MAX)
                        
                    x_actual, p_actual = compute_actual_x_and_p(
                        x_target=raw_x_p, t_alloc=raw_t_p,
                        W=Config.BANDWIDTH, g=Config.G_IJ,
                        N0=Config.NOISE_POWER, p_max=Config.EDGE_P_MAX
                    )
                    x_peer[i, neighbor_id] = x_actual
                    p_peer[i, neighbor_id] = p_actual

        return {
            'f_edge': f_edge, 'f_cloud': f_cloud, 'x_cloud': x_cloud,
            'p_cloud': p_cloud, 'x_peer': x_peer, 'p_peer': p_peer,
            'penalties': penalties
        }

    def _softmax_with_mask(self, logits, mask):
        safe_logits = np.where(mask, logits, -1e9)
        e_x = np.exp(safe_logits - np.max(safe_logits))
        e_x = np.where(mask, e_x, 0.0)
        sum_e_x = np.sum(e_x)
        return e_x / (sum_e_x + 1e-9)
        
    def _calc_p_req(self, x, t, W, g, N0):
        if t <= 1e-9 or x <= 1e-9:
            return 0.0
        return (2**(x / (W * t)) - 1) * (N0 / g)