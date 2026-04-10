from config import Config
import numpy as np

def compute_actual_x_and_p(x_target, t_alloc, W, g, N0, p_max):
    """
    Following Lemma 1 and Constraints C4, C5
    (Extract from MADDPG.py)
    """
    if t_alloc <= 1e-9 or x_target <= 1e-9:
        return 0.0, 0.0
    
    R_max = W * np.log2(1 + (g * p_max) / N0)
    x_max_possible = R_max * t_alloc
    x_actual = min(x_target, x_max_possible)
    p_actual = (2**(x_actual / (W * t_alloc)) - 1) * (N0 / g)
    
    return x_actual, min(p_actual, p_max)


class BaseActionDecoder:
    def get_action_dim(self, num_neighbors):
        """Define the output dimension for the neural network"""
        raise NotImplementedError
        
    def decode(self, state, raw_actions, num_edge, neighbors_map):
        """Rescale raw actions from [0, 1] to the environment's physical action range"""
        raise NotImplementedError


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