import numpy as np
from config import Config
from .base_decoder import BaseActionDecoder
from ..utils import compute_actual_x_and_p

class AO_XPDecoder(BaseActionDecoder):
    """
    Modify from XPDecoder
    """
    def get_action_dim(self, num_neighbors):
        # Remove f_edge, f_cloud
        return 2 + (2 * num_neighbors)

    def decode(self, state, raw_actions, num_edge, neighbors_map):
        post_Q_edge = state['Q_edge']
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))

        for i in range(num_edge):
            action = raw_actions[i]
            
            # Cloud Transmission
            raw_x_cloud = action[0] * post_Q_edge[i]
            p_cloud[i] = action[1] * Config.EDGE_P_MAX

            # Peer Transmission
            raw_x_peers = []
            idx = 2
            for neighbor_id in neighbors_map.get(i, []):
                if post_Q_edge[i] > post_Q_edge[neighbor_id]:
                    raw_x_peers.append((neighbor_id, action[idx] * post_Q_edge[i], action[idx+1] * Config.EDGE_P_MAX))
                else:
                    raw_x_peers.append((neighbor_id, 0.0, 0.0))
                idx += 2
            
            total_x_request = raw_x_cloud + sum([x for _, x, _ in raw_x_peers])
            scale_factor = min(1.0, post_Q_edge[i] / (total_x_request + 1e-9))
            
            x_cloud[i] = raw_x_cloud * scale_factor
            for neighbor_id, x_p, p_p in raw_x_peers:
                x_peer[i, neighbor_id] = x_p * scale_factor
                p_peer[i, neighbor_id] = p_p

        return {
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'x_peer': x_peer,
            'p_peer': p_peer
        }


class AO_XTDecoder(BaseActionDecoder):
    """
    Modify from XTDecoder
    """
    def get_action_dim(self, num_neighbors):
        # x_cloud, tau_cloud + (x_peer, tau_peer) * N
        return 2 + (2 * num_neighbors)

    def decode(self, state, raw_actions, num_edge, neighbors_map):
        post_Q_edge = state['Q_edge']
        
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))

        SAFE_T_OFF = Config.TIME_SLOT_DURATION * 0.99999

        for i in range(num_edge):
            action = raw_actions[i]
            
            raw_x_cloud = action[0] * post_Q_edge[i]
            raw_tau_cloud = action[1]

            raw_x_peers = []
            idx = 2
            for neighbor_id in neighbors_map.get(i, []):
                if post_Q_edge[i] > post_Q_edge[neighbor_id]:
                    raw_x_peers.append((neighbor_id, action[idx] * post_Q_edge[i], action[idx+1]))
                else:
                    raw_x_peers.append((neighbor_id, 0.0, 0.0))
                idx += 2

            total_x_request = raw_x_cloud + sum([x for _, x, _ in raw_x_peers])
            scale_factor = min(1.0, post_Q_edge[i] / (total_x_request + 1e-9))
            
            active_taus = [raw_tau_cloud] + [tau for _, _, tau in raw_x_peers]
            total_tau = sum(active_taus) + 1e-9
            
            t_cloud = (raw_tau_cloud / total_tau) * SAFE_T_OFF
            
            x_cloud[i], p_cloud[i] = compute_actual_x_and_p(
                x_target=raw_x_cloud * scale_factor, t_alloc=t_cloud,
                W=Config.BANDWIDTH, g=Config.G_IC, N0=Config.NOISE_POWER, p_max=Config.EDGE_P_MAX
            )
            
            for neighbor_id, x_p, tau_p in raw_x_peers:
                if x_p > 0 and tau_p > 0:
                    t_peer = (tau_p / total_tau) * SAFE_T_OFF
                    x_actual, p_actual = compute_actual_x_and_p(
                        x_target=x_p * scale_factor, t_alloc=t_peer,
                        W=Config.BANDWIDTH, g=Config.G_IJ, N0=Config.NOISE_POWER, p_max=Config.EDGE_P_MAX
                    )
                    x_peer[i, neighbor_id] = x_actual
                    p_peer[i, neighbor_id] = p_actual

        return {
            'x_cloud': x_cloud, 'p_cloud': p_cloud,
            'x_peer': x_peer, 'p_peer': p_peer
        }