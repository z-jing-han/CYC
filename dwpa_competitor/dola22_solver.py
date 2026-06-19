import numpy as np
from config import Config

class DOLA22Solver:
    """
    Implementation of the DOLA22 Algorithm.
    Focus: Queue Stability and Load Balancing.
    Logic derived from 'should_offload_to_edge' and 'should_offload_to_cloud' in DOLA22.py.
    """
    def __init__(self, env):
        self.env = env
        self.t_off_dict = {}

    def solve(self, state):
        Q_edge = state['Q_edge'].copy()
        Q_cloud = state['Q_cloud'].copy()
        CI_edge = state['CI_edge']
        CI_cloud = state['CI_cloud']
        neighbors_map = state['Graph']
        num_edge = len(Q_edge)
        
        f_edge = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)

        # Handle Noise Power Unit Conversion
        N0_watts = Config.NOISE_POWER
        if N0_watts < 0: 
            N0_watts = 10 ** ((N0_watts - 30) / 10)

        # 1. Local Computation Resource Allocation
        # Logic: f = sqrt(Q / (3 * V * CI * zeta * phi))
        for i in range(num_edge):
            denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_opt = np.sqrt(Q_edge[i] / denom)
                f_edge[i] = np.clip(f_opt, 0, Config.EDGE_F_MAX)
            else:
                f_edge[i] = Config.EDGE_F_MAX
            # f_edge[i] = Config.EDGE_F_MAX
            # Update virtual queue
            bits_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            Q_edge[i] = max(0, Q_edge[i] - bits_local)

        # 2. Peer Offloading (Queue Balancing)

        for i in range(num_edge):
            if Config.TIME_SLOT_ADJUST == "scale":
                eligible_neighbors = sum(1 for j in neighbors_map[i] if Q_edge[i] > Q_edge[j])
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (eligible_neighbors + 1)
            elif Config.TIME_SLOT_ADJUST == "fix_time_slot" :
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (num_edge + 1)

        for i in range(num_edge):
            for j in neighbors_map[i]:
                # DOLA Strategy: Balance the queues.
                # Optimal offload amount is half the difference to equalize Q_i and Q_j
                if Q_edge[i] <= Q_edge[j]:
                    continue

                optimal_offload_bits = (Q_edge[i] - Q_edge[j])# / 2.0
                
                if optimal_offload_bits <= 0:
                    continue

                # Calculate Power required for this transfer (Lyapunov derivation from source)
                a_param = Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * self.t_off_dict[i]
                b_param = (Q_edge[j] - Q_edge[i]) * Config.BANDWIDTH * self.t_off_dict[i]
                c_param = Config.G_IJ / N0_watts

                if a_param > 0 and c_param > 0:
                    term1 = - (b_param / (a_param * np.log(2)))
                    p_opt = term1 - (1.0 / c_param)
                    
                    # Clamp power to max capacity
                    actual_power = np.clip(p_opt, 0, Config.EDGE_P_MAX)
                    
                    if actual_power > 0:
                        # Calculate achievable rate
                        snr = (Config.G_IJ * actual_power) / N0_watts
                        max_bits_by_power = Config.BANDWIDTH * np.log2(1 + snr) * self.t_off_dict[i]
                        
                        # DOLA Logic: min(Target Balance Amount, Max Possible by Power)
                        final_bits = min(optimal_offload_bits, max_bits_by_power)
                        
                        # Ensure we don't offload more than we have
                        final_bits = min(final_bits, Q_edge[i])
                        
                        x_peer[i, j] = final_bits
                        p_peer[i, j] = actual_power

        # Update Q for Cloud Decision
        tx_sum = np.sum(x_peer, axis=1)
        rx_sum = np.sum(x_peer, axis=0)
        Q_edge_post_peer = np.maximum(0, Q_edge - tx_sum + rx_sum)

        # 3. Cloud Offloading (Queue Balancing)
        for i in range(num_edge):
            curr_q_edge = Q_edge_post_peer[i]
            curr_q_cloud = Q_cloud[i]

            if curr_q_edge <= curr_q_cloud:
                continue

            # Target: Equalize Edge and Cloud Queues
            optimal_offload_bits = (curr_q_edge - curr_q_cloud)# / 2.0
            
            if optimal_offload_bits <= 0:
                continue
            
            a_param = Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * self.t_off_dict[i]
            b_param = (curr_q_cloud - curr_q_edge) * Config.BANDWIDTH * self.t_off_dict[i]
            c_param = Config.G_IC / N0_watts

            if a_param > 0 and c_param > 0:
                term1 = - (b_param / (a_param * np.log(2)))
                p_opt = term1 - (1.0 / c_param)
                
                actual_power = np.clip(p_opt, 0, Config.EDGE_P_MAX)
                
                if actual_power > 0:
                    snr = (Config.G_IC * actual_power) / N0_watts
                    max_bits_by_power = Config.BANDWIDTH * np.log2(1 + snr) * self.t_off_dict[i]
                    
                    final_bits = min(optimal_offload_bits, max_bits_by_power)
                    final_bits = min(final_bits, curr_q_edge)
                    
                    x_cloud[i] = final_bits
                    p_cloud[i] = actual_power

        # 4. Cloud Computation
        Q_cloud_post_offload = Q_cloud + x_cloud
        for i in range(num_edge):
            denom = 3 * Config.V * CI_cloud[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_c_opt = np.sqrt(Q_cloud_post_offload[i] / denom)
                f_cloud[i] = np.clip(f_c_opt, 0, Config.CLOUD_F_MAX)
            else:
                f_cloud[i] = Config.CLOUD_F_MAX

        return {
            'f_edge': f_edge,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'f_cloud': f_cloud
        }