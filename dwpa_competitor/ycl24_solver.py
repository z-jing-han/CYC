import numpy as np
from config import Config

class YCL24Solver:
    """
    Implementation of the YCL24 Algorithm.
    Focus: Lyapunov Optimization (Similar to DCWA) but Cloud-Centric.
    Logic: The provided YCL24.py implementation only executes cloud offloading 
    in its main simulation loop.
    """
    def __init__(self, env):
        self.env = env
        self.t_off_dict = {}

    def solve(self, state):
        Q_edge = state['Q_edge'].copy()
        Q_cloud = state['Q_cloud'].copy()
        CI_edge = state['CI_edge'] # Note: YCL24 reads fixed CI from file usually
        CI_cloud = state['CI_cloud']
        neighbors_map = state['Graph']
        num_edge = len(Q_edge)
        
        f_edge = np.zeros(num_edge)
        x_peer = np.zeros((num_edge, num_edge))
        p_peer = np.zeros((num_edge, num_edge))
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)

        N0_watts = Config.NOISE_POWER

        # 1. Local Computation
        for i in range(num_edge):
            denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_opt = np.sqrt(Q_edge[i] / denom)
                f_edge[i] = np.clip(f_opt, 0, Config.EDGE_F_MAX)
            else:
                f_edge[i] = Config.EDGE_F_MAX
            
            bits_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            Q_edge[i] = max(0, Q_edge[i] - bits_local)

        # 2. Peer Offloading 
        # Explicitly skipped based on YCL24.py 'simulate_edge_cloud_system' loop 
        # which only calls 'should_offload_to_cloud'.

        # 3. Cloud Offloading (Lyapunov Optimization)
        # Uses the same derived formula as DCWA/DOLA for Power
        for i in range(num_edge):
            if Config.TIME_SLOT_ADJUST == "scale":
                eligible_neighbors = sum(1 for j in neighbors_map[i] if Q_edge[i] > Q_edge[j])
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (eligible_neighbors + 1)
            elif Config.TIME_SLOT_ADJUST == "fix_time_slot" :
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (num_edge + 1)

        for i in range(num_edge):
            curr_q_edge = Q_edge[i]
            curr_q_cloud = Q_cloud[i]
            
            self.t_off_dict[i] = Config.TIME_SLOT_DURATION
            a_param = Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * self.t_off_dict[i]
            b_param = (curr_q_cloud - curr_q_edge) * Config.BANDWIDTH * self.t_off_dict[i]
            c_param = Config.G_IC / N0_watts

            if a_param > 0 and c_param > 0:
                # p* = - (b / a ln 2) - (1/c)
                term1 = - (b_param / (a_param * np.log(2)))
                p_opt = term1 - (1.0 / c_param)
                
                # Check valid range
                if 0 <= p_opt <= Config.EDGE_P_MAX:
                    transmission_power = p_opt
                else:
                    transmission_power = 0
                
                if transmission_power > 0:
                    # G_i_c calculation check (Drift + Penalty check)
                    snr = (Config.G_IC * transmission_power) / N0_watts
                    rate = Config.BANDWIDTH * np.log2(1 + snr)
                    
                    # Lyapunov Drift check: (V * CI * p)/R + Q_cloud - Q_edge <= 0
                    drift = (Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * transmission_power) / rate + curr_q_cloud - curr_q_edge
                    
                    if drift <= 0:
                        x_cloud_star = rate * self.t_off_dict[i]
                        # Limit to available bits
                        x_cloud[i] = min(x_cloud_star, curr_q_edge)
                        p_cloud[i] = transmission_power

        # 4. Cloud Computation (Standard)
        # YCL24.py uses a simple processing loop, here we apply the standard dynamic freq allocation
        # compatible with the framework
        Q_cloud_post_offload = Q_cloud + x_cloud
        for i in range(num_edge):
             # YCL uses simplistic processing in file, but we align with the Solver API's expectation
            f_cloud[i] = Config.CLOUD_F_MAX 

        return {
            'f_edge': f_edge,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'f_cloud': f_cloud
        }