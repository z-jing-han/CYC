import numpy as np
from config import Config

class FIXTIMESolver:
    def __init__(self, env, variant='default'):
        self.env = env
        self.variant = variant
        self.t_off_dict = {}
        self.lambda_tolerance = 1e-4
        self.max_bisect_iter = 50

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

        # =========================================================================
        # Step 1: Local Computation Resource Allocation
        # Ref: Pub. Algorithm 1 (P1a) Line 2 - 3, Eq (15)
        # =========================================================================
        for i in range(num_edge):
            denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_opt = np.sqrt(Q_edge[i] / denom)
                f_edge[i] = np.clip(f_opt, 0, Config.EDGE_F_MAX)
            else:
                f_edge[i] = Config.EDGE_F_MAX
            
            bits_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            Q_edge[i] = max(0, Q_edge[i] - bits_local)

        # =========================================================================
        # Step 2: Cloud Computation Resource Allocation
        # Ref: Pub. Algorithm 1 (P1b) Line 5, Eq (16)
        # =========================================================================
        for i in range(num_edge):
            denom = 3 * Config.V * CI_cloud[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_c_opt = np.sqrt(Q_cloud[i] / denom)
                f_cloud[i] = np.clip(f_c_opt, 0, Config.CLOUD_F_MAX) 
            else:
                f_cloud[i] = Config.CLOUD_F_MAX

        # =========================================================================
        # Step 3 & 4: Joint Peer and Cloud Offloading with Queue Constraint (KKT)
        # Ref: Problem 4 Reformulation (Water-filling approach)
        # =========================================================================

        for i in range(num_edge):
            if Config.TIME_SLOT_ADJUST == "scale":
                eligible_neighbors = sum(1 for j in neighbors_map[i] if Q_edge[i] > Q_edge[j])
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (eligible_neighbors + 1)
            elif Config.TIME_SLOT_ADJUST == "fix_time_slot" :
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (num_edge + 1)

        def calc_offloading(i, lambda_val):
            """Calculate optimal x and p for a given lambda (water-level)"""
            curr_x_peer = np.zeros(num_edge)
            curr_p_peer = np.zeros(num_edge)
            curr_x_cloud = 0.0
            curr_p_cloud = 0.0
            
            common_term_denom = Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.NOISE_POWER * np.log(2)

            # 1. Evaluate Peer Offloading
            for j in neighbors_map[i]:
                term = Q_edge[i] - Q_edge[j] - lambda_val
                if term > 0:
                    val = (Config.BANDWIDTH * Config.G_IJ * term) / common_term_denom
                    if val > 1:
                        x_val = Config.BANDWIDTH * self.t_off_dict[i] * np.log2(val)
                        p_req = (2 ** (x_val / (Config.BANDWIDTH * self.t_off_dict[i])) - 1) * (Config.NOISE_POWER / Config.G_IJ)
                        
                        # Clip based on Physical Power Constraint
                        if p_req > Config.EDGE_P_MAX:
                            p_req = Config.EDGE_P_MAX
                            x_val = Config.BANDWIDTH * self.t_off_dict[i] * np.log2(1 + Config.EDGE_P_MAX * Config.G_IJ / Config.NOISE_POWER)
                        
                        curr_x_peer[j] = x_val
                        curr_p_peer[j] = p_req

            # 2. Evaluate Cloud Offloading
            term_c = Q_edge[i] - Q_cloud[i] - lambda_val
            if term_c > 0:
                val_c = (Config.BANDWIDTH * Config.G_IC * term_c) / common_term_denom
                if val_c > 1:
                    x_c_val = Config.BANDWIDTH * self.t_off_dict[i] * np.log2(val_c)
                    p_c_req = (2 ** (x_c_val / (Config.BANDWIDTH * self.t_off_dict[i])) - 1) * (Config.NOISE_POWER / Config.G_IC)
                    
                    # Clip based on Physical Power Constraint
                    if p_c_req > Config.EDGE_P_MAX:
                        p_c_req = Config.EDGE_P_MAX
                        x_c_val = Config.BANDWIDTH * self.t_off_dict[i] * np.log2(1 + Config.EDGE_P_MAX * Config.G_IC / Config.NOISE_POWER)
                    
                    curr_x_cloud = x_c_val
                    curr_p_cloud = p_c_req

            return curr_x_peer, curr_p_peer, curr_x_cloud, curr_p_cloud

        for i in range(num_edge):
            # Try unconstrained case first (lambda = 0)
            x_p, p_p, x_c, p_c = calc_offloading(i, 0.0)
            total_offload = np.sum(x_p) + x_c

            if total_offload <= Q_edge[i]:
                # Constraint inactive: Local queue has enough data
                x_peer[i, :] = x_p
                p_peer[i, :] = p_p
                x_cloud[i] = x_c
                p_cloud[i] = p_c
            else:
                # Constraint active: Apply Bisection Search to find optimal lambda > 0
                lambda_low = 0.0
                lambda_high = Q_edge[i]  # Lambda won't exceed local queue size practically
                
                for _ in range(self.max_bisect_iter):
                    lambda_mid = (lambda_low + lambda_high) / 2.0
                    x_p, p_p, x_c, p_c = calc_offloading(i, lambda_mid)
                    total_offload = np.sum(x_p) + x_c

                    if abs(total_offload - Q_edge[i]) < self.lambda_tolerance:
                        break
                    
                    if total_offload > Q_edge[i]:
                        lambda_low = lambda_mid
                    else:
                        lambda_high = lambda_mid
                
                # Assign finalized constrained results
                x_peer[i, :] = x_p
                p_peer[i, :] = p_p
                x_cloud[i] = x_c
                p_cloud[i] = p_c

        return {
            'f_edge': f_edge,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'f_cloud': f_cloud
        }