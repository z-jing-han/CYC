import numpy as np
from config import Config

class DWPASolver:
    def __init__(self, env, variant='default'):
        self.env = env
        self.epsilon = 1e-6
        self.max_iter = 10
        self.variant = variant
        self.t_off_dict = {}
    
    def _task_allocate(self, CI, trans_power, Q_self, Q_target, gain, t_off):
        snr = (gain * trans_power) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        if rate < 1e-9:
            return 0
        a_param = Config.V * CI * Config.CONST_JOULE_TO_KWH * trans_power / rate
        b_param = Q_self - Q_target
        return rate * t_off if a_param < b_param else 0
    
    def _power_allocate(self, CI, Q_self, Q_target, gain, t_off):
        a_param = Config.V * CI * Config.CONST_JOULE_TO_KWH * t_off
        b_param = (Q_target - Q_self) * Config.BANDWIDTH * t_off
        c_param = gain / Config.NOISE_POWER
        p_opt = - (b_param / (a_param * np.log(2))) - (1.0 / c_param)
        return np.clip(p_opt, 0, Config.EDGE_P_MAX)

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
        #      Thesis Algorithm 3 Line 4, Eq (5.2)
        # =========================================================================
        for i in range(num_edge):
            # denom = 3 * V * k * xi * phi
            denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_opt = np.sqrt(Q_edge[i] / denom)
                f_edge[i] = np.clip(f_opt, 0, Config.EDGE_F_MAX)
            else:
                f_edge[i] = Config.EDGE_F_MAX
            
            # Pub. Algorithm 1 Line 3
            # bits = cycles / (cycles/bit)
            bits_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            Q_edge[i] = max(0, Q_edge[i] - bits_local)

        # Pub. Algorithm 1 Line 4 (Update Q_cloud[i] from offloading): Environment setup complete

        # =========================================================================
        # Step 2: Cloud Computation Resource Allocation
        # Ref: Pub. Algorithm 1 (P1b) Line 5, Eq (16)
        #      Thesis Algorithm 3 Line 7, Eq (5.12)
        # =========================================================================
        for i in range(num_edge):
            denom = 3 * Config.V * CI_cloud[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_c_opt = np.sqrt(Q_cloud[i] / denom)
                f_cloud[i] = np.clip(f_c_opt, 0, Config.CLOUD_F_MAX) 
            else:
                f_cloud[i] = Config.CLOUD_F_MAX

        # Pub. Algorithm 1 Line 6 (Update Q_cloud[i] after cloud compute): To be executed by the environment

        # =========================================================================
        # Step 3: Peer Offloading (P2)
        # Ref: Pub. Algorithm 1 Line 7 - 10, Eq (17), Eq (18)
        #      Thesis Algorithm 3 Line 5, Algorithm 1, Eq (5.3) - (5.6)
        # =========================================================================
        
        for i in range(num_edge):
            if Config.TIME_SLOT_ADJUST == "scale":
                eligible_neighbors = sum(1 for j in neighbors_map[i] if Q_edge[i] > Q_edge[j])
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (eligible_neighbors + 1)
            elif Config.TIME_SLOT_ADJUST == "fix_time_slot" :
                self.t_off_dict[i] = Config.TIME_SLOT_DURATION / (num_edge + 1)
        
        for iter in range(self.max_iter):
            # Pub. Algorithm 1 Line 2 - 5
            max_err = 0.0
            for i in range(num_edge):
                for j in neighbors_map[i]:
                    if Q_edge[i] <= Q_edge[j]:
                        x_peer[i, j] = 0
                        p_peer[i, j] = 0
                        continue

                    old_x = x_peer[i, j]
                    old_p = p_peer[i, j]

                    # Step 3-1
                    # Pub. Algorithm 1 (P2a) Line 8, Eq (17)
                    # Thesis Algorithm 1 Line 7, Eq (5.3), Eq (5.4)
                    x_peer[i, j] = self._task_allocate(CI_edge[i], p_peer[i, j], Q_edge[i], Q_edge[j], Config.G_IJ, self.t_off_dict[i])

                    # Step 3-2
                    # Pub. Algorithm 1 (P2b) Line 8, Eq (18)
                    # Thesis Algorithm 1 Line 9, Eq (5.5), Eq (5.6)
                    p_peer[i, j] = self._power_allocate(CI_edge[i], Q_edge[i], Q_edge[j], Config.G_IJ, self.t_off_dict[i])

                    err_x = abs(x_peer[i, j] - old_x) / (old_x + 1e-9)
                    err_p = abs(p_peer[i, j] - old_p) / (old_p + 1e-9)
                    
                    max_err = max(max_err, err_x, err_p)
            
            if max_err <= self.epsilon:
                break

        # Pub. Algorithm 1 Line 11
        tx_sum = np.sum(x_peer, axis=1)
        rx_sum = np.sum(x_peer, axis=0)
        Q_edge_post_peer = np.maximum(0, Q_edge - tx_sum + rx_sum)

        # =========================================================================
        # Step 4: Cloud Offloading (P3)
        # Ref: Pub. Algorithm 1 Line 12 - 15, Eq (17), Eq (18)
        #      Thesis Algorithm 3 Line 5, Algorithm 1, Eq (5.3) - (5.6)
        # =========================================================================
        for iter in range(self.max_iter):
            max_err = 0.0
            for i in range(num_edge):
                curr_q_edge = Q_edge_post_peer[i]
                curr_q_cloud = Q_cloud[i]

                if curr_q_edge <= curr_q_cloud:
                    x_cloud[i] = 0
                    p_cloud[i] = 0
                    continue
                
                old_x = x_cloud[i]
                old_p = p_cloud[i]

                # Step 4-1
                # Pub. Algorithm 1 (P3a) Line 13, Eq (20)
                # Thesis Algorithm 2 Line 7, Eq (5.7), Eq (5.8)
                x_cloud[i] = self._task_allocate(CI_edge[i], p_cloud[i], curr_q_edge, curr_q_cloud, Config.G_IC, self.t_off_dict[i])
                x_cloud[i] = min(x_cloud[i], curr_q_edge)
                # x_cloud[i] = max(x_cloud[i], curr_q_edge)

                # Step 4-2
                # Pub. Algorithm 1 (P3b) Line 14, Eq (21)
                # Thesis Algorithm 2 Line 9, Eq (5.9), Eq (5.10)
                p_cloud[i] = self._power_allocate(CI_edge[i], curr_q_edge, curr_q_cloud, Config.G_IC, self.t_off_dict[i])
                

                err_x = abs(x_cloud[i] - old_x) / (old_x + 1e-9)
                err_p = abs(p_cloud[i] - old_p) / (old_p + 1e-9)
                
                max_err = max(max_err, err_x, err_p)

            if max_err <= self.epsilon:
                break
        
        # Pub. Algorithm 1 Line 16 (Update Q_edge[i] after cloud offload): To be executed by the environment

        # Variant
        if self.variant == "VO":
            x_peer = np.zeros((num_edge, num_edge))
            p_peer = np.zeros((num_edge, num_edge))
        elif self.variant == "HF":
            for i in range(num_edge):
                if Q_edge_post_peer[i] <= Config.EEDGE_Q_CAPACITY[i]:
                    x_cloud[i] = 0
                    p_cloud[i] = 0
                else:
                    x_cloud[i] = Q_edge_post_peer[i] - Config.EEDGE_Q_CAPACITY[i]
                    req_p = (Config.NOISE_POWER / Config.G_IC) * (2 ** ((x_cloud[i] / self.t_off_dict[i]) / Config.BANDWIDTH) - 1)
                    p_cloud[i] = np.clip(req_p, 0, Config.EDGE_P_MAX)
        elif self.variant == "LF":
            x_peer = np.zeros((num_edge, num_edge))
            p_peer = np.zeros((num_edge, num_edge))
            for i in range(num_edge):
                if Q_edge[i] <= Config.EEDGE_Q_CAPACITY[i]:
                    x_cloud[i] = 0
                    p_cloud[i] = 0
                else:
                    x_cloud[i] = Q_edge[i] - Config.EEDGE_Q_CAPACITY[i]

        return {
            'f_edge': f_edge,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'f_cloud': f_cloud
        }