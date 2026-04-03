import numpy as np
from config import Config

class AOSolver:
    def __init__(self, env, variant='joint_opt'):
        self.env = env
        self.variant = variant
        self.T_off_total = Config.TIME_SLOT_DURATION
        self.lambda_tolerance = 1e-12
        self.max_bisect_iter = 50
        self.max_alt_iter = 15

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
        t_peer_out = np.zeros((num_edge, num_edge))
        
        x_cloud = np.zeros(num_edge)
        p_cloud = np.zeros(num_edge)
        t_cloud_out = np.zeros(num_edge)
        f_cloud = np.zeros(num_edge)

        # =========================================================================
        # Step 1 & 2: Local & Cloud Computation Resource Allocation
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

        for i in range(num_edge):
            denom_c = 3 * Config.V * CI_cloud[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom_c > 1e-40:
                f_c_opt = np.sqrt(Q_cloud[i] / denom_c)
                f_cloud[i] = np.clip(f_c_opt, 0, Config.CLOUD_F_MAX) 
            else:
                f_cloud[i] = Config.CLOUD_F_MAX

        # =========================================================================
        # Step 3 & 4: Exact Joint Data-Time Allocation (Alternating Bisection)
        # =========================================================================
        R_max_peer = Config.BANDWIDTH * np.log2(1 + Config.EDGE_P_MAX * Config.G_IJ / Config.NOISE_POWER)
        R_max_cloud = Config.BANDWIDTH * np.log2(1 + Config.EDGE_P_MAX * Config.G_IC / Config.NOISE_POWER)

        for i in range(num_edge):
            active_peers = [j for j in neighbors_map[i] if Q_edge[i] > Q_edge[j]]
            active_cloud = (Q_edge[i] > Q_cloud[i])
            num_active = len(active_peers) + (1 if active_cloud else 0)
            
            if num_active == 0:
                continue
            
            t_p = np.zeros(num_edge)
            t_c = 0.0
            initial_t = self.T_off_total / num_active
            for j in active_peers: t_p[j] = initial_t
            if active_cloud: t_c = initial_t
            
            common_denom = Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.NOISE_POWER * np.log(2)
            A_peer = (Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.NOISE_POWER) / Config.G_IJ
            A_cloud = (Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.NOISE_POWER) / Config.G_IC

            curr_x_p = np.zeros(num_edge)
            curr_x_c = 0.0

            for alt_iter in range(self.max_alt_iter):
                prev_x_p = curr_x_p.copy()
                prev_x_c = curr_x_c
                
                # -----------------------------------------------------------------
                # [Phase 1]: Data Allocation (λ-Bisection for fixed time t)
                # -----------------------------------------------------------------
                def calc_data_given_lambda(lambda_val):
                    temp_x_p = np.zeros(num_edge)
                    temp_x_c = 0.0
                    
                    for j in active_peers:
                        if t_p[j] < 1e-9: continue
                        term = Q_edge[i] - Q_edge[j] - lambda_val
                        if term > 0:
                            val = (Config.BANDWIDTH * Config.G_IJ * term) / common_denom
                            if val > 1:
                                x_val = Config.BANDWIDTH * t_p[j] * np.log2(val)
                                max_transfer_p = (Q_edge[i] - Q_edge[j]) / 2.0 
                                temp_x_p[j] = np.clip(x_val, 0, max_transfer_p)
                                
                    if active_cloud and t_c >= 1e-9:
                        term_c = Q_edge[i] - Q_cloud[i] - lambda_val
                        if term_c > 0:
                            val_c = (Config.BANDWIDTH * Config.G_IC * term_c) / common_denom
                            if val_c > 1:
                                x_c_val = Config.BANDWIDTH * t_c * np.log2(val_c)
                                max_transfer_c = (Q_edge[i] - Q_cloud[i]) / 2.0
                                temp_x_c = np.clip(x_c_val, 0, max_transfer_c)
                                
                    return temp_x_p, temp_x_c

                xp_test, xc_test = calc_data_given_lambda(0.0)
                if np.sum(xp_test) + xc_test <= Q_edge[i]:
                    curr_x_p, curr_x_c = xp_test, xc_test
                else:
                    lambda_low, lambda_high = 0.0, Q_edge[i]
                    for bis in range(self.max_bisect_iter):
                        lambda_mid = (lambda_low + lambda_high) / 2.0
                        xp_tmp, xc_tmp = calc_data_given_lambda(lambda_mid)
                        if np.sum(xp_tmp) + xc_tmp > Q_edge[i]:
                            lambda_low = lambda_mid
                        else:
                            lambda_high = lambda_mid
                    curr_x_p, curr_x_c = calc_data_given_lambda(lambda_high)

                total_data = np.sum(curr_x_p) + curr_x_c
                if total_data < 1e-9:
                    break
                
                min_t_req = sum(curr_x_p[j] / R_max_peer for j in active_peers) + (curr_x_c / R_max_cloud if active_cloud else 0)
                if min_t_req > self.T_off_total:
                    scale = self.T_off_total / min_t_req
                    curr_x_p *= scale
                    curr_x_c *= scale
                else:
                    for j in active_peers:
                        curr_x_p[j] = min(curr_x_p[j], R_max_peer * self.T_off_total)
                    curr_x_c = min(curr_x_c, R_max_cloud * self.T_off_total)

                # -----------------------------------------------------------------
                # [Phase 2]: Exact Time Allocation (μ-Bisection for fixed data x)
                # -----------------------------------------------------------------
                def get_t_for_mu(mu_val):
                    temp_t_p = np.zeros(num_edge)
                    temp_t_c = 0.0
                    t_tot = 0.0
                    
                    # g(z) = 1 + z * ln(2) * 2^z - 2^z
                    def g_func(z):
                        return 1.0 + z * np.log(2) * (2**z) - (2**z)
                        
                    for j in active_peers:
                        xj = curr_x_p[j]
                        if xj < 1e-9: continue
                        target = mu_val / A_peer
                        
                        r_low = xj / self.T_off_total
                        r_high = R_max_peer
                        
                        if g_func(r_high / Config.BANDWIDTH) <= target:
                            r_opt = r_high
                        elif g_func(r_low / Config.BANDWIDTH) >= target:
                            r_opt = r_low
                        else:
                            for _ in range(25):
                                r_mid = (r_low + r_high) / 2
                                if g_func(r_mid / Config.BANDWIDTH) < target:
                                    r_low = r_mid
                                else:
                                    r_high = r_mid
                            r_opt = (r_low + r_high) / 2
                            
                        tj = xj / r_opt
                        temp_t_p[j] = tj
                        t_tot += tj
                        
                    if active_cloud and curr_x_c >= 1e-9:
                        xc = curr_x_c
                        target = mu_val / A_cloud
                        r_low = xc / self.T_off_total
                        r_high = R_max_cloud
                        
                        if g_func(r_high / Config.BANDWIDTH) <= target:
                            r_opt = r_high
                        elif g_func(r_low / Config.BANDWIDTH) >= target:
                            r_opt = r_low
                        else:
                            for _ in range(25):
                                r_mid = (r_low + r_high) / 2
                                if g_func(r_mid / Config.BANDWIDTH) < target:
                                    r_low = r_mid
                                else:
                                    r_high = r_mid
                            r_opt = (r_low + r_high) / 2
                            
                        tc = xc / r_opt
                        temp_t_c = tc
                        t_tot += tc
                        
                    return t_tot, temp_t_p, temp_t_c
                
                mu_low = 0.0
                mu_high = 1e-12
                while True:
                    t_tot, _, _ = get_t_for_mu(mu_high)
                    if t_tot <= self.T_off_total + 1e-9:
                        break
                    mu_high *= 10.0
                    
                for bis in range(self.max_bisect_iter):
                    mu_mid = (mu_low + mu_high) / 2.0
                    t_tot, _, _ = get_t_for_mu(mu_mid)
                    if t_tot > self.T_off_total:
                        mu_low = mu_mid
                    else:
                        mu_high = mu_mid
                        
                _, new_t_p, new_t_c = get_t_for_mu(mu_high)

                diff_t = np.sum(np.abs(new_t_p - t_p)) + abs(new_t_c - t_c)
                diff_x = np.sum(np.abs(curr_x_p - prev_x_p)) + abs(curr_x_c - prev_x_c)

                if diff_t < 1e-6 and diff_x < 1e-6:
                    t_p = new_t_p
                    t_c = new_t_c
                    break
                
                alpha = 0.6              
                old = sum(abs(new_t_p-t_p))+abs(new_t_c-t_c)
                t_p = alpha * new_t_p + (1 - alpha) * t_p
                t_c = alpha * new_t_c + (1 - alpha) * t_c
            
            x_peer[i, :] = curr_x_p
            x_cloud[i] = curr_x_c
            t_peer_out[i, :] = t_p
            t_cloud_out[i] = t_c
            
            # From Lemma 1
            for j in active_peers:
                if t_p[j] > 1e-9 and curr_x_p[j] > 1e-9:
                    p_req = (2 ** (curr_x_p[j] / (Config.BANDWIDTH * t_p[j])) - 1) * (Config.NOISE_POWER / Config.G_IJ)
                    p_peer[i, j] = np.clip(p_req, 0, Config.EDGE_P_MAX)
                    
            if t_c > 1e-9 and curr_x_c > 1e-9:
                p_c_req = (2 ** (curr_x_c / (Config.BANDWIDTH * t_c)) - 1) * (Config.NOISE_POWER / Config.G_IC)
                p_cloud[i] = np.clip(p_c_req, 0, Config.EDGE_P_MAX)

        return {
            'f_edge': f_edge,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud,
            'f_cloud': f_cloud,
            't_peer': t_peer_out, 
            't_cloud': t_cloud_out
        }
