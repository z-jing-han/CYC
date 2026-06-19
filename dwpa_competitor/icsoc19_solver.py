import numpy as np
from config import Config

class ICSOC19Solver:
    """
    Implementation of the ICSOC19 Algorithm.
    Focus: Carbon Emission Minimization via Search.
    Logic: Iterates through potential offloading amounts to find the one 
    that minimizes Total Carbon (Transmission + Remote Execution).
    """
    def __init__(self, env):
        self.env = env

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

        N0_watts = Config.NOISE_POWER
        if N0_watts < 0: 
            N0_watts = 10 ** ((N0_watts - 30) / 10)

        # 1. Local Computation (Initial calculation required for comparison)
        # Standard Lyapunov-based freq, used as baseline "Local Emission"
        for i in range(num_edge):
            denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
            if denom > 1e-40:
                f_opt = np.sqrt(Q_edge[i] / denom)
                f_edge[i] = np.clip(f_opt, 0, Config.EDGE_F_MAX)
            else:
                f_edge[i] = Config.EDGE_F_MAX

        # 2. Peer Offloading (Greedy Search)

        for i in range(num_edge):
            if Q_edge[i] <= 0: continue

            # Baseline: Local Emission
            local_emission = CI_edge[i] * (f_edge[i] ** 3) * Config.ZETA * Config.CONST_JOULE_TO_KWH

            best_target = -1
            best_bits = 0
            best_power = 0
            min_total_emission = local_emission

            # Check all neighbors
            for j in neighbors_map[i]:
                # Heuristic search step size from ICSOC19.py
                step = max(1, int(Q_edge[i] // 10))
                
                for bits in range(0, int(Q_edge[i]) + step, step):
                    if bits <= 0: continue
                    bits = min(bits, Q_edge[i])

                    # Calculate required power to send 'bits' in 1 timeslot
                    required_rate = bits / Config.TIME_SLOT_DURATION
                    # Shannon inverse: P = (2^(R/W) - 1) * N0 / G
                    p_required = (2 ** (required_rate / Config.BANDWIDTH) - 1) * N0_watts / Config.G_IJ
                    
                    if p_required <= Config.EDGE_P_MAX:
                        # Transmission Emission
                        tx_emission = CI_edge[i] * p_required * Config.TIME_SLOT_DURATION * Config.CONST_JOULE_TO_KWH
                        
                        # Remote Processing Emission (Estimate using simple model as per paper code)
                        # Note: The paper assumes a simple cubic model for the destination
                        # We use the destination's current freq for estimation or max
                        # Replicating code: dest_emission = dest_CI * (freq^3) * bits * const
                        # The code uses bits directly in the cubic formula which implies normalized units
                        # We will use the standard cubic energy model: E = k * f^2 * bits (simplified)
                        # or strictly follow code: CI * (f^3) * bits * const. 
                        # Assuming destination runs at current f_edge[j]
                        dest_emission = CI_edge[j] * (f_edge[j]**3) * bits * Config.CONST_JOULE_TO_KWH # Simplified based on code logic
                        
                        total = tx_emission + dest_emission
                        
                        if total < min_total_emission:
                            min_total_emission = total
                            best_bits = bits
                            best_power = p_required
                            best_target = j
            
            if best_target != -1:
                x_peer[i, best_target] = best_bits
                p_peer[i, best_target] = best_power
                # Deduct from local queue immediately for Cloud step
                Q_edge[i] -= best_bits

        # 3. Cloud Offloading (Greedy Search)
        for i in range(num_edge):
            if Q_edge[i] <= 0: continue
            
            # Recalculate local emission with remaining queue
            local_emission = CI_edge[i] * (f_edge[i] ** 3) * Config.ZETA * Config.CONST_JOULE_TO_KWH
            
            best_bits = 0
            best_power = 0
            min_total_emission = local_emission
            
            step = max(1, int(Q_edge[i] // 10))
            
            for bits in range(0, int(Q_edge[i]) + step, step):
                if bits <= 0: continue
                bits = min(bits, Q_edge[i])
                
                required_rate = bits / Config.TIME_SLOT_DURATION
                p_required = (2 ** (required_rate / Config.BANDWIDTH) - 1) * N0_watts / Config.G_IC
                
                if p_required <= Config.EDGE_P_MAX:
                    tx_emission = CI_edge[i] * p_required * Config.TIME_SLOT_DURATION * Config.CONST_JOULE_TO_KWH
                    # Cloud processing emission
                    cloud_emission = CI_cloud[i] * (Config.CLOUD_F_MAX**3) * bits * Config.CONST_JOULE_TO_KWH
                    
                    total = tx_emission + cloud_emission
                    
                    if total < min_total_emission:
                        min_total_emission = total
                        best_bits = bits
                        best_power = p_required
            
            if best_bits > 0:
                x_cloud[i] = best_bits
                p_cloud[i] = best_power

        # 4. Cloud Frequencies
        # Standard update
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