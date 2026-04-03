import numpy as np
import gurobipy as gp
from gurobipy import GRB

from config import Config

class GurobiSolver:
    def __init__(self, env, variant='joint_opt'):
        self.env = env
        self.variant = variant
        self.T_off_total = Config.TIME_SLOT_DURATION

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
        # Step 1 & 2: Local & Cloud Computation Resource Allocation (Closed-form)
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
        # Step 3 & 4: Joint Data & Time Allocation via Gurobi General Constraints
        # =========================================================================
        X_SCALE = Config.MB_TO_BITS

        for i in range(num_edge):
            active_peers = [j for j in neighbors_map[i] if Q_edge[i] > Q_edge[j]]
            active_cloud = (Q_edge[i] > Q_cloud[i])
            
            targets = [('peer', j) for j in active_peers]
            if active_cloud:
                targets.append(('cloud', i))
                
            M = len(targets)
            if M == 0:
                continue
            
            # 1. Extract coefficients for dynamic scaling calculation
            E_factors = []
            Q_diffs = []
            R_maxs = []
            
            for ttype, tgt in targets:
                g = Config.G_IJ if ttype == 'peer' else Config.G_IC
                Q_tgt = Q_edge[tgt] if ttype == 'peer' else Q_cloud[tgt]
                
                E_factors.append(Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.NOISE_POWER / g)
                Q_diffs.append(Q_tgt - Q_edge[i])
                R_maxs.append(Config.BANDWIDTH * np.log2(1 + Config.EDGE_P_MAX * g / Config.NOISE_POWER))

            max_coef = max(max(E_factors), max([abs(qd * X_SCALE) for qd in Q_diffs]))
            OBJ_SCALE = max_coef if max_coef > 1e-9 else 1.0

            try:
                # 2. Initialize the Gurobi model
                m = gp.Model(f"Edge_{i}")
                m.setParam('OutputFlag', 0)
                m.setParam('NonConvex', 2)     # enable bilinear constraints (x = u * t)
                m.setParam('TimeLimit', GRB.INFINITY)
                m.setParam('MIPGap', 1e-3)
                
                # 3. x_vars (MB), t_vars (sec)
                x_vars = m.addVars(M, lb=0, name="x")
                t_vars = m.addVars(M, lb=1e-4, ub=self.T_off_total, name="t")
                
                # Shannon Capacity's variable: u = rate/W, v = 2^u
                u_vars = m.addVars(M, lb=0, name="u")
                v_vars = m.addVars(M, lb=1, name="v")
                
                p_vars = m.addVars(M, lb=0, ub=Config.EDGE_P_MAX, name="p")
                e_vars = m.addVars(M, lb=0, name="e")

                # 4. Set up constraints
                m.addConstr(gp.quicksum(x_vars[idx] for idx in range(M)) <= Q_edge[i] / X_SCALE)
                m.addConstr(gp.quicksum(t_vars[idx] for idx in range(M)) <= self.T_off_total)

                obj_expr = 0
                for idx, (ttype, tgt) in enumerate(targets):
                    g = Config.G_IJ if ttype == 'peer' else Config.G_IC
                    
                    m.addConstr(x_vars[idx] <= Q_edge[i] / X_SCALE)
                    m.addConstr(x_vars[idx] * X_SCALE <= R_maxs[idx] * t_vars[idx])
                    
                    # From Shannon: x = t * W * log2(1 + gP/N) => u = log2(1 + gP/N) => x = u * t * W (Bilinear Constraint)
                    m.addConstr(x_vars[idx] * X_SCALE == u_vars[idx] * t_vars[idx] * Config.BANDWIDTH)
                    
                    # Set an upper bound for u to accelerate convergence
                    u_max = R_maxs[idx] / Config.BANDWIDTH
                    u_vars[idx].ub = u_max
                    
                    # Exponential constraint: v = 2^u
                    m.addGenConstrExpA(u_vars[idx], v_vars[idx], 2.0, name=f"exp_{idx}")
                    
                    # Power constraint: P = (N/g) * (v - 1)
                    m.addConstr(p_vars[idx] == (Config.NOISE_POWER / g) * (v_vars[idx] - 1))
                    
                    # Energy constraint: E = P * t (Bilinear Constraint)
                    m.addConstr(e_vars[idx] == p_vars[idx] * t_vars[idx])
                    
                    # Objective
                    cost_E = (Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH) * e_vars[idx]
                    cost_Q = Q_diffs[idx] * x_vars[idx] * X_SCALE
                    obj_expr += (cost_E + cost_Q) / OBJ_SCALE

                # 5. Objective and optimaization
                m.setObjective(obj_expr, GRB.MINIMIZE)
                m.optimize()

                # 6. Extract and reconstruct results
                if m.status in [GRB.OPTIMAL, GRB.SUBOPTIMAL, GRB.TIME_LIMIT] and m.SolCount > 0:
                    for idx, (ttype, tgt) in enumerate(targets):
                        curr_x = max(x_vars[idx].X, 0.0) * X_SCALE
                        curr_t = max(t_vars[idx].X, 0.0)
                        
                        if curr_x > 1e-9 and curr_t > 1e-9:
                            g = Config.G_IJ if ttype == 'peer' else Config.G_IC
                            p_req = (2 ** (curr_x / (Config.BANDWIDTH * curr_t)) - 1) * (Config.NOISE_POWER / g)
                            p_req = np.clip(p_req, 0, Config.EDGE_P_MAX)
                            
                            if ttype == 'peer':
                                x_peer[i, tgt] = curr_x
                                p_peer[i, tgt] = p_req
                                t_peer_out[i, tgt] = curr_t
                            else:
                                x_cloud[i] = curr_x
                                p_cloud[i] = p_req
                                t_cloud_out[i] = curr_t
            except Exception as e:
                print(f"[Gurobi] Solver failed for Edge {i}: {e}")

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