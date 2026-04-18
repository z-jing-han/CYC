import numpy as np
from config import Config

# def calculate_rewards(state, next_state, info, carbon, decisions, V_param=Config.MARL_V, penalty_weight=1e6):
#     rewards = {}
#     num_edge = len(state['Q_edge'])
#     edge_metrics = info.get('edge_metrics', [])
#     cloud_metrics = info.get('cloud_metrics', [])
#     penalties = decisions.get('penalties', np.zeros(num_edge))

#     for i in range(num_edge):
#         drift_edge = 0.5 * (((next_state['Q_edge'][i] / (8 * 1024 * 1024)) ** 2) - ((state['Q_edge'][i] / (8 * 1024 * 1024)) ** 2))
#         drift_cloud = 0.5 * (((next_state['Q_cloud'][i] / (8 * 1024 * 1024)) ** 2) - ((state['Q_cloud'][i] / (8 * 1024 * 1024)) ** 2))
#         drift_penalty = drift_edge + drift_cloud
#         absolute_penalty = 0.01 * ((next_state['Q_edge'][i] / (8 * 1024 * 1024)) + (next_state['Q_cloud'][i] / (8 * 1024 * 1024)))
#         # q_penalty = next_state['Q_edge'][i] - state['Q_edge'][i] + next_state['Q_cloud'][i] - state['Q_cloud'][i]
#         # q_penalty = next_state['Q_edge'][i] + next_state['Q_cloud'][i]
#         carbon_penalty = 0.0
#         if i < len(edge_metrics):
#             carbon_penalty += edge_metrics[i]['carbon']
#         if i < len(cloud_metrics):
#             carbon_penalty += cloud_metrics[i]['carbon']
#         power_violation_penalty = penalty_weight * penalties[i]
        
#         rewards[i] = - (drift_penalty + absolute_penalty + V_param * carbon_penalty + power_violation_penalty)
#         print("C:", carbon_penalty)
#         print("Q:", drift_penalty)
#         print("A:", absolute_penalty)
#     print(rewards)
#     return rewards

def calculate_rewards(state, next_state, info, carbon, decisions, V_param=Config.MARL_V, penalty_weight=1e6):
    rewards = {}
    num_edge = len(state['Q_edge'])
    edge_metrics = info.get('edge_metrics', [])
    cloud_metrics = info.get('cloud_metrics', [])
    penalties = decisions.get('penalties', np.zeros(num_edge))

    BITS_TO_MB = 8 * 1024 * 1024
    TOLERANCE_MB = 15000.0 

    for i in range(num_edge):
        q_edge_t_minus_1 = max(0.0, (state['Q_edge'][i] / BITS_TO_MB) - TOLERANCE_MB)
        q_cloud_t_minus_1 = max(0.0, (state['Q_cloud'][i] / BITS_TO_MB) - TOLERANCE_MB)
        
        q_edge_t = max(0.0, (next_state['Q_edge'][i] / BITS_TO_MB) - TOLERANCE_MB)
        q_cloud_t = max(0.0, (next_state['Q_cloud'][i] / BITS_TO_MB) - TOLERANCE_MB)
        
        drift_edge = 0.5 * ((q_edge_t ** 2) - (q_edge_t_minus_1 ** 2))
        drift_cloud = 0.5 * ((q_cloud_t ** 2) - (q_cloud_t_minus_1 ** 2))
        drift_penalty = drift_edge + drift_cloud
        
        # absolute_penalty = 0.0001 * ((next_state['Q_edge'][i] / BITS_TO_MB) + (next_state['Q_cloud'][i] / BITS_TO_MB))
        
        carbon_penalty = 0.0
        if i < len(edge_metrics):
            carbon_penalty += edge_metrics[i]['carbon']
        if i < len(cloud_metrics):
            carbon_penalty += cloud_metrics[i]['carbon']
            
        power_violation_penalty = penalty_weight * penalties[i]
        
        rewards[i] = - (drift_penalty + V_param * carbon_penalty + power_violation_penalty)
        
    return rewards

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

def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    """Compute Generalized Advantage Estimation"""
    advs = []
    gae = 0
    for step in reversed(range(len(rewards))):
        if step == len(rewards) - 1:
            delta = rewards[step] + gamma * next_value * (1 - dones[step]) - values[step]
        else:
            delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advs.insert(0, gae)
    return advs

class RunningMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, np.float32)
        self.var = np.ones(shape, np.float32)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

def compute_ao_state(state):
    Q_edge = state['Q_edge'].copy()
    Q_cloud = state['Q_cloud'].copy()
    CI_edge = state['CI_edge']
    CI_cloud = state['CI_cloud']
    num_edge = len(Q_edge)
    
    f_edge = np.zeros(num_edge)
    f_cloud = np.zeros(num_edge)
    
    for i in range(num_edge):
        # Local Computing
        denom = 3 * Config.V * CI_edge[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
        if denom > 1e-40:
            f_edge[i] = np.clip(np.sqrt(Q_edge[i] / denom), 0, Config.EDGE_F_MAX)
        else:
            f_edge[i] = Config.EDGE_F_MAX
        bits_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
        Q_edge[i] = max(0, Q_edge[i] - bits_local)
        
        # Cloud computing
        denom_c = 3 * Config.V * CI_cloud[i] * Config.CONST_JOULE_TO_KWH * Config.ZETA * Config.PHI
        if denom_c > 1e-40:
            f_cloud[i] = np.clip(np.sqrt(Q_cloud[i] / denom_c), 0, Config.CLOUD_F_MAX)
        else:
            f_cloud[i] = Config.CLOUD_F_MAX
        bits_cloud = (f_cloud[i] / Config.PHI) * Config.TIME_SLOT_DURATION
        Q_cloud[i] = max(0, Q_cloud[i] - bits_cloud)
        
    post_comp_state = state.copy()
    post_comp_state['Q_edge'] = Q_edge
    post_comp_state['Q_cloud'] = Q_cloud
    
    return f_edge, f_cloud, post_comp_state