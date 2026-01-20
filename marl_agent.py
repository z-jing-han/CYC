import numpy as np
from config import Config

class RandomAgent:
    def __init__(self, agent_id, num_agents):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.action_dim = 1 + 1 + (num_agents - 1)

    def get_action(self, state):
        raw = np.random.rand(self.action_dim)
        action_probs = raw / np.sum(raw)
        return action_probs
    
    def learn(self, state, action, reward, next_state):
        pass

class QLearningAgent:
    def __init__(self, agent_id, num_agents, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.num_peers = num_agents - 1
        self.num_actions = 3 + self.num_peers 
        self.q_table = {}

    def _discretize_state(self, observation):
        q_val = observation['Q_edge'][self.agent_id]
        ci_val = observation['CI_edge'][self.agent_id]
        q_bin = int(q_val // 1e7) 
        ci_bin = int(ci_val * 10) 
        return (q_bin, ci_bin)

    def _get_ratios_from_action(self, action_idx):
        ratios = np.zeros(1 + 1 + self.num_peers)
        if action_idx == 0: ratios[0] = 1.0
        elif action_idx == 1: ratios[1] = 1.0
        elif action_idx == 2: ratios[0] = 0.5; ratios[1] = 0.5
        else:
            peer_target_idx = action_idx - 3
            if 0 <= peer_target_idx < self.num_peers:
                ratios[2 + peer_target_idx] = 1.0
            else:
                ratios[0] = 1.0
        return ratios

    def get_action(self, state):
        state_key = self._discretize_state(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = np.argmax(self.q_table[state_key])
            
        self.last_action = action_idx
        self.last_state = state_key
        return self._get_ratios_from_action(action_idx)

    def learn(self, state, ratios_action, reward, next_state):
        state_key = self.last_state
        next_state_key = self._discretize_state(next_state)
        action_idx = self.last_action
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
            
        best_next_q = np.max(self.q_table[next_state_key])
        current_q = self.q_table[state_key][action_idx]
        
        new_q = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

class MARLController:
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.num_edge = Config.NUM_EDGE_SERVERS

    def get_decisions(self, state):
        f_edge = np.zeros(self.num_edge)
        x_peer = np.zeros((self.num_edge, self.num_edge))
        p_peer = np.zeros((self.num_edge, self.num_edge))
        x_cloud = np.zeros(self.num_edge)
        p_cloud = np.zeros(self.num_edge)
        f_cloud = np.zeros(self.num_edge) 
        
        Q_edge = state['Q_edge']
        
        for i in range(self.num_edge):
            agent = self.agents[i]
            probs = agent.get_action(state)
            
            ratio_local = probs[0]
            ratio_cloud = probs[1]
            ratio_peers = probs[2:] 
            
            task_load = Q_edge[i]
            
            # --- 1. Local Processing ---
            f_edge[i] = Config.EDGE_F_MAX * ratio_local
            
            # --- 2. Cloud Offloading ---
            x_cloud[i] = task_load * ratio_cloud
            if x_cloud[i] > 1e-3:
                p_cloud[i] = Config.EDGE_P_MAX 
                f_cloud[i] = Config.CLOUD_F_MAX
            
            # --- 3. Peer Offloading ---
            peer_indices = [k for k in range(self.num_edge) if k != i]
            
            for idx, neighbor_id in enumerate(peer_indices):
                ratio = ratio_peers[idx]
                amount = task_load * ratio
                
                x_peer[i, neighbor_id] = amount
                if amount > 1e-3:
                    p_peer[i, neighbor_id] = Config.EDGE_P_MAX
                    
        return {
            'f_edge': f_edge,
            'f_cloud': f_cloud,
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud
        }

    def update_agents(self, state, decisions, rewards, next_state):
        for i in range(self.num_edge):
            agent = self.agents[i]
            agent.learn(state, None, rewards[i], next_state)