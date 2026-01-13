import numpy as np
from config import Config

class RandomAgent:
    """
    Random Policy Agent
    Output:
    1. Local Ratio
    2. Cloud Ratio
    3. Peer Ratio (Distributed to remaining N-1 neighbors)
    """
    def __init__(self, agent_id, num_agents):
        self.agent_id = agent_id
        self.num_agents = num_agents
        
        # Action: Local + Cloud + (N-1) Peers
        self.action_dim = 1 + 1 + (num_agents - 1)

    def get_action(self, state):
        # Randomly generate ratios
        raw = np.random.rand(self.action_dim)
        # Softmax-like normalization
        action_probs = raw / np.sum(raw)
        return action_probs
    
    def learn(self, state, action, reward, next_state):
        # Random agent does not learn
        pass

class QLearningAgent:
    """
    Simple Q-Learning Agent (Discrete Action Space)
    Maps discrete actions to continuous ratios for compatibility.
    """
    def __init__(self, agent_id, num_agents, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.agent_id = agent_id
        self.num_agents = num_agents
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Simplified Discrete Action Space
        # 0: 100% Local
        # 1: 100% Cloud
        # 2: 50% Local, 50% Cloud
        # 3 ~ 3+(N-2): 100% to specific neighbor (Simplified)
        self.num_peers = num_agents - 1
        self.num_actions = 3 + self.num_peers 
        
        # Q-Table: Dictionary mapping state (tuple) to action values
        # State: (Discretized Queue Level, Discretized CI Level)
        self.q_table = {}

    def _discretize_state(self, observation):
        """
        Convert continuous observation to a discrete tuple for Q-table key.
        Observation includes Global state, but agent focuses on local info.
        """
        q_val = observation['Q_edge'][self.agent_id]
        ci_val = observation['CI_edge'][self.agent_id]
        
        # Simple discretization bins
        q_bin = int(q_val // 1e7) # e.g., 10Mb chunks
        ci_bin = int(ci_val * 10) # e.g., 0.1 steps
        
        return (q_bin, ci_bin)

    def _get_ratios_from_action(self, action_idx):
        """
        Maps discrete action index to [Local, Cloud, Peer1, Peer2...] ratios
        """
        # Format: [Local, Cloud, Peer_0, Peer_1, ...] (Peer list excludes self)
        ratios = np.zeros(1 + 1 + self.num_peers)
        
        if action_idx == 0: # All Local
            ratios[0] = 1.0
        elif action_idx == 1: # All Cloud
            ratios[1] = 1.0
        elif action_idx == 2: # 50/50 Local/Cloud
            ratios[0] = 0.5
            ratios[1] = 0.5
        else:
            # Offload to specific peer
            peer_target_idx = action_idx - 3
            if 0 <= peer_target_idx < self.num_peers:
                ratios[2 + peer_target_idx] = 1.0
            else:
                # Fallback to local if index error
                ratios[0] = 1.0
                
        return ratios

    def get_action(self, state):
        state_key = self._discretize_state(state)
        
        # Initialize Q-values for this state if not exist
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.num_actions)
        
        # Epsilon-Greedy
        if np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = np.argmax(self.q_table[state_key])
            
        # Store last action for learning update (if needed externally, though typically passed in loop)
        self.last_action = action_idx
        self.last_state = state_key
            
        return self._get_ratios_from_action(action_idx)

    def learn(self, state, ratios_action, reward, next_state):
        """
        Update Q-Table
        Note: 'ratios_action' is what the environment received. 
        We rely on self.last_action to know which discrete choice was made.
        """
        state_key = self.last_state
        next_state_key = self._discretize_state(next_state)
        action_idx = self.last_action
        
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.num_actions)
            
        # Q-Learning Update Rule
        # Q(s,a) = Q(s,a) + lr * (R + gamma * max(Q(s',a')) - Q(s,a))
        best_next_q = np.max(self.q_table[next_state_key])
        current_q = self.q_table[state_key][action_idx]
        
        new_q = current_q + self.lr * (reward + self.gamma * best_next_q - current_q)
        self.q_table[state_key][action_idx] = new_q

class MARLController:
    """
    Converts multiple Agents' outputs into environment-friendly 'decisions' dictionary
    and handles Physical Constraints.
    """
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.num_edge = Config.NUM_EDGE_SERVERS

    def get_decisions(self, state):
        """
        Convert Agent ratio decisions to physical quantities:
        - f_edge (CPU frequency)
        - x_peer, x_cloud (Offloading bits)
        - p_peer, p_cloud (Transmission power)
        """
        f_edge = np.zeros(self.num_edge)
        x_peer = np.zeros((self.num_edge, self.num_edge))
        p_peer = np.zeros((self.num_edge, self.num_edge))
        x_cloud = np.zeros(self.num_edge)
        p_cloud = np.zeros(self.num_edge)
        f_cloud = np.zeros(self.num_edge) # Cloud CPU frequency for each edge user
        
        Q_edge = state['Q_edge']
        
        for i in range(self.num_edge):
            agent = self.agents[i]
            # Pass state to agent (Required for RL Agents)
            probs = agent.get_action(state)
            
            # Parse ratios
            ratio_local = probs[0]
            ratio_cloud = probs[1]
            ratio_peers = probs[2:] # List of (N-1) peers
            
            # 1. Local Frequency Computation
            # Strategy: Occupy CPU capacity proportional to assigned local ratio
            f_edge[i] = Config.EDGE_F_MAX * ratio_local
            
            # Current Queue Load
            task_load = Q_edge[i]
            
            # 2. Cloud Offloading
            x_cloud[i] = task_load * ratio_cloud
            if x_cloud[i] > 1e-3:
                # If decided to transmit, assume max power
                p_cloud[i] = Config.EDGE_P_MAX 
                # [BUG FIX] Must set Cloud frequency, otherwise Cloud energy is 0
                # Assume Cloud runs at standard Edge frequency or faster
                f_cloud[i] = Config.EDGE_F_MAX 
            
            # 3. Peer Offloading
            # Find neighbor indices (excluding self)
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
        """
        Trigger learning for all agents.
        """
        for i in range(self.num_edge):
            agent = self.agents[i]
            # RandomAgent might assume decisions are passed differently, 
            # but QLearningAgent needs the specific logic handled internally or here.
            # Here we just pass the signal to learn.
            # Note: 'decisions' here contains the physical values, not the ratios.
            # Ideally, agents stored their last action internally.
            agent.learn(state, None, rewards[i], next_state)