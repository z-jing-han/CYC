import numpy as np
from config import Config

class RandomAgent:
    """
    Random Strategy Agent
    Output:
    1. Local Ratio
    2. Cloud Ratio
    3. Peer Ratio (Allocated to the remaining N-1 neighbors)
    """
    def __init__(self, agent_id, num_agents):
        self.agent_id = agent_id
        self.num_agents = num_agents
        
        # Action: Local + Cloud + (N-1) Peers
        self.action_dim = 1 + 1 + (num_agents - 1)

    def get_action(self, state):
        # Randomly generate ratios
        raw = np.random.rand(self.action_dim)
        action_probs = raw / np.sum(raw)
        return action_probs

class MARLController:
    """
    Convert the outputs of multiple agents into a 'decisions' dictionary acceptable to the environment and handle physical constraints.
    """
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.num_edge = Config.NUM_EDGE_SERVERS

    def get_decisions(self, state):
        # 1. Collect actions (ratios) from all agents.
        # It is assumed here that the states have already been assigned to each agent; for simplicity, the global state is read directly.
        # In actual RL training, the state must be decomposed/split for each agent.
        
        f_edge = np.zeros(self.num_edge)
        f_cloud = np.zeros(self.num_edge)
        x_peer = np.zeros((self.num_edge, self.num_edge))
        p_tx_peer = np.zeros((self.num_edge, self.num_edge))
        x_cloud = np.zeros(self.num_edge)
        p_tx_cloud = np.zeros(self.num_edge)
        
        Q_edge = state['Q_edge']
        
        for i in range(self.num_edge):
            agent = self.agents[i]
            # Simplification: Agents observe either global information or only their own.
            # For a Random Agent, the content of the State is irrelevant.
            probs = agent.get_action(None)
            
            # Parse ratios
            ratio_local = probs[0]
            ratio_cloud = probs[1]
            ratio_peers = probs[2:]
            
            # --- Convert to physical quantities ---
            
            # 1. Local Frequency
            # If local processing is decided, assume a frequency proportional to the queue or a fixed frequency is used.
            # Here, it is simply set to: allocate the maximum processing capacity according to the ratio, or set it based on current Queue * Ratio.
            # For simplicity and to maintain Random behavior, it is set as Max * Ratio.
            f_edge[i] = Config.EDGE_F_MAX * ratio_local
            
            # 2. Cloud Offloading
            # etermine the transmission volume x_{cloud}
            # Total task volume = current Queue size
            task_load = Q_edge[i]
            x_cloud[i] = task_load * ratio_cloud
            # Set the transmission power (if transmission occurs, set it to maximum)
            if x_cloud[i] > 1e-3:
                p_tx_cloud[i] = Config.CLOUD_P_TX_MAX
                # Cloud CPU Assume on-demand allocation
                f_cloud[i] = Config.CLOUD_F_MAX * 0.2 # Assumed values
            
            # 3. Peer Offloading
            peer_indices = [k for k in range(self.num_edge) if k != i]
            for idx, neighbor_id in enumerate(peer_indices):
                ratio = ratio_peers[idx]
                amount = task_load * ratio
                x_peer[i, neighbor_id] = amount
                if amount > 1e-3:
                    p_tx_peer[i, neighbor_id] = Config.EDGE_P_TX_MAX
                    
        return {
            'f_edge': f_edge,
            'f_cloud': f_cloud,
            'p_tx_peer': p_tx_peer,
            'x_peer': x_peer,
            'p_tx_cloud': p_tx_cloud,
            'x_cloud': x_cloud
        }