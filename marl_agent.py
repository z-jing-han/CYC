import numpy as np
from config import Config

class RandomAgent:
    """
    隨機策略 Agent
    輸出:
    1. Local Ratio
    2. Cloud Ratio
    3. Peer Ratio (分配給其餘 N-1 個鄰居)
    """
    def __init__(self, agent_id, num_agents):
        self.agent_id = agent_id
        self.num_agents = num_agents
        
        # Action: Local + Cloud + (N-1) Peers
        self.action_dim = 1 + 1 + (num_agents - 1)

    def get_action(self, state):
        # 隨機產生比例 (未來可替換為 RL Policy 輸出)
        raw = np.random.rand(self.action_dim)
        # Softmax-like normalization
        action_probs = raw / np.sum(raw)
        return action_probs

class MARLController:
    """
    將多個 Agent 的輸出轉換為環境可接受的 'decisions' 字典
    並處理物理限制 (Physical Constraints)
    """
    def __init__(self, env, agents):
        self.env = env
        self.agents = agents
        self.num_edge = Config.NUM_EDGE_SERVERS

    def get_decisions(self, state):
        """
        將 Agent 的比例決策轉換為環境需要的物理量：
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
            # 這裡傳入 state (若是 RL Agent 則需要 observation)
            # RandomAgent 目前不使用 input state
            probs = agent.get_action(None)
            
            # 解析比例
            ratio_local = probs[0]
            ratio_cloud = probs[1]
            ratio_peers = probs[2:] # List of (N-1) peers
            
            # 1. Local Frequency Computation
            # 策略：根據分配到的本地比例，佔用對應比例的 CPU 能力
            f_edge[i] = Config.EDGE_F_MAX * ratio_local
            
            # 當前佇列總量
            task_load = Q_edge[i]
            
            # 2. Cloud Offloading
            x_cloud[i] = task_load * ratio_cloud
            if x_cloud[i] > 1e-3:
                # 若決定傳輸，假設使用最大功率發送
                p_cloud[i] = Config.EDGE_P_MAX 
                # [BUG FIX] 必須設定 Cloud 處理該任務的頻率，否則 Cloud 能耗為 0
                # 這裡假設 Cloud 以標準 Edge 頻率或更快速度處理
                f_cloud[i] = Config.EDGE_F_MAX 
            
            # 3. Peer Offloading
            # 找出鄰居索引 (排除自己)
            peer_indices = [k for k in range(self.num_edge) if k != i]
            
            for idx, neighbor_id in enumerate(peer_indices):
                ratio = ratio_peers[idx]
                amount = task_load * ratio
                
                x_peer[i, neighbor_id] = amount
                if amount > 1e-3:
                    p_peer[i, neighbor_id] = Config.EDGE_P_MAX
                    
        return {
            'f_edge': f_edge,
            'f_cloud': f_cloud, # 回傳 f_cloud
            'x_peer': x_peer,
            'p_peer': p_peer,
            'x_cloud': x_cloud,
            'p_cloud': p_cloud
        }