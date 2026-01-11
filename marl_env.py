import numpy as np
from config import Config

class CloudEdgeEnvironment:
    def __init__(self, data_loader):
        # 載入資料
        self.task_data, self.ci_hist_raw, self.ci_pred_raw, self.edge_graph = data_loader.load_data()
        
        self.num_edge = Config.NUM_EDGE_SERVERS
        self.time_step = 0
        
        if self.task_data:
            first_key = list(self.task_data.keys())[0]
            self.max_time_steps = len(self.task_data[first_key])
        else:
            self.max_time_steps = 1000 
        
        # State Variables
        self.queues_edge = np.zeros(self.num_edge)
        self.queues_cloud = np.zeros(self.num_edge) 
        
        # Carbon Intensity State (History & Prediction)
        # 用來模擬 DCWA.py 的 Alpha Smoothing 狀態
        # [current, previous]
        self.ci_hist_state = np.zeros((self.num_edge, 2)) 
        self.ci_pred_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_hist_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_pred_state = np.zeros((self.num_edge, 2))

    def reset(self):
        self.time_step = 0
        self.queues_edge.fill(40000000) # Preload tasks similar to DCWA logs
        self.queues_cloud.fill(0)
        
        # Init CI state
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            if edge_name in self.ci_hist_raw:
                val = self.ci_hist_raw[edge_name][0]
                self.ci_hist_state[i] = [val, val]
                val_pred = self.ci_pred_raw[edge_name][0]
                self.ci_pred_state[i] = [val_pred, val_pred]
        
        return self._get_state()

    def _get_state(self):
        # 1. Update CI (Alpha Smoothing logic simulated from DCWA)
        t = self.time_step % self.max_time_steps
        
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        
        arrival_sizes = np.zeros(self.num_edge)
        
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            
            # CI
            if edge_name in self.ci_hist_raw:
                ci_edge[i] = self.ci_hist_raw[edge_name][t]
            else:
                ci_edge[i] = 0.5
                
            cloud_name = f"Cloud Server {i+1}"
            if cloud_name in self.ci_hist_raw: 
                ci_cloud[i] = self.ci_hist_raw[cloud_name][t]
            else:
                ci_cloud[i] = ci_edge[i] # Fallback
                
            # Task Arrival
            if edge_name in self.task_data:
                arrival_sizes[i] = self.task_data[edge_name][t]
        
        # Construct State
        return {
            'Q_edge': self.queues_edge.copy(),
            'Q_cloud': self.queues_cloud.copy(),
            'CI_edge': ci_edge,
            'CI_cloud': ci_cloud,
            'Arrival': arrival_sizes,
            'Graph': self.edge_graph
        }

    def compute_transmission_rate(self, p_tx, is_cloud=False):
        # Shannon Formula: R = B * log2(1 + P*h / N0)
        gain = Config.G_IC if is_cloud else Config.G_IJ
        snr = (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        return rate

    def step(self, decisions):
        """
        執行一步模擬
        decisions: 包含 f_edge, x_peer, p_peer, x_cloud, p_cloud
        """
        f_edge = decisions['f_edge']
        x_peer = decisions['x_peer'] # 這是 Agent "想要" 傳輸的量
        p_peer = decisions['p_peer']
        x_cloud = decisions['x_cloud'] # 這是 Agent "想要" 傳輸的量
        p_cloud = decisions['p_cloud'] 
        f_cloud = decisions.get('f_cloud', np.zeros(self.num_edge))

        state = self._get_state()
        ci_edge = state['CI_edge']
        ci_cloud = state['CI_cloud']
        arrival = state['Arrival']
        
        total_carbon = 0.0
        
        # --- 1. Process Local Tasks ---
        capacity_local = (f_edge / Config.PHI) * Config.TIME_SLOT_DURATION
        bits_processed_local = np.minimum(self.queues_edge, capacity_local)
        
        # --- 2. Calculate Carbon Emission (Local) ---
        for i in range(self.num_edge):
            # [FIXED] 改回使用 ZETA，避免 1e27 的數值爆炸
            # 雖然 DCWA.py 寫 bits，但其實際 Log 數值與 ZETA 公式較吻合 (1425 vs 546)
            e_local = ci_edge[i] * (f_edge[i]**3) * Config.ZETA * Config.CONST_EMISSION_COMPUTATION
            total_carbon += e_local

        # --- 3. Cloud Processing (Remote) ---
        capacity_cloud = (f_cloud / Config.PHI) * Config.TIME_SLOT_DURATION 
        bits_processed_cloud = np.minimum(self.queues_cloud, capacity_cloud)
        
        for i in range(self.num_edge):
            e_cloud = ci_cloud[i] * (f_cloud[i]**3) * Config.ZETA * Config.CONST_EMISSION_COMPUTATION
            total_carbon += e_cloud

        # --- 4. Transmission Physics Check (CRITICAL) ---
        # 必須計算物理上的最大傳輸量 (Rate * Time)，不能讓 Agent 隨意傳輸無限大
        
        # A. Peer Offloading Limit
        real_x_peer = np.zeros_like(x_peer)
        for i in range(self.num_edge):
            for j in range(self.num_edge):
                if i == j or x_peer[i, j] <= 0: continue
                
                # 計算該鏈路的物理極限速率
                if p_peer[i, j] > 0:
                    rate = self.compute_transmission_rate(p_peer[i, j], is_cloud=False)
                    max_bits = rate * Config.TIME_SLOT_DURATION
                    # 實際傳輸 = min(想傳的, 物理極限)
                    real_x_peer[i, j] = min(x_peer[i, j], max_bits)
                    
                    # Carbon (Tx)
                    e_tx = ci_edge[i] * p_peer[i, j] * Config.TIME_SLOT_DURATION * Config.CONST_EMISSION_TRANSMISSION
                    total_carbon += e_tx
                else:
                    real_x_peer[i, j] = 0

        # B. Cloud Offloading Limit
        real_x_cloud = np.zeros_like(x_cloud)
        for i in range(self.num_edge):
            if x_cloud[i] > 0 and p_cloud[i] > 0:
                rate = self.compute_transmission_rate(p_cloud[i], is_cloud=True)
                max_bits = rate * Config.TIME_SLOT_DURATION
                real_x_cloud[i] = min(x_cloud[i], max_bits)
                
                # Carbon (Tx)
                e_tx = ci_edge[i] * p_cloud[i] * Config.TIME_SLOT_DURATION * Config.CONST_EMISSION_TRANSMISSION
                total_carbon += e_tx
            else:
                real_x_cloud[i] = 0
        
        # --- 5. Queue Update ---
        # 1. 扣除本地處理
        self.queues_edge -= bits_processed_local
        
        # 2. 執行卸載 (Out)
        # 檢查：總卸載量不能超過剩餘 Queue
        total_out_req = np.sum(real_x_peer, axis=1) + real_x_cloud
        actual_out_ratio = np.ones(self.num_edge)
        mask = total_out_req > self.queues_edge
        actual_out_ratio[mask] = self.queues_edge[mask] / (total_out_req[mask] + 1e-9)
        
        final_x_peer = real_x_peer * actual_out_ratio[:, np.newaxis]
        final_x_cloud = real_x_cloud * actual_out_ratio
        
        self.queues_edge -= (np.sum(final_x_peer, axis=1) + final_x_cloud)
        
        # 3. 接收卸載 (In) 與 新任務 (Arrival)
        peer_in = np.sum(final_x_peer, axis=0)
        self.queues_edge += arrival + peer_in
        
        # Cloud Queue Update
        self.queues_cloud -= bits_processed_cloud
        self.queues_cloud = np.maximum(self.queues_cloud, 0)
        self.queues_cloud += final_x_cloud
        
        self.time_step += 1
        
        # Info stats
        info = {
            'carbon': total_carbon,
            'q_avg_total': np.mean(self.queues_edge),
            'processed_local': np.mean(bits_processed_local),
            'processed_cloud': np.mean(bits_processed_cloud),
            'offloaded_cloud': np.mean(final_x_cloud)
        }
        
        done = self.time_step >= self.max_time_steps
        
        return self._get_state(), total_carbon, done, info