import numpy as np
from config import Config

class CloudEdgeEnvironment:
    def __init__(self, data_loader):
        # Load Data
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
        # Used to simulate Alpha Smoothing state from DCWA.py
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
        Execute one simulation step.
        decisions: contains f_edge, x_peer, p_peer, x_cloud, p_cloud
        """
        f_edge = decisions['f_edge']
        x_peer = decisions['x_peer'] # Amount Agent "wants" to transmit
        p_peer = decisions['p_peer']
        x_cloud = decisions['x_cloud'] # Amount Agent "wants" to transmit
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
            # [FIXED] Switched back to ZETA to avoid 1e27 value explosion
            # Although DCWA.py says bits, the actual Log values match ZETA formula (1425 vs 546) better
            e_local = ci_edge[i] * (f_edge[i]**3) * Config.ZETA * Config.CONST_EMISSION_COMPUTATION
            total_carbon += e_local

        # --- 3. Cloud Processing (Remote) ---
        capacity_cloud = (f_cloud / Config.PHI) * Config.TIME_SLOT_DURATION 
        bits_processed_cloud = np.minimum(self.queues_cloud, capacity_cloud)
        
        for i in range(self.num_edge):
            e_cloud = ci_cloud[i] * (f_cloud[i]**3) * Config.ZETA * Config.CONST_EMISSION_COMPUTATION
            total_carbon += e_cloud

        # --- 4. Transmission Physics Check (CRITICAL) ---
        # Must calculate physical max transmission limit (Rate * Time)
        # Cannot allow Agent to transmit infinite amount
        
        # A. Peer Offloading Limit
        real_x_peer = np.zeros_like(x_peer)
        for i in range(self.num_edge):
            for j in range(self.num_edge):
                if i == j or x_peer[i, j] <= 0: continue
                
                # Calculate physical limit rate for this link
                if p_peer[i, j] > 0:
                    rate = self.compute_transmission_rate(p_peer[i, j], is_cloud=False)
                    max_bits = rate * Config.TIME_SLOT_DURATION
                    # Actual Tx = min(Desired, Physical Limit)
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
        # 1. Deduct Local Processing
        self.queues_edge -= bits_processed_local
        
        # 2. Execute Offloading (Out)
        # Check: Total offload cannot exceed remaining Queue
        total_out_req = np.sum(real_x_peer, axis=1) + real_x_cloud
        actual_out_ratio = np.ones(self.num_edge)
        mask = total_out_req > self.queues_edge
        actual_out_ratio[mask] = self.queues_edge[mask] / (total_out_req[mask] + 1e-9)
        
        final_x_peer = real_x_peer * actual_out_ratio[:, np.newaxis]
        final_x_cloud = real_x_cloud * actual_out_ratio
        
        self.queues_edge -= (np.sum(final_x_peer, axis=1) + final_x_cloud)
        
        # 3. Receive Offloading (In) and New Tasks (Arrival)
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