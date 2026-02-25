import numpy as np
from config import Config

class CloudEdgeEnvironment:
    def __init__(self, data_loader, logger=None):
        self.task_data, self.ci_hist_raw, self.ci_pred_raw, self.edge_graph = data_loader.load_data()
        self.logger = logger
        
        self.num_edge = Config.NUM_EDGE_SERVERS
        self.time_step = 0
        
        if self.task_data:
            first_key = list(self.task_data.keys())[0]
            self.max_time_steps = len(self.task_data[first_key])
        else:
            self.max_time_steps = 1000 
        
        self.queues_edge = np.zeros(self.num_edge)
        self.queues_cloud = np.zeros(self.num_edge) 
        
        # State placeholders
        self.ci_hist_state = np.zeros((self.num_edge, 2)) 
        self.ci_pred_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_hist_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_pred_state = np.zeros((self.num_edge, 2))

    def reset(self):
        self.time_step = 0
        # Reset Queues: 0 MB Initial
        self.queues_edge.fill(0.0) 
        self.queues_cloud.fill(0)
        
        return self._get_state()

    def _get_state(self):
        t = self.time_step % self.max_time_steps
        
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        arrival_sizes = np.zeros(self.num_edge)
        
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            
            # Change to Predict State
            if edge_name in self.ci_pred_raw:
                ci_edge[i] = self.ci_pred_raw[edge_name][t]
            else:
                raise ValueError(f"Missing edge_name {edge_name}")
                
            cloud_name = f"Cloud Server {i+1}"
            if cloud_name in self.ci_pred_raw: 
                ci_cloud[i] = self.ci_pred_raw[cloud_name][t]
            else:
                raise ValueError(f"Missing cloud_name {cloud_name}")
                
            if edge_name in self.task_data:
                arrival_sizes[i] = self.task_data[edge_name][t]
        
        return {
            'Q_edge': self.queues_edge.copy(),
            'Q_cloud': self.queues_cloud.copy(),
            'CI_edge': ci_edge,
            'CI_cloud': ci_cloud,
            'Arrival': arrival_sizes,
            'Graph': self.edge_graph
        }

    def compute_transmission_rate(self, p_tx, is_cloud=False):
        """
        Shannon Capacity: R (bits/sec) = W (Hz) * log2(1 + S (W) /N (W))
                          S = g (no unit) * p (W)
        """
        gain = Config.G_IC if is_cloud else Config.G_IJ
        SNR = 0 if Config.NOISE_POWER <= 0 else (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + SNR)
        return rate
    
    def compute_energy_carbon(self, factor, ci, t, is_cmp=False):
        energy = ((factor ** 3 * Config.ZETA) if is_cmp else factor) * t
        carbon = ci * energy * Config.CONST_JOULE_TO_KWH
        return energy, carbon

    def step(self, decisions):
        """
        Executes one time slot with Concurrent Transmission and Computation logic.
        1. Transmit to Cloud/Peers. Maximum available time is T_slot.
        2. Compute locally concurrently. Maximum available time is also T_slot.
        3. Update Queue and Carbon.
        """
        
        # 1. Extract Decisions
        f_edge = decisions['f_edge'].copy()
        x_peer = decisions['x_peer'].copy()
        p_peer = decisions['p_peer'].copy()
        x_cloud = decisions['x_cloud'].copy()
        p_cloud = decisions['p_cloud'].copy() 
        f_cloud = decisions.get('f_cloud', np.zeros(self.num_edge)).copy()

        if self.logger:
            self.logger.log_algorithm_decisions(self.time_step % self.max_time_steps, decisions)

        # 2. Get Environment State
        state = self._get_state()
        t_actual = self.time_step % self.max_time_steps

        # 2-1. Get CI histroy data, Use real data, not alpha smooth one
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        for i in range(self.num_edge):
            ci_edge[i] = self.ci_hist_raw[f"Edge Server {i+1}"][t_actual]
            ci_cloud[i] = self.ci_hist_raw[f"Cloud Server {i+1}"][t_actual]
        
        arrival = state['Arrival']
        
        # Initialize Metrics
        m_edge_proc_local = np.zeros(self.num_edge)
        m_edge_tx_peer = np.zeros(self.num_edge)
        m_edge_tx_cloud = np.zeros(self.num_edge)
        m_edge_energy_comp = np.zeros(self.num_edge)
        m_edge_energy_tx = np.zeros(self.num_edge)
        m_edge_carbon = np.zeros(self.num_edge)
        m_edge_q_pre = self.queues_edge.copy()
        
        m_cloud_proc = np.zeros(self.num_edge)
        m_cloud_rx_edge = np.zeros(self.num_edge)
        m_cloud_energy_comp = np.zeros(self.num_edge)
        m_cloud_carbon = np.zeros(self.num_edge)
        m_cloud_q_pre = self.queues_cloud.copy()

        total_carbon = 0.0
        
        # Actual executed values for queue update
        actual_x_peer = np.zeros_like(x_peer)
        actual_x_cloud = np.zeros_like(x_cloud)
        actual_ec = np.zeros(self.num_edge)

        for i in range(self.num_edge):
            # =================================================================
            # Phase 1: Edge Server - Serial Transmission First
            # =================================================================
            # 1-A. Validate & Limit by Current Queue
            current_q = self.queues_edge[i]
            req_x_peer_sum = np.sum(x_peer[i])
            req_x_cloud = x_cloud[i]
            total_req_offload = req_x_peer_sum + req_x_cloud
            
            # ************ TODO START *********************** 
            # Scale down if request > queue
            scale_factor_q = 1.0
            if total_req_offload > current_q:
                scale_factor_q = current_q / (total_req_offload + 1e-9)
            
            valid_x_peer = x_peer[i] * scale_factor_q
            valid_x_cloud = x_cloud[i] * scale_factor_q
            # ************ TODO END   ***********************

            # 1-B. Calculate Transmission Time & Check Time Constraint
            t_needed_peer = np.zeros(self.num_edge)
            t_needed_cloud = 0.0
            
            # Peer Transmissions
            for j in range(self.num_edge):
                if i != j and valid_x_peer[j] > 1e-9 and p_peer[i, j] > 1e-9:
                    rate = self.compute_transmission_rate(p_peer[i, j], is_cloud=False)
                    if rate > 1e-9:
                        t_needed_peer[j] = valid_x_peer[j] / rate
                    else:
                        valid_x_peer[j] = 0 # Cannot transmit with 0 rate
            
            # Cloud Transmission
            if valid_x_cloud > 1e-9 and p_cloud[i] > 1e-9:
                rate_c = self.compute_transmission_rate(p_cloud[i], is_cloud=True)
                if rate_c > 1e-9:
                    t_needed_cloud = valid_x_cloud / rate_c
                else:
                    valid_x_cloud = 0

            # Check Time Overflow
            total_tx_time = np.sum(t_needed_peer) + t_needed_cloud
            
            # How to change to fix transmite size?
            # ************ TODO START *********************** 
            scale_factor_time = 1.0
            if total_tx_time > Config.TIME_SLOT_DURATION:
                scale_factor_time = Config.TIME_SLOT_DURATION / (total_tx_time + 1e-9)
            
            # 1-C. Finalize Transmission Values
            # Apply time scaling
            final_x_peer = valid_x_peer * scale_factor_time
            final_x_cloud = valid_x_cloud * scale_factor_time
            final_t_peer = t_needed_peer * scale_factor_time
            final_t_cloud = t_needed_cloud * scale_factor_time
            # ************ TODO END   *********************** 
            
            # Store Actuals
            actual_x_peer[i] = final_x_peer
            actual_x_cloud[i] = final_x_cloud
            m_edge_tx_peer[i] = np.sum(final_x_peer)
            m_edge_tx_cloud[i] = final_x_cloud
            
            # 1-D. Calculate Transmission Energy/Carbon
            # Peer Part
            for j in range(self.num_edge):
                if final_t_peer[j] > 0:
                    e_j, carb = self.compute_energy_carbon(p_peer[i, j], ci_edge[i], final_t_peer[j])
                    m_edge_energy_tx[i] += e_j
                    m_edge_carbon[i] += carb
                    total_carbon += carb
            
            # Cloud Part
            if final_t_cloud > 0:
                e_c, carb = self.compute_energy_carbon(p_cloud[i], ci_edge[i], final_t_cloud)
                m_edge_energy_tx[i] += e_c
                m_edge_carbon[i] += carb
                total_carbon += carb

            # =================================================================
            # Phase 2: Edge Server - Computation
            # =================================================================
            t_cmp = Config.TIME_SLOT_DURATION
            
            # Compute Capacity: EC_i = f * T_cmp / phi
            ec_capacity = (f_edge[i] * t_cmp) / Config.PHI
            
            # Available data in queue (after transmission)
            q_remaining = max(0, current_q - np.sum(final_x_peer) - final_x_cloud)
            
            # Actual Processed
            processed_bits = min(q_remaining, ec_capacity)
            actual_ec[i] = processed_bits
            m_edge_proc_local[i] = processed_bits
            
            # Computation Energy
            # Formula: U_i(t) := k_i * xi * f^3 * T_i^cmp
            if f_edge[i] > 0 and t_cmp > 0:
                t_cmp = min(t_cmp, m_edge_proc_local[i] / (f_edge[i] / Config.PHI))
                e_cmp, carb = self.compute_energy_carbon(f_edge[i], ci_edge[i], t_cmp, True)
                m_edge_energy_comp[i] += e_cmp
                m_edge_carbon[i] += carb
                total_carbon += carb
        
        # =================================================================
        # Phase 3: Cloud Server Computation
        # =================================================================
        for i in range(self.num_edge):
            # Compute Capacity: EC_i = f * T_cmp / phi
            capacity_cloud = (f_cloud[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            processed_cloud = min(self.queues_cloud[i], capacity_cloud)
            m_cloud_proc[i] = processed_cloud
            
            # Energy (only computation)
            if f_cloud[i] > 0:
                t_active_cloud = processed_cloud / (f_cloud[i] / Config.PHI)
                e_c, carb = self.compute_energy_carbon(f_cloud[i], ci_cloud[i], t_active_cloud, True)
                m_cloud_energy_comp[i] += e_c
                m_cloud_carbon[i] += carb
                total_carbon += carb
            
            # Store Cloud Receive Metrics (from Edge Phase)
            m_cloud_rx_edge[i] = actual_x_cloud[i]

        # =================================================================
        # Phase 4: Queue Updates
        # =================================================================
        # Q_i(t+1) = [Q_i(t) - EC_i(t) - x_cloud - sum(x_peer)]^+ + A_i(t) + sum(x_peer_in)
        
        # Calculate Peer Inflow (sum of column j)
        peer_inflow = np.sum(actual_x_peer, axis=0)
        
        # Update Edge Queues
        term_removal = actual_ec + actual_x_cloud + np.sum(actual_x_peer, axis=1)
        self.queues_edge = np.maximum(0, self.queues_edge - term_removal)
        
        # Add Arrivals and Peer Inflow
        self.queues_edge += arrival + peer_inflow
        
        # Update Cloud Queues
        # Q_cloud(t+1) = Q_cloud(t) - Processed + Inflow(from Edge)
        self.queues_cloud = np.maximum(0, self.queues_cloud - m_cloud_proc)
        self.queues_cloud += actual_x_cloud
        
        # =================================================================
        # Logging & Info
        # =================================================================
        self.time_step += 1
        
        info = {
            'carbon': total_carbon,
            'q_avg_total': np.mean(self.queues_edge),
            'processed_local': np.mean(actual_ec),
            'processed_cloud': np.mean(m_cloud_proc),
            'offloaded_cloud': np.mean(actual_x_cloud)
        }
        
        if self.logger:
            edge_metrics = []
            for i in range(self.num_edge):
                edge_metrics.append({
                    'arrival': arrival[i],
                    'q_pre': m_edge_q_pre[i],
                    'q_post': self.queues_edge[i],
                    'proc_local': actual_ec[i],
                    'tx_peer': m_edge_tx_peer[i],
                    'tx_cloud': m_edge_tx_cloud[i],
                    'pow_peer': np.sum(p_peer[i]),
                    'pow_cloud': p_cloud[i],
                    'energy_comp': m_edge_energy_comp[i],
                    'energy_tx': m_edge_energy_tx[i],
                    'carbon': m_edge_carbon[i]
                })
            
            cloud_metrics = []
            for i in range(self.num_edge):
                cloud_metrics.append({
                    'q_pre': m_cloud_q_pre[i],
                    'q_post': self.queues_cloud[i],
                    'proc': m_cloud_proc[i],
                    'rx_edge': m_cloud_rx_edge[i],
                    'energy_comp': m_cloud_energy_comp[i],
                    'carbon': m_cloud_carbon[i]
                })
                
            metrics = {
                'time_step': self.time_step - 1,
                'edge_metrics': edge_metrics,
                'cloud_metrics': cloud_metrics,
                'global_metrics': {
                    'total_carbon': total_carbon,
                    'avg_q': np.mean(self.queues_edge)
                }
            }
            self.logger.log_step(metrics)
        
        done = self.time_step >= self.max_time_steps
        
        return self._get_state(), total_carbon, done, info