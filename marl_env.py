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
        # Reset Queues: 5 MB Initial
        self.queues_edge.fill(5.0 * Config.MB_TO_BITS) 
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
        gain = Config.G_IC if is_cloud else Config.G_IJ
        # Shannon Capacity: W * log2(1 + S/N)
        # S = g (no unit) * p (W)
        # R (bits/sec) = W (Hz) * log2(1 + S (W) /N (W))
        snr = 0 if Config.NOISE_POWER <= 0 else (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        return rate

    def step(self, decisions):
        """
        Executes one time slot with the Serial Transmission -> Computation logic.
        Ref: 
        1. Transmit to Cloud/Peer (Serial). T_off = x / R.
        2. Compute with remaining time. T_cmp = T_slot - sum(T_off).
        3. Update Queue and Carbon.
        """
        
        # 1. Extract Decisions
        f_edge = decisions['f_edge'].copy()
        x_peer = decisions['x_peer'].copy()
        p_peer = decisions['p_peer'].copy()
        x_cloud = decisions['x_cloud'].copy()
        p_cloud = decisions['p_cloud'].copy() 
        f_cloud = decisions.get('f_cloud', np.zeros(self.num_edge)).copy()

        # 2. Get Environment State
        state = self._get_state()
        # ci_edge = state['CI_edge']
        # ci_cloud = state['CI_cloud']
        # Environment need to use history data
        t_actual = self.time_step % self.max_time_steps
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
        
        # Temporary arrays to hold actual executed values for queue update
        actual_x_peer = np.zeros_like(x_peer)
        actual_x_cloud = np.zeros_like(x_cloud)
        actual_ec = np.zeros(self.num_edge)

        # =================================================================
        # Phase 1: Edge Server - Serial Transmission First
        # =================================================================
        for i in range(self.num_edge):
            # --- A. Validate & Limit by Current Queue ---
            # You cannot transmit more than you have. 
            # Logic: Priority to Offloading (as per "First transmit...")
            current_q = self.queues_edge[i]
            
            # Calculate total requested offload
            req_x_peer_sum = np.sum(x_peer[i])
            req_x_cloud = x_cloud[i]
            total_req_offload = req_x_peer_sum + req_x_cloud
            
            # Scale down if request > queue
            scale_factor_q = 1.0
            if total_req_offload > current_q:
                scale_factor_q = current_q / (total_req_offload + 1e-9)
            
            # Apply Queue Constraint
            valid_x_peer = x_peer[i] * scale_factor_q
            valid_x_cloud = x_cloud[i] * scale_factor_q
            
            # --- B. Calculate Transmission Time & Check Time Constraint ---
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
            
            scale_factor_time = 1.0
            if total_tx_time > Config.TIME_SLOT_DURATION:
                scale_factor_time = Config.TIME_SLOT_DURATION / (total_tx_time + 1e-9)
            
            # --- C. Finalize Transmission Values ---
            # Apply time scaling
            final_x_peer = valid_x_peer * scale_factor_time
            final_x_cloud = valid_x_cloud * scale_factor_time
            final_t_peer = t_needed_peer * scale_factor_time
            final_t_cloud = t_needed_cloud * scale_factor_time
            
            # Store Actuals
            actual_x_peer[i] = final_x_peer
            actual_x_cloud[i] = final_x_cloud
            m_edge_tx_peer[i] = np.sum(final_x_peer)
            m_edge_tx_cloud[i] = final_x_cloud
            
            # --- D. Calculate Transmission Energy/Carbon ---
            # U_off = k * p * T_off
            # Peer Energy
            for j in range(self.num_edge):
                if final_t_peer[j] > 0:
                    # gCO2 = k (g/kWh) * (kWh/J = 1 / (3600 * 1000) = 2.778e-7) *  W (J/sec) * sec
                    e_j = p_peer[i, j] * final_t_peer[j]
                    m_edge_energy_tx[i] += e_j
                    carb = ci_edge[i] * e_j * Config.CONST_JOULE_TO_KWH
                    m_edge_carbon[i] += carb
                    total_carbon += carb
            
            # Cloud Energy
            if final_t_cloud > 0:
                # Same as peer
                e_c = p_cloud[i] * final_t_cloud
                m_edge_energy_tx[i] += e_c
                carb = ci_edge[i] * e_c * Config.CONST_JOULE_TO_KWH
                m_edge_carbon[i] += carb
                total_carbon += carb

            # =================================================================
            # Phase 2: Edge Server - Computation (Remaining Time)
            # =================================================================
            # Time for computation
            total_tx_time_final = np.sum(final_t_peer) + final_t_cloud
            t_cmp = max(0, Config.TIME_SLOT_DURATION - total_tx_time_final)
            
            # Compute Capacity: EC_i = f * T_cmp / phi
            ec_capacity = (f_edge[i] * t_cmp) / Config.PHI
            
            # Available data in queue (after transmission)
            # Note: We already scaled x to fit in Q, so remaining >= 0
            q_remaining = max(0, current_q - np.sum(final_x_peer) - final_x_cloud)
            
            # Actual Processed
            processed_bits = min(q_remaining, ec_capacity)
            actual_ec[i] = processed_bits
            m_edge_proc_local[i] = processed_bits
            
            # Computation Energy
            # Formula: U_i(t) := k_i * xi * f^3 * T_i^cmp
            # NOTE: This formula implies energy is spent for the allocated time window T_cmp,
            # regardless of whether the queue empties early. This is common in time-slotted models
            # where the server is "on" for that duration.
            if f_edge[i] > 0 and t_cmp > 0:
                # gCO2 = k (g/kWh) * zeta * f^3 (sec)^3 * sec
                t_cmp = min(t_cmp, m_edge_proc_local[i] / (f_edge[i] / Config.PHI))
                e_cmp = (f_edge[i]**3) * Config.ZETA * t_cmp
                m_edge_energy_comp[i] += e_cmp
                carb = ci_edge[i] * e_cmp * Config.CONST_JOULE_TO_KWH
                m_edge_carbon[i] += carb
                total_carbon += carb
        
        # =================================================================
        # Phase 3: Cloud Server Computation
        # =================================================================
        # Cloud uses full time slot
        for i in range(self.num_edge):
            # Cloud processes its own queue
            capacity_cloud = (f_cloud[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            processed_cloud = min(self.queues_cloud[i], capacity_cloud)
            
            m_cloud_proc[i] = processed_cloud
            
            # Energy (Cloud)
            if f_cloud[i] > 0:
                # Assuming Cloud runs for the time needed or full slot? 
                # Usually cloud models are simpler. Let's assume proportional to work 
                # OR full slot if consistent with Edge.
                # Let's use proportional time for Cloud to be fair, or full slot if specified.
                # User says: "Total time length 1 Time slot".
                # So we calculate energy based on full slot if f > 0?
                # Let's assume standard model: Energy = Power * Time.
                # Time = bits / Rate.
                # Same as edge computation
                t_active_cloud = processed_cloud / (f_cloud[i] / Config.PHI) if f_cloud[i] > 0 else 0
                e_c = (f_cloud[i]**3) * Config.ZETA * t_active_cloud
                
                m_cloud_energy_comp[i] += e_c
                carb = ci_cloud[i] * e_c * Config.CONST_JOULE_TO_KWH
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
        # We perform the subtraction using the actuals we calculated
        # Since we enforced (EC + x + sum_x) <= Q_pre implies the [ ]^+ term is 0 if empty
        # But we use max(0, ...) just to be safe against float errors
        
        # Note: actual_ec was limited by q_remaining.
        # q_remaining was Q - x_peer - x_cloud.
        # So Q - x_peer - x_cloud - actual_ec >= 0.
        
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