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

        # neighbors map
        self.neighbors_map = {}
        for i in range(self.num_edge):
            server_name = f"Edge Server {i+1}"
            neighbors = self.edge_graph.get(server_name, [])
            n_indices = []
            for n in neighbors:
                try: n_indices.append(int(n.split()[-1]) - 1)
                except: pass
            self.neighbors_map[i] = n_indices

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
            'Graph': self.neighbors_map
        }

    def compute_transmission_rate(self, p_tx, is_cloud=False):
        """
        Shannon Capacity: R (bits/sec) = W (Hz) * log2(1 + S (W) /N (W))
                          S = g (no unit) * p (W)
        """
        gain = Config.G_IC if is_cloud else Config.G_IJ
        snr = 0 if Config.NOISE_POWER <= 0 else (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        return rate
    
    def compute_energy_carbon(self, factor, ci, t, is_cmp=False):
        energy = ((factor ** 3 * Config.ZETA) if is_cmp else factor) * t
        carbon = ci * energy * Config.CONST_JOULE_TO_KWH
        return energy, carbon

    def _adjust_scale_factor(self, i, current_q, x_peer, x_cloud, p_peer, p_cloud):
        req_x_peer_sum = np.sum(x_peer)
        req_x_cloud = x_cloud
        total_req_offload = req_x_peer_sum + req_x_cloud
        
        # First scale down for check queue size
        scale_factor_q = 1.0
        if total_req_offload > current_q:
            scale_factor_q = current_q / (total_req_offload + 1e-9)
        valid_x_peer = x_peer * scale_factor_q
        valid_x_cloud = x_cloud * scale_factor_q

        t_needed_peer = np.zeros(self.num_edge)
        t_needed_cloud = 0.0
        
        # Compute Require time
        for j in range(self.num_edge):
            if i != j and valid_x_peer[j] > 1e-9 and p_peer[j] > 1e-9:
                rate = self.compute_transmission_rate(p_peer[j], is_cloud=False)
                if rate > 1e-9:
                    t_needed_peer[j] = valid_x_peer[j] / rate
                else:
                    valid_x_peer[j] = 0 # Cannot transmit with 0 rate
        
        # Cloud Transmission
        if valid_x_cloud > 1e-9 and p_cloud > 1e-9:
            rate = self.compute_transmission_rate(p_cloud, is_cloud=True)
            if rate > 1e-9:
                t_needed_cloud = valid_x_cloud / rate
            else:
                valid_x_cloud = 0

        # Second Scale down by time slot
        total_tx_time = np.sum(t_needed_peer) + t_needed_cloud
        
        scale_factor_time = 1.0
        if total_tx_time > Config.TIME_SLOT_DURATION:
            scale_factor_time = Config.TIME_SLOT_DURATION / (total_tx_time + 1e-9)
        
        final_x_peer = valid_x_peer * scale_factor_time
        final_x_cloud = valid_x_cloud * scale_factor_time
        final_t_peer = t_needed_peer * scale_factor_time
        final_t_cloud = t_needed_cloud * scale_factor_time

        return final_x_peer, final_x_cloud, final_t_peer, final_t_cloud
    
    def _adjust_fix_time_slot(self, i, current_q, x_peer, x_cloud, p_peer, p_cloud):
        slot_time = Config.TIME_SLOT_DURATION / self.num_edge

        final_t_peer = np.zeros(self.num_edge)
        final_t_cloud = 0.0
        
        valid_x_peer = np.zeros(self.num_edge)
        valid_x_cloud = 0.0

        for j in range(self.num_edge):
            if i != j:
                if x_peer[j] > 1e-9 and p_peer[j] > 1e-9:
                    rate = self.compute_transmission_rate(p_peer[j], is_cloud=False)
                    if rate > 1e-9:
                        max_x = rate * slot_time
                        valid_x_peer[j] = min(x_peer[j], max_x)
                        final_t_peer[j] = slot_time 
                    else:
                        valid_x_peer[j] = 0.0
                        final_t_peer[j] = 0.0 
                else:
                    valid_x_peer[j] = 0.0
                    final_t_peer[j] = 0.0

        if x_cloud > 1e-9 and p_cloud > 1e-9:
            rate = self.compute_transmission_rate(p_cloud, is_cloud=True)
            if rate > 1e-9:
                max_x_c = rate * slot_time
                valid_x_cloud = min(x_cloud, max_x_c)
                final_t_cloud = slot_time
            else:
                valid_x_cloud = 0.0
                final_t_cloud = 0.0
        else:
            valid_x_cloud = 0.0
            final_t_cloud = 0.0
        
        total_req_offload = np.sum(valid_x_peer) + valid_x_cloud
        
        if total_req_offload > current_q:
            scale_factor_q = current_q / (total_req_offload + 1e-9)
            valid_x_peer *= scale_factor_q
            valid_x_cloud *= scale_factor_q
        
        final_x_peer = valid_x_peer
        final_x_cloud = valid_x_cloud

        return final_x_peer, final_x_cloud, final_t_peer, final_t_cloud

    def adjust_transmission_power_time(self, i, current_q, x_peer, x_cloud, p_peer, p_cloud):
        if Config.TIME_SLOT_ADJUST == 'scale':
            return self._adjust_scale_factor(i, current_q, x_peer, x_cloud, p_peer, p_cloud)
        elif Config.TIME_SLOT_ADJUST == 'fix_time_slot':
            return self._adjust_fix_time_slot(i, current_q, x_peer, x_cloud, p_peer, p_cloud)
        else:
            raise ValueError(f"Unknown Config.TIME_SLOT_ADJUST: {Config.TIME_SLOT_ADJUST}")

    def step(self, decisions):
        """
        Executes one time slot with Concurrent Transmission and Computation logic.
        1. Transmit to Cloud/Peers. Maximum available time is T_slot.
        2. Compute locally concurrently. Maximum available time is also T_slot.
        3. Update Queue and Carbon.
        """
        
        # Extract Decisions
        f_edge = decisions['f_edge'].copy()
        x_peer = decisions['x_peer'].copy()
        p_peer = decisions['p_peer'].copy()
        x_cloud = decisions['x_cloud'].copy()
        p_cloud = decisions['p_cloud'].copy() 
        f_cloud = decisions.get('f_cloud', np.zeros(self.num_edge)).copy()

        if self.logger:
            self.logger.log_algorithm_decisions(self.time_step % self.max_time_steps, decisions)

        # Get Environment State
        state = self._get_state()
        t_actual = self.time_step % self.max_time_steps

        # Use real data, not alpha smooth one
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        for i in range(self.num_edge):
            ci_edge[i] = self.ci_hist_raw[f"Edge Server {i+1}"][t_actual]
            ci_cloud[i] = self.ci_hist_raw[f"Cloud Server {i+1}"][t_actual]
        
        arrival = state['Arrival']
        
        # Initialize Metrics
        edge_proc_local = np.zeros(self.num_edge)
        edge_tx_peer = np.zeros(self.num_edge)
        edge_tx_cloud = np.zeros(self.num_edge)
        edge_energy_comp = np.zeros(self.num_edge)
        edge_energy_tx = np.zeros(self.num_edge)
        edge_carbon = np.zeros(self.num_edge)
        edge_q_pre = self.queues_edge.copy()

        cloud_proc = np.zeros(self.num_edge)
        cloud_rx_edge = np.zeros(self.num_edge)
        cloud_energy_comp = np.zeros(self.num_edge)
        cloud_carbon = np.zeros(self.num_edge)
        cloud_q_pre = self.queues_cloud.copy()

        total_carbon = 0.0
        
        # Actual executed values for queue update
        actual_x_peer = np.zeros_like(x_peer)
        actual_x_cloud = np.zeros_like(x_cloud)
        actual_ec = np.zeros(self.num_edge)

        for i in range(self.num_edge):
            # =================================================================
            # Phase 1: Edge Server - Serial Transmission First
            # =================================================================
            # Adjust Transmission Power and Transmission Time
            current_q = self.queues_edge[i]
            final_x_peer, final_x_cloud, final_t_peer, final_t_cloud = self.adjust_transmission_power_time(
                i, current_q, x_peer[i], x_cloud[i], p_peer[i], p_cloud[i]
            )
            
            actual_x_peer[i] = final_x_peer
            actual_x_cloud[i] = final_x_cloud
            edge_tx_peer[i] = np.sum(final_x_peer)
            edge_tx_cloud[i] = final_x_cloud
            
            # Calculate Transmission Energy/Carbon
            for j in range(self.num_edge):
                if final_t_peer[j] > 0:
                    energy_peer, carbon = self.compute_energy_carbon(p_peer[i, j], ci_edge[i], final_t_peer[j])
                    edge_energy_tx[i] += energy_peer
                    edge_carbon[i] += carbon
                    total_carbon += carbon
            
            if final_t_cloud > 0:
                energy_cloud, carbon = self.compute_energy_carbon(p_cloud[i], ci_edge[i], final_t_cloud)
                edge_energy_tx[i] += energy_cloud
                edge_carbon[i] += carbon
                total_carbon += carbon

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
            edge_proc_local[i] = processed_bits
            
            # Computation Energy
            # Formula: U_i(t) := k_i * xi * f^3 * T_i^cmp
            if f_edge[i] > 0 and t_cmp > 0:
                t_cmp = min(t_cmp,edge_proc_local[i] / (f_edge[i] / Config.PHI))
                energy_compute, carbon = self.compute_energy_carbon(f_edge[i], ci_edge[i], t_cmp, True)
                edge_energy_comp[i] += energy_compute
                edge_carbon[i] += carbon
                total_carbon += carbon
        
        # =================================================================
        # Phase 3: Cloud Server Computation
        # =================================================================
        for i in range(self.num_edge):
            # Compute Capacity: EC_i = f * T_cmp / phi
            capacity_cloud = (f_cloud[i] / Config.PHI) * Config.TIME_SLOT_DURATION
            processed_cloud = min(self.queues_cloud[i], capacity_cloud)
            cloud_proc[i] = processed_cloud
            
            # Energy (only computation)
            if f_cloud[i] > 0:
                t_active_cloud = processed_cloud / (f_cloud[i] / Config.PHI)
                energy_compute, carbon = self.compute_energy_carbon(f_cloud[i], ci_cloud[i], t_active_cloud, True)
                cloud_energy_comp[i] += energy_compute
                cloud_carbon[i] += carbon
                total_carbon += carbon
            
            # Store Cloud Receive Metrics (from Edge Phase)
            cloud_rx_edge[i] = actual_x_cloud[i]

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
        self.queues_cloud = np.maximum(0, self.queues_cloud -cloud_proc)
        self.queues_cloud += actual_x_cloud
        
        # =================================================================
        # Logging & Info
        # =================================================================
        self.time_step += 1
        
        info = {
            'carbon': total_carbon,
            'q_avg_total': np.mean(self.queues_edge),
            'processed_local': np.mean(actual_ec),
            'processed_cloud': np.mean(cloud_proc),
            'offloaded_cloud': np.mean(actual_x_cloud)
        }
        
        if self.logger:
            edge_metrics = []
            for i in range(self.num_edge):
                edge_metrics.append({
                    'arrival': arrival[i],
                    'q_pre':edge_q_pre[i],
                    'q_post': self.queues_edge[i],
                    'proc_local': actual_ec[i],
                    'tx_peer':edge_tx_peer[i],
                    'tx_cloud':edge_tx_cloud[i],
                    'pow_peer': np.sum(p_peer[i]),
                    'pow_cloud': p_cloud[i],
                    'energy_comp':edge_energy_comp[i],
                    'energy_tx':edge_energy_tx[i],
                    'carbon':edge_carbon[i]
                })
            
            cloud_metrics = []
            for i in range(self.num_edge):
                cloud_metrics.append({
                    'q_pre':cloud_q_pre[i],
                    'q_post': self.queues_cloud[i],
                    'proc':cloud_proc[i],
                    'rx_edge':cloud_rx_edge[i],
                    'energy_comp':cloud_energy_comp[i],
                    'carbon':cloud_carbon[i]
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