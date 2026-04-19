import numpy as np
import random
from config import Config

class CloudEdgeEnvironment:
    def __init__(self, data_loader, warning_log_file=None, logger=None, is_training=False):
        self.task_data, self.ci_hist_raw, self.ci_pred_raw, self.edge_graph = data_loader.load_data()
        self.logger = logger

        self.warning_log_file = warning_log_file
        if self.warning_log_file:
            with open(self.warning_log_file, "w", encoding="utf-8") as f:
                f.write("=== Simulation Constraint Warnings Log ===\n")
        
        self.num_edge = Config.NUM_EDGE_SERVERS
        self.time_step = 0
        self.is_training = is_training
        self.start_offset = 0
        
        if self.ci_hist_raw:
            first_key = list(self.ci_hist_raw.keys())[0]
            self.total_data_length = len(self.ci_hist_raw[first_key]) 
        else:
            self.total_data_length = 1000
        self.episode_length = 1000 
        
        if self.task_data:
            first_key = list(self.task_data.keys())[0]
            self.task_length = len(self.task_data[first_key])
        else:
            self.task_length = 1000
        
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
    
    def _log_warning(self, message):
        if not self.warning_log_file:
            return
        with open(self.warning_log_file, "a", encoding="utf-8") as f:
            f.write(message + "\n")

    def reset(self):
        self.time_step = 0
        if self.is_training:
            max_start = max(0, self.total_data_length - self.episode_length)
            self.start_offset = random.randint(0, max_start) 
        else:
            self.start_offset = 0
        # Reset Queues: 0 MB Initial
        self.queues_edge.fill(0.0) 
        self.queues_cloud.fill(0)
        
        return self._get_state()

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
            self.logger.log_algorithm_decisions(self.time_step, decisions)

        # Get Environment State
        state = self._get_state()
        # t_actual = self.time_step % self.max_time_steps
        t_actual = (self.time_step + self.start_offset) % self.total_data_length

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

        # =================================================================
        # Phase 1: Computation Model - Include Cloud and Edge Server
        # =================================================================

        # Check Freq Constraint
        f_cloud, f_edge = self._check_freq_constraint(f_cloud, f_edge)

        for i in range(self.num_edge):
            # Phase 1-1: Cloud Server
            # Compute Capacity: EC_i = f * T_cmp / phi
            capacity_cloud = (f_cloud[i] * Config.TIME_SLOT_DURATION) / Config.PHI
            processed_cloud = min(self.queues_cloud[i], capacity_cloud)
            cloud_proc[i] = processed_cloud
            
            # Cloud Computation Energy
            if f_cloud[i] > 0:
                t_active_cloud = processed_cloud / (f_cloud[i] / Config.PHI)
                energy_compute, carbon = self._compute_energy_carbon(f_cloud[i], ci_cloud[i], t_active_cloud, True)
                cloud_energy_comp[i] += energy_compute
                cloud_carbon[i] += carbon
                total_carbon += carbon

            # Phase 1-2: Edge Server
            # Compute Capacity: EC_i = f * T_cmp / phi
            ec_capacity = (f_edge[i] * Config.TIME_SLOT_DURATION) / Config.PHI
            processed_bits = min(self.queues_edge[i], ec_capacity)
            actual_ec[i] = processed_bits
            edge_proc_local[i] = processed_bits
            
            # Computation Energy
            # Formula: U_i(t) := k_i * xi * f^3 * T_i^cmp
            if f_edge[i] > 0:
                t_activate_edge = edge_proc_local[i] / (f_edge[i] / Config.PHI)
                energy_compute, carbon = self._compute_energy_carbon(f_edge[i], ci_edge[i], t_activate_edge, True)
                edge_energy_comp[i] += energy_compute
                edge_carbon[i] += carbon
                total_carbon += carbon
        
        # =================================================================
        # Phase 2: Edge Server - Serial Transmission First
        # =================================================================
        current_q = self.queues_edge - actual_ec
        for i in range(self.num_edge):

            # Check offloading constraint
            p_cloud[i], p_peer[i], final_x_cloud, final_x_peer, final_t_cloud, final_t_peer = self._check_offloading_constraint(
                i, p_cloud[i], p_peer[i], x_cloud[i], x_peer[i], current_q[i]
            )

            actual_x_peer[i] = final_x_peer
            actual_x_cloud[i] = final_x_cloud
            edge_tx_peer[i] = np.sum(final_x_peer)
            edge_tx_cloud[i] = final_x_cloud

            # Store Cloud Receive Metrics (from Edge Phase)
            cloud_rx_edge[i] = actual_x_cloud[i]
            
            # Calculate Transmission Energy/Carbon
            for j in range(self.num_edge):
                if final_t_peer[j] > 0:
                    energy_peer, carbon = self._compute_energy_carbon(p_peer[i, j], ci_edge[i], final_t_peer[j])
                    edge_energy_tx[i] += energy_peer
                    edge_carbon[i] += carbon
                    total_carbon += carbon
            
            if final_t_cloud > 0:
                energy_cloud, carbon = self._compute_energy_carbon(p_cloud[i], ci_edge[i], final_t_cloud)
                edge_energy_tx[i] += energy_cloud
                edge_carbon[i] += carbon
                total_carbon += carbon

        # =================================================================
        # Phase 3: Queue Updates
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
            'offloaded_cloud': np.mean(actual_x_cloud),
            'edge_metrics': [{'carbon': edge_carbon[i]} for i in range(self.num_edge)],
            'cloud_metrics': [{'carbon': cloud_carbon[i]} for i in range(self.num_edge)]
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
        
        done = self.time_step >= self.episode_length
        
        return self._get_state(), total_carbon, done, info

    def _get_state(self):
        t_actual = (self.time_step + self.start_offset) % self.total_data_length
        t_task = (self.time_step + self.start_offset) % self.task_length
        
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        arrival_sizes = np.zeros(self.num_edge)
        
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            
            if edge_name in self.ci_pred_raw:
                ci_edge[i] = self.ci_pred_raw[edge_name][t_actual]
            else:
                raise ValueError(f"Missing edge_name {edge_name}")
                
            cloud_name = f"Cloud Server {i+1}"
            if cloud_name in self.ci_pred_raw: 
                ci_cloud[i] = self.ci_pred_raw[cloud_name][t_actual]
            else:
                raise ValueError(f"Missing cloud_name {cloud_name}")
                
            if edge_name in self.task_data:
                arrival_sizes[i] = self.task_data[edge_name][t_task]
        
        return {
            'Q_edge': self.queues_edge.copy(),
            'Q_cloud': self.queues_cloud.copy(),
            'CI_edge': ci_edge,
            'CI_cloud': ci_cloud,
            'Arrival': arrival_sizes,
            'Graph': self.neighbors_map
        }

    def _check_freq_constraint(self, f_cloud, f_edge):
        cloud_exceed_idx = np.where(f_cloud > Config.CLOUD_F_MAX)[0]
        edge_exceed_idx = np.where(f_edge > Config.EDGE_F_MAX)[0]

        for i in cloud_exceed_idx:
            self._log_warning(f"[Warning] Step {self.time_step}:")
            self._log_warning(f"\tCloud Server {i+1} request freq {f_cloud[i]:.4f} exceed MAX {Config.CLOUD_F_MAX}")

        for i in edge_exceed_idx:
            self._log_warning(f"[Warning] Step {self.time_step}:")
            self._log_warning(f"\tEdge Server {i+1} request freq {f_edge[i]:.4f} exceed MAX {Config.EDGE_F_MAX}")
        
        np.clip(f_cloud, a_min=0.0, a_max=Config.CLOUD_F_MAX, out=f_cloud)
        np.clip(f_edge, a_min=0.0, a_max=Config.EDGE_F_MAX, out=f_edge)
        
        return f_cloud, f_edge
    
    def _check_offloading_constraint(self, i, p_cloud, p_peer, x_cloud, x_peer, current_q):
        # 1. Transmission Power Constraint Check
        if p_cloud > Config.EDGE_P_MAX:
            self._log_warning(f"[Warning] Step {self.time_step}:")
            self._log_warning(f"\tEdge Server {i+1} request transmission power to cloud {p_cloud:.4f} exceed MAX {Config.EDGE_P_MAX}")

        power_exceed_idx = np.where(p_peer > Config.EDGE_P_MAX)[0]
        for j in power_exceed_idx:
            self._log_warning(f"[Warning] Step {self.time_step}:")
            self._log_warning(f"\tEdge Server {i+1} request transmission power to Server {j+1} {p_peer[j]:.4f} exceed MAX {Config.EDGE_P_MAX}")
        
        p_cloud = min(p_cloud, Config.EDGE_P_MAX)
        np.clip(p_peer, a_min=0.0, a_max=Config.EDGE_P_MAX, out=p_peer)

        # 2. Transmisson Time Constraint Check
        t_needed_cloud = 0.0
        t_needed_peer = np.zeros(self.num_edge)

        if x_cloud > 1e-9 and p_cloud > 1e-9:
            rate = self._compute_transmission_rate(p_cloud, is_cloud=True)
            if rate > 1e-9:
                t_needed_cloud = x_cloud / rate
        
        for j in range(self.num_edge):
            if i != j and x_peer[j] > 1e-9 and p_peer[j] > 1e-9:
                rate = self._compute_transmission_rate(p_peer[j], is_cloud=False)
                if rate > 1e-9:
                    t_needed_peer[j] = x_peer[j] / rate
        
        if Config.TIME_SLOT_ADJUST == "fix_time_slot":
            MAX_FIX_TIME_SLOT_DURATION = Config.TIME_SLOT_DURATION / self.num_edge + 1e-6
            if t_needed_cloud > MAX_FIX_TIME_SLOT_DURATION:
                self._log_warning(f"[Warning] Step {self.time_step}:")
                self._log_warning(f"\tEdge Server {i+1} request transmission time to cloud {t_needed_cloud:.4f} exceed MAX {MAX_FIX_TIME_SLOT_DURATION}")

            time_exceed_idx = np.where(t_needed_peer > MAX_FIX_TIME_SLOT_DURATION)[0]
            for j in time_exceed_idx:
                self._log_warning(f"[Warning] Step {self.time_step}:")
                self._log_warning(f"\tEdge Server {i+1} request transmission time to Server {j+1} {t_needed_peer[j]:.4f} exceed MAX {MAX_FIX_TIME_SLOT_DURATION}")
            
            t_needed_cloud = min(t_needed_cloud, MAX_FIX_TIME_SLOT_DURATION)
            np.clip(t_needed_peer, a_min=0.0, a_max=MAX_FIX_TIME_SLOT_DURATION, out=t_needed_peer)
        
        elif Config.TIME_SLOT_ADJUST == "scale":
            scale_factor_time = 1.0
            total_tx_time = np.sum(t_needed_peer) + t_needed_cloud
            if total_tx_time > Config.TIME_SLOT_DURATION + 1e-6:
                self._log_warning(f"[Warning] Step {self.time_step}:")
                self._log_warning(f"\tEdge Server {i+1} total offload time ({total_tx_time:.4f}s) exceeds Time Slot ({Config.TIME_SLOT_DURATION}s)")
                scale_factor_time = Config.TIME_SLOT_DURATION / (total_tx_time + 1e-9)
            t_needed_cloud *= scale_factor_time
            t_needed_peer *= scale_factor_time
        
        # 3. check available offloading available size
        final_x_cloud = 0.0
        final_x_peer = np.zeros(self.num_edge)
        
        if x_cloud > 1e-9 and p_cloud > 1e-9:
            rate_cloud = self._compute_transmission_rate(p_cloud, is_cloud=True)
            max_x_cloud = rate_cloud * t_needed_cloud
            final_x_cloud = min(x_cloud, max_x_cloud)
        
        for j in range(self.num_edge):
            if i != j and x_peer[j] > 1e-9 and p_peer[j] > 1e-9:
                rate_peer = self._compute_transmission_rate(p_peer[j], is_cloud=False)
                max_x_peer = rate_peer * t_needed_peer[j]
                final_x_peer[j] = min(x_peer[j], max_x_peer)
            
        total_requested = final_x_cloud + np.sum(final_x_peer)
        if total_requested > current_q + 1e-9:
            scale_q = current_q / (total_requested + 1e-9)
            self._log_warning(f"[Warning] Step {self.time_step}:")
            self._log_warning(f"\tEdge Server {i+1} total offload request ({total_requested:.4f}MB) exceeds current Queue ({current_q:.4f}MB)")
            final_x_cloud *= scale_q
            final_x_peer *= scale_q
        
        return p_cloud, p_peer, final_x_cloud, final_x_peer, t_needed_cloud, t_needed_peer

    def _compute_energy_carbon(self, factor, ci, t, is_cmp=False):
        energy = ((factor ** 3 * Config.ZETA) if is_cmp else factor) * t
        carbon = ci * energy * Config.CONST_JOULE_TO_KWH
        return energy, carbon

    def _compute_transmission_rate(self, p_tx, is_cloud=False):
        """
        Shannon Capacity: R (bits/sec) = W (Hz) * log2(1 + S (W) /N (W))
                          S = g (no unit) * p (W)
        """
        gain = Config.G_IC if is_cloud else Config.G_IJ
        snr = 0 if Config.NOISE_POWER <= 0 else (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        return rate
