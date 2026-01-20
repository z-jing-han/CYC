import numpy as np
from config import Config

class CloudEdgeEnvironment:
    def __init__(self, data_loader, logger=None, enable_dvfs=True):
        self.task_data, self.ci_hist_raw, self.ci_pred_raw, self.edge_graph = data_loader.load_data()
        self.logger = logger
        self.enable_dvfs = enable_dvfs
        
        self.num_edge = Config.NUM_EDGE_SERVERS
        self.time_step = 0
        
        if self.task_data:
            first_key = list(self.task_data.keys())[0]
            self.max_time_steps = len(self.task_data[first_key])
        else:
            self.max_time_steps = 1000 
        
        self.queues_edge = np.zeros(self.num_edge)
        self.queues_cloud = np.zeros(self.num_edge) 
        
        self.ci_hist_state = np.zeros((self.num_edge, 2)) 
        self.ci_pred_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_hist_state = np.zeros((self.num_edge, 2))
        self.ci_cloud_pred_state = np.zeros((self.num_edge, 2))

    def reset(self):
        self.time_step = 0
        # Reset Queues: 5 MB Initial
        self.queues_edge.fill(5.0 * Config.MB_TO_BITS) 
        self.queues_cloud.fill(0)
        
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            if edge_name in self.ci_hist_raw:
                val = self.ci_hist_raw[edge_name][0]
                self.ci_hist_state[i] = [val, val]
                val_pred = self.ci_pred_raw[edge_name][0]
                self.ci_pred_state[i] = [val_pred, val_pred]
        
        return self._get_state()

    def _get_state(self):
        t = self.time_step % self.max_time_steps
        
        ci_edge = np.zeros(self.num_edge)
        ci_cloud = np.zeros(self.num_edge)
        arrival_sizes = np.zeros(self.num_edge)
        
        for i in range(self.num_edge):
            edge_name = f"Edge Server {i+1}"
            
            if edge_name in self.ci_hist_raw:
                ci_edge[i] = self.ci_hist_raw[edge_name][t]
            else:
                ci_edge[i] = 0.5
                
            cloud_name = f"Cloud Server {i+1}"
            if cloud_name in self.ci_hist_raw: 
                ci_cloud[i] = self.ci_hist_raw[cloud_name][t]
            else:
                ci_cloud[i] = ci_edge[i] 
                
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
        snr = 0 if Config.NOISE_POWER <= 0 else (p_tx * gain) / Config.NOISE_POWER
        rate = Config.BANDWIDTH * np.log2(1 + snr)
        return rate

    def step(self, decisions):
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

        f_edge = decisions['f_edge'].copy()
        x_peer = decisions['x_peer']
        p_peer = decisions['p_peer']
        x_cloud = decisions['x_cloud']
        p_cloud = decisions['p_cloud'] 
        f_cloud = decisions.get('f_cloud', np.zeros(self.num_edge)).copy()

        # --- DVFS Logic ---
        if self.enable_dvfs:
            for i in range(self.num_edge):
                # Edge
                capacity_local = (f_edge[i] / Config.PHI) * Config.TIME_SLOT_DURATION
                actual_load_local = min(self.queues_edge[i], capacity_local)
                if actual_load_local > 0:
                    f_needed = (actual_load_local * Config.PHI) / Config.TIME_SLOT_DURATION
                    f_edge[i] = min(f_edge[i], f_needed)
                else:
                    f_edge[i] = 0
                
                # Cloud
                capacity_cloud = (f_cloud[i] / Config.PHI) * Config.TIME_SLOT_DURATION
                actual_load_cloud = min(self.queues_cloud[i], capacity_cloud)
                if actual_load_cloud > 0:
                    f_c_needed = (actual_load_cloud * Config.PHI) / Config.TIME_SLOT_DURATION
                    f_cloud[i] = min(f_cloud[i], f_c_needed)
                else:
                    f_cloud[i] = 0

        state = self._get_state()
        ci_edge = state['CI_edge']
        ci_cloud = state['CI_cloud']
        arrival = state['Arrival']
        
        total_carbon = 0.0
        
        # --- 1. Process Local Tasks ---
        capacity_local = (f_edge / Config.PHI) * Config.TIME_SLOT_DURATION
        # Cannot exceed the Queue len
        bits_processed_local = np.minimum(self.queues_edge, capacity_local)
        
        for i in range(self.num_edge):
            m_edge_proc_local[i] = bits_processed_local[i]
            
            if f_edge[i] > 1e-6:
                processing_rate = f_edge[i] / Config.PHI
                # Time = Work / Rate
                time_active = bits_processed_local[i] / processing_rate
                time_active = min(time_active, Config.TIME_SLOT_DURATION)
                
                # Energy = xi * f^3 * t
                energy_joules = (f_edge[i]**3) * Config.ZETA * time_active
                m_edge_energy_comp[i] = energy_joules
                
                c_val = ci_edge[i] * energy_joules * Config.CONST_JOULE_TO_KWH
                m_edge_carbon[i] += c_val
                total_carbon += c_val

        # --- 2. Cloud Processing ---
        capacity_cloud = (f_cloud / Config.PHI) * Config.TIME_SLOT_DURATION
        bits_processed_cloud = np.minimum(self.queues_cloud, capacity_cloud)
        
        for i in range(self.num_edge):
            m_cloud_proc[i] = bits_processed_cloud[i]
            
            if f_cloud[i] > 1e-6:
                processing_rate = f_cloud[i] / Config.PHI
                time_active = bits_processed_cloud[i] / processing_rate
                time_active = min(time_active, Config.TIME_SLOT_DURATION)
                
                energy_joules = (f_cloud[i]**3) * Config.ZETA * time_active
                m_cloud_energy_comp[i] = energy_joules
                
                c_val = ci_cloud[i] * energy_joules * Config.CONST_JOULE_TO_KWH
                m_cloud_carbon[i] = c_val
                total_carbon += c_val

        # --- 3. Transmission Physics ---
        real_x_peer = np.zeros_like(x_peer)
        
        for i in range(self.num_edge):
            for j in range(self.num_edge):
                if i == j or x_peer[i, j] <= 0: continue
                
                if p_peer[i, j] > 0:
                    rate = self.compute_transmission_rate(p_peer[i, j], is_cloud=False)
                    max_bits = rate * Config.TIME_SLOT_DURATION
                    real_x_peer[i, j] = min(x_peer[i, j], max_bits)
                    
                    if rate > 0 and real_x_peer[i, j] > 0:
                        time_tx = real_x_peer[i, j] / rate
                        energy_joules = p_peer[i, j] * time_tx
                        m_edge_energy_tx[i] += energy_joules
                        
                        c_val = ci_edge[i] * energy_joules * Config.CONST_JOULE_TO_KWH
                        m_edge_carbon[i] += c_val
                        total_carbon += c_val
        
        real_x_cloud = np.zeros_like(x_cloud)
        for i in range(self.num_edge):
            if x_cloud[i] > 0 and p_cloud[i] > 0:
                rate = self.compute_transmission_rate(p_cloud[i], is_cloud=True)
                max_bits = rate * Config.TIME_SLOT_DURATION
                real_x_cloud[i] = min(x_cloud[i], max_bits)
                
                if rate > 0 and real_x_cloud[i] > 0:
                    time_tx = real_x_cloud[i] / rate
                    energy_joules = p_cloud[i] * time_tx
                    m_edge_energy_tx[i] += energy_joules 
                    
                    c_val = ci_edge[i] * energy_joules * Config.CONST_JOULE_TO_KWH
                    m_edge_carbon[i] += c_val
                    total_carbon += c_val

        # --- 4. Queue Update ---
        # 1. excluding local processing
        self.queues_edge -= bits_processed_local
        
        # 2. excluding the offloaded portion (ensuring it does not become negative; i.e., the offloaded amount cannot exceed the current amount)
        total_out_req = np.sum(real_x_peer, axis=1) + real_x_cloud
        actual_out_ratio = np.ones(self.num_edge)
        
        # check for exceeding the available amount
        mask = total_out_req > (self.queues_edge + 1e-9)
        actual_out_ratio[mask] = self.queues_edge[mask] / (total_out_req[mask] + 1e-9)
        
        final_x_peer = real_x_peer * actual_out_ratio[:, np.newaxis]
        final_x_cloud = real_x_cloud * actual_out_ratio
        
        # record the actual offloaded amount
        for i in range(self.num_edge):
            m_edge_tx_peer[i] = np.sum(final_x_peer[i])
            m_edge_tx_cloud[i] = final_x_cloud[i]
            m_cloud_rx_edge[i] = final_x_cloud[i]
        
        # Update the edge queue: subtract the offloaded amount, add arrivals, and add tasks offloaded from other nodes
        self.queues_edge -= (np.sum(final_x_peer, axis=1) + final_x_cloud)
        peer_in = np.sum(final_x_peer, axis=0)
        self.queues_edge += arrival + peer_in
        
        # ensure that floating-point errors do not result in tiny negative values
        self.queues_edge = np.maximum(self.queues_edge, 0.0)
        
        # Update Cloud Queue
        self.queues_cloud -= bits_processed_cloud
        self.queues_cloud = np.maximum(self.queues_cloud, 0.0)
        self.queues_cloud += final_x_cloud
        
        self.time_step += 1
        
        info = {
            'carbon': total_carbon,
            'q_avg_total': np.mean(self.queues_edge),
            'processed_local': np.mean(bits_processed_local),
            'processed_cloud': np.mean(bits_processed_cloud),
            'offloaded_cloud': np.mean(final_x_cloud)
        }
        
        if self.logger:
            edge_metrics = []
            for i in range(self.num_edge):
                edge_metrics.append({
                    'arrival': arrival[i],
                    'q_pre': m_edge_q_pre[i],
                    'q_post': self.queues_edge[i],
                    'proc_local': m_edge_proc_local[i],
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