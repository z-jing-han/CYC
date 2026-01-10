import numpy as np
import random
from config import Config

class CloudEdgeEnvironment:
    def __init__(self):
        self.num_edge = Config.NUM_EDGE_SERVERS
        self.time_step = 0
        
        # State variables: The queue for each edge device (Q_i) and its corresponding queue in the cloud (Q_i^{cloud})
        # Unit: bits
        self.queues_edge = np.zeros(self.num_edge)
        self.queues_cloud = np.zeros(self.num_edge)
        
        # Carbon Intensity (k_i(t))
        self.carbon_intensity_edge = np.zeros(self.num_edge)
        self.carbon_intensity_cloud = 0.0
        
        # Channel Gains (g_{i,j}, g_{i,c})
        self.g_edge_edge = np.zeros((self.num_edge, self.num_edge))
        self.g_edge_cloud = np.zeros(self.num_edge)
        
        # Workload Arrival (A_i(t))
        self.arrival_tasks = np.zeros(self.num_edge)

    def reset(self):
        self.time_step = 0
        self.queues_edge.fill(0.0)
        self.queues_cloud.fill(0.0)
        self._update_environment_factors()
        return self._get_state()

    def _update_environment_factors(self):
        """Update random variables: CI, Channel Gain, Task Arrival"""
        # 1. Carbon Intensity (Simulate time-varying and location-varying factors)
        self.carbon_intensity_edge = np.random.normal(Config.CI_MEAN, Config.CI_VAR, self.num_edge)
        self.carbon_intensity_edge = np.maximum(self.carbon_intensity_edge, 0.1)
        self.carbon_intensity_cloud = np.random.normal(Config.CI_MEAN, Config.CI_VAR)
        
        # 2. Channel Gains
        for i in range(self.num_edge):
            self.g_edge_cloud[i] = Config.get_channel_gain()
            for j in range(self.num_edge):
                if i != j:
                    self.g_edge_edge[i, j] = Config.get_channel_gain()
                    
        # 3. Task Arrival (On-Off Model as per paper)
        # Simplification: Use Poisson or Gaussian distribution for simulation
        for i in range(self.num_edge):
            if random.random() < 0.5: # ON state
                self.arrival_tasks[i] = np.random.normal(Config.TASK_ARRIVAL_MEAN_ON, Config.TASK_ARRIVAL_MEAN_ON*0.1)
            else: # OFF state
                self.arrival_tasks[i] = np.random.normal(Config.TASK_ARRIVAL_MEAN_OFF, Config.TASK_ARRIVAL_MEAN_OFF*0.1)
            self.arrival_tasks[i] = max(0, self.arrival_tasks[i])

    def _get_state(self):
        """System state returned to the Agent or Solver"""
        return {
            'Q_edge': self.queues_edge.copy(),
            'Q_cloud': self.queues_cloud.copy(),
            'CI_edge': self.carbon_intensity_edge.copy(),
            'CI_cloud': self.carbon_intensity_cloud,
            'G_ee': self.g_edge_edge.copy(),
            'G_ec': self.g_edge_cloud.copy(),
            'Arrival': self.arrival_tasks.copy()
        }

    def calculate_transmission_rate(self, p_tx, gain):
        """Shannon Formula: R = W * log2(1 + g*p / N0)"""
        snr = (gain * p_tx) / Config.NOISE_POWER
        return Config.BANDWIDTH * np.log2(1 + snr)

    def step(self, decisions):
        """
        Execute one step of the simulatio
        decisions is a dictionary containing the decision variables for each Edge:
        - f_edge: [num_edge] (Local CPU freq)
        - f_cloud: [num_edge] (Cloud CPU freq allocated to each edge task)
        - p_tx_peer: [num_edge, num_edge] (Tx power from i to j)
        - x_peer: [num_edge, num_edge] (Offloading amount bits from i to j)
        - p_tx_cloud: [num_edge] (Tx power to cloud)
        - x_cloud: [num_edge] (Offloading amount bits to cloud)
        """
        
        # Unpack decisions
        f_edge = decisions['f_edge']
        f_cloud = decisions['f_cloud']
        p_tx_peer = decisions['p_tx_peer']
        x_peer = decisions['x_peer']
        p_tx_cloud = decisions['p_tx_cloud']
        x_cloud = decisions['x_cloud']
        
        # --- 1. Processed bits ---
        # Formula (2): EC_i(t) = f * T / phi
        ec_local = (f_edge * Config.TIME_SLOT_DURATION) / Config.PHI_EDGE
        # Formula (5): CC_i(t)
        cc_cloud = (f_cloud * Config.TIME_SLOT_DURATION) / Config.PHI_CLOUD
        
        # --- 2. Objective ---
        # Local Comp Carbon: k * xi * f^3 * T
        carbon_local = self.carbon_intensity_edge * Config.KAPPA_EDGE * (f_edge**3) * Config.TIME_SLOT_DURATION
        
        # Cloud Comp Carbon
        carbon_cloud_comp = self.carbon_intensity_cloud * Config.KAPPA_CLOUD * (f_cloud**3) * Config.TIME_SLOT_DURATION
        
        # Transmission Carbon (Peer)
        # Time for tx = x / R
        carbon_tx_peer = 0
        for i in range(self.num_edge):
            for j in range(self.num_edge):
                if i != j and x_peer[i, j] > 0:
                    rate = self.calculate_transmission_rate(p_tx_peer[i, j], self.g_edge_edge[i, j])
                    if rate > 0:
                        t_tx = x_peer[i, j] / rate
                        carbon_tx_peer += self.carbon_intensity_edge[i] * p_tx_peer[i, j] * t_tx

        # Transmission Carbon (Cloud)
        carbon_tx_cloud = 0
        for i in range(self.num_edge):
            if x_cloud[i] > 0:
                rate = self.calculate_transmission_rate(p_tx_cloud[i], self.g_edge_cloud[i])
                if rate > 0:
                    t_tx = x_cloud[i] / rate
                    carbon_tx_cloud += self.carbon_intensity_edge[i] * p_tx_cloud[i] * t_tx

        total_carbon = np.sum(carbon_local) + np.sum(carbon_cloud_comp) + carbon_tx_peer + carbon_tx_cloud

        # --- 3. Queue Dynamics ---
        # Formula (1): Q_i(t+1)
        # Note: x_i(t) = x_cloud + sum(x_peer_out)
        # incoming_peer = sum(x_peer_in)
        
        x_out_total = x_cloud + np.sum(x_peer, axis=1) # Total amount transferred out from each Edge
        x_in_peer = np.sum(x_peer, axis=0)             # Total amount received by each Edge
        
        # Edge Queue Update
        # [Q - EC - x_out]+ + A + x_in
        # Note: The actual processing volume cannot exceed the current queue size
        processed_and_offloaded = ec_local + x_out_total
        # Ensure the processed amount does not exceed the current queue size (Physical constraint).
        # However, in Lyapunov formulas, full-speed processing is typically assumed; here, we apply a physical limit truncation.
        real_processed = np.minimum(self.queues_edge, processed_and_offloaded)
        
        # remaining
        remaining = self.queues_edge - real_processed
        # Add new
        self.queues_edge = remaining + self.arrival_tasks + x_in_peer
        
        # Cloud Queue Update Formula (4)
        # Cloud Queue processes tasks from Edge i
        real_cloud_processed = np.minimum(self.queues_cloud, cc_cloud)
        self.queues_cloud = (self.queues_cloud - real_cloud_processed) + x_cloud

        # --- 4. Calculate the next state ---
        self.time_step += 1
        self._update_environment_factors()
        next_state = self._get_state()
        
        info = {
            'carbon': total_carbon,
            'q_avg': np.mean(self.queues_edge),
            'q_cloud_avg': np.mean(self.queues_cloud)
        }
        
        return next_state, total_carbon, info