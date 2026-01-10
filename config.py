import numpy as np

class Config:
    # --- System architecture settings ---
    NUM_EDGE_SERVERS = 5       # The paper sets up 5 Edge Servers
    NUM_CLOUD_SERVERS = 1      # 1 Cloud Server
    TIME_SLOT_DURATION = 1.0   # 1 hour per slot in paper, but usually 1 sec/min for simulation speed
    
    # --- Computation and energy consumption parameters (based on paper Section VI) ---
    # CPU Frequency (Cycles/s)
    EDGE_F_MAX = 10.0 * 1e9    # 10 GHz
    CLOUD_F_MAX = 80.0 * 1e9   # 80 GHz
    
    # Computational Intensity (Cycles/bit)
    PHI_EDGE = 1000.0          # cycles/bit
    PHI_CLOUD = 1000.0         # cycles/bit
    
    # Effective Capacitance Coefficient (for Energy = k * f^3 * t)
    KAPPA_EDGE = 1e-27         # paper uses \xi = 10^-18 for capacitance, scaled for numerical stability if needed
                               # Note: Power = \xi * f^3. If f is 10^9, f^3 is 10^27. 
                               # 10^-18 * 10^27 = 10^9 Watts (Too high). 
                               # Usually \xi is around 1e-28. Let's assume paper meant 1e-27 or units are different.
                               # Let's align with common values: 1e-28
    KAPPA_CLOUD = 1e-27
    
    # Transmission Power (Watts)
    EDGE_P_TX_MAX = 1.0        # 1 Watt
    CLOUD_P_TX_MAX = 1.0       # Assumptions for constraints on cloud feedback or edge-to-cloud transmission
    
    # Communication
    BANDWIDTH = 1e6            # W = 1 MHz
    NOISE_POWER = 1e-13        # N0 = -130 dBm = 10^-13 Watts
    CHANNEL_GAIN_AVG = 1e-6    # Average g (assumed)
    
    # --- Tasks and Queues ---
    TASK_ARRIVAL_MEAN_ON = 64.0 * 8 * 1e6  # 64 MB in bits
    TASK_ARRIVAL_MEAN_OFF = 3.2 * 8 * 1e6  # 3.2 MB in bits
    
    # --- Lyapunov Optimization parameters (V) ---
    # The larger $V is, the more emphasis is placed on carbon reduction, which may lead to higher queue latency.
    LYAPUNOV_V = 1e12          # Adjust based on the order of magnitude of the cost
    
    # --- Carbon Intensity (gCO2/Joule) ---
    # Simulate randomly varying Carbon Intensity (CI)
    CI_MEAN = 0.5 
    CI_VAR = 0.1

    @staticmethod
    def get_channel_gain():
        # Simple simulation of channel gain variations
        return np.random.exponential(Config.CHANNEL_GAIN_AVG)