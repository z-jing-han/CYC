import numpy as np
import os

class Config:
    # All paramenter will be reload via config.json
    # --- File Path Settings ---
    BASE_DIR = os.getcwd()
    CONFIG_JSON = 'config.json' 
    TASK_FILE = 'data_arrival.csv'
    CARBON_FILE = 'carbon_intensity.csv'
    
    # --- Default System Parameters ---
    NUM_EDGE_SERVERS = 5
    NUM_CLOUD_SERVERS = 5
    TIME_SLOT_DURATION = 3600.0     # Unit: seconds (s)
    TIME_SLOT_ADJUST = 'scale'      # 'scale' or 'fix_time_slot'
    
    # --- Physical Parameters ---
    EDGE_F_MAX = 10e9          # 10 GHz
    CLOUD_F_MAX = 80e9         # 80 GHz
    EDGE_P_MAX = 1.0           # Unit: Watts (W)
    PHI = 1000.0               # Unit: cycles/bit
    BANDWIDTH = 1e6            # Unit: Hz (1 MHz)
    NOISE_POWER = 1e-12        # Unit: Watts (W)
    G_IJ = 1e-8                # Channel Gain Edge-Edge
    G_IC = 1e-8                # Channel Gain Edge-Cloud
    ZETA = 1e-28               # Unit: Effective Capacitance coefficient

    # --- Energy Conversion Constants ---
    # Carbon (g) = Carbon Intensity (g/kWh) * Energy (J) * (kWh/J)
    CONST_JOULE_TO_KWH = 2.778e-7

    # --- Lyapunov Optimization Parameters ---
    V = 1e21                   # Control parameter (Trade-off between Energy and Queue)

    # --- Data Unit Scaling ---
    MB_TO_BITS = 8 * 1e6

    # --- Local Queue Capacity ---
    EEDGE_Q_CAPACITY = []

    # MARL
    MARL_LR_ACTOR = 1e-4
    MARL_LR_CRITIC = 1e-3
    MARL_GAMMA = 0.99
    MARL_BATCH_SIZE = 64
    MARL_EPISODES = 15
    MARL_BUFFER_SIZE = 10000
    MARL_NOISE = 0.05