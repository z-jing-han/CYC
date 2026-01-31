import numpy as np
import os

class Config:
    # --- File Path Settings ---
    BASE_DIR = os.getcwd()
    CONFIG_JSON = 'config.json' 
    TASK_FILE = 'data_arrival.csv'
    CARBON_FILE = 'carbon_intensity.csv'

    # (will be overwritten by config.json)
    # --- Default System Parameters ---
    NUM_EDGE_SERVERS = 5
    NUM_CLOUD_SERVERS = 5
    TIME_SLOT_DURATION = 3600.0   # Unit: seconds (s)
    
    # --- Physical Parameters ---
    EDGE_F_MAX = 10e9          # 10 GHz
    CLOUD_F_MAX = 80e9         # 80 GHz
    EDGE_P_MAX = 1.0           # Unit: Watts (W)
    PHI = 1000.0               # Unit: cycles/bit
    BANDWIDTH = 1e6            # Unit: Hz (1 MHz)
    NOISE_POWER = 1e-16        # Unit: Watts (W)
    G_IJ = 1e-4                # Channel Gain Edge-Edge
    G_IC = 1e-4                # Channel Gain Edge-Cloud
    ZETA = 1e-18               # Unit: Effective Capacitance coefficient

    # --- Energy Conversion Constants ---
    # Carbon (g) = Carbon Intensity (g/kWh) * Energy (J) * (kWh/J)
    CONST_JOULE_TO_KWH = 2.778e-7

    # --- Lyapunov Optimization Parameters ---
    V = 1e12                   # Control parameter (Trade-off between Energy and Queue)
    
    # --- Alpha Smoothing (Can be updated from config.json if needed) ---
    ALPHAS = list()
    ALPHA_CLOUD = 0.93

    # --- Data Unit Scaling ---
    MB_TO_BITS = 8 * 1e6