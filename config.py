import numpy as np
import os

class Config:
    # --- File Path Settings ---
    BASE_DIR = os.getcwd()
    CONFIG_FILE = 'InputData/DCWA-6-4-mb-0-2_1.txt'
    HISTORY_FILE = 'InputData/Sprint.txt'
    PREDICT_FILE = 'InputData/Sprint_predict.txt'

    # --- System Parameters ---
    NUM_EDGE_SERVERS = 5
    NUM_CLOUD_SERVERS = 5
    
    # Time Slot Duration: paper p.6 "Each time slot was one hour."
    TIME_SLOT_DURATION = 3600.0   # Unit: seconds (s)
    
    # --- Physical Parameters ---
    # paper p.39: fi_max = 10 GHz, fi_cmax = 80 GHz
    EDGE_F_MAX = 10e9          # 10 GHz
    CLOUD_F_MAX = 80e9         # 80 GHz
    
    # Power limit: paper p.39 pi_max = 1 W
    EDGE_P_MAX = 1.0           # Unit: Watts (W)
    
    # Computational Intensity: paper p.39 phi = 1000
    PHI = 1000.0               # Unit: cycles/bit
    
    # Effective Capacitance (ZETA)
    # paper p.39 xi = 10^-18 (It is usually MH Level Unit).
    # If We consider Hz (10^9)，ZETA = 1e-28 should be a reason Watt Level Power
    # Power = ZETA * f^3 = 1e-28 * (10^10)^3 = 1e-28 * 10^30 = 100 Watts
    # [Check Again]
    ZETA = 1e-28               # Unit: Effective Capacitance coefficient
    
    # --- Communication Parameters ---
    # paper p.39 Wi = 1 MHz
    BANDWIDTH = 1e6            # Unit: Hz (1 MHz)
    
    # Noise Power: paper p.39 N0 = -130 dBm
    # -130 dBm = 10^(-130/10) mW = 10^-13 mW = 10^-16 W
    NOISE_POWER = 1e-16        # Unit: Watts (W)
    
    # Channel Gain
    G_IJ = 1e-4                # Channel Gain Edge-Edge
    G_IC = 1e-4                # Channel Gain Edge-Cloud
    
    # --- Energy Conversion Constants ---
    # Carbon (g) = Carbon Intensity (g/kWh) * Energy (J) * (kWh/J)
    # 1 Joule = 2.77778e-7 kWh
    CONST_JOULE_TO_KWH = 2.778e-7

    # --- Lyapunov Optimization Parameters ---
    # V: Control parameter (Trade-off between Energy and Queue)
    V = 1e12                   # Unit: Dimensionless weight factor
    
    # --- Alpha Smoothing ---
    ALPHAS = [0.86, 0.75, 0.95, 0.97, 0.65]
    ALPHA_CLOUD = 0.93

    # --- Data Unit Scaling ---
    # 1 MB = 8 * 10^6 bits
    DATA_SCALE_FACTOR = 8.0
    MB_TO_BITS = 8 * 1e6