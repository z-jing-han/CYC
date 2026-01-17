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
    TIME_SLOT_DURATION = 1.0   # Unit: seconds (s)
    
    # --- Physical Parameters ---
    # Frequency limit
    EDGE_F_MAX = 1e9           # 1 GHz
    # [FIX] Added Cloud Frequency Limit from Paper (80 GHz)
    # Without this, Cloud is capped at 1GHz, causing queue buildup.
    CLOUD_F_MAX = 8e10         # 80 GHz 
    
    # Power limit: 1 Watt
    EDGE_P_MAX = 1.0           # Unit: Watts (W)
    
    # Computational Intensity
    PHI = 1000.0               # Unit: cycles/bit (CPU cycles required per bit)
    
    # Effective Capacitance (ZETA)
    # [FIX] Corrected ZETA for physical realism (Watts ~ 1-10 range)
    # 1e-18 with 1e9^3 Hz gives 1e9 Watts. 1e-28 gives ~1 Watt.
    ZETA = 1e-28               # Unit: Effective Capacitance coefficient
    
    # --- Communication Parameters ---
    BANDWIDTH = 10e6           # Unit: Hz (10 MHz)
    
    # Noise Power
    # [FIX] -130 dBm = 10^-16 W. -100 dBm = 10^-13 W.
    # Using 1e-13 to ensure reasonable Shannon Capacity.
    NOISE_POWER = 1e-13        # Unit: Watts (W)
    
    G_IJ = 1e-7                # Channel Gain Edge-Edge (Linear scale)
    G_IC = 1e-7                # Channel Gain Edge-Cloud (Linear scale)
    
    # --- Energy Conversion Constants ---
    # Energy (Joules) = Power (Watts) * Time (Seconds)
    # Carbon (g) = Carbon Intensity (g/kWh) * Energy (kWh)
    # 1 Joule = 2.77778e-7 kWh
    CONST_JOULE_TO_KWH = 2.778e-7 

    # --- Lyapunov Optimization Parameters ---
    # V: Control parameter for tradeoff between Queue and Carbon
    # Larger V -> Less Carbon, Larger Queue
    V = 1e9                    # Unit: Dimensionless weight factor (Updated to 1e9 to match DCWA.py)
    
    # --- Alpha Smoothing ---
    ALPHAS = [0.88, 0.81, 0.95, 0.97, 0.9] 
    ALPHA_CLOUD = 0.82