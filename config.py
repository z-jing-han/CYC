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
    NUM_CLOUD_SERVERS = 5      # In DCWA code, Cloud count equals Edge count (one-to-one mapping logic)
    TIME_SLOT_DURATION = 1.0   
    
    # --- Physical Parameters (Based on DCWA.py original settings) ---
    EDGE_F_MAX = 1e9           # 1 GHz
    EDGE_P_MAX = 1e9           # Max transmission power
    
    # Computational Intensity
    PHI = 1000.0               # Cycles per bit (cpu_cycles_per_bit)
    ZETA = 1e-18               # Effective Capacitance
    
    # Communication Parameters
    BANDWIDTH = 10e6           # 10 MHz
    NOISE_POWER = 130          # N0
    G_IJ = 1e-8                # Channel Gain Edge-Edge
    G_IC = 1e-8                # Channel Gain Edge-Cloud
    
    # --- Key Correction: Energy Emission Constants (Source: DCWA.py) ---
    # Constants used in DCWA.py; note that computation and transmission are different
    CONST_EMISSION_COMPUTATION = 2.78 * 10**(-13) 
    CONST_EMISSION_TRANSMISSION = 2.78 * 10**(-15)

    # --- Lyapunov Optimization Parameters ---
    V = 1.0                    # Weight control for Queue vs Carbon
    
    # --- Alpha Smoothing ---
    # Hardcoded values from DCWA.py
    ALPHAS = [0.88, 0.81, 0.95, 0.97, 0.9] # Corresponding to Edge 1-5
    ALPHA_CLOUD = 0.82