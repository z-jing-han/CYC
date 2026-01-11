import numpy as np
import os

class Config:
    # --- 檔案路徑設定 ---
    BASE_DIR = os.getcwd()
    CONFIG_FILE = 'DCWA-6-4-mb-0-2_1.txt' 
    HISTORY_FILE = 'Sprint.txt'
    PREDICT_FILE = 'Sprint_predict.txt'

    # --- 系統參數 ---
    NUM_EDGE_SERVERS = 5       
    NUM_CLOUD_SERVERS = 5      # DCWA 程式碼中 Cloud 數量等於 Edge 數量 (一對一配對邏輯)
    TIME_SLOT_DURATION = 1.0   
    
    # --- 物理參數 (依照 DCWA.py 原始設定) ---
    EDGE_F_MAX = 1e9           # 1 GHz
    EDGE_P_MAX = 1e9           # 傳輸功率上限
    
    # 運算強度
    PHI = 1000.0               # Cycles per bit (cpu_cycles_per_bit)
    ZETA = 1e-18               # Effective Capacitance
    
    # 通訊參數
    BANDWIDTH = 10e6           # 10 MHz
    NOISE_POWER = 130          # N0
    G_IJ = 1e-8                # Channel Gain Edge-Edge
    G_IC = 1e-8                # Channel Gain Edge-Cloud
    
    # --- 關鍵修正：能耗轉換常數 (源自 DCWA.py) ---
    # DCWA.py 使用的常數，注意 computation 和 transmission 不同
    CONST_EMISSION_COMPUTATION = 2.78 * 10**(-13) 
    CONST_EMISSION_TRANSMISSION = 2.78 * 10**(-15)

    # --- Lyapunov 優化參數 ---
    V = 1.0                    # 控制 Queue vs Carbon 的權重
    
    # --- Alpha Smoothing ---
    # 依照 DCWA.py 中的寫死數值
    ALPHAS = [0.88, 0.81, 0.95, 0.97, 0.9] # 對應 Edge 1-5
    ALPHA_CLOUD = 0.82