import os
import numpy as np
from config import Config

class DataLoader:
    def __init__(self):
        # 從 Config 取得檔名
        self.config_file = Config.CONFIG_FILE
        self.history_file = Config.HISTORY_FILE
        self.predict_file = Config.PREDICT_FILE

    def load_data(self):
        """
        載入資料的主入口。
        """
        print(f"Initializing data loader...")
        print(f"Target Config File: {self.config_file}")
        
        # 1. 讀取 Config 檔案 (任務到達與拓樸)
        # 這是最關鍵的部分，若失敗則無法進行正確模擬
        if os.path.exists(self.config_file):
            try:
                task_arrival, edge_graph = self._read_dcwa_config(self.config_file)
                print(f"Successfully loaded Config/Task data from {self.config_file}")
            except Exception as e:
                print(f"Error parsing config file: {e}")
                print("Falling back to synthetic task data.")
                task_arrival, _, _, edge_graph = self._generate_synthetic_data()
        else:
            print(f"Config file '{self.config_file}' not found. Using synthetic task data.")
            task_arrival, _, _, edge_graph = self._generate_synthetic_data()

        # 2. 嘗試讀取 CI 檔案 (Carbon Intensity)
        ci_history = {}
        ci_predict = {}
        if os.path.exists(self.history_file) and os.path.exists(self.predict_file):
            try:
                ci_history, ci_predict = self._read_ci_files(self.history_file, self.predict_file)
                print(f"Successfully loaded CI data from {self.history_file} and {self.predict_file}")
            except Exception as e:
                print(f"Error parsing CI files: {e}. Using synthetic CI data.")
                _, syn_hist, syn_pred, _ = self._generate_synthetic_data()
                ci_history, ci_predict = syn_hist, syn_pred
        else:
            print(f"CI files not found. Using synthetic CI data.")
            _, syn_hist, syn_pred, _ = self._generate_synthetic_data()
            ci_history, ci_predict = syn_hist, syn_pred
            
        return task_arrival, ci_history, ci_predict, edge_graph

    def _read_dcwa_config(self, filepath):
        """
        專門解析 DCWA-6-4-mb-0-2_1.txt 格式
        格式:
        Line 0: Num TimeSlots (e.g., 1000)
        Line 1: Num Servers (e.g., 5)
        Line 2: Neighbors (e.g., "00100 00001 ...")
        Line 3 ~ 3+N-1: Edge Specs (Ignored here, using Config class)
        Line 3+N ~ 3+2N-1: Cloud Specs (Ignored)
        Line 3+2N ~ End: Task Arrivals (TimeSlots lines)
        """
        task_arrival_data = {}
        edge_graph = {}
        
        with open(filepath, 'r') as file:
            lines = [l.strip() for l in file.readlines() if l.strip()]
            
            # Header
            try:
                num_timeslots = int(lines[0])
                num_servers = int(lines[1])
            except ValueError:
                raise ValueError("Header format error. Expected integers at line 0 and 1.")

            # Parse Neighbors (Line 2)
            # 格式可能是 "00100 00001 10011..." (空格分隔)
            neighbor_parts = lines[2].split()
            
            # 驗證格式長度
            if len(neighbor_parts) != num_servers:
                # 有時候可能是連在一起的長字串? 如果是這樣需要額外處理
                # 但根據 snippet 應該是分開的
                # 如果只有一個元素但長度是 N*N，則切分
                if len(neighbor_parts) == 1 and len(neighbor_parts[0]) == num_servers * num_servers:
                    full_str = neighbor_parts[0]
                    neighbor_parts = [full_str[i*num_servers:(i+1)*num_servers] for i in range(num_servers)]
            
            for i in range(num_servers):
                edge_server_name = f"Edge Server {i + 1}"
                edge_graph[edge_server_name] = []
                
                # 讀取第 i 個 server 的鄰居關係字串
                if i < len(neighbor_parts):
                    relation_str = neighbor_parts[i]
                    for j in range(num_servers):
                        if j < len(relation_str) and relation_str[j] == '1':
                            edge_graph[edge_server_name].append(f"Edge Server {j + 1}")

            # Parse Task Arrivals
            # 起始行數: 3 (Header+Neighbors) + 2 * num_servers (Specs)
            # 在您提供的檔案中，第3行開始是規格，共有 5+5=10 行規格
            # 所以 Task Arrival 從 3 + 10 = 13 行開始 (0-indexed)
            start_line = 3 + 2 * num_servers
            
            # 確保不會超出檔案範圍
            end_line = min(start_line + num_timeslots, len(lines))
            
            # 初始化 list
            for i in range(num_servers):
                task_arrival_data[f"Edge Server {i + 1}"] = []
                
            for line_idx in range(start_line, end_line):
                parts = lines[line_idx].split()
                for i in range(num_servers):
                    if i < len(parts):
                        server_name = f"Edge Server {i + 1}"
                        try:
                            val = float(parts[i]) # 讀取並轉為 float
                            task_arrival_data[server_name].append(val)
                        except ValueError:
                            task_arrival_data[server_name].append(0.0)
                            
        return task_arrival_data, edge_graph

    def _read_ci_files(self, hist_path, pred_path):
        """讀取 CI 檔案 (Sprint.txt)，這部分維持不變，假設格式為 CSV"""
        ci_history = {}
        ci_predict = {}
        num_servers = Config.NUM_EDGE_SERVERS
        
        # Helper to read csv-like lines
        def read_file_to_dict(path):
            data_dict = {}
            with open(path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                # Edge Servers (前 N 行)
                for i in range(num_servers):
                    if i < len(lines):
                        vals = list(map(float, lines[i].split(',')))
                        data_dict[f"Edge Server {i + 1}"] = vals
                
                # Cloud Servers
                # 假設 Cloud 資料在 Edge 之後。如果只有一行，則共用。
                # 如果有 N 行 Cloud，則一一對應。
                remaining_lines = len(lines) - num_servers
                if remaining_lines >= num_servers:
                    for i in range(num_servers):
                        vals = list(map(float, lines[num_servers + i].split(',')))
                        data_dict[f"Cloud Server {i + 1}"] = vals
                elif remaining_lines > 0:
                    # 共用最後一行
                    vals = list(map(float, lines[-1].split(',')))
                    for i in range(num_servers):
                        data_dict[f"Cloud Server {i + 1}"] = vals
                else:
                    # 若無 Cloud 資料，複製 Edge 1 或生成預設
                    print("Warning: No separate Cloud CI data found. Using Edge CI as fallback.")
                    for i in range(num_servers):
                         data_dict[f"Cloud Server {i + 1}"] = data_dict[f"Edge Server 1"]
            return data_dict

        ci_history = read_file_to_dict(hist_path)
        ci_predict = read_file_to_dict(pred_path)
                    
        return ci_history, ci_predict

    def _generate_synthetic_data(self):
        # Fallback generator
        num_timeslots = 100
        num_edge = Config.NUM_EDGE_SERVERS
        
        edge_graph = {}
        for i in range(num_edge):
            name = f"Edge Server {i+1}"
            neighbors = [f"Edge Server {j+1}" for j in range(num_edge) if i!=j]
            edge_graph[name] = neighbors

        task_arrival = {}
        for i in range(num_edge):
            name = f"Edge Server {i+1}"
            task_arrival[name] = np.random.randint(1e6, 5e6, size=num_timeslots).tolist()

        ci_history = {}
        ci_predict = {}
        all_servers = [f"Edge Server {i+1}" for i in range(num_edge)] + \
                      [f"Cloud Server {i+1}" for i in range(Config.NUM_CLOUD_SERVERS)]
        
        for name in all_servers:
            base = np.random.uniform(0.2, 0.6)
            trend = np.sin(np.linspace(0, 10, num_timeslots)) * 0.1
            vals = np.clip(base + trend + np.random.normal(0, 0.05, num_timeslots), 0.1, 1.0).tolist()
            ci_history[name] = vals
            ci_predict[name] = vals 

        return task_arrival, ci_history, ci_predict, edge_graph