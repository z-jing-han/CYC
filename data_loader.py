import os
import numpy as np
from config import Config

class DataLoader:
    def __init__(self):
        self.config_file = Config.CONFIG_FILE
        self.history_file = Config.HISTORY_FILE
        self.predict_file = Config.PREDICT_FILE

    def load_data(self):
        print(f"Initializing data loader...")
        print(f"Target Config File: {self.config_file}")
        
        # 1. Load Task Data
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

        # 2. Load CI Data
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
        task_arrival_data = {}
        edge_graph = {}
        
        with open(filepath, 'r') as file:
            lines = [l.strip() for l in file.readlines() if l.strip()]
            
            try:
                num_timeslots = int(lines[0])
                num_servers = int(lines[1])
            except ValueError:
                raise ValueError("Header format error.")

            # Neighbors
            neighbor_parts = lines[2].split()
            if len(neighbor_parts) != num_servers:
                if len(neighbor_parts) == 1 and len(neighbor_parts[0]) == num_servers * num_servers:
                    full_str = neighbor_parts[0]
                    neighbor_parts = [full_str[i*num_servers:(i+1)*num_servers] for i in range(num_servers)]
            
            for i in range(num_servers):
                edge_server_name = f"Edge Server {i + 1}"
                edge_graph[edge_server_name] = []
                if i < len(neighbor_parts):
                    relation_str = neighbor_parts[i]
                    for j in range(num_servers):
                        if j < len(relation_str) and relation_str[j] == '1':
                            edge_graph[edge_server_name].append(f"Edge Server {j + 1}")

            # Task Arrivals
            start_line = 3 + 2 * num_servers
            end_line = min(start_line + num_timeslots, len(lines))
            
            for i in range(num_servers):
                task_arrival_data[f"Edge Server {i + 1}"] = []
                
            for line_idx in range(start_line, end_line):
                parts = lines[line_idx].split()
                for i in range(num_servers):
                    if i < len(parts):
                        server_name = f"Edge Server {i + 1}"
                        try:
                            val_mb = float(parts[i]) 
                            
                            # [Check Again]
                            if val_mb > 5000:
                                val_mb = val_mb / 1e6 
                            
                            # MB -> Mb -> bits
                            val_mbits = val_mb * Config.DATA_SCALE_FACTOR
                            val_bits = val_mbits * 1e6 
                            
                            task_arrival_data[server_name].append(val_bits)
                        except ValueError:
                            task_arrival_data[server_name].append(0.0)
                            
        return task_arrival_data, edge_graph

    def _read_ci_files(self, hist_path, pred_path):
        ci_history = {}
        ci_predict = {}
        num_servers = Config.NUM_EDGE_SERVERS
        
        def read_file_to_dict(path):
            data_dict = {}
            with open(path, 'r') as f:
                lines = [l.strip() for l in f.readlines() if l.strip()]
                for i in range(num_servers):
                    if i < len(lines):
                        vals = list(map(float, lines[i].split(',')))
                        data_dict[f"Edge Server {i + 1}"] = vals
                
                remaining_lines = len(lines) - num_servers
                if remaining_lines >= num_servers:
                    for i in range(num_servers):
                        vals = list(map(float, lines[num_servers + i].split(',')))
                        data_dict[f"Cloud Server {i + 1}"] = vals
                elif remaining_lines > 0:
                    vals = list(map(float, lines[-1].split(',')))
                    for i in range(num_servers):
                        data_dict[f"Cloud Server {i + 1}"] = vals
                else:
                    for i in range(num_servers):
                         data_dict[f"Cloud Server {i + 1}"] = data_dict[f"Edge Server 1"]
            return data_dict

        ci_history = read_file_to_dict(hist_path)
        ci_predict = read_file_to_dict(pred_path)
        return ci_history, ci_predict

    def _generate_synthetic_data(self):
        print("Generating synthetic data according to Thesis parameters (Mean 64MB)...")
        num_timeslots = 1000
        num_edge = Config.NUM_EDGE_SERVERS
        
        edge_graph = {}
        for i in range(num_edge):
            name = f"Edge Server {i+1}"
            neighbors = [f"Edge Server {j+1}" for j in range(num_edge) if i!=j]
            edge_graph[name] = neighbors

        task_arrival = {}
        for i in range(num_edge):
            name = f"Edge Server {i+1}"
            # Paper p.39: On/Off Model, Mean On=64MB, Off=3.2MB
            # Random gernerate 10MB ~ 100MB
            val_mb = np.random.uniform(10.0, 100.0, size=num_timeslots)
            val_bits = val_mb * Config.DATA_SCALE_FACTOR * 1e6
            task_arrival[name] = val_bits.tolist()

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