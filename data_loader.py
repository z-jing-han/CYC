import os
import json
import numpy as np
import pandas as pd
from config import Config

class DataLoader:
    def __init__(self):
        self.config_json_path = Config.CONFIG_JSON
        self.task_csv_path = Config.TASK_FILE
        self.carbon_csv_path = Config.CARBON_FILE

    def load_data(self):
        """
        Loads data from config.json, data_arrival.csv, and carbon_data.csv.
        Returns:
            task_arrival (dict): {ServerName: [bits_t0, ...]}
            ci_history (dict): {ServerName: [val_t0, ...]}
            ci_predict (dict): {ServerName: [val_t0, ...]}
            edge_graph (dict): {ServerName: [NeighborName, ...]}
        """
        
        edge_graph = self._load_topology(self.config_json_path)
        task_arrival = self._load_tasks(self.task_csv_path)
        ci_history, ci_predict = self._load_carbon(self.carbon_csv_path)
        
        # Verify alignment
        if task_arrival and ci_history:
            t_len = len(next(iter(task_arrival.values())))
            c_len = len(next(iter(ci_history.values())))
        
        return task_arrival, ci_history, ci_predict, edge_graph

    def _load_topology(self, json_path):
        """Reads config.json to build the edge server topology."""
        edge_graph = {}
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Error: {json_path} not found.")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # Read Global Config
            if 'system_settings' in config_data:
                Config.NUM_EDGE_SERVERS = config_data['system_settings'].get('num_edge_servers', Config.NUM_EDGE_SERVERS)
                Config.NUM_CLOUD_SERVERS = Config.NUM_EDGE_SERVERS
                Config.TIME_SLOT_DURATION = config_data['system_settings'].get('duration_seconds', Config.TIME_SLOT_DURATION)
                Config.CLOUD_F_MAX = config_data['system_settings'].get('cloud_max_freq_Hz', Config.CLOUD_F_MAX)
                Config.EDGE_F_MAX = config_data['system_settings'].get('edge_max_freq_Hz', Config.EDGE_F_MAX)
                Config.EDGE_P_MAX = config_data['system_settings'].get('edge_max_trans_power_W', Config.EDGE_P_MAX)
                Config.PHI = config_data['system_settings'].get('cpu_cycles_per_bit', Config.PHI)
                Config.ZETA = config_data['system_settings'].get('effective_capacitance_Zeta', Config.ZETA)
                Config.BANDWIDTH = config_data['system_settings'].get('edge_bandwith_Hz', Config.BANDWIDTH)
                # Conver dBm to W
                Config.NOISE_POWER = 10 ** ((config_data['system_settings'].get('noise_power_dBm', -130) - 30) / 10)
                Config.G_IJ = config_data['system_settings'].get('channel_gain', Config.G_IJ)
                Config.G_IC = config_data['system_settings'].get('channel_gain', Config.G_IC)
            else:
                print("system_settings lost in config.json, using default setting")

            # Parse Adjacency Matrix
            matrix = config_data.get('topology', {}).get('matrix', [])
            edge_servers = config_data.get('servers', {}).get('edge_servers', [])
            
            num_servers = len(matrix)
            for i in range(num_servers):
                curr_name = f"Edge Server {i+1}"
                neighbors = []
                for j in range(num_servers):
                    if i != j and matrix[i][j] == 1:
                        neighbors.append(f"Edge Server {j+1}")
                edge_graph[curr_name] = neighbors
            
        except Exception as e:
            print(f"Failed to parse config.json: {e}")
            raise e

        return edge_graph

    def _load_tasks(self, csv_path):
        """Reads data_arrival.csv."""
        task_arrival = {}
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: {csv_path} not found.")

        try:
            df = pd.read_csv(csv_path)
            
            # normalize keys to "Edge Server 1"
            
            for col in df.columns:
                # Normalize key: replace underscore with space if it looks like a server name
                if "Edge" in col:
                    clean_name = col.replace('_', ' ').strip()
                    # Ensure "Edge Server X" format
                    if "EdgeServer" in clean_name: # Handle case "EdgeServer 1"
                         clean_name = clean_name.replace("EdgeServer", "Edge Server")
                    
                    # Bits
                    values_bits = df[col].values
                    task_arrival[clean_name] = values_bits.tolist()
            
        except Exception as e:
            print(f"Failed to parse data_arrival.csv: {e}")
            raise e
            
        return task_arrival

    def _load_carbon(self, csv_path):
        """Reads carbon_data.csv for History and Prediction."""
        ci_history = {}
        ci_predict = {}
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: {csv_path} not found.")

        try:
            df = pd.read_csv(csv_path)
            
            # Expected columns: TimeSlot, Edge_1_History, ..., Cloud_Shared_History
            
            # 1. Parse Edge Data
            for col in df.columns:
                if "Edge" in col and "History" in col:
                    # Format: "Edge_1_History" -> extract ID -> "Edge Server 1"
                    try:
                        # Split by underscore
                        parts = col.split('_') # ['Edge', '1', 'History']
                        if len(parts) >= 2 and parts[1].isdigit():
                            server_id = parts[1]
                            server_name = f"Edge Server {server_id}"
                            
                            vals = df[col].values.tolist()
                            ci_history[server_name] = vals
                            ci_predict[server_name] = vals # Assume perfect prediction for now
                    except:
                        print(f"Skipping unknown column format: {col}")

            # 2. Parse Cloud Data
            # If there is a 'Cloud_Shared_History' or similar
            cloud_col = next((c for c in df.columns if 'Cloud' in c), None)
            
            if cloud_col:
                cloud_vals = df[cloud_col].values.tolist()
                # Broadcast to all cloud servers
                for i in range(Config.NUM_CLOUD_SERVERS):
                    c_name = f"Cloud Server {i+1}"
                    ci_history[c_name] = cloud_vals
                    ci_predict[c_name] = cloud_vals
            else:
                raise ValueError("No Cloud Carbon column found")

        except Exception as e:
            print(f"Failed to parse carbon_data.csv: {e}")
            raise e
            
        return ci_history, ci_predict