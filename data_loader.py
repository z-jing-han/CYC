import os
import json
import numpy as np
import pandas as pd
from config import Config

class DataLoader:
    def __init__(self, config_path=None, carbon_path=None, task_path=None):
        self.config_json_path = config_path if config_path else Config.CONFIG_JSON
        self.task_csv_path = task_path if task_path else Config.TASK_FILE
        self.carbon_csv_path = carbon_path if carbon_path else Config.CARBON_FILE
        
        self.edge_servers_metadata = []
        self.cloud_servers_metadata = []

    def load_data(self):
        """
        Loads data from config.json, data_arrival.csv, and carbon_data.csv.
        Returns:
            task_arrival (dict): {ServerName: [bits_t0, ...]}
            ci_history (dict): {ServerName: [val_t0, ...]}
            ci_predict (dict): {ServerName: [val_t0, ...]}
            edge_graph (dict): {ServerName: [NeighborName, ...]}
        """
        
        self._load_config(self.config_json_path)
        edge_graph = self._load_topology(self.config_json_path)
        task_arrival = self._load_tasks(self.task_csv_path)
        ci_history, ci_predict = self._load_carbon(self.carbon_csv_path)
        
        if task_arrival and ci_history:
            t_len = len(next(iter(task_arrival.values())))
            c_len = len(next(iter(ci_history.values())))
        
        return task_arrival, ci_history, ci_predict, edge_graph

    def _load_config(self, json_path):
        """Reads config.json to get all config information and server metadata"""
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Error: {json_path} not found.")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            Config.NUM_EDGE_SERVERS = config_data['system_settings']['num_edge_servers']
            cloud_list = config_data['servers'].get('cloud_servers', [])
            Config.NUM_CLOUD_SERVERS = len(cloud_list) 
            Config.TIME_SLOT_DURATION = config_data['system_settings']['duration_seconds']
            Config.CLOUD_F_MAX = config_data['system_settings']['cloud_max_freq_Hz']
            Config.EDGE_F_MAX = config_data['system_settings']['edge_max_freq_Hz']
            Config.EDGE_P_MAX = config_data['system_settings']['edge_max_trans_power_W']
            Config.PHI = config_data['system_settings']['cpu_cycles_per_bit']
            Config.ZETA = config_data['system_settings']['effective_capacitance_Zeta']
            Config.BANDWIDTH = config_data['system_settings']['edge_bandwith_Hz']

            Config.NOISE_POWER = 10 ** ((config_data['system_settings']['noise_power_dBm'] - 30) / 10)
            Config.G_IJ = config_data['system_settings']['channel_gain_peer']
            Config.G_IC = config_data['system_settings']['channel_gain_cloud']
            Config.V = config_data['system_settings']['trade_off_V']

            self.edge_servers_metadata = config_data['servers']['edge_servers']
            self.cloud_servers_metadata = config_data['servers']['cloud_servers']

            # Populate Alphas directly to Config class for global access
            Config.ALPHAS = [] # Reset first
            for s in self.edge_servers_metadata:
                Config.ALPHAS.append(s['alpha'])
            
            if self.cloud_servers_metadata:
                Config.ALPHA_CLOUD = self.cloud_servers_metadata[0]['alpha']

        except Exception as e:
            print(f"Failed to parse config.json in _load_config: {e}")
            raise e

    def _load_topology(self, json_path):
        """Reads config.json to build the edge server topology."""
        edge_graph = {}
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Error: {json_path} not found.")

        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            matrix = config_data.get('topology', {}).get('matrix', [])
            num_servers = len(matrix)
            
            for i in range(num_servers):
                curr_name = self.edge_servers_metadata[i]['name']
                neighbors = []
                for j in range(num_servers):
                    if i != j and matrix[i][j] == 1:
                        neighbor_name = self.edge_servers_metadata[j]['name']
                        neighbors.append(neighbor_name)
                edge_graph[curr_name] = neighbors
            
        except Exception as e:
            print(f"Failed to parse topology in _load_topology: {e}")
            raise e

        return edge_graph

    def _load_tasks(self, csv_path):
        """Reads data_arrival.csv."""
        task_arrival = {}
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: {csv_path} not found.")

        try:
            df = pd.read_csv(csv_path)
            for col in df.columns:
                if "Edge" in col:
                    clean_name = col.replace('_', ' ').strip()
                    if "EdgeServer" in clean_name:
                         clean_name = clean_name.replace("EdgeServer", "Edge Server")
                    
                    values_bits = df[col].values
                    task_arrival[clean_name] = values_bits.tolist()
            
        except Exception as e:
            print(f"Failed to parse data_arrival.csv: {e}")
            raise e
            
        return task_arrival

    def _load_carbon(self, csv_path):
        """
        Reads carbon_data.csv based on City Names defined in Config.
        """
        ci_history = {}
        ci_predict = {}
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: {csv_path} not found.")

        try:
            df = pd.read_csv(csv_path)
            df.columns = df.columns.str.strip()

            def calculate_alpha_pred(actuals, alpha):
                preds = []
                if not actuals:
                    return preds
                last_pred = actuals[0]
                preds.append(last_pred)
                for t in range(1, len(actuals)):
                    new_pred = alpha * actuals[t-1] + (1 - alpha) * last_pred
                    preds.append(new_pred)
                    last_pred = new_pred
                return preds
            
            for server_info in self.edge_servers_metadata:
                s_name = server_info['name']       # e.g., "Edge Server 1"
                s_city = server_info['city_name']  # e.g., "Seattle (OR)"
                s_alpha = server_info['alpha']
                
                if s_city in df.columns:
                    vals = df[s_city].values.tolist()
                    ci_history[s_name] = vals
                    ci_predict[s_name] = calculate_alpha_pred(vals, s_alpha)
                else:
                    print(f"Warning: City '{s_city}' for {s_name} not found in Carbon CSV columns: {df.columns.tolist()}")
            
            for server_info in self.cloud_servers_metadata:
                c_name = server_info['name']       # e.g., "Cloud Server 1"
                c_city = server_info['city_name']  # e.g., "Berkeley County (SC)"
                c_alpha = server_info['alpha']
                
                target_col = None
                if c_city in df.columns:
                    target_col = c_city
                else:
                    import difflib
                    matches = difflib.get_close_matches(c_city, df.columns, n=1, cutoff=0.8)
                    if matches:
                        target_col = matches[0]
                        print(f"Notice: Mapped config city '{c_city}' to CSV column '{target_col}'")
                
                if target_col:
                    vals = df[target_col].values.tolist()
                    ci_history[c_name] = vals
                    ci_predict[c_name] = calculate_alpha_pred(vals, c_alpha)
                else:
                     print(f"Warning: City '{c_city}' for {c_name} not found in Carbon CSV.")

        except Exception as e:
            print(f"Failed to parse carbon_data.csv: {e}")
            raise e
            
        return ci_history, ci_predict