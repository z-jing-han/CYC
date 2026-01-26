import json
import csv
import numpy as np
import random
import argparse
import os
import sys

def parse_arguments():
    parser = argparse.ArgumentParser(description="Traffic Data Generator")
    parser.add_argument('--dir', type=str, required=True, 
                        help="Target directory containing config.json. The output data_arrival.csv will also be saved here.")
    args = parser.parse_args()
    return args

def generate_traffic_data(config_path, output_path):
    try:
        if not os.path.exists(config_path):
             print(f"Error: Config file not found at {config_path}")
             return

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            
    except FileNotFoundError:
        print(f"File Not Found Error: {config_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from {config_path}")
        return
    
    try:
        num_timeslots = config.get('system_settings', {}).get('num_timeslots', 100)
        num_servers = config.get('system_settings', {}).get('num_edge_servers', 5)
        
        model_config = config.get('data_arrival_model', {})
        selected_model = model_config.get('selected_model', 'poisson-distribution')
    except AttributeError:
        print("Error: Invalid config structure.")
        return
    
    all_servers_data = []

    if selected_model == 'on-off-model':
        params = model_config['models']['on-off-model']

        p_on_to_off = params['transition_probability']['on_to_off']
        p_off_to_on = params['transition_probability']['off_to_on']

        on_mean = params['on_state_params']['mean_bits']
        on_std = params['on_state_params']['std_dev_bits']
        
        off_mean = params['off_state_params']['mean_bits']
        off_std = params['off_state_params']['std_dev_bits']

        for server_idx in range(num_servers):
            server_trace = []
            # Random initial state
            is_on = random.choice([True, False])
            
            for _ in range(num_timeslots):
                current_val = 0.0
                if is_on:
                    current_val = np.random.normal(on_mean, on_std)
                    if random.random() < p_on_to_off:
                        is_on = False
                else:
                    current_val = np.random.normal(off_mean, off_std)
                    if random.random() < p_off_to_on:
                        is_on = True
                
                server_trace.append(int(max(0.0, current_val)))
            
            all_servers_data.append(server_trace)

    elif selected_model == 'poisson-distribution':
        params = model_config['models'].get('poisson-distribution', {'lambda_bits': 5e6})
        lam = params.get('lambda_bits', 5e6)

        for server_idx in range(num_servers):
            server_trace = np.random.poisson(lam, num_timeslots)
            all_servers_data.append(server_trace)
    
    else:
        print(f"Error: Unknown select model {selected_model}")
        return
    
    transposed_data = np.array(all_servers_data).T

    headers = [f"Edge_Server_{i+1}" for i in range(num_servers)]

    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(transposed_data)
    except Exception as e:
        print(f"Error writing output file: {e}")
    

if __name__ == "__main__":
    args = parse_arguments()
    
    target_dir = args.dir
    
    if not os.path.exists(target_dir):
        print(f"Error: Target directory '{target_dir}' does not exist.")
        sys.exit(1)
    
    config_file = os.path.join(target_dir, 'config.json')
    output_file = os.path.join(target_dir, 'data_arrival.csv')
    
    generate_traffic_data(config_file, output_file)