import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

from data_loader import DataLoader

try:
    sns.set_theme(style="whitegrid")
except:
    plt.style.use('seaborn-whitegrid')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulation Result Plotter")
    parser.add_argument('--input_dir', type=str, required=False, 
                        help="Directory containing raw input data (e.g., carbon_intensity.csv)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Directory where 'csv' folder is located and where 'figures' folder will be created")
    
    parser.add_argument('--plot_tradeoff', action='store_true', 
                        help="Plot trade-off for V parameter experiments")
    parser.add_argument('--tradeoff_data_dir', type=str, 
                        help="Directory containing all V_*_Input and V_*_Output folders")
    
    args = parser.parse_args()
    return args

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_tradeoff(tradeoff_data_dir, output_dir):
    import glob
    
    input_dirs = glob.glob(os.path.join(tradeoff_data_dir, 'V_*_Input'))
    results = []
    
    for in_dir in input_dirs:
        config_path = os.path.join(in_dir, 'config.json')
        dir_name = os.path.basename(in_dir)
        tag = dir_name.replace('V_', '').replace('_Input', '')
        out_dir = os.path.join(tradeoff_data_dir, f'V_{tag}_Output')
        
        csv_path = os.path.join(out_dir, 'csv', 'stats_DWPA.csv') 
        
        if not os.path.exists(config_path) or not os.path.exists(csv_path):
            continue
            
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            v_val = config.get('system_settings', {}).get('trade_off_V', None)
            
        if v_val is not None:
            try:
                df = pd.read_csv(csv_path)
                carbon_cols = [c for c in df.columns if 'Carbon(g)' in c]
                total_carbon = df[carbon_cols].sum().sum()
                
                queue_cols = [c for c in df.columns if 'Q_Post(bits)' in c]
                total_queue = df[queue_cols].sum().sum()
                
                results.append((v_val, total_carbon, total_queue))
            except Exception as e:
                print(f"[Warning] Failed to process {csv_path}: {e}")
                
    if not results:
        print("No data found for trade-off plot.")
        return
    
    results.sort(key=lambda x: x[0])
    v_vals = [x[0] for x in results]
    carbons = [x[1] for x in results]
    queues = [x[2] for x in results]
    
    ensure_dir(output_dir)
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:red'
    ax1.set_xlabel('V Parameter (Log Scale)')
    ax1.set_ylabel('Total Carbon Emission (g)', color=color1)
    ax1.plot(v_vals, carbons, marker='o', color=color1, label='Carbon Emission')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log')
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Total Queue Length (bits)', color=color2)  
    ax2.plot(v_vals, queues, marker='s', color=color2, label='Queue Length')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Trade-off between Carbon Emission and Queue Length under varying V')
    fig.tight_layout()  
    plt.savefig(os.path.join(output_dir, 'tradeoff_V_dual_axis.png'), dpi=300)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(carbons, queues, marker='^', linestyle='-', color='purple')
    for i, txt in enumerate(v_vals):
        plt.annotate(f'V={txt:.0e}', (carbons[i], queues[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    plt.xlabel('Total Carbon Emission (g)')
    plt.ylabel('Total Queue Length (bits)')
    plt.title('Pareto Trade-off: Carbon Emission vs Queue Length')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tradeoff_pareto.png'), dpi=300)
    plt.close()
    
    print(f"Trade-off plots successfully saved to {output_dir}")

def plot_carbon_intensity(ci_history, figures_dir, config_data=None):
    """
    Plots carbon intensity traces for all servers.
    Legend is horizontal, expanded to fill the plot width, and placed below the title.
    """
    
    if not ci_history:
        print("[Plot Warning] ci_history is empty! Skipping carbon intensity plot.")
        return
    
    valid_traces = [t for t in ci_history.values() if len(t) > 0]
    if not valid_traces:
        print(f"[Plot Warning] ci_history has keys {list(ci_history.keys())} but all traces are empty.")
        return
    
    label_map = {}
    if config_data:
        try:
            for s in config_data.get('servers', {}).get('edge_servers', []):
                short_name = s['name'].replace("Edge Server", "Edge").strip()
                label_map[s['name']] = f"{s['city_name']} ({short_name})"
            
            cloud_servers = config_data.get('servers', {}).get('cloud_servers', [])
            if cloud_servers:
                for s in cloud_servers:
                    label_map[s['name']] = f"{s['city_name']} (Cloud)"
        except Exception as e:
            print(f"[Plot Warning] Failed to create label map from config: {e}. Using default names.")
    
    plt.figure(figsize=(15, 8)) 
    
    plotted_something = False
    cloud_plotted = False

    # Loop 1: Edge Servers
    for s_name, traces in ci_history.items():
        if "Edge" in s_name and len(traces) > 0:
            label = label_map.get(s_name, s_name)
            plt.plot(traces, label=label, alpha=0.8, linewidth=2)
            plotted_something = True

    # Loop 2: Cloud Servers
    for s_name, traces in ci_history.items():
        if "Cloud" in s_name and len(traces) > 0:
            if not cloud_plotted:
                label = label_map.get(s_name, s_name)
                plt.plot(traces, label=label, alpha=0.8, linewidth=2)
                
                plotted_something = True
                cloud_plotted = True
    
    plt.title("Carbon Intensity Traces Across Locations", fontsize=18, fontweight='bold', pad=60) 
    
    plt.xlabel("Time Slot", fontsize=14)
    plt.ylabel("Intensity (gCO2/kWh)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if plotted_something:
        plt.legend(
            bbox_to_anchor=(0., 1.02, 1., .102), 
            loc='lower left',
            ncol=6, 
            mode="expand", 
            borderaxespad=0.,
            fontsize=12,
            frameon=True
        )
        
        plt.tight_layout()
        
        save_path = os.path.join(figures_dir, "carbon_intensity_trace.png")
        plt.savefig(save_path, dpi=300)
    else:
        print("[Plot Warning] No 'Edge' or 'Cloud' servers found in ci_history to plot.")
    
    plt.close()

def get_algorithm_name(filename):
    """Extract algorithm 'stats_DWPA.csv' -> 'DWPA'"""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    if name.startswith('stats_'):
        return name.replace('stats_', '')
    return name

def process_and_plot_simulation_details(csv_dir, figures_dir, config_data):
    algo_groups = config_data.get('algorithms', {}).get('plot_groups', {
        'default': ['DWPA']
    })

    # find all csv
    csv_files = glob.glob(os.path.join(csv_dir, "stats_*.csv"))
    if not csv_files:
        print(f"[Warning] No CSV files found in {csv_dir}")
        return
    
    entities = {
        'Total':  {'carbon_col': 'Total_Carbon(g)', 'queue_col': 'Avg_System_Q(bits)', 'label': 'Total System'},
        'Edge1':  {'carbon_col': 'Edge1_Carbon(g)', 'queue_col': 'Edge1_Q_Post(bits)', 'label': 'Edge Server 1'},
        'Edge2':  {'carbon_col': 'Edge2_Carbon(g)', 'queue_col': 'Edge2_Q_Post(bits)', 'label': 'Edge Server 2'},
        'Edge3':  {'carbon_col': 'Edge3_Carbon(g)', 'queue_col': 'Edge3_Q_Post(bits)', 'label': 'Edge Server 3'},
        'Edge4':  {'carbon_col': 'Edge4_Carbon(g)', 'queue_col': 'Edge4_Q_Post(bits)', 'label': 'Edge Server 4'},
        'Edge5':  {'carbon_col': 'Edge5_Carbon(g)', 'queue_col': 'Edge5_Q_Post(bits)', 'label': 'Edge Server 5'},
        'Cloud1': {'carbon_col': 'Cloud1_Carbon(g)', 'queue_col': 'Cloud1_Q_Post(bits)', 'label': 'Cloud Server 1'},
        'Cloud2': {'carbon_col': 'Cloud2_Carbon(g)', 'queue_col': 'Cloud2_Q_Post(bits)', 'label': 'Cloud Server 2'},
        'Cloud3': {'carbon_col': 'Cloud3_Carbon(g)', 'queue_col': 'Cloud3_Q_Post(bits)', 'label': 'Cloud Server 3'},
        'Cloud4': {'carbon_col': 'Cloud4_Carbon(g)', 'queue_col': 'Cloud4_Q_Post(bits)', 'label': 'Cloud Server 4'},
        'Cloud5': {'carbon_col': 'Cloud5_Carbon(g)', 'queue_col': 'Cloud5_Q_Post(bits)', 'label': 'Cloud Server 5'}
    }

    data_store = {
        'Carbon': {k: {} for k in entities.keys()},
        'Queue': {k: {} for k in entities.keys()},
        'Fairness': {'Total': {}}  # 系統層級指標
    }
    
    time_index = None

    for fpath in csv_files:
        algo_name = get_algorithm_name(fpath)
        try: 
            df = pd.read_csv(fpath)
            if time_index is None:
                time_index = df['TimeSlot'] if 'TimeSlot' in df.columns else df.index
            
            for entity_key, config in entities.items():
                carbon_col = config['carbon_col']
                queue_col = config['queue_col']
                
                if carbon_col in df.columns:
                    data_store['Carbon'][entity_key][algo_name] = df[carbon_col]
                
                if queue_col in df.columns:
                    data_store['Queue'][entity_key][algo_name] = df[queue_col] / (8 * 1024 * 1024)

            # Edge Servers Jain's Fairness Index
            edge_q_cols = [c for c in df.columns if 'Q_Post(bits)' in c and 'Edge' in c]
            if edge_q_cols:
                sum_q = df[edge_q_cols].sum(axis=1)
                sum_q_sq = (df[edge_q_cols] ** 2).sum(axis=1)
                n_edges = len(edge_q_cols)
                jfi = (sum_q ** 2) / (n_edges * sum_q_sq)
                jfi = jfi.fillna(1.0)
                data_store['Fairness']['Total'][algo_name] = jfi

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
    
    def plot_metric_group(metric_name, output_folder, y_label, file_suffix, allowed_algos):
        """
        metric_name: 'Carbon', 'Queue', or 'Fairness'
        output_folder: specific folder path (e.g. figures/dwpa/Carbon_Emission)
        allowed_algos: list of algorithms to plot for this group
        """
        
        for entity_key, algo_data in data_store[metric_name].items():
            # Filter: Only include algos that are in the allowed list AND exist in data
            filtered_data = {algo: series for algo, series in algo_data.items() if algo in allowed_algos}
            
            if not filtered_data:
                continue

            plt.figure(figsize=(10, 6))
            
            # Sort keys to ensure consistent color assignment if needed, or just iterate
            for algo in sorted(filtered_data.keys()):
                series = filtered_data[algo]
                plt.plot(time_index, series, label=algo, linewidth=2, alpha=0.8)
            
            entity_label = entities[entity_key]['label']
            plt.title(f'{entity_label} - {metric_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Time Slot', fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.legend(title="Algorithm", loc='upper right', frameon=True)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            if metric_name == 'Fairness':
                plt.ylim(-0.05, 1.05)
                
            plt.tight_layout()

            filename = f"{entity_key}_{file_suffix}.png"
            out_path = os.path.join(output_folder, filename)
            try:
                plt.savefig(out_path, dpi=300)
            except Exception as e:
                print(f"  -> Error saving {filename}: {e}")
            finally:
                plt.close()
    
    # 2. Iterate through groups and plot
    for group_name, allowed_list in algo_groups.items():
        group_base_dir = os.path.join(figures_dir, group_name)
        carbon_dir = os.path.join(group_base_dir, 'Carbon_Emission')
        queue_dir = os.path.join(group_base_dir, 'Queue_Len')
        fairness_dir = os.path.join(group_base_dir, 'Queue_Fairness')
        
        ensure_dir(carbon_dir)
        ensure_dir(queue_dir)
        ensure_dir(fairness_dir)

        plot_metric_group(
            metric_name='Carbon',
            output_folder=carbon_dir,
            y_label='Carbon Emission (g)',
            file_suffix='carbon',
            allowed_algos=allowed_list
        )

        plot_metric_group(
            metric_name='Queue',
            output_folder=queue_dir,
            y_label='Queue Length (MB)',
            file_suffix='queue',
            allowed_algos=allowed_list
        )
        
        plot_metric_group(
            metric_name='Fairness',
            output_folder=fairness_dir,
            y_label="Jain's Fairness Index",
            file_suffix='fairness',
            allowed_algos=allowed_list
        )

if __name__ == "__main__":
    args = parse_arguments()

    if args.plot_tradeoff:
        if not args.tradeoff_data_dir:
            print("Error: --tradeoff_data_dir is required for plotting trade-off.")
            exit(1)
        plot_tradeoff(args.tradeoff_data_dir, args.output_dir)
        exit(0)
    
    figures_dir = os.path.join(args.output_dir, 'figures')
    ensure_dir(figures_dir)
    
    stats_csv_dir = os.path.join(args.output_dir, 'csv')

    ci_input_path = os.path.join(args.input_dir, 'carbon_intensity.csv')
    config_input_path = os.path.join(args.input_dir, 'config.json')
    task_input_path = os.path.join(args.input_dir, 'data_arrival.csv') 

    config_data_for_plot = None
    try:
        with open(config_input_path, 'r', encoding='utf-8') as f:
            config_data_for_plot = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load config.json for plotting labels: {e}")
    
    try:
        loader = DataLoader(
            config_path=config_input_path, 
            carbon_path=ci_input_path,
            task_path=task_input_path
        )
        
        _, ci_history, _, _ = loader.load_data()
        plot_carbon_intensity(ci_history, figures_dir, config_data=config_data_for_plot)
        
    except Exception as e:
        print(f"[Warning] Failed to load data via DataLoader: {e}")
        print("Skipping Carbon Intensity Trace plot.")
    
    if os.path.exists(stats_csv_dir):
        process_and_plot_simulation_details(stats_csv_dir, figures_dir, config_data_for_plot)
    else:
        print(f"[Error] CSV directory not found: {stats_csv_dir}")