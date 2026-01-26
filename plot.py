import argparse
import os
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    sns.set_theme(style="whitegrid")
except:
    plt.style.use('seaborn-whitegrid')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Simulation Result Plotter")
    parser.add_argument('--input_dir', type=str, required=True, 
                        help="Directory containing raw input data (e.g., carbon_intensity.csv)")
    parser.add_argument('--output_dir', type=str, required=True, 
                        help="Directory where 'csv' folder is located and where 'figures' folder will be created")
    
    args = parser.parse_args()
    return args

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def plot_carbon_intensity(input_path, output_path):
    if not os.path.exists(input_path):
        print(f"Error: File not found: {input_path}")
        return

    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return
    
    plt.figure(figsize=(14, 8))
    
    # Label Mapping
    labels = {
        'Edge_1_History': 'Seattle (OR)',
        'Edge_4_History': 'New York (NY)',
        'Edge_3_History': 'Chicago (PJM)',
        'Edge_5_History': 'Kansas City (KS)',
        'Cloud_Shared_History': 'Berkeley County (SC)',
        'Edge_2_History': 'Los Angeles (CA)'
    }

    for col, label in labels.items():
        if col in df.columns:
            plt.plot(df.index, df[col], label=label, linewidth=2, alpha=0.8)
    
    plt.title('Carbon Intensity Over Time (24 Hours)', fontsize=18, fontweight='bold')
    plt.xlabel('Time Slot (5-min intervals)', fontsize=14)
    plt.ylabel('Carbon Intensity (gCO2eq/kWh)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300)
    except Exception as e:
        print(f"Error saving carbon intensity figure: {e}")
    finally:
        plt.close()

def get_algorithm_name(filename):
    """Extract algorithm 'stats_DWPA.csv' -> 'DWPA'"""
    basename = os.path.basename(filename)
    name = os.path.splitext(basename)[0]
    if name.startswith('stats_'):
        return name.replace('stats_', '')
    return name

def process_and_plot_simulation_details(csv_dir, figures_dir):
    # Define output subdir
    carbon_dir = os.path.join(figures_dir, 'Carbon_Emission')
    queue_dir = os.path.join(figures_dir, 'Queue_Len')
    ensure_dir(carbon_dir)
    ensure_dir(queue_dir)

    # find all csv
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    if not csv_files:
        print(f"[Warning] No CSV files found in {csv_dir}")
        return
    
    entities = {
        'Total':  {'type': 'sum', 'label': 'Total System'},
        'Edge1':  {'type': 'single', 'prefix': 'Edge1', 'label': 'Edge Server 1'},
        'Edge2':  {'type': 'single', 'prefix': 'Edge2', 'label': 'Edge Server 2'},
        'Edge3':  {'type': 'single', 'prefix': 'Edge3', 'label': 'Edge Server 3'},
        'Edge4':  {'type': 'single', 'prefix': 'Edge4', 'label': 'Edge Server 4'},
        'Edge5':  {'type': 'single', 'prefix': 'Edge5', 'label': 'Edge Server 5'},
        'Cloud':  {'type': 'single', 'prefix': 'Cloud', 'label': 'Cloud Server'}
    }

    data_store = {
        'Carbon': {k: {} for k in entities.keys()},
        'Queue': {k: {} for k in entities.keys()}
    }
    
    time_index = None

    for fpath in csv_files:
        algo_name = get_algorithm_name(fpath)
        try:
            df = pd.read_csv(fpath)
            if time_index is None:
                time_index = df['TimeSlot'] if 'TimeSlot' in df.columns else df.index
            
            cloud_prefix = 'Cloud'
            
            for entity_key, config in entities.items():
                if config['type'] == 'single':
                    prefix = config['prefix']
                    col_name = next((c for c in df.columns if c.startswith(prefix) and 'Carbon' in c), None)
                    if col_name:
                        data_store['Carbon'][entity_key][algo_name] = df[col_name]
                
                elif config['type'] == 'sum':
                    cols = [c for c in df.columns if ('Edge' in c or 'Cloud' in c) and 'Carbon' in c]
                    if cols:
                        data_store['Carbon'][entity_key][algo_name] = df[cols].sum(axis=1)
                if config['type'] == 'single':
                    prefix = config['prefix']
                    col_name = next((c for c in df.columns if c.startswith(prefix) and ('Q_Post' in c or 'Queue' in c)), None)
                    if col_name:
                        # bits to Mb
                        data_store['Queue'][entity_key][algo_name] = df[col_name] / 1e6
                
                elif config['type'] == 'sum':
                    cols = [c for c in df.columns if ('Edge' in c or 'Cloud' in c) and ('Q_Post' in c or 'Queue' in c)]
                    if cols:
                        data_store['Queue'][entity_key][algo_name] = df[cols].sum(axis=1) / 1e6

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
    
    def plot_metric_group(metric_name, output_folder, y_label, file_suffix):
        """
        metric_name: 'Carbon' or 'Queue'
        output_folder: folder path
        y_label: label for Y axis
        file_suffix: 'carbon' or 'queue'
        """
        
        for entity_key, algo_data in data_store[metric_name].items():
            if not algo_data:
                continue

            plt.figure(figsize=(10, 6))
            
            for algo, series in algo_data.items():
                plt.plot(time_index, series, label=algo, linewidth=2, alpha=0.8)
            
            entity_label = entities[entity_key]['label']
            plt.title(f'{entity_label} - {metric_name}', fontsize=16, fontweight='bold')
            plt.xlabel('Time Slot', fontsize=12)
            plt.ylabel(y_label, fontsize=12)
            plt.legend(title="Algorithm", loc='upper right', frameon=True)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            filename = f"{entity_key}_{file_suffix}.png"
            out_path = os.path.join(output_folder, filename)
            try:
                plt.savefig(out_path, dpi=300)
            except Exception as e:
                print(f"  -> Error saving {filename}: {e}")
            finally:
                plt.close()
    
    plot_metric_group(
        metric_name='Carbon',
        output_folder=carbon_dir,
        y_label='Carbon Emission (g)',
        file_suffix='carbon'
    )

    plot_metric_group(
        metric_name='Queue',
        output_folder=queue_dir,
        y_label='Queue Length (Mb)',
        file_suffix='queue'
    )

if __name__ == "__main__":
    args = parse_arguments()
    
    figures_dir = os.path.join(args.output_dir, 'figures')
    ensure_dir(figures_dir)
    
    stats_csv_dir = os.path.join(args.output_dir, 'csv')

    ci_input_path = os.path.join(args.input_dir, 'carbon_intensity.csv')
    ci_output_path = os.path.join(figures_dir, 'carbon_intensity.png')
    
    if os.path.exists(ci_input_path):
        plot_carbon_intensity(ci_input_path, ci_output_path)
    else:
        print(f"[Info] Carbon intensity file not found at {ci_input_path}, skipping.")
    
    if os.path.exists(stats_csv_dir):
        process_and_plot_simulation_details(stats_csv_dir, figures_dir)
    else:
        print(f"[Error] CSV directory not found: {stats_csv_dir}")