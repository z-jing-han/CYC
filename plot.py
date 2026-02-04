import argparse
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 引入 DataLoader
from data_loader import DataLoader

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

def plot_carbon_intensity(ci_history, figures_dir, config_data=None):
    """
    Plots carbon intensity traces for all servers.
    Legend is horizontal, expanded to fill the plot width, and placed below the title.
    """
    
    # --- Debug: 檢查數據是否真的有傳進來 ---
    if not ci_history:
        print("[Plot Warning] ci_history is empty! Skipping carbon intensity plot.")
        return
    
    # 檢查是否有任何有效的 Trace (長度 > 0)
    valid_traces = [t for t in ci_history.values() if len(t) > 0]
    if not valid_traces:
        print(f"[Plot Warning] ci_history has keys {list(ci_history.keys())} but all traces are empty.")
        return
    # ---------------------------------------

    # --- 建立標籤映射字典 ---
    label_map = {}
    if config_data:
        try:
            # 處理 Edge Servers
            for s in config_data.get('servers', {}).get('edge_servers', []):
                short_name = s['name'].replace("Edge Server", "Edge").strip()
                label_map[s['name']] = f"{s['city_name']} ({short_name})"
            
            # 處理 Cloud Servers
            cloud_servers = config_data.get('servers', {}).get('cloud_servers', [])
            if cloud_servers:
                for s in cloud_servers:
                    label_map[s['name']] = f"{s['city_name']} (Cloud)"
        except Exception as e:
            print(f"[Plot Warning] Failed to create label map from config: {e}. Using default names.")

    # [調整] 加大寬度 (15, 8) 以確保橫向圖例有足夠空間展開而不擁擠
    plt.figure(figsize=(15, 8)) 
    
    plotted_something = False
    cloud_plotted = False

    # Loop 1: Edge Servers
    for s_name, traces in ci_history.items():
        if "Edge" in s_name and len(traces) > 0:
            label = label_map.get(s_name, s_name)
            plt.plot(traces, label=label, alpha=0.8, linewidth=2)
            plotted_something = True

    # Loop 2: Cloud Servers (Cloud 只畫一次，且樣式正常)
    for s_name, traces in ci_history.items():
        if "Cloud" in s_name and len(traces) > 0:
            if not cloud_plotted:
                label = label_map.get(s_name, s_name)
                # Cloud 樣式與 Edge 相同
                plt.plot(traces, label=label, alpha=0.8, linewidth=2)
                
                plotted_something = True
                cloud_plotted = True

    # [調整] 設定標題與軸標籤
    # pad=60: 將標題向上推，留出中間的空間給圖例
    plt.title("Carbon Intensity Traces Across Locations", fontsize=18, fontweight='bold', pad=60) 
    
    plt.xlabel("Time Slot", fontsize=14)
    plt.ylabel("Intensity (gCO2/kWh)", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    if plotted_something:
        # [修改重點] 圖例設定：對齊數據寬度
        # bbox_to_anchor=(0., 1.02, 1., .102):
        #   前兩個參數 (0, 1.02) 定義左下角座標 (在圖表區域上方一點點)
        #   第三個參數 (1.) 定義寬度為 100% (與圖表同寬)
        # mode="expand": 強制圖例內容平均分散填滿這個寬度
        # ncol=6: 設定為 6 欄 (5 Edge + 1 Cloud)，確保排成一列
        
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
        # print(f"[Plot] Saved carbon intensity trace to {save_path}")
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

def process_and_plot_simulation_details(csv_dir, figures_dir):
    algo_groups = {
        'dwpa': ['DWPA', 'DWPALF', 'DWPAVO', 'DWPAHF'],
        'competitor': ['DWPA', 'DOLA22', 'ICSOC19', 'YCL24']
    }

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
                        data_store['Queue'][entity_key][algo_name] = df[col_name] / 1e6 # bits to Mb
                elif config['type'] == 'sum':
                    cols = [c for c in df.columns if ('Edge' in c or 'Cloud' in c) and ('Q_Post' in c or 'Queue' in c)]
                    if cols:
                        data_store['Queue'][entity_key][algo_name] = df[cols].sum(axis=1) / 1e6

        except Exception as e:
            print(f"Error processing file {fpath}: {e}")
    
    def plot_metric_group(metric_name, output_folder, y_label, file_suffix, allowed_algos):
        """
        metric_name: 'Carbon' or 'Queue'
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
        
        ensure_dir(carbon_dir)
        ensure_dir(queue_dir)

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
            y_label='Queue Length (Mb)',
            file_suffix='queue',
            allowed_algos=allowed_list
        )

if __name__ == "__main__":
    # 1. 讀取參數
    args = parse_arguments()
    
    figures_dir = os.path.join(args.output_dir, 'figures')
    ensure_dir(figures_dir)
    
    stats_csv_dir = os.path.join(args.output_dir, 'csv')

    # 2. 建構正確的完整路徑 (指向 Data/Base_Input)
    ci_input_path = os.path.join(args.input_dir, 'carbon_intensity.csv')
    config_input_path = os.path.join(args.input_dir, 'config.json')
    task_input_path = os.path.join(args.input_dir, 'data_arrival.csv') 

    # 新增：讀取 Config Data 以供繪圖標籤使用
    config_data_for_plot = None
    try:
        with open(config_input_path, 'r', encoding='utf-8') as f:
            config_data_for_plot = json.load(f)
    except Exception as e:
        print(f"[Warning] Failed to load config.json for plotting labels: {e}")

    # 3. 初始化 DataLoader 並傳入所有路徑
    try:
        # print(f"Loading data via DataLoader from: {args.input_dir} ...")
        
        loader = DataLoader(
            config_path=config_input_path, 
            carbon_path=ci_input_path,
            task_path=task_input_path
        )
        
        _, ci_history, _, _ = loader.load_data()
        
        # 繪圖 (傳入 config_data)
        plot_carbon_intensity(ci_history, figures_dir, config_data=config_data_for_plot)
        
    except Exception as e:
        print(f"[Warning] Failed to load data via DataLoader: {e}")
        # import traceback
        # traceback.print_exc() 
        print("Skipping Carbon Intensity Trace plot.")

    # 4. 處理模擬結果 CSV
    if os.path.exists(stats_csv_dir):
        process_and_plot_simulation_details(stats_csv_dir, figures_dir)
    else:
        print(f"[Error] CSV directory not found: {stats_csv_dir}")