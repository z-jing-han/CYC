import argparse
import os
import sys
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
    
    labels = {
        'Edge_1_History': 'Seattle (OR)',
        'Edge_4_History': 'New York (NY)',
        'Edge_3_History': 'Chicago (PJM)',
        'Edge_5_History': 'Kansas City (KS)',
        'Cloud_Shared_History': 'Berkeley County (SC)',
        'Edge_2_History': 'Cheyenne (WY)'
    }

    colors = {
        'Edge_1_History': '#1f77b4',
        'Edge_4_History': '#2ca02c',
        'Edge_3_History': '#ff7f0e',
        'Edge_5_History': '#d68910',
        'Cloud_Shared_History': '#17becf',
        'Edge_2_History': '#d62728'
    }

    x = df.get('TimeSlot', df.index)

    has_plotted = False
    for column, label in labels.items():
        if column in df.columns:
            plt.plot(x, df[column], label=label, color=colors.get(column), linewidth=2, alpha=0.8)
            has_plotted = True
    
    if not has_plotted:
        print("Warning: No matching columns found in carbon_intensity.csv to plot.")
        plt.close()
        return

    plt.xlabel('Time Slot', fontsize=12)
    plt.ylabel('Carbon Intensity (gCO2/kWh)', fontsize=12)
    plt.legend(bbox_to_anchor=(0.5, 1.08), loc='upper center', ncol=6, frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"[Success] Carbon Intensity plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")
    finally:
        plt.close()

def plot_simulation_comparison(csv_dir, output_path):

    file_pattern = os.path.join(csv_dir, "stats_*.csv")
    files = glob.glob(file_pattern)
    
    print(f"[Plotting] Searching for stats files in: {file_pattern}")
    
    if not files:
        print(f"No files found matching '{file_pattern}'. Skipping comparison plot.")
        return

    print(f"Found {len(files)} files: {[os.path.basename(f) for f in files]}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    for filepath in files:
        try:
            df = pd.read_csv(filepath)
            filename = os.path.basename(filepath)
            
            # --- 解析演算法名稱 ---
            # 優先嘗試提取 stats_ 與 .csv 中間的內容
            # 例如: stats_DWPA.csv -> DWPA
            # 例如: stats_QLearning_2023.csv -> QLearning_2023
            name_part = os.path.splitext(filename)[0] # remove extension
            if name_part.startswith('stats_'):
                algo_name = name_part[6:] # remove 'stats_'
            else:
                algo_name = name_part
            
            carbon_cols = [c for c in df.columns if 'Carbon' in c and 'Total' not in c]

            if not carbon_cols and 'Total_Carbon(g)' in df.columns:
                 total_carbon = df['Total_Carbon(g)']
            elif carbon_cols:
                 total_carbon = df[carbon_cols].sum(axis=1)
            else:
                print(f"Skipping {filename}: No Carbon columns found.")
                continue
            
            queue_cols = [c for c in df.columns if 'Q_Post' in c]
            
            x_axis = df.get('TimeSlot', df.index)
            
            # 計算 Queue 總合 (Mb)
            if queue_cols:
                total_queue_mb = df[queue_cols].sum(axis=1) / 1e6 
            elif 'Avg_System_Q(bits)' in df.columns:
                 # 如果只有平均值，乘以 Edge 數量 (假設為 5) 估算總和，或者直接畫平均
                 # 這裡為了通用性，若只有平均，我們畫平均但標註
                 total_queue_mb = df['Avg_System_Q(bits)'] / 1e6
                 algo_name += " (Avg)"
            else:
                 print(f"Skipping {filename}: No Queue columns found.")
                 continue
            
            ax1.plot(x_axis, total_carbon, label=algo_name, linewidth=2, alpha=0.8)
            ax2.plot(x_axis, total_queue_mb, label=algo_name, linewidth=2, alpha=0.8)

        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    
    ax1.set_title('Total Carbon Emission over Time', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Carbon Emission (g)', fontsize=12)
    ax1.legend(title="Algorithm", loc='upper right', frameon=True)
    ax1.grid(True, linestyle='--', alpha=0.7)

    ax2.set_title('Total Queue Length over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Time Slot', fontsize=12)
    ax2.set_ylabel('Queue Length (Mb)', fontsize=12)
    ax2.legend(title="Algorithm", loc='upper right', frameon=True)
    ax2.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    
    try:
        plt.savefig(output_path, dpi=300)
        print(f"[Success] Simulation comparison plot saved to: {output_path}")
    except Exception as e:
        print(f"Error saving comparison figure: {e}")
    finally:
        plt.close()

if __name__ == "__main__":
    args = parse_arguments()
    
    figures_dir = os.path.join(args.output_dir, 'figures')
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        print(f"Created figures directory: {figures_dir}")
    
    ci_input_path = os.path.join(args.input_dir, 'carbon_intensity.csv')
    ci_output_path = os.path.join(figures_dir, 'carbon_intensity.png')

    stats_csv_dir = os.path.join(args.output_dir, 'csv')
    stats_output_path = os.path.join(figures_dir, 'simulation_comparison.png')
    
    print("--- 1. Plotting Carbon Intensity ---")
    plot_carbon_intensity(ci_input_path, ci_output_path)
    
    print("\n--- 2. Plotting Simulation Stats ---")
    if os.path.exists(stats_csv_dir):
        plot_simulation_comparison(stats_csv_dir, stats_output_path)
    else:
        print(f"Error: Stats CSV directory not found: {stats_csv_dir}")
        print("Please run the simulation (main.py) first to generate results.")