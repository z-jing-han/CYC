import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) > 1:
    x_labels = sys.argv[1:]
else:
    x_labels = ['5e-8', '1e-7', '5e-7', '1e-6', '5e-6']

file_paths = [f"Data/Thesis_Output-{label}/csv/stats_MARPPO_XT_CTDE.csv" for label in x_labels]

total_carbon_list = []
avg_queue_list = []
valid_x_labels = []

dfs_for_timeseries = [] 

for label, file_path in zip(x_labels, file_paths):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        df['Avg_System_Q(MB)'] = df['Avg_System_Q(bits)'] / (8 * 1024 * 1024)
        
        total_carbon = df['Total_Carbon(g)'].sum()
        avg_queue = df['Avg_System_Q(MB)'].mean()
        
        total_carbon_list.append(total_carbon)
        avg_queue_list.append(avg_queue)
        valid_x_labels.append(label)
        dfs_for_timeseries.append((label, df))
    else:
        print(f"Warning: Can not find {file_path}, Skip")

if total_carbon_list:
    os.makedirs('Data', exist_ok=True)
    
    plot_x_labels = []
    for x in valid_x_labels:
        val = float(x) / 0.1
        sci_str = f"{val:.1e}"
        base, exp = sci_str.split('e')
        latex_label = rf"${float(base):g} \times 10^{{{int(exp)}}}$"
        plot_x_labels.append(latex_label)
        
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:red'
    ax1.set_xlabel(r'$\beta_2/\beta_1$ ratio', fontsize=12)
    ax1.set_ylabel('Total Carbon (g) [Sum]', color=color1, fontsize=12)
    line1 = ax1.plot(plot_x_labels, total_carbon_list, marker='o', color=color1, linewidth=2, label='Total Carbon')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Average Queue (MB) [Mean]', color=color2, fontsize=12)
    line2 = ax2.plot(plot_x_labels, avg_queue_list, marker='s', color=color2, linewidth=2, label='Avg Queue')
    ax2.tick_params(axis='y', labelcolor=color2)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', ncol=2)
    
    save_path1 = os.path.join('Data', 'tradeoff_plot.pdf')
    fig1.savefig(save_path1, bbox_inches='tight')
    plt.close(fig1)
    print(f"Trade-off plot successfully saved to {save_path1}")

    # =========================================================
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for label, df in dfs_for_timeseries:
        val = float(label) / 0.1
        sci_str = f"{val:.1e}"
        base, exp = sci_str.split('e')
        latex_label = rf"$\beta_2/\beta_1 = {float(base):g} \times 10^{{{int(exp)}}}$"
        time_axis = df.index 
        ax3.plot(time_axis, df['Avg_System_Q(MB)'], linewidth=1.2, label=latex_label, alpha=0.85)
        ax4.plot(time_axis, df['Total_Carbon(g)'], linewidth=1.2, label=latex_label, alpha=0.85)

    ax3.set_title('System Queue Length Over Time', fontsize=14, fontweight='bold')
    ax3.set_ylabel('Average Queue (MB)', fontsize=12)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=10)

    ax4.set_title('Total Carbon Emission Over Time', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Time Slot (Hours)', fontsize=12)
    ax4.set_ylabel('Total Carbon (g)', fontsize=12)
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='upper right', fontsize=10)

    fig2.tight_layout()
    save_path2 = os.path.join('Data', 'time_series_comparison.pdf')
    fig2.savefig(save_path2, bbox_inches='tight')
    plt.close(fig2)
    print(f"Time-series plots successfully saved to {save_path2}")

else:
    print("Can not find any data")