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

for label, file_path in zip(x_labels, file_paths):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        
        total_carbon = df['Total_Carbon(g)'].sum()
        avg_queue = df['Avg_System_Q(bits)'].mean()
        
        total_carbon_list.append(total_carbon)
        avg_queue_list.append(avg_queue)
        valid_x_labels.append(label)
    else:
        print(f"Warning: Can not find {file_path}, Skip")

if total_carbon_list:
    plot_x_labels = []
    for x in valid_x_labels:
        val = float(x) / 0.1
        sci_str = f"{val:.1e}"
        base, exp = sci_str.split('e')
        latex_label = rf"${float(base):g} \times 10^{{{int(exp)}}}$"
        plot_x_labels.append(latex_label)
    fig, ax1 = plt.subplots(figsize=(12, 6))
    color1 = 'tab:red'
    ax1.set_xlabel(r'$\beta_2/\beta_1$ ratio', fontsize=12)
    ax1.set_ylabel('Total Carbon (g) [Sum]', color=color1, fontsize=12)
    line1 = ax1.plot(plot_x_labels, total_carbon_list, marker='o', color=color1, linewidth=2, label='Total Carbon')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2 = ax1.twinx()  
    color2 = 'tab:blue'
    ax2.set_ylabel('Average Queue (bits) [Mean]', color=color2, fontsize=12)
    line2 = ax2.plot(plot_x_labels, avg_queue_list, marker='s', color=color2, linewidth=2, label='Avg Queue')
    ax2.tick_params(axis='y', labelcolor=color2)
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper center', ncol=2)
    os.makedirs('Data', exist_ok=True)
    save_path = os.path.join('Data', 'tradeoff_plot.pdf')
    plt.savefig(save_path, bbox_inches='tight')
else:
    print("Can not find any data")