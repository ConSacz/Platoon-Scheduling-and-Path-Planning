try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset', '-f')
except:
    pass
# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from utils.workspace_functions import load_mat
from utils.fitness_functions import sep_fitness, ori_fitness, fitness
from utils.graph_functions import create_graph

# %% DATA AUTO IMPORT
Trials = 50
algorithm_configs = {
    'TLBO': 'TLBO'
}
case_configs = {
    'N600': 'case_600',
    'N800': 'case_800',
    'N1000': 'case_1000',  
}

results = {}
ori_results = {}
sep_results = {}
runtimes = {}

for algo_name, algo_folder in algorithm_configs.items():
    results[algo_name] = {}
    ori_results[algo_name] = {}
    sep_results[algo_name] = {}
    runtimes[algo_name] = {}

    for case_name, case_folder in case_configs.items():
        weighted_costs = []
        ori_weighted_costs = []
        final_sep_costs = []
        final_runtime = []

        for i in range(Trials):
            folder_name = os.path.join("data", case_folder, algo_folder)
            file_name = f"{algo_folder}_{i}.mat"

            data = load_mat(folder_name, file_name)

            BestCostIt = data['BestCostIt'].flatten()  # đảm bảo là vector 1D
            weighted_fit = BestCostIt[-1]  # lấy iteration cuối
            
            ind = data['best']
            init = data['init']
            ARRIVAL_TIMES = data['ARRIVAL_TIMES']
            G, paths_SH, paths_HD = create_graph()
            sep_fit = sep_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
            ori_fit = ori_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)

            weighted_costs.append(weighted_fit)
            ori_weighted_costs.append(ori_fit)
            final_sep_costs.append(sep_fit)
            final_runtime.append(data['runtime'])
            
        results[algo_name][case_name] = np.array(weighted_costs)
        ori_results[algo_name][case_name] = np.array(ori_weighted_costs)
        sep_results[algo_name][case_name] = np.array(final_sep_costs)
        runtimes[algo_name][case_name] = np.array(final_runtime)

del algo_folder, algo_name, BestCostIt, case_folder, case_name, data, file_name, weighted_costs, weighted_fit, folder_name, i, final_runtime
del ARRIVAL_TIMES, final_sep_costs, G, ind, init, paths_HD, paths_SH, sep_fit, ori_weighted_costs, ori_fit

# %% scatter 2 fitness
fig, ax = plt.subplots(figsize=(8, 5))
markers = {
    'N600': 'o',
    'N800': 's',
    'N1000': '^'
}
colors = {
    'N600': 'blue',
    'N800': 'salmon',
    'N1000': 'green'
}
for algo in algorithm_configs:
    for idx, case in enumerate(case_configs):
        data = sep_results[algo][case]
        fuel = data[:, 0]
        wait = data[:, 1]

        ax.scatter(wait, fuel, label=case, alpha=0.7, marker=markers[case], color=colors[case])
    # ax.set_title(case, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    
ax.set_ylabel('Fuel Cost', fontsize=16)
ax.set_xlabel('Wait Cost', fontsize=16)
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07),fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del algo, ax, case, colors, data, fig, fuel, handles, idx, labels
del markers, wait
# %% fitness in TLBO
fig, ax = plt.subplots(figsize=(8, 5))

fuel_means = []
wait_means = []
ori_costs = []
reductions = []

for idx, case in enumerate(case_configs):

    data = sep_results['TLBO'][case]

    fuel_mean = np.mean(data[:, 0])
    wait_mean = np.mean(data[:, 1])

    final_cost = fuel_mean + wait_mean

    ori_cost = np.mean(ori_results['TLBO'][case])

    reduction = (ori_cost - final_cost) / ori_cost * 100

    fuel_means.append(fuel_mean)
    wait_means.append(wait_mean)
    ori_costs.append(ori_cost)
    reductions.append(reduction)

x = np.arange(len(case_configs))
ax.bar(x, fuel_means, width=0.6, label='Fuel Cost')
ax.bar(x, wait_means, width=0.6, bottom=fuel_means, label='Wait Cost')

for i in range(len(case_configs)):
    if i == 0:
        ax.hlines(ori_costs[i], x[i] - 0.3, x[i] + 0.3, linestyles='--', linewidth=2, label='Non-platoon Cost')
    else:
        ax.hlines(ori_costs[i], x[i] - 0.3, x[i] + 0.3, linestyles='--', linewidth=2)
    ax.text(x[i], (fuel_means[i] + wait_means[i]) * 1.05, f'-{reductions[i]:.1f}%', ha='center', va='bottom', fontsize=16)

ax.set_xticks(x)
ax.set_xticklabels(case_configs)
ax.set_ylabel('Cost', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y')
ax.legend(fontsize=16)
plt.tight_layout()

plt.show()

del ax, case, data, fig, final_cost, fuel_mean, fuel_means, i, idx, ori_cost, ori_costs, reduction, reductions
del wait_mean, wait_means, x

# %% platoon size vs fitness: bubble chart
from utils.platoon_functions import decode, get_platoons
from utils.graph_functions import create_graph

G, paths_SH, paths_HD = create_graph()
# cases = ['N60', 'N80', 'N100']
start_data = []
hub_data = []

for idx, case in enumerate(case_configs):
    start_counts = []
    hub_counts = []
    folder_name = os.path.join("data", case_configs[case], "TLBO")

    for trial in range(50):
        file_name = f"TLBO_{trial}.mat"
        data = load_mat(folder_name, file_name)
        ind = data['best']
        init = data['init']
        ARRIVAL_TIMES = data['ARRIVAL_TIMES']
        routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
        S_depart_times = ARRIVAL_TIMES + np.array(ind["wait_S"])
        H_arrival_times = S_depart_times + np.array(times_SH)
        H_depart_times = H_arrival_times + np.array(ind["wait_H"])
        start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times, ind["prior_S"], ind["prior_H"])
        start_platoons = [g for g in start_dict.values() if len(g) > 1]
        hub_platoons = [g for g in hub_dict.values() if len(g) > 1]
        start_counts.append(len(start_platoons))
        hub_counts.append(len(hub_platoons))

    start_data.append(start_counts)
    hub_data.append(hub_counts)

fig, ax = plt.subplots(figsize=(8, 5))

positions_start = [1, 4, 7]
positions_hub = [2, 5, 8]

bp1 = ax.boxplot(start_data, positions=positions_start, widths=0.7, patch_artist=True, medianprops=dict(color='black', linewidth=3))
bp2 = ax.boxplot(hub_data, positions=positions_hub, widths=0.7, patch_artist=True, medianprops=dict(color='black', linewidth=3))

for patch in bp1['boxes']:
    patch.set_facecolor('royalblue')
for patch in bp2['boxes']:
    patch.set_facecolor('salmon')

ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels(case_configs)
ax.set_ylabel('Number of Platoons', fontsize=16)
ax.set_xlabel('Problem Size', fontsize=16)
# ax.set_title('Distribution of Platoon Counts Across Different Cases', fontsize=18)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', alpha=0.3)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='royalblue', edgecolor='black', label='Start Platoons'),
    Patch(facecolor='salmon', edgecolor='black', label='Hub Platoons')
]

ax.legend(handles=legend_elements, fontsize=16)
plt.tight_layout()
plt.show()

del G, paths_SH, paths_HD, start_data, hub_data, case, start_counts, hub_counts, idx
del folder_name, trial, file_name, data, ind, init, ARRIVAL_TIMES, routes, times_SH, times_HD
del S_depart_times, H_arrival_times, H_depart_times, start_dict, hub_dict, start_platoons, hub_platoons
del fig, ax, positions_start, positions_hub, bp1, bp2, patch, legend_elements

#%% number of platoons
from utils.platoon_functions import decode, get_platoons
from utils.graph_functions import create_graph

G, paths_SH, paths_HD = create_graph()

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

start_color = 'royalblue'
hub_color = 'forestgreen'

for idx, case in enumerate(case_configs):
    ax = axes[idx]
    start_sizes = []
    hub_sizes = []
    folder_name = os.path.join("data", case_configs[case], "TLBO")

    for trial in range(50):
        file_name = f"TLBO_{trial}.mat"
        data = load_mat(folder_name, file_name)
        ind = data['best']
        init = data['init']
        ARRIVAL_TIMES = data['ARRIVAL_TIMES']
        
        routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
        S_depart_times = ARRIVAL_TIMES + np.array(ind["wait_S"])
        H_arrival_times = S_depart_times + np.array(times_SH)
        H_depart_times = H_arrival_times + np.array(ind["wait_H"])
        start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times, ind["prior_S"], ind["prior_H"])

        start_platoons = [g for g in start_dict.values() if 6 > len(g) > 1]
        hub_platoons = [g for g in hub_dict.values() if  6 > len(g) > 1]
        start_sizes.extend([len(g) for g in start_platoons])
        hub_sizes.extend([len(g) for g in hub_platoons])

    bins = np.arange(2, max(start_sizes + hub_sizes) + 2) - 0.5
    
    from collections import Counter

    start_count = Counter(start_sizes)
    hub_count = Counter(hub_sizes)
    x = sorted(set(start_count.keys()) | set(hub_count.keys()))
    start_y = [start_count.get(i, 0) for i in x]
    hub_y = [hub_count.get(i, 0) for i in x]
    
    w = 0.35
    ax.bar(np.array(x) - w/2, start_y, width=w, color=start_color, edgecolor='black', label='Start Platoons')
    ax.bar(np.array(x) + w/2, hub_y, width=w, color=hub_color, edgecolor='black', label='Hub Platoons')
    
    ax.set_title(case, fontsize=16)
axes[0].set_ylabel('Frequency', fontsize=16)
axes[1].set_xlabel('Platoon Size', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='y', alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, fontsize=16, bbox_to_anchor=(0.5, 1.05))
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del G, paths_SH, paths_HD, fig, axes, ax, start_color, hub_color, idx, case, start_sizes, hub_sizes
del data, ind, init, ARRIVAL_TIMES, routes, times_SH, times_HD, S_depart_times, H_arrival_times, H_depart_times
del start_dict, hub_dict, start_platoons, hub_platoons, bins, handles, folder_name, trial, file_name, labels
del hub_count, hub_y, start_count, start_y, w, x
