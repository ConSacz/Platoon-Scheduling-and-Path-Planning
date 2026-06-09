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
from utils.fitness_functions import sep_fitness, ori_fitness
from utils.graph_functions import create_graph

# %% DATA AUTO IMPORT
Trials = 50
algorithm_configs = {
    'GA': 'GA',
    'PSO': 'PSO',
    'TLBO': 'TLBO'
}
case_configs = {
    'N60': 'case_60',
    'N80': 'case_80',
    'N100': 'case_100',  
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
        final_costs = []
        ori_final_costs = []
        final_sep_costs = []
        final_runtime = []

        for i in range(Trials):
            folder_name = os.path.join("data", case_folder, algo_folder)
            file_name = f"{algo_folder}_{i}.mat"

            data = load_mat(folder_name, file_name)

            BestCostIt = data['BestCostIt'].flatten()  # đảm bảo là vector 1D
            final_value = BestCostIt[-1]  # lấy iteration cuối
            
            ind = data['best']
            init = data['init']
            ARRIVAL_TIMES = data['ARRIVAL_TIMES']
            G, paths_SH, paths_HD = create_graph()
            sep_fit = sep_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)
            ori_fit = ori_fitness(ind, init, ARRIVAL_TIMES, paths_SH, paths_HD, G)

            final_costs.append(final_value)
            ori_final_costs.append(ori_fit)
            final_sep_costs.append(sep_fit)
            final_runtime.append(data['runtime'])
            
        results[algo_name][case_name] = np.array(final_costs)
        ori_results[algo_name][case_name] = np.array(ori_final_costs)
        sep_results[algo_name][case_name] = np.array(final_sep_costs)
        runtimes[algo_name][case_name] = np.array(final_runtime)

del algo_folder, algo_name, BestCostIt, case_folder, case_name, data, file_name, final_costs, final_value, folder_name, i, final_runtime
del ARRIVAL_TIMES, final_sep_costs, G, ind, init, paths_HD, paths_SH, sep_fit, ori_final_costs, ori_fit

# %% box plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
# fig, axes = plt.subplots(1, 4, figsize=(20, 4))
colors = ['lightblue', 'salmon', 'lightgreen']

# flatten axes để duyệt dễ hơn
axes = axes.flatten()
for idx, case in enumerate(case_configs):
    ax = axes[idx]
    data_to_plot = []
    labels = []
    for algo in algorithm_configs:
        data_to_plot.append(results[algo][case])
        labels.append(algo)
    bp = ax.boxplot(
        data_to_plot,
        patch_artist=True,
        medianprops=dict(color='black', linewidth=2)
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax.set_title(case, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)

legend_elements = [
    Patch(facecolor=colors[i], label=labels[i])
    for i in range(len(labels))
]

axes[0].set_ylabel('Best Cost', fontsize=16)

fig.legend(
    handles=legend_elements,
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.5, 1.07),
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del algo, ax, axes, bp, case, color, colors, data_to_plot, fig, idx,labels, legend_elements, patch
#%% line plot
from utils.test import first_pop
colors = {
    'GA': 'blue',
    'PSO': 'salmon',
    'TLBO': 'green'
}
line_styles = {
    'GA': '--',
    'PSO': '-.',
    'TLBO': '-'
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# chuyển axes thành vector 1 chiều
axes = axes.flatten()

for idx, case in enumerate(case_configs):
    ax = axes[idx]
    for algo in algorithm_configs:
        all_trials = []
        for i in range(Trials):
            folder = os.path.join(
                "data",
                case_configs[case],
                algorithm_configs[algo]
            )
            file = f"{algorithm_configs[algo]}_{i}.mat"
            data = load_mat(folder, file)
            BestCostIt = data['BestCostIt'].flatten()
            first_cost = first_pop(data['best']['wait_S'].shape[0], data['ARRIVAL_TIMES'], data['init'])
            BestCostIt = np.concatenate(([first_cost], BestCostIt))
            all_trials.append(BestCostIt)
        all_trials = np.array(all_trials)
        mean = np.mean(all_trials, axis=0)
        std = np.std(all_trials, axis=0)
        x = np.arange(len(mean))
        ax.plot(
            x,
            mean,
            label=algo,
            linestyle=line_styles[algo],
            linewidth=2,
            color=colors[algo]
        )
        ax.fill_between(
            x,
            mean - std,
            mean + std,
            alpha=0.2
        )
    ax.set_title(case, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
# ylabel cho cột bên trái
axes[0].set_ylabel('Best Cost', fontsize=16)
# axes[2].set_ylabel('Best Cost')

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=3,
    bbox_to_anchor=(0.5, 1.07),
    fontsize=16
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del algo, ax, axes, all_trials, case, line_styles, mean, x, std, colors, BestCostIt
del fig, file, folder, handles, i, idx, labels, data, first_cost

# %% scatter 2 fitness
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
markers = {
    'GA': 'o',
    'PSO': 's',
    'TLBO': '^'
}
colors = {
    'GA': 'blue',
    'PSO': 'salmon',
    'TLBO': 'green'
}

for idx, case in enumerate(case_configs):
    ax = axes[idx]
    for algo in algorithm_configs:
        data = sep_results[algo][case]
        fuel = data[:, 0]
        wait = data[:, 1]

        ax.scatter(wait, fuel, label=algo, alpha=0.7, marker=markers[algo], color=colors[algo])
    ax.set_title(case, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True)
    
axes[0].set_ylabel('Fuel Cost', fontsize=16)
axes[1].set_xlabel('Wait Cost', fontsize=16)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.07),fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del algo, ax, axes, case, colors, data, fig, fuel, handles, idx, labels
del markers, wait

# %% bar chart
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for idx, case in enumerate(case_configs):
    ax = axes[idx]
    fuel_means = []
    wait_means = []
    for algo in algorithm_configs:
        data = sep_results[algo][case]
        fuel_means.append(np.mean(data[:, 0]))
        wait_means.append(np.mean(data[:, 1]))

    x = np.arange(len(algorithm_configs))
    ax.bar(x, fuel_means, label='Fuel Cost')
    ax.bar(x, wait_means, bottom=fuel_means, label='Wait Cost')
    
    ax.set_xticks(x)
    ax.set_xticklabels(algorithm_configs)
    ax.set_title(case, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.grid(True, axis='y')
    
axes[0].set_ylabel('Cost', fontsize=16)
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc='upper center',
    ncol=2,
    bbox_to_anchor=(0.5, 1.07),
    fontsize=16
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

del algo, ax, axes, case, data, fig, fuel_means, handles
del idx, labels, wait_means, x

# %% runtime comparison
fig, ax = plt.subplots(figsize=(7, 5))

problem_sizes = [60, 80, 100]
colors = {
    'GA': 'blue',
    'PSO': 'salmon',
    'TLBO': 'green'
}
line_styles = {
    'GA': '--',
    'PSO': '-.',
    'TLBO': '-'
}

for algo in algorithm_configs:
    mean_runtimes = []
    std_runtimes = []
    for case in case_configs:
        data = np.array(runtimes[algo][case])
        mean_runtimes.append(np.mean(data))
        std_runtimes.append(np.std(data))

    mean_runtimes = np.array(mean_runtimes)
    std_runtimes = np.array(std_runtimes)
    ax.plot(problem_sizes, mean_runtimes, marker='o', linewidth=2, linestyle=line_styles[algo], color=colors[algo], label=algo)
    ax.fill_between(problem_sizes, mean_runtimes - std_runtimes, mean_runtimes + std_runtimes, alpha=0.2, color=colors[algo])

ax.set_xlabel('Number of Vehicles', fontsize=16)
ax.set_ylabel('Runtime (m)', fontsize=16)
ax.tick_params(axis='both', labelsize=16)
ax.grid(True)
ax.legend(fontsize=16)
plt.tight_layout()

plt.show()

del algo, ax,case, colors, data, fig, line_styles, mean_runtimes, problem_sizes, std_runtimes
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

# %% gantt chart
from utils.platoon_functions import decode, get_platoons
from utils.graph_functions import create_graph

folder_name = os.path.join("data", "case_60", "TLBO")
file_name = "TLBO_0.mat"

data = load_mat(folder_name, file_name)

ind = data['best']
init = data['init']
ARRIVAL_TIMES = data['ARRIVAL_TIMES']
G, paths_SH, paths_HD = create_graph()
N = data['best']['wait_S'].shape[0]

routes, times_SH, times_HD = decode(ind, init, paths_SH, paths_HD, G)
S_depart_times = ARRIVAL_TIMES + np.array(ind["wait_S"])
H_arrival_times = S_depart_times + np.array(times_SH)
H_depart_times = H_arrival_times + np.array(ind["wait_H"])
D_arrival_times = H_depart_times + np.array(times_HD)

ARRIVAL_TIMES   = ARRIVAL_TIMES/2
S_depart_times  = S_depart_times/2
H_arrival_times = H_arrival_times/2
H_depart_times  = H_depart_times/2
D_arrival_times = D_arrival_times/2

start_dict, hub_dict = get_platoons(routes, S_depart_times, H_depart_times, ind["prior_S"], ind["prior_H"])

fig, ax = plt.subplots(figsize=(25, 25))

spacing = 7
cmap = plt.cm.tab20
default_color = 'gray'


vehicle_colors = {}
all_platoons = []
for group in start_dict.values():
    if len(group) > 1:
        all_platoons.append(group)

for group in hub_dict.values():
    if len(group) > 1:
        all_platoons.append(group)
        
# draw by platoon
drawn = set()
for pid, group in enumerate(all_platoons):
    color = cmap(pid % 20)
    for veh in group:
        if veh in drawn:
            continue

        drawn.add(veh)
        y = veh * spacing

        ax.hlines(y, 0, np.max(D_arrival_times) + 5, linestyle='--', linewidth=0.6, color='gray', alpha=0.4)
        ax.barh(y, S_depart_times[veh] - ARRIVAL_TIMES[veh], left=ARRIVAL_TIMES[veh], height=1.5, color='lightgray', linestyle='--', edgecolor='black')
        ax.barh(y, H_arrival_times[veh] - S_depart_times[veh], left=S_depart_times[veh], height=1.5, color='blue', edgecolor=color)
        ax.barh(y, H_depart_times[veh] - H_arrival_times[veh], left=H_arrival_times[veh], height=1.5, color='khaki', linestyle='--', edgecolor='black')
        ax.barh(y, D_arrival_times[veh] - H_depart_times[veh], left=H_depart_times[veh], height=1.5, color='blue', edgecolor=color)

# draw solo vehicles
for veh in range(N):
    color = cmap(pid % 20)
    if veh in drawn:
        continue

    y = veh * spacing
    
    ax.hlines(y, 0, np.max(D_arrival_times) + 5, linestyle='--', linewidth=0.6, color='gray', alpha=0.4)
    ax.barh(y, S_depart_times[veh] - ARRIVAL_TIMES[veh], left=ARRIVAL_TIMES[veh], height=1.5, color='lightgray', linestyle='--', edgecolor='black')
    ax.barh(y, H_arrival_times[veh] - S_depart_times[veh], left=S_depart_times[veh], height=1.5, color=color, edgecolor='black')
    ax.barh(y, H_depart_times[veh] - H_arrival_times[veh], left=H_arrival_times[veh], height=1.5, color='khaki', linestyle='--', edgecolor='black')
    ax.barh(y, D_arrival_times[veh] - H_depart_times[veh], left=H_depart_times[veh], height=1.5, color=color, edgecolor='black')

# platoon labels
for pid, group in enumerate(all_platoons):

    color = cmap(pid % 20)

    if group in start_dict.values():
        depart_time = S_depart_times[group[0]]
        for veh in group:
            y = veh * spacing
            ax.text(depart_time + 0.2, y + 1.8, f'S:{group}', fontsize=16, color=color)

    if group in hub_dict.values():
        depart_time = H_depart_times[group[0]]
        for veh in group:
            y = veh * spacing
            ax.text(depart_time + 0.2, y + 1.8, f'H:{group}', fontsize=16, color=color)

ax.set_xlim(0, 20)
ax.set_xlabel('Time', fontsize=20)
ax.set_ylabel('Vehicle ID', fontsize=20)
ax.set_title('Vehicle Scheduling Gantt Chart', fontsize=24)
ax.set_yticks([i * spacing for i in range(N)])
ax.set_yticklabels([f'V{i}' for i in range(N)])
ax.tick_params(axis='both', labelsize=16)
ax.grid(True, axis='x', alpha=0.3)

from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor='lightgray', edgecolor='black', label='Waiting at S'),
    Patch(facecolor='khaki', edgecolor='black', label='Waiting at H'),
    Patch(facecolor='tab:blue', edgecolor='black', label='Platoon Travel')
]

ax.legend(handles=legend_elements, fontsize=16)
plt.tight_layout()
plt.show()

del folder_name, file_name, data, ind, init, ARRIVAL_TIMES
del G, paths_SH, paths_HD, N, routes, times_SH, times_HD
del S_depart_times, H_arrival_times, H_depart_times, D_arrival_times
del start_dict, hub_dict, fig, ax, spacing, cmap, default_color
del vehicle_colors, all_platoons, pid, group, color, drawn
del veh, y, legend_elements

# %% platoon number box chart
from utils.platoon_functions import decode, get_platoons
from utils.graph_functions import create_graph

G, paths_SH, paths_HD = create_graph()
cases = ['N60', 'N80', 'N100']
start_data = []
hub_data = []

for case in cases:
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

fig, ax = plt.subplots(figsize=(10, 6))

positions_start = [1, 4, 7]
positions_hub = [2, 5, 8]

bp1 = ax.boxplot(start_data, positions=positions_start, widths=0.7, patch_artist=True, medianprops=dict(color='black', linewidth=3))
bp2 = ax.boxplot(hub_data, positions=positions_hub, widths=0.7, patch_artist=True, medianprops=dict(color='black', linewidth=3))

for patch in bp1['boxes']:
    patch.set_facecolor('royalblue')
for patch in bp2['boxes']:
    patch.set_facecolor('salmon')

ax.set_xticks([1.5, 4.5, 7.5])
ax.set_xticklabels(cases)
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

del G, paths_SH, paths_HD, cases, start_data, hub_data, case, start_counts, hub_counts
del folder_name, trial, file_name, data, ind, init, ARRIVAL_TIMES, routes, times_SH, times_HD
del S_depart_times, H_arrival_times, H_depart_times, start_dict, hub_dict, start_platoons, hub_platoons
del fig, ax, positions_start, positions_hub, bp1, bp2, patch, legend_elements

#%% platoon size bar chart
from utils.platoon_functions import decode, get_platoons
from utils.graph_functions import create_graph

G, paths_SH, paths_HD = create_graph()

cases = ['N60', 'N80', 'N100']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

start_color = 'royalblue'
hub_color = 'forestgreen'

for idx, case in enumerate(cases):
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

del G, paths_SH, paths_HD, cases, fig, axes, ax, start_color, hub_color, idx, case, start_sizes, hub_sizes
del data, ind, init, ARRIVAL_TIMES, routes, times_SH, times_HD, S_depart_times, H_arrival_times, H_depart_times
del start_dict, hub_dict, start_platoons, hub_platoons, bins, handles, folder_name, trial, file_name, labels