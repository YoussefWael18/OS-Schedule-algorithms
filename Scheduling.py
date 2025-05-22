import pandas as pd
from typing import List
import matplotlib.pyplot as plt
from collections import deque

df = pd.read_csv("D:/Scheduling algorithms/chrome_process_data.csv")
print("Original Reading from Chrome:")
print(df)

processes = df.groupby("PID").agg({   # grouping by PID, keeping the minimum Arrival_Time (earliest appearance) and maximum Burst_Time (peak CPU demand)
    "Arrival_Time": "min",
    "Burst_Time": "max"
}).reset_index()
processes = processes.sort_values("Arrival_Time").reset_index(drop=True) # sorting the dataset

print(processes)
pid_list = processes['PID'].tolist()        # create a list of  process ID
arrival_time_list = processes['Arrival_Time'].tolist() # create a list of arrival time
burst_time_list = processes['Burst_Time'].tolist()     # create a list of burst time 

# Visualization for SRTF
def visualize_srtf(processes_metrics: dict, execution_order: List[int], time_step: float):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.3)

    ax_gantt = fig.add_subplot(gs[0])
    unique_pids = sorted(list(set([p for p in execution_order if p is not None])))
    colors = plt.cm.get_cmap('tab10', max(len(unique_pids), 2))
    pid_color_map = {pid: colors(i) for i, pid in enumerate(unique_pids)}
    pid_color_map[None] = 'lightgray'

    current_pid = execution_order[0]
    start_block_time = 0.0
    block_duration_steps = 0

    for i in range(len(execution_order)):
        if execution_order[i] == current_pid:
            block_duration_steps += 1
        else:
            duration = block_duration_steps * time_step
            if duration > 1e-9:
                ax_gantt.barh(0, duration, left=start_block_time, height=0.4,
                             color=pid_color_map[current_pid], label=f'P{current_pid}' if current_pid is not None else 'Idle')
            current_pid = execution_order[i]
            start_block_time += duration
            block_duration_steps = 1
    duration = block_duration_steps * time_step
    if duration > 1e-9:
        ax_gantt.barh(0, duration, left=start_block_time, height=0.4,
                     color=pid_color_map[current_pid], label=f'P{current_pid}' if current_pid is not None else 'Idle')

    ax_gantt.set_yticks([])
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_title("SRTF Gantt Chart")
    handles, labels = ax_gantt.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    sorted_legend = sorted(zip(unique_labels, unique_handles), key=lambda x: float('inf') if x[0] == 'Idle' else int(x[0][1:]))
    unique_labels, unique_handles = zip(*sorted_legend)
    ax_gantt.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    total_sim_time = len(execution_order) * time_step
    ax_gantt.set_xlim(0, total_sim_time)
    ax_gantt.xaxis.grid(True, linestyle='--', alpha=0.6)

    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    metrics = {
        'PID': [],
        'Arrival Time': [],
        'Burst Time': [],
        'Completion Time': [],
        'Turnaround Time': [],
        'Waiting Time': []
    }
    for pid in sorted(processes_metrics.keys()):
        metrics['PID'].append(pid)
        metrics['Arrival Time'].append(processes_metrics[pid]['Arrival_Time'])
        metrics['Burst Time'].append(processes_metrics[pid]['Burst_Time'])
        metrics['Completion Time'].append(round(processes_metrics[pid].get('Completion_Time', 0), 3))
        metrics['Turnaround Time'].append(round(processes_metrics[pid].get('Turnaround_Time', 0), 3))
        metrics['Waiting Time'].append(round(processes_metrics[pid].get('Waiting_Time', 0), 3))
    df_metrics = pd.DataFrame(metrics)
    df_metrics['Arrival Time'] = df_metrics['Arrival Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Burst Time'] = df_metrics['Burst Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Completion Time'] = df_metrics['Completion Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Turnaround Time'] = df_metrics['Turnaround Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Waiting Time'] = df_metrics['Waiting Time'].apply(lambda x: f'{x:.3f}')
    table = ax_table.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                          loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('srtf_gantt_chart.png')
    plt.show()

# Scheduling algorithm 1 - SRTF
def SRTF(PID: List[int], Arrival_Time: List[float], Burst_Time: List[float]):
    processes = {}
    for pid, at, bt in zip(PID, Arrival_Time, Burst_Time):
        processes[pid] = {
            'Arrival_Time': at,
            'Burst_Time': bt,
            'Remaining_Burst_Time': bt,
            'Completion_Time': 0,
            'Turnaround_Time': 0,
            'Waiting_Time': 0
        }
    current_time = 0
    completed = 0
    execution_order = []
    n = len(processes)
    while completed < n:
        Shortest_time = float('inf')
        Shortest_pid = None
        ready_queue = [pid for pid in processes 
                      if processes[pid]['Arrival_Time'] <= current_time 
                      and processes[pid]['Remaining_Burst_Time'] > 0]
        if not ready_queue:
            execution_order.append(None)
            current_time += 0.01
        else:
            for pid in ready_queue:
                if processes[pid]['Remaining_Burst_Time'] < Shortest_time:
                    Shortest_time = processes[pid]['Remaining_Burst_Time']
                    Shortest_pid = pid
            processes[Shortest_pid]['Remaining_Burst_Time'] -= 0.01
            execution_order.append(Shortest_pid)
            current_time += 0.01
            if processes[Shortest_pid]['Remaining_Burst_Time'] <= 0:
                completed += 1
                processes[Shortest_pid]['Completion_Time'] = current_time
                processes[Shortest_pid]['Turnaround_Time'] = processes[Shortest_pid]['Completion_Time'] - processes[Shortest_pid]['Arrival_Time']
                processes[Shortest_pid]['Waiting_Time'] = processes[Shortest_pid]['Turnaround_Time'] - processes[Shortest_pid]['Burst_Time']
    return processes, execution_order

def visualize_round_robin(pid_list, arrival_time_list, burst_time_list, execution_order, quantum):
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.3)
    ax_gantt = fig.add_subplot(gs[0])
    unique_pids = sorted(list(set([p for p in execution_order if p is not None])))
    colors = plt.cm.get_cmap('tab10', max(len(unique_pids), 2))
    pid_color_map = {pid: colors(i) for i, pid in enumerate(unique_pids)}
    pid_color_map[None] = 'lightgray'
    current_pid = execution_order[0]
    start_block_time = 0.0
    block_duration_steps = 0
    for i in range(len(execution_order)):
        if execution_order[i] == current_pid:
            block_duration_steps += 1
        else:
            duration = block_duration_steps * quantum if current_pid is not None else block_duration_steps
            if duration > 1e-9:
                ax_gantt.barh(0, duration, left=start_block_time, height=0.4,
                             color=pid_color_map[current_pid],
                             label=f'P{current_pid}' if current_pid is not None else 'Idle')
            start_block_time += duration
            current_pid = execution_order[i]
            block_duration_steps = 1
    duration = block_duration_steps * quantum if current_pid is not None else block_duration_steps
    if duration > 1e-9:
        ax_gantt.barh(0, duration, left=start_block_time, height=0.4,
                     color=pid_color_map[current_pid],
                     label=f'P{current_pid}' if current_pid is not None else 'Idle')
    ax_gantt.set_yticks([])
    ax_gantt.set_xlabel("Time")
    ax_gantt.set_title("Round Robin Gantt Chart")
    handles, labels = ax_gantt.get_legend_handles_labels()
    unique_labels = []
    unique_handles = []
    for h, l in zip(handles, labels):
        if l not in unique_labels:
            unique_labels.append(l)
            unique_handles.append(h)
    sorted_legend = sorted(zip(unique_labels, unique_handles), key=lambda x: float('inf') if x[0] == 'Idle' else int(x[0][1:]))
    unique_labels, unique_handles = zip(*sorted_legend)
    ax_gantt.legend(unique_handles, unique_labels, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=1, fontsize=8)
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')
    metrics = {'PID': [], 'Arrival Time': [], 'Burst Time': [], 'Completion Time': [], 'Turnaround Time': [], 'Waiting Time': []}
    remaining_burst_times = {pid: burst for pid, burst in zip(pid_list, burst_time_list)}
    current_time = 0
    completion_times = {}
    for i, pid in enumerate(execution_order):
        if pid is not None:
            time_slice = min(quantum, remaining_burst_times[pid])
            current_time += time_slice
            remaining_burst_times[pid] -= time_slice
            # Use a small epsilon to handle floating-point precision issues
            if remaining_burst_times[pid] <= 1e-9:  # Consider as finished if very close to 0
                completion_times[pid] = current_time
        else:
            current_time += 1
    for i, pid in enumerate(pid_list):
        completion_time = completion_times.get(pid, 0)
        turnaround_time = completion_time - arrival_time_list[i]
        waiting_time = turnaround_time - burst_time_list[i]
        metrics['PID'].append(pid)
        metrics['Arrival Time'].append(arrival_time_list[i])
        metrics['Burst Time'].append(burst_time_list[i])
        metrics['Completion Time'].append(round(completion_time, 3))
        metrics['Turnaround Time'].append(round(turnaround_time, 3))
        metrics['Waiting Time'].append(round(waiting_time, 3))
    df_metrics = pd.DataFrame(metrics)
    df_metrics['Arrival Time'] = df_metrics['Arrival Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Burst Time'] = df_metrics['Burst Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Completion Time'] = df_metrics['Completion Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Turnaround Time'] = df_metrics['Turnaround Time'].apply(lambda x: f'{x:.3f}')
    df_metrics['Waiting Time'] = df_metrics['Waiting Time'].apply(lambda x: f'{x:.3f}')
    table = ax_table.table(cellText=df_metrics.values, colLabels=df_metrics.columns,
                          loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.2)
    for (i, j), cell in table.get_celld().items():
        cell.set_edgecolor('black')
        cell.set_linewidth(0.5)
    plt.tight_layout()
    plt.savefig('rr_gantt_chart.png')
    plt.show()

# Scheduling algorithm 2 - Round Robin
def round_robin(pid_list, arrival_time_list, burst_time_list):
    processes = {}
    for pid, at, bt in zip(pid_list, arrival_time_list, burst_time_list):
        processes[pid] = {'Arrival_Time': at, 'Burst_Time': bt, 'Remaining_Burst_Time': bt, 'Completion_Time': 0}
    ready_queue = []
    completed = 0
    current_time = 0
    quantum = 0.5
    execution_order = []
    n = len(pid_list)
    
    while completed < n:
        # Add processes that have arrived and are not yet completed
        for i, pid in enumerate(pid_list):
            if (arrival_time_list[i] <= current_time and 
                processes[pid]['Remaining_Burst_Time'] > 0 and 
                pid not in ready_queue):
                ready_queue.append(pid)
        
        if ready_queue:
            current_pid = ready_queue.pop(0)
            if processes[current_pid]['Remaining_Burst_Time'] > 0:
                time_slice = min(quantum, processes[current_pid]['Remaining_Burst_Time'])
                execution_order.append(current_pid)
                current_time += time_slice
                processes[current_pid]['Remaining_Burst_Time'] -= time_slice
                if processes[current_pid]['Remaining_Burst_Time'] <= 1e-9:  # Handle floating-point precision
                    processes[current_pid]['Completion_Time'] = current_time
                    completed += 1
                else:
                    ready_queue.append(current_pid)
        else:
            # Advance to the next arrival time 
            next_arrival = min([at for i, at in enumerate(arrival_time_list) 
                              if processes[pid_list[i]]['Remaining_Burst_Time'] > 0 
                              and at > current_time], default=current_time + 0.1)
            execution_order.append(None)
            current_time = next_arrival
    
    return execution_order

processes, execution_order = SRTF(pid_list, arrival_time_list, burst_time_list)
visualize_srtf(processes, execution_order, 0.01)
execution_order_RR = round_robin(pid_list, arrival_time_list, burst_time_list)
visualize_round_robin(pid_list, arrival_time_list, burst_time_list, execution_order_RR, 0.5)