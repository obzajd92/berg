"""
Asynchronous Dashboard Listener with PER, Live Logging, and Adaptive Anomaly Detection
========================================================================================
Maintains a dual-thread model to animate live processing metrics. Features:
1. Non-blocking asynchronous writing of latency logs to a CSV ledger.
2. Rolling window moving-average filters tracking standard deviation (+3-sigma) 
   boundaries to detect processing latency spikes immediately.

Author: AI Collaborator
Date: October 2026
"""

import sys
import os
import time
import re
import queue
import threading
import asyncio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from typing import Dict, Any, List

# Ensure package paths resolve properly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from text_anomaly_pkg.inference import LiveTextInferenceRunner
except ImportError:
    from text_anomaly_pkg.text_anomaly_pkg.inference import LiveTextInferenceRunner

# Thread-safe global queue to bridge background parser thread and main Matplotlib window thread
metrics_shuttle_queue = queue.Queue()

# -------------------------------------------------------------------------
# MODULE 1: ASYNCHRONOUS NON-BLOCKING METRICS LOGGER
# -------------------------------------------------------------------------

class AsyncMetricsLogger:
    """Handles thread-safe asynchronous disk writes for telemetry metrics via an internal event loop."""
    def __init__(self, log_filepath: str = "generated/latency_performance_log.csv"):
        self.log_filepath = log_filepath
        self.queue: asyncio.Queue = None
        self.loop: asyncio.AbstractEventLoop = None
        self._ensure_file_headers()

    def _ensure_file_headers(self):
        """Pre-populates structural columns on disk if the CSV is newly created."""
        os.makedirs(os.path.dirname(self.log_filepath), exist_ok=True)
        if not os.path.exists(self.log_filepath):
            df = pd.DataFrame(columns=["Timestamp", "Tokens", "Queue_Wait_ms", "Neural_Latency_ms", "Is_Spike"])
            df.to_csv(self.log_filepath, index=False)

    def start_logger_thread(self):
        """Spawns an isolated background thread dedicated to processing disk write tasks."""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._worker_orchestrator())

        t = threading.Thread(target=run_loop, daemon=True)
        t.start()

    async def _worker_orchestrator(self):
        self.queue = asyncio.Queue()
        print(f"-> Asynchronous disk logging pipeline activated: {self.log_filepath}")
        while True:
            log_item = await self.queue.get()
            try:
                df = pd.DataFrame([log_item])
                df.to_csv(self.log_filepath, mode='a', header=False, index=False)
            except Exception as e:
                print(f"[LOG ERROR] Failed writing data entry: {e}")
            self.queue.task_done()

    def log_entry_async(self, tokens: int, queue_wait: float, neural_lat: float, is_spike: int):
        """Exposes a non-blocking call interface to safely queue items from other threads."""
        if self.loop and self.queue:
            entry = {
                "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Tokens": tokens,
                "Queue_Wait_ms": round(queue_wait, 3),
                "Neural_Latency_ms": round(neural_lat, 3),
                "Is_Spike": is_spike
            }
            self.loop.call_soon_threadsafe(self.queue.put_nowait, entry)

# Initialize global log dispatcher instance
async_disk_logger = AsyncMetricsLogger()

# -------------------------------------------------------------------------
# MODULE 2: BACKGROUND ASYNC STREAM PARSER & WORKER
# -------------------------------------------------------------------------

def run_background_input_worker(model_path: str):
    """Worker loop running in a background thread to safely block on sys.stdin inputs."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    runner = LiveTextInferenceRunner(model_path=model_path)
    async_disk_logger.start_logger_thread()

    print("=== Production Live Listener Active (PER / D3QN) ===")
    print("Pipe your live file or stream text blocks below...")

    while True:
        line = sys.stdin.readline()
        if not line:
            time.sleep(0.05)
            continue
            
        ingest_time = time.perf_counter()
        cleaned_text = line.strip()
        
        if cleaned_text:
            start_proc = time.perf_counter()
            analysis = runner.process_stream_frame(cleaned_text)
            end_proc = time.perf_counter()
            
            queue_wait = (start_proc - ingest_time) * 1000.0
            neural_latency = (end_proc - start_proc) * 1000.0
            
            metrics_shuttle_queue.put({
                "text": cleaned_text,
                "tokens": analysis["raw_token_count"],
                "queue_wait": queue_wait,
                "neural_latency": neural_latency,
                "action": analysis["selected_action"]
            })

# -------------------------------------------------------------------------
# MODULE 3: MOVING-AVERAGE FILTER BOUNDARY & REAL-TIME DASHBOARD
# -------------------------------------------------------------------------

plot_window_size = 50
latency_history: List[float] = []
action_distribution = {0: 0, 1: 0, 2: 0}
ma_window_size = 10  

def update_live_dashboard(frame, ax1, ax2, line_main, line_ma, line_upper, anomaly_scatter, bar_container):
    """Animate callback loop running natively on the UI main thread frame-by-frame."""
    global latency_history, action_distribution
    
    new_data_arrived = False
    while not metrics_shuttle_queue.empty():
        try:
            item = metrics_shuttle_queue.get_nowait()
            latency_history.append(item["neural_latency"])
            if item["action"] in action_distribution:
                action_distribution[item["action"]] += 1
            new_data_arrived = True
            
            is_spike_flag = 0
            if len(latency_history) >= ma_window_size:
                recent_window = latency_history[-ma_window_size:]
                current_ma = np.mean(recent_window)
                current_std = np.std(recent_window)
                upper_bound = current_ma + (3.0 * current_std)
                
                if item["neural_latency"] > upper_bound:
                    is_spike_flag = 1
                    print(f"⚠️ [ANOMALY DETECTED] Processing spike isolated! Latency: {item['neural_latency']:.2f}ms")
                    
            async_disk_logger.log_entry_async(item["tokens"], item["queue_wait"], item["neural_latency"], is_spike_flag)
            
        except queue.Empty:
            break

    if not new_data_arrived:
        return line_main, line_ma, line_upper, anomaly_scatter

    if len(latency_history) > plot_window_size:
        latency_history = latency_history[-plot_window_size:]

    n_points = len(latency_history)
    x_axis = np.arange(n_points)
    
    ma_curve = []
    upper_curve = []
    anomaly_x, anomaly_y = [], []
    
    for idx in range(n_points):
        if idx < ma_window_size:
            ma_curve.append(latency_history[idx])
            upper_curve.append(latency_history[idx] + 2.0)
        else:
            slice_window = latency_history[max(0, idx - ma_window_size):idx + 1]
            win_mean = np.mean(slice_window)
            win_std = np.std(slice_window)
            win_upper = win_mean + (3.0 * win_std)
            
            ma_curve.append(win_mean)
            upper_curve.append(win_upper)
            
            if latency_history[idx] > win_upper:
                anomaly_x.append(idx)
                anomaly_y.append(latency_history[idx])

    line_main.set_data(x_axis, latency_history)
    line_ma.set_data(x_axis, ma_curve)
    line_upper.set_data(x_axis, upper_curve)
    anomaly_scatter.set_offsets(np.column_stack([anomaly_x, anomaly_y]) if anomaly_x else np.empty((0, 2)))

    ax1.set_xlim(0, max(plot_window_size, n_points))
    if latency_history:
        ax1.set_ylim(0, max(np.max(latency_history) * 1.2, 5.0))

    for rect, val in zip(bar_container, action_distribution.values()):
        rect.set_height(val)
    ax2.set_ylim(0, max(np.max(list(action_distribution.values())) * 1.2, 10))

    return line_main, line_ma, line_upper, anomaly_scatter

# -------------------------------------------------------------------------
# MODULE 4: ORCHESTRATOR FRAMEWORK INITIALIZATION
# -------------------------------------------------------------------------

if __name__ == "__main__":
    weights_path = os.path.join("generated", "dueling_double_dqn_model.pt")
    
    bg_thread = threading.Thread(target=run_background_input_worker, args=(weights_path,), daemon=True)
    bg_thread.start()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    l_main, = ax1.plot([], [], color='teal', lw=1.5, label='Raw Processing Latency')
    l_ma, = ax1.plot([], [], color='blue', lw=2.0, linestyle='-', label=f'Moving Average (n={ma_window_size})')
    l_upper, = ax1.plot([], [], color='crimson', lw=1.5, linestyle='--', label='Anomaly Upper Bound (+3σ)')
    scat = ax1.scatter([], [], color='red', s=80, marker='X', zorder=5, label='Isolated Latency Spike')
    
    ax1.set_title('Real-Time Neural Inference Performance\nAdaptive Moving-Average Filter Boundaries', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Ingested Live Text Frame Sequence Index', fontsize=10)
    ax1.set_ylabel('Execution Processing Latency (ms)', fontsize=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.legend(loc='upper left', fontsize=9)

action_keys = ['Code 0\n[Overlap]', 'Code 1\n[Nodes]', 'Code 2\n[Dual Target]']
bars = ax2.bar(action_keys, [0, 0, 0], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', width=0.5)
ax2.set_title('Dueling Double-DQN Policy Engine\nCumulative Action Selection Footprint', fontsize=11, fontweight='bold')
ax2.set_ylabel('Total Distribution Count', fontsize=10)
ax2.grid(True, axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
ani = animation.FuncAnimation(
fig, update_live_dashboard,
fargs=(ax1, ax2, l_main, l_ma, l_upper, scat, bars),
interval=100, save_count=100, blit=False
)
plt.show()