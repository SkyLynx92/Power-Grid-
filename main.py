import numpy as np
import matplotlib.pyplot as plt
import os

def daily_consumption(data):
    """Return total energy per day (shape: 30,)"""
    return np.sum(data, axis=1)

def monthly_consumption(data):
    """Return total energy used in the entire dataset."""
    return np.sum(data)

def find_peak_hours(data):
    """
    For each day, find hours where usage > that day's average.
    Returns: {day_index: [list_of_peak_hours]}
    """
    peak_dict = {}
    for i, day_data in enumerate(data):
        avg = np.mean(day_data)
        peak_hours = np.where(day_data > avg)[0].tolist()
        peak_dict[i + 1] = peak_hours
    return peak_dict

def predict_next_day_usage(data):
    """Predict next day's total energy using simple 7-day moving average."""
    last_7_days = data[-7:]
    daily_totals = np.sum(last_7_days, axis=1)
    return np.mean(daily_totals)

# ---------- Visualization through Plotting Section ----------

def plot_daily_usage(daily_usage, save_path="outputs/daily_usage_plot.png"):
    """Plot daily total consumption."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(daily_usage)+1), daily_usage, marker='o')
    plt.title("Daily Energy Consumption (kWh)")
    plt.xlabel("Day")
    plt.ylabel("Energy Used (kWh)")
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def plot_peak_hours(day_index, day_data, save_path="outputs/peak_hours_day1.png"):
    """Plot hourly usage for one day highlighting peak hours."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    avg = np.mean(day_data)
    hours = np.arange(24)
    plt.figure(figsize=(10, 5))
    plt.bar(hours, day_data, color=['red' if x > avg else 'skyblue' for x in day_data])
    plt.axhline(y=avg, color='black', linestyle='--', label='Daily Average')
    plt.title(f"Peak Hours Visualization — Day {day_index}")
    plt.xlabel("Hour of the Day")
    plt.ylabel("Energy Used (kWh)")
    plt.legend()
    plt.savefig(save_path)
    plt.close()
