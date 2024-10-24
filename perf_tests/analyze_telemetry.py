import json
import glob
import numpy as np
import pandas as pd
import os
from datetime import datetime

def load_telemetry_data(pattern='telemetry_data_*.json'):
    telemetry_files = glob.glob(pattern)
    all_data = []
    
    for file in telemetry_files:
        with open(file, 'r') as f:
            data = json.load(f)
            all_data.extend(data)
    
    return all_data

def analyze_telemetry(all_data):
    if not all_data:
        print("No telemetry data to analyze.")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure 'total_time_ms' is numeric
    df['total_time_ms'] = pd.to_numeric(df['total_time_ms'], errors='coerce')
    
    # Calculate metrics
    avg_time = df['total_time_ms'].mean()
    median_time = df['total_time_ms'].median()
    p95_time = df['total_time_ms'].quantile(0.95)
    total_requests = len(df)
    successful_requests = df['total_time_ms'].notna().sum()
    error_rate = ((total_requests - successful_requests) / total_requests) * 100 if total_requests > 0 else 0.0
    
    summary = {
        'Average Response Time (ms)': avg_time,
        'Median Response Time (ms)': median_time,
        '95th Percentile Response Time (ms)': p95_time,
        'Total Requests': total_requests,
        'Successful Requests': successful_requests,
        'Error Rate (%)': error_rate,
    }
    
    # Save summary to JSON
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    summary_filename = f"telemetry_summary_{timestamp}.json"
    with open(summary_filename, 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Print summary
    print("Telemetry Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    all_data = load_telemetry_data()
    analyze_telemetry(all_data)
