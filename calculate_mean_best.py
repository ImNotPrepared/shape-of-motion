import os
import re
import numpy as np
import json
import csv

# Define the root directory
root_dir = "./results_dance"
output_json_file = "latest_and_best_results.json"
output_csv_file = "latest_and_best_results.csv"

# Initialize dictionaries to store results
latest_results = {}
best_results = {}

# Iterate through all subdirectories inside the root directory
for subdir, _, files in os.walk(root_dir):
    try:
      validation_files = [f for f in files if f.startswith("validation_")]

      # Initialize variables to track metrics
      latest_epoch = -1
      latest_metrics_sum = None
      latest_metrics_count = 0
      best_metrics_sum = None
      best_metrics_count = 0
      metric_keys = []
      
      # Iterate through validation files to accumulate metrics
      for validation_file in validation_files:
          file_path = os.path.join(subdir, validation_file)
          with open(file_path, "r") as f:
              content = f.read()
              matches = re.findall(r"Epoch (\d+)\n(.*?)\n\n", content, re.DOTALL)
              
              for match in matches:
                  epoch = int(match[0])
                  metrics_str = match[1]
                  
                  # Extract metric keys and values
                  metrics_lines = metrics_str.splitlines()
                  metrics_dict = {}
                  for line in metrics_lines:
                      key, value = line.split(": ")
                      metrics_dict[key] = float(value)
                      if key not in metric_keys:
                          metric_keys.append(key)
                  metrics_values = np.array(list(metrics_dict.values()))
                  
                  # Update latest metrics (average over the latest epoch)
                  if epoch > latest_epoch:
                      latest_epoch = epoch
                      latest_metrics_sum = metrics_values
                      latest_metrics_count = 1
                  elif epoch == latest_epoch:
                      latest_metrics_sum += metrics_values
                      latest_metrics_count += 1
                  
                  # Update best metrics (average over all validation files, considering lpips lower is better, others higher is better)
                  if best_metrics_sum is None:
                      best_metrics_sum = metrics_values
                      best_metrics_count = 1
                  else:
                      best_metrics_sum += metrics_values
                      best_metrics_count += 1
    except:
        continue
    
    # Calculate averaged latest and best metrics
    if latest_metrics_count > 0:
        latest_metrics_avg = latest_metrics_sum / latest_metrics_count
    else:
        latest_metrics_avg = None
    
    if best_metrics_count > 0:
        best_metrics_avg = best_metrics_sum / best_metrics_count
    else:
        best_metrics_avg = None
    
    # Store results in dictionaries
    if latest_metrics_avg is not None and best_metrics_avg is not None:
        latest_results[subdir] = {
            "latest_epoch": latest_epoch,
            "latest_metrics": dict(zip(metric_keys, latest_metrics_avg)),
            "best_metrics": dict(zip(metric_keys, best_metrics_avg))
        }

# Write results to JSON file
with open(output_json_file, "w") as json_file:
    json.dump(latest_results, json_file, indent=4)

# Write results to CSV file
with open(output_csv_file, "w", newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Folder", "Latest Epoch", "Metric Keys", "Latest Metrics", "Best Metrics"])
    for folder, metrics in latest_results.items():
        csv_writer.writerow([
            folder,
            metrics["latest_epoch"],
            metric_keys,
            metrics["latest_metrics"],
            metrics["best_metrics"]
        ])

print(f"Results have been saved to {output_json_file} and {output_csv_file}")
