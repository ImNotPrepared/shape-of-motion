import os
import numpy as np

results_dance_path = '/data3/zihanwa3/Capstone-DSR/shape-of-motion/results_dance'

# List all subdirectories in results_dance_path
subdirs = [d for d in os.listdir(results_dance_path) if os.path.isdir(os.path.join(results_dance_path, d))]

for subdir in subdirs:
    subdir_path = os.path.join(results_dance_path, subdir)
    
    # Get list of validation_metrics_cam*.txt files
    txt_files = [os.path.join(subdir_path, f'validation_metrics_cam{i}.txt') for i in range(4)]
    
    # Check if files exist
    missing_files = [f for f in txt_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files in {subdir}: {missing_files}")
        continue
    
    # Initialize data structure
    data = {}  # key: epoch number, value: dict of metric name to list of values
    
    for txt_file in txt_files:
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            current_epoch = None
            for line in lines:
                line = line.strip()
                if line.startswith('Epoch'):
                    # Extract epoch number
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            current_epoch = int(parts[1])
                        except ValueError:
                            print(f"Invalid epoch number in file {txt_file}: {line}")
                            continue
                elif ':' in line and current_epoch is not None:
                    # Metric line
                    metric_name, value_str = line.split(':', 1)
                    metric_name = metric_name.strip()
                    value_str = value_str.strip()
                    # Convert value to float
                    try:
                        if value_str.lower() == 'nan':
                            value = np.nan
                        else:
                            value = float(value_str)
                    except ValueError:
                        print(f"Invalid value in file {txt_file}: {line}")
                        continue
                    # Store the value
                    if current_epoch not in data:
                        data[current_epoch] = {}
                    if metric_name not in data[current_epoch]:
                        data[current_epoch][metric_name] = []
                    data[current_epoch][metric_name].append(value)
                else:
                    # Ignore other lines
                    pass
    # Now compute averages
    averaged_data = {}  # key: epoch number, value: dict of metric name to average value
    for epoch in sorted(data.keys()):
        averaged_data[epoch] = {}
        for metric_name, values in data[epoch].items():
            # Convert list to numpy array
            values_array = np.array(values)
            # Compute nan-aware mean
            avg_value = np.nanmean(values_array)
            averaged_data[epoch][metric_name] = avg_value
    
    # Write the averaged metrics to a file
    output_file = os.path.join(subdir_path, 'validation_metrics_general.txt')
    with open(output_file, 'w') as f_out:
        for epoch in sorted(averaged_data.keys()):
            f_out.write(f'Epoch {epoch}\n')
            for metric_name, avg_value in averaged_data[epoch].items():
                if np.isnan(avg_value):
                    value_str = 'nan'
                else:
                    value_str = str(avg_value)
                f_out.write(f'{metric_name}: {value_str}\n')
            f_out.write('\n')  # Add an empty line between epochs
    print(f"Processed {subdir}, averaged metrics saved to {output_file}")
