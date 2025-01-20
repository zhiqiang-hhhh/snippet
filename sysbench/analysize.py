import re
import numpy as np
import matplotlib.pyplot as plt

def extract_time(line):
    time_match = re.search(r"Time\(ms\)=([0-9]+)", line)
    if time_match:
        return int(time_match.group(1))
    return None

def process_audit_log(input_file):
    times = []
    with open(input_file, 'r') as infile:
        for line in infile:
            time_ms = extract_time(line)
            if time_ms is not None:
                times.append(time_ms)
    return times

def calculate_statistics(times):
    average_time = np.mean(times)
    max_time = np.max(times)
    p95_time = np.percentile(times, 95)
    return average_time, max_time, p95_time

def plot_distribution(times, output_file):
    plt.hist(times, bins=50, edgecolor='black')
    plt.xlabel('Time(ms)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Time(ms)')
    plt.savefig(output_file)
    plt.show()

if __name__ == "__main__":
    input_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.1.processed'
    output_image = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/time_distribution.1.png'

    times = process_audit_log(input_file)
    average_time, max_time, p95_time = calculate_statistics(times)

    print(f"Average Time(ms): {average_time}")
    print(f"Max Time(ms): {max_time}")
    print(f"P95 Time(ms): {p95_time}")

    plot_distribution(times, output_image)
    print(f"Distribution plot saved to {output_image}")