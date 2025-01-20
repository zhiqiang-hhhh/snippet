import re

def extract_time_and_stmt(line):
    time_match = re.search(r"Time\(ms\)=([0-9]+)", line)
    stmt_match = re.search(r"Stmt=([^|]+)", line)
    
    if time_match and stmt_match:
        time_ms = int(time_match.group(1))
        stmt = stmt_match.group(1)
        return time_ms, stmt
    return None, None

def filter_queries_by_time(input_file, output_file, min_time, max_time):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            time_ms, stmt = extract_time_and_stmt(line)
            if time_ms is not None and min_time <= time_ms <= max_time:
                outfile.write(f"Time(ms)={time_ms} | Stmt={stmt}\n")

if __name__ == "__main__":
    input_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0'
    output_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0.filtered'
    min_time = 500
    max_time = 600

    filter_queries_by_time(input_file, output_file, min_time, max_time)
    print(f"Filtered queries saved to {output_file}")