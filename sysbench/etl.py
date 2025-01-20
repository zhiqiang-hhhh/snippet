import re

def extract_time_and_stmt(line):
    time_match = re.search(r"Time\(ms\)=([0-9]+)", line)
    stmt_match = re.search(r"Stmt=([^|]+)", line)
    
    if time_match and stmt_match:
        time_ms = time_match.group(1)
        stmt = stmt_match.group(1)
        if 'count(k)' in stmt:
            return f"Time(ms)={time_ms} | Stmt={stmt}"
    return None

def process_audit_log(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            result = extract_time_and_stmt(line)
            if result:
                outfile.write(result + '\n')

if __name__ == "__main__":
    input_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.1'
    output_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.1.processed'
    process_audit_log(input_file, output_file)
    print(f"Processed log saved to {output_file}")