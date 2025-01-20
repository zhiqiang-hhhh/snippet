import re

def clean_sql(stmt):
    # Remove \n and excessive spaces
    stmt = stmt.replace('\\n', ' ')
    stmt = re.sub(r'\s+', ' ', stmt).strip()
    return stmt

def process_filtered_log(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            stmt_match = re.search(r"Stmt=([^|]+)", line)
            if stmt_match:
                stmt = stmt_match.group(1)
                cleaned_stmt = clean_sql(stmt)
                outfile.write(cleaned_stmt + '\n')

if __name__ == "__main__":
    input_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0 copy.filtered'
    output_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0.cleaned'
    
    process_filtered_log(input_file, output_file)
    print(f"Cleaned SQL queries saved to {output_file}")