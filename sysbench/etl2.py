input_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0.cleaned'
output_file = '/mnt/disk1/hezhiqiang/Code/snippet/sysbench/fe.audit.log.0.cleaned.modified'

with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
    for line in infile:
        line = line.strip()
        if line:
            outfile.write(f'"{line}",\n')