#!/usr/bin/env python3
import time
import sys

def get_page_cache():
    """从/proc/meminfo获取Cached和Buffers的内存统计"""
    cached = buffers = 0
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if line.startswith('Cached:'):
                    cached = int(line.split()[1])
                elif line.startswith('Buffers:'):
                    buffers = int(line.split()[1])
    except Exception as e:
        print(f"Error reading /proc/meminfo: {e}", file=sys.stderr)
        return None
    
    # 将KB转换为MB并返回总和
    return (cached + buffers) / 1024.0

def main(interval=1):
    print("Monitoring Page Cache changes. Press Ctrl+C to exit...")
    print("-" * 65)
    print(f"{'Time':<20} | {'Cache(MB)':>10} | {'Delta(MB)':>10} | {'Rate(MB/s)':>12}")
    print("-" * 65)
    
    prev_cache = get_page_cache()
    if prev_cache is None:
        sys.exit(1)
    
    prev_time = time.time()

    try:
        while True:
            time.sleep(interval)
            
            current_cache = get_page_cache()
            if current_cache is None:
                continue
                
            current_time = time.time()
            time_diff = current_time - prev_time
            
            # 计算变化量和速率
            delta = current_cache - prev_cache
            rate = delta / time_diff if time_diff > 0 else 0
            
            # 格式化输出
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"{timestamp} | {current_cache:10.2f} | {delta:10.2f} | {rate:12.2f}")
            
            # 更新前值
            prev_cache = current_cache
            prev_time = current_time
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    # 使用示例：默认1秒间隔，可添加参数控制间隔时间
    interval = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    main(interval)