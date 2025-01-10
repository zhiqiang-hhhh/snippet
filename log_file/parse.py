import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# 解析日志文件函数（基于空格分隔）
def parse_log_file(file_path, time_idx, value_idx):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()  # 按空格分割
            if len(parts) > max(time_idx, value_idx):  # 确保索引合法
                time_str = parts[1]
                value_str = parts[value_idx]  # 提取数值字段
                if value_str.isdigit():  # 检查是否是数字
                    value = int(value_str)
                    time = datetime.strptime(time_str, "%H:%M:%S.%f")  # 转换为时间
                    data.append({'time': time, 'value': value})
                else:
                    print(f"Skipping line due to non-numeric value: {line.strip()}")

    df = pd.DataFrame(data)
    print(f"Parsed DataFrame from {file_path}:\n", df.head())  # 打印解析后的 DataFrame
    return df

# 文件路径
file_a = "/mnt/disk1/hezhiqiang/Code/snippet/log_file/current"
file_b = "/mnt/disk1/hezhiqiang/Code/snippet/log_file/remaing"

df_concurrency = parse_log_file(file_a, time_idx=0, value_idx=7)
df_task_queue = parse_log_file(file_a, time_idx=0, value_idx=9)
df_schedule_task = parse_log_file(file_a, time_idx=0, value_idx=11)
df_task_to_submit = parse_log_file(file_a, time_idx=0, value_idx=11)


# B 文件中：时间戳在第 0 和第 1 列，并发数在第 8 列
df_b = parse_log_file(file_b, time_idx=0, value_idx=13)

# 为区分数据来源，加上一个来源列
# df_concurrency['source'] = 'current_concurrency'
df_task_queue['source'] = 'current_task_queue'
df_schedule_task['source'] = 'current_schedule_task'
# df_task_to_submit['source'] = 'current_task_to_submit'
df_b['source'] = 'remaing_scanner'

# 合并数据
df = pd.concat([df_task_queue, df_schedule_task, df_b], ignore_index=True)

# 按时间排序
df = df.sort_values('time')

# 绘图
plt.figure(figsize=(48, 24))
for source, group in df.groupby('source'):
    plt.plot(group['time'], group['value'], label=f"Source {source}")

plt.title("Time Series Data from Files A and B (Microsecond Precision)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.grid(True)

# 保存图表到文件
output_file = "output_plot.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')  # 保存为高分辨率 PNG 文件
print(f"Plot saved to {output_file}")
