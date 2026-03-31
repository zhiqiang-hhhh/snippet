# Node Exporter + Prometheus + Grafana 一键部署

基于 shell 脚本 + parallel-ssh 的二进制自动化部署方案，适用于无外网 Docker 环境的内网机器。

## 架构

```
                   +--------------------------+
                   |   172.20.56.74           |
                   |   /mnt/disk2/hzq         |
                   |                          |
                   |  +--------+  +--------+  |
  浏览器 --------->|  |Grafana |  |Promethe|  |
  :3000            |  | :3000  |  |us :9090|  |
                   |  +--------+  +---+----+  |
                   |  +-------------+ |       |
                   |  |Node Exporter| |       |
                   |  |   :9100     | |       |
                   |  +-------------+ |       |
                   +------------------+-------+
                                      | scrape (每15s)
              +-----------------------+-----------------------+
              |                       |                       |
  +-----------+--------+  +-----------+--------+  +-----------+--------+
  | 172.20.56.83       |  | 172.20.56.84       |  | 172.20.56.85       |
  | /mnt/disk1/hzq     |  | /mnt/disk1/hzq     |  | /mnt/disk1/hzq     |
  | +----------------+ |  | +----------------+ |  | +----------------+ |
  | |Node Exporter   | |  | |Node Exporter   | |  | |Node Exporter   | |
  | |   :9100        | |  | |   :9100        | |  | |   :9100        | |
  | +----------------+ |  | +----------------+ |  | +----------------+ |
  +--------------------+  +--------------------+  +--------------------+
```

## 组件版本

| 组件 | 版本 | 说明 |
|---|---|---|
| Node Exporter | v1.8.2 | 系统指标采集 |
| Prometheus | v2.53.3 | 指标存储与查询 |
| Grafana OSS | v11.4.0 | 可视化面板 |

## 机器清单

| 机器 | 部署组件 | 安装目录 | 端口 |
|---|---|---|---|
| 172.20.56.74 | Prometheus + Grafana + Node Exporter | `/mnt/disk2/hzq` | 9090 / 3000 / 9100 |
| 172.20.56.83 | Node Exporter | `/mnt/disk1/hzq` | 9100 |
| 172.20.56.84 | Node Exporter | `/mnt/disk1/hzq` | 9100 |
| 172.20.56.85 | Node Exporter | `/mnt/disk1/hzq` | 9100 |

## 前置条件

- 74 机器已配置到 83/84/85 的 **root 免密 SSH**
- 74 机器有 **sudo/root** 权限（注册 systemd 服务需要）
- 所有机器为 **linux amd64** 架构
- 端口 3000 / 9090 / 9100 未被占用

## 使用方法

### 1. 将本目录拷贝到 74 机器

```bash
scp -r deploy_monitoring root@172.20.56.74:/mnt/disk2/hzq/
```

### 2. 在 74 上执行脚本

```bash
ssh root@172.20.56.74
cd /mnt/disk2/hzq/deploy_monitoring
bash deploy_monitoring.sh
```

### 3. 脚本执行流程

脚本全自动完成以下 6 个步骤：

| 步骤 | 内容 |
|---|---|
| **Step 0** | 前置检查：自动安装 pssh（如缺失）、SSH 连通性测试、创建目录 |
| **Step 1** | 下载二进制包：Node Exporter / Prometheus / Grafana（优先国内 GitHub 镜像，Grafana 走官方 CDN `dl.grafana.com`），已有包则跳过 |
| **Step 2** | 部署 Node Exporter 到全部 4 台机器（pssh 批量分发到 83/84/85 + 本机 74），注册 systemd 服务并启动 |
| **Step 3** | 部署 Prometheus 到 74，自动生成 `prometheus.yml`（包含 4 个 node exporter targets），注册 systemd 服务并启动 |
| **Step 4** | 部署 Grafana OSS 到 74，生成 `custom.ini` 配置 + provisioning 自动配置 Prometheus 数据源，注册 systemd 服务并启动 |
| **Step 5** | 导入 Node Exporter Full Dashboard（Grafana ID: 1860），下载失败则自动生成简化版 Dashboard（CPU / Memory / Disk / Network / Load / Uptime 6 个面板） |
| **Step 6** | 最终验证：逐一检查所有服务端口可达性，输出访问地址汇总 |

### 4. 手动提前下载包（可选）

如果机器完全无法访问外网，可以手动下载以下 3 个包，放到脚本同目录的 `packages/` 下：

```
deploy_monitoring/
├── deploy_monitoring.sh
└── packages/
    ├── node_exporter-1.8.2.linux-amd64.tar.gz
    ├── prometheus-2.53.3.linux-amd64.tar.gz
    └── grafana-11.4.0.linux-amd64.tar.gz
```

下载地址：

- Node Exporter: https://github.com/prometheus/node_exporter/releases/download/v1.8.2/node_exporter-1.8.2.linux-amd64.tar.gz
- Prometheus: https://github.com/prometheus/prometheus/releases/download/v2.53.3/prometheus-2.53.3.linux-amd64.tar.gz
- Grafana OSS: https://dl.grafana.com/oss/release/grafana-11.4.0.linux-amd64.tar.gz

脚本检测到 `packages/` 目录下已有对应文件时，会自动跳过下载步骤。

## 部署完成后

### 访问地址

```
Grafana:    http://172.20.56.74:3000    (用户名: admin / 密码: admin，首次登录需修改密码)
Prometheus: http://172.20.56.74:9090
```

### 目录结构

```
# 74 机器 (/mnt/disk2/hzq)
├── node_exporter/
│   └── node_exporter                    # 二进制文件
├── prometheus/
│   ├── prometheus                       # 二进制文件
│   ├── promtool                         # 配置校验工具
│   ├── prometheus.yml                   # 采集配置（可编辑添加 targets）
│   ├── consoles/                        # 内置 console 模板
│   ├── console_libraries/
│   └── data/                            # TSDB 时序数据（默认保留 30 天）
└── grafana/
    ├── bin/grafana                       # 二进制文件
    ├── conf/
    │   ├── defaults.ini                  # 默认配置（勿改）
    │   └── custom.ini                   # 自定义配置（改这个）
    ├── data/                             # SQLite DB + 日志 + 插件
    │   ├── grafana.db
    │   └── log/
    └── provisioning/
        ├── datasources/prometheus.yml   # 自动配置的 Prometheus 数据源
        └── dashboards/
            ├── default.yml              # Dashboard provisioning 配置
            └── json/
                └── node-exporter-full.json  # 预装的 Dashboard

# 83/84/85 机器 (/mnt/disk1/hzq)
└── node_exporter/
    └── node_exporter                    # 二进制文件
```

### 服务管理

所有组件通过 systemd 管理，支持开机自启。

```bash
# 在 74 上
systemctl {start|stop|restart|status} prometheus
systemctl {start|stop|restart|status} grafana
systemctl {start|stop|restart|status} node_exporter

# 在 83/84/85 上（单台）
ssh root@172.20.56.83 "systemctl restart node_exporter"

# 批量操作（在 74 上执行）
echo -e "172.20.56.83\n172.20.56.84\n172.20.56.85" > /tmp/ne_hosts
pssh -h /tmp/ne_hosts -l root "systemctl restart node_exporter"
```

### 查看日志

```bash
journalctl -u prometheus -f
journalctl -u grafana -f
journalctl -u node_exporter -f

# 远程查看
ssh root@172.20.56.83 "journalctl -u node_exporter -f"
```

## 日常运维

### 添加新的监控机器

1. 在新机器上部署 Node Exporter（参考脚本中 `deploy_node_exporter` 函数）
2. 编辑 Prometheus 配置，添加 target：

```bash
vim /mnt/disk2/hzq/prometheus/prometheus.yml
```

在 `node_exporter` job 的 `targets` 列表中添加新 IP：

```yaml
  - job_name: "node_exporter"
    static_configs:
      - targets:
        - '172.20.56.74:9100'
        - '172.20.56.83:9100'
        - '172.20.56.84:9100'
        - '172.20.56.85:9100'
        - '172.20.56.NEW:9100'    # 新增
```

3. 热加载配置（无需重启）：

```bash
curl -X POST http://localhost:9090/-/reload
```

> 已通过 `--web.enable-lifecycle` 启用热加载。

### 修改 Prometheus 数据保留时间

默认保留 30 天。如需调整，编辑 systemd unit 文件：

```bash
vim /etc/systemd/system/prometheus.service
# 修改 --storage.tsdb.retention.time=30d 为期望值
systemctl daemon-reload
systemctl restart prometheus
```

### 手动导入 Grafana Dashboard

如果自动导入的 Dashboard 不满足需求，可在 Grafana UI 中手动导入：

1. 访问 http://172.20.56.74:3000
2. 左侧菜单 -> Dashboards -> Import
3. 输入 Dashboard ID（推荐）：
   - **1860** - Node Exporter Full（最全面）
   - **11074** - Node Exporter for Prometheus（简洁版）
   - **13978** - Node Exporter Quickstart
4. 选择 Prometheus 数据源 -> Import

### 配置告警（可选扩展）

在 `prometheus.yml` 同级目录创建告警规则文件：

```bash
mkdir -p /mnt/disk2/hzq/prometheus/rules
cat > /mnt/disk2/hzq/prometheus/rules/node_alerts.yml << 'EOF'
groups:
  - name: node_alerts
    rules:
      - alert: HighCpuUsage
        expr: 100 - (avg by(instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "CPU 使用率过高 ({{ $labels.instance }})"
          description: "CPU 使用率超过 85%，当前值: {{ $value }}%"

      - alert: HighMemoryUsage
        expr: (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100 > 90
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "内存使用率过高 ({{ $labels.instance }})"
          description: "内存使用率超过 90%，当前值: {{ $value }}%"

      - alert: DiskSpaceLow
        expr: (1 - node_filesystem_avail_bytes{fstype!="tmpfs"} / node_filesystem_size_bytes{fstype!="tmpfs"}) * 100 > 85
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "磁盘空间不足 ({{ $labels.instance }})"
          description: "磁盘使用率超过 85%，挂载点: {{ $labels.mountpoint }}，当前值: {{ $value }}%"

      - alert: NodeDown
        expr: up{job="node_exporter"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "节点离线 ({{ $labels.instance }})"
          description: "Node Exporter 无法连接超过 1 分钟"
EOF
```

然后取消 `prometheus.yml` 中 `rule_files` 的注释：

```yaml
rule_files:
  - "rules/*.yml"
```

热加载生效：

```bash
curl -X POST http://localhost:9090/-/reload
```

## 故障排查

### 服务启动失败

```bash
# 查看详细错误
journalctl -u <服务名> --no-pager -n 50

# 检查端口占用
ss -tlnp | grep -E '3000|9090|9100'

# 手动启动查看错误输出
/mnt/disk2/hzq/prometheus/prometheus --config.file=/mnt/disk2/hzq/prometheus/prometheus.yml
```

### Prometheus targets 显示 DOWN

1. 检查目标机器 Node Exporter 是否运行：`ssh root@目标IP "systemctl status node_exporter"`
2. 检查网络连通性：`curl http://目标IP:9100/metrics`
3. 检查防火墙：`iptables -L -n | grep 9100`

### Grafana 数据源连接失败

1. 确认 Prometheus 运行正常：`curl http://localhost:9090/-/ready`
2. 在 Grafana 中：Settings -> Data Sources -> Prometheus -> Test，查看错误信息

### 包下载失败

脚本会依次尝试以下下载源：
1. 国内 GitHub 镜像（ghproxy.com 等）
2. GitHub 官方直连
3. Grafana 官方 CDN（`dl.grafana.com`）

如果全部失败，手动下载后放入 `packages/` 目录重新执行即可（脚本检测到文件存在会跳过下载）。
