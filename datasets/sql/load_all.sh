#!/usr/bin/env bash
# ============================================================
# 批量 Stream Load + ETL 导入脚本
# 用法: bash load_all.sh <fe_host> <http_port> <query_port> <db> <user> [password]
# 示例: bash load_all.sh 127.0.0.1 8930 9930 test_demo root
#
# 流程:
#   Step 0: 创建数据库
#   Step 1: 建表 (执行 0*.sql DDL)
#   Step 2: Stream Load 导入本地数据文件
#            - 128D: 全部 4 个表 (1w/10w/50w/100w) 直接 stream load
#            - 768D: 只 stream load cohere_user_1w
#   Step 3: ETL 扩展 768D 表
#            - cohere_user_1w  -> cohere_user_10w  (INSERT INTO SELECT x10)
#            - cohere_user_10w -> cohere_user_50w  (INSERT INTO SELECT x5)
#            - cohere_user_50w -> cohere_user_100w (INSERT INTO SELECT x2)
#   Step 4: 验证数据量
#
# 可选环境变量:
#   PARALLEL=4        并行导入路数 (默认4)
#   SKIP_DDL=1        跳过建库/建表步骤
#   TABLES="1w 10w"   只导入指定规模的表 (默认全部)
#   DIMS="768d"       只导入指定维度: 128d, 768d, all (默认 all)
# ============================================================

set -euo pipefail

FE_HOST="${1:?用法: $0 <fe_host> <http_port> <query_port> <db> <user> [password]}"
HTTP_PORT="${2:?缺少 http_port (Stream Load 用)}"
QUERY_PORT="${3:?缺少 query_port (MySQL 协议端口)}"
DB="${4:?缺少数据库名}"
USER="${5:?缺少用户名}"
PASSWORD="${6:-}"
PARALLEL="${PARALLEL:-4}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$(dirname "$SCRIPT_DIR")"

BASE_URL="http://${FE_HOST}:${HTTP_PORT}/api/${DB}"

# 构建 mysql 命令（密码为空时不加 -p）
if [ -n "$PASSWORD" ]; then
  MYSQL_CMD="mysql -h ${FE_HOST} -P ${QUERY_PORT} -u ${USER} -p${PASSWORD}"
else
  MYSQL_CMD="mysql -h ${FE_HOST} -P ${QUERY_PORT} -u ${USER}"
fi

# --------------------------------------------------
# 创建临时的 shard 导入脚本 (避免 export -f 的兼容性问题)
# --------------------------------------------------
LOAD_HELPER=$(mktemp /tmp/load_shard_XXXXXX.sh)
trap 'rm -f "$LOAD_HELPER"' EXIT

cat > "$LOAD_HELPER" <<'HELPER_EOF'
#!/usr/bin/env bash
# 参数: <table> <shard_file> <base_url> <user> <password>
set -euo pipefail

TABLE="$1"
SHARD_FILE="$2"
BASE_URL="$3"
LOAD_USER="$4"
LOAD_PASSWORD="$5"

SHARD_NAME="$(basename "$SHARD_FILE")"
LABEL="${TABLE}_${SHARD_NAME%.tsv.gz}_$(date +%s%N)"

echo "[LOAD] ${TABLE} <- ${SHARD_NAME}"

result=$(zcat "$SHARD_FILE" | \
  curl --location-trusted -s \
    -u "${LOAD_USER}:${LOAD_PASSWORD}" \
    -H "column_separator:\t" \
    -H "columns: user_id, id, embedding" \
    -H "label:${LABEL}" \
    -T - \
    "${BASE_URL}/${TABLE}/_stream_load" 2>&1)

# 用 python3 提取 Status 字段; 如果 JSON 解析失败则用 grep 兜底
status=$(echo "$result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('Status', 'UNKNOWN'))
except Exception:
    print('PARSE_ERROR')
" 2>/dev/null)

if [ -z "$status" ]; then
  status="PARSE_ERROR"
fi

if [ "$status" = "Success" ]; then
  # 提取行数
  loaded=$(echo "$result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    print(d.get('NumberLoadedRows', '?'))
except Exception:
    print('?')
" 2>/dev/null)
  echo "[OK]   ${TABLE} <- ${SHARD_NAME} (${loaded} rows)"
else
  echo "[FAIL] ${TABLE} <- ${SHARD_NAME}: status=${status}"
  # 截取前500字符输出方便调试
  echo "       ${result:0:500}"
  exit 1
fi
HELPER_EOF
chmod +x "$LOAD_HELPER"

# --------------------------------------------------
# 定义表与数据集的映射
# --------------------------------------------------
# --------------------------------------------------
# 128D (SIFT) 表映射 — 全部 4 个规模都有本地数据文件
# --------------------------------------------------
declare -A TABLE_MAP_128D=(
  ["1w"]="sift_user_1w"
  ["10w"]="sift_user_10w"
  ["50w"]="sift_user_50w"
  ["100w"]="sift_user_100w"
)
declare -A DATASET_MAP_128D=(
  ["1w"]="dataset_1w"
  ["10w"]="dataset_10w"
  ["50w"]="dataset_50w"
  ["100w"]="dataset_100w"
)

# --------------------------------------------------
# 768D (Cohere) 表映射
# Stream load 只导入 1w; 10w/50w/100w 由 Step 3 ETL 扩展
# --------------------------------------------------
declare -A TABLE_MAP_768D=(
  ["1w"]="cohere_user_1w"
  ["10w"]="cohere_user_10w"
  ["50w"]="cohere_user_50w"
  ["100w"]="cohere_user_100w"
)
declare -A DATASET_MAP_768D=(
  ["1w"]="dataset_768d_1w"
)

# 支持 TABLES 环境变量指定只导入部分规模 (默认全部)
SCALE_LIST="${TABLES:-1w 10w 50w 100w}"
# 支持 DIMS 环境变量指定维度: 128d, 768d, 或 all (默认 all)
DIMS="${DIMS:-all}"

# --------------------------------------------------
# Step 0: 创建数据库
# --------------------------------------------------
if [ "${SKIP_DDL:-0}" != "1" ]; then
  echo "========================================"
  echo "Step 0: 创建数据库 ${DB}"
  echo "========================================"
  ${MYSQL_CMD} -e "CREATE DATABASE IF NOT EXISTS ${DB};" 2>&1 && \
    echo "[OK] 数据库 ${DB} 已就绪" || \
    echo "[WARN] 创建数据库失败，请手动执行: CREATE DATABASE IF NOT EXISTS ${DB};"

  # --------------------------------------------------
  # Step 1: 建表
  # --------------------------------------------------
  echo ""
  echo "========================================"
  echo "Step 1: 建表"
  echo "========================================"
  for sql_file in "${SCRIPT_DIR}"/0*.sql; do
    echo "[SQL] 执行 $(basename "$sql_file")"
    ${MYSQL_CMD} -D "${DB}" < "$sql_file" 2>&1 && \
      echo "[OK]   $(basename "$sql_file")" || \
      echo "[FAIL] $(basename "$sql_file") - 请手动执行"
  done
fi

# --------------------------------------------------
# Step 2: Stream Load 导入本地数据文件
# --------------------------------------------------
LOAD_ERRORS=0

# 构建要导入的 (table, dataset) 列表
declare -a LOAD_PAIRS=()

for scale in ${SCALE_LIST}; do
  # 128D: 所有规模都有本地数据文件
  if [ "$DIMS" = "all" ] || [ "$DIMS" = "128d" ]; then
    if [ -n "${TABLE_MAP_128D[$scale]+x}" ] && [ -n "${DATASET_MAP_128D[$scale]+x}" ]; then
      table="${TABLE_MAP_128D[$scale]}"
      dataset="${DATASET_MAP_128D[$scale]}"
      LOAD_PAIRS+=("${table}:${dataset}")
    fi
  fi
  # 768D: 只有 1w 有本地数据文件 (10w/50w/100w 由 ETL 扩展)
  if [ "$DIMS" = "all" ] || [ "$DIMS" = "768d" ]; then
    if [ -n "${TABLE_MAP_768D[$scale]+x}" ] && [ -n "${DATASET_MAP_768D[$scale]+x}" ]; then
      table="${TABLE_MAP_768D[$scale]}"
      dataset="${DATASET_MAP_768D[$scale]}"
      LOAD_PAIRS+=("${table}:${dataset}")
    fi
  fi
done

for pair in "${LOAD_PAIRS[@]}"; do
  table="${pair%%:*}"
  dataset="${pair##*:}"
  shard_dir="${DATA_DIR}/${dataset}"

  if [ ! -d "$shard_dir" ]; then
    echo "[SKIP] ${shard_dir} 不存在，跳过 ${table}"
    continue
  fi

  shard_count=$(find "${shard_dir}" -name 'shard_*.tsv.gz' | wc -l)

  echo ""
  echo "========================================"
  echo "Step 2: 导入 ${table} (${shard_count} shards, ${PARALLEL} parallel)"
  echo "========================================"

  # 使用 find + xargs 调用外部脚本，避免 export -f 问题
  # xargs 会将退出码传播回来，-P 控制并行度
  find "${shard_dir}" -name 'shard_*.tsv.gz' -print0 | sort -z | \
    xargs -0 -I{} -P "${PARALLEL}" \
      bash "$LOAD_HELPER" "$table" "{}" "$BASE_URL" "$USER" "$PASSWORD" \
    || LOAD_ERRORS=$((LOAD_ERRORS + 1))

  echo "[DONE] ${table} 导入完成"
done

# --------------------------------------------------
# Step 3: ETL 扩展 768D 表 (1w -> 10w -> 50w -> 100w)
#
# 通过 INSERT INTO SELECT 复制数据并偏移 id，避免本地生成大文件。
# 每一步从较小的表复制多批到较大的表:
#   cohere_user_1w  (100用户 x 10,000)  --x10--> cohere_user_10w  (100用户 x 100,000)
#   cohere_user_10w (100用户 x 100,000) --x5-->  cohere_user_50w  (100用户 x 500,000)
#   cohere_user_50w (100用户 x 500,000) --x2-->  cohere_user_100w (100用户 x 1,000,000)
#
# 每批 id 偏移 = batch_index * (源表每用户向量数), 保证 id 不冲突。
# 注意: 扩展后的数据向量值有重复 (同一向量不同 id), 不影响暴力搜索基准测试。
# --------------------------------------------------
if [ "$DIMS" = "all" ] || [ "$DIMS" = "768d" ]; then
  echo ""
  echo "========================================"
  echo "Step 3: ETL 扩展 768D 表 (cohere_user_1w -> 10w -> 50w -> 100w)"
  echo "========================================"

  # --- 1w -> 10w: 复制 10 批, 每批偏移 10000 (1w 每用户有 10,000 向量) ---
  echo ""
  echo "[ETL] cohere_user_1w -> cohere_user_10w (10 batches, id offset 10000)"
  for batch in $(seq 0 9); do
    offset=$((batch * 10000))
    echo "  [ETL] batch ${batch}/9, id offset = ${offset}"
    ${MYSQL_CMD} -D "${DB}" -e "
      INSERT INTO cohere_user_10w (user_id, id, embedding)
      SELECT user_id, id + ${offset}, embedding
      FROM cohere_user_1w;
    " 2>&1 || { echo "  [FAIL] ETL batch ${batch} for cohere_user_10w"; LOAD_ERRORS=$((LOAD_ERRORS + 1)); }
  done
  echo "[DONE] cohere_user_10w ETL 完成"

  # --- 10w -> 50w: 复制 5 批, 每批偏移 100000 (10w 每用户有 100,000 向量) ---
  echo ""
  echo "[ETL] cohere_user_10w -> cohere_user_50w (5 batches, id offset 100000)"
  for batch in $(seq 0 4); do
    offset=$((batch * 100000))
    echo "  [ETL] batch ${batch}/4, id offset = ${offset}"
    ${MYSQL_CMD} -D "${DB}" -e "
      INSERT INTO cohere_user_50w (user_id, id, embedding)
      SELECT user_id, id + ${offset}, embedding
      FROM cohere_user_10w;
    " 2>&1 || { echo "  [FAIL] ETL batch ${batch} for cohere_user_50w"; LOAD_ERRORS=$((LOAD_ERRORS + 1)); }
  done
  echo "[DONE] cohere_user_50w ETL 完成"

  # --- 50w -> 100w: 复制 2 批, 每批偏移 500000 (50w 每用户有 500,000 向量) ---
  echo ""
  echo "[ETL] cohere_user_50w -> cohere_user_100w (2 batches, id offset 500000)"
  for batch in $(seq 0 1); do
    offset=$((batch * 500000))
    echo "  [ETL] batch ${batch}/1, id offset = ${offset}"
    ${MYSQL_CMD} -D "${DB}" -e "
      INSERT INTO cohere_user_100w (user_id, id, embedding)
      SELECT user_id, id + ${offset}, embedding
      FROM cohere_user_50w;
    " 2>&1 || { echo "  [FAIL] ETL batch ${batch} for cohere_user_100w"; LOAD_ERRORS=$((LOAD_ERRORS + 1)); }
  done
  echo "[DONE] cohere_user_100w ETL 完成"
fi

# --------------------------------------------------
# Step 4: 验证
# --------------------------------------------------
echo ""
echo "========================================"
echo "Step 4: 验证数据量"
echo "========================================"

EXPECTED=(
  "sift_user_1w:1000000"
  "sift_user_10w:10000000"
  "sift_user_50w:50000000"
  "sift_user_100w:100000000"
  "cohere_user_1w:1000000"
  "cohere_user_10w:10000000"
  "cohere_user_50w:50000000"
  "cohere_user_100w:100000000"
)

for entry in "${EXPECTED[@]}"; do
  table="${entry%%:*}"
  expected="${entry##*:}"
  count=$(${MYSQL_CMD} -D "${DB}" -N -e "SELECT COUNT(*) FROM ${table};" 2>/dev/null || echo "N/A")
  if [ "$count" = "$expected" ]; then
    echo "  [OK]   ${table}: ${count} rows (expected ${expected})"
  else
    echo "  [WARN] ${table}: ${count} rows (expected ${expected})"
  fi
done

echo ""
if [ "$LOAD_ERRORS" -gt 0 ]; then
  echo "完成，但有 ${LOAD_ERRORS} 个表导入过程中出现错误，请检查上面的日志。"
  exit 1
else
  echo "全部完成!"
fi
