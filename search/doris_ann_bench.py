import logging
from query import get_conn
from faiss.contrib.datasets import DatasetSIFT1M
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

dataSet = DatasetSIFT1M()
query = dataSet.get_queries()
groundtruth = dataSet.get_groundtruth()
num_queries, dimension = query.shape

# Connect to Doris
conn = get_conn()
cursor = conn.cursor()

# cursor.execute("DROP DATABASE IF EXISTS sift")
# cursor.execute("CREATE DATABASE sift")
cursor.execute("USE sift_1m")

# hnsw_ef_search = [16, 32, 64, 128, 256]
hnsw_ef_search = [64]
hnsw_bounded_queue = [True]
topN_list = [20]

for ef_search in hnsw_ef_search:
    for bounded_queue in hnsw_bounded_queue:
        logging.info(f"Testing HNSW Flat with efSearch={ef_search}, bounded_queue={bounded_queue}")

        # 存储每个topN的检索结果
        retrieved_ids_dict = {n: [] for n in topN_list}

        for q_idx, q in enumerate(query):
            for topN in topN_list:
                # 假设 embedding 是向量字段，l2_distance_approximate 可用
                # 需要将q转为合适的格式传递给SQL
                q_str = ','.join([str(float(x)) for x in q])
                sql = (
                    f"SELECT id FROM sift_1m_dim_128_num_1000000_2 "
                    f"ORDER BY l2_distance_approximate(embedding, [{q_str}]) ASC LIMIT {topN}"
                )
                logging.info(f"Executing SQL: {sql}")
                cursor.execute(sql)
                ids = [row[0] for row in cursor.fetchall()]
                # 只保留id
                retrieved_ids_dict[topN].append(ids)

        # 计算recall@N
        for topN in topN_list:
            retrieved_ids = np.array(retrieved_ids_dict[topN])  # shape: (num_queries, topN)
            gt_topN = groundtruth[:, :topN]  # shape: (num_queries, topN)
            # 判断每个query的检索结果是否命中任一真实近邻
            hit = np.array([
                len(set(retrieved_ids[i]) & set(gt_topN[i])) > 0
                for i in range(num_queries)
            ])
            recall = hit.sum() / num_queries
            logging.info(f"Recall@{topN}: {recall:.4f}")
