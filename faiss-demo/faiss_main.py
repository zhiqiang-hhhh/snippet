import numpy as np
import faiss

# 数据库中的向量（4 个二维向量）
xb = np.array([
    [1.0, 0.0],   # inner product = 1
    [0.0, 1.0],   # inner product = 1
    [1.0, 1.0],   # inner product = 2
    [-1.0, -1.0], # inner product = -2
], dtype='float32')

# 查询向量（1 个）
xq = np.array([
    [1.0, 1.0]
], dtype='float32')

# 创建索引
index = faiss.IndexFlatIP(2)
index.add(xb)

# 做 range search，设置 inner product 阈值为 1.5
threshold = 1.5
lims, distances, labels = index.range_search(xq, threshold)

print("labels (indices):", labels)
print("distances (inner products):", distances)
print("lims:", lims)
