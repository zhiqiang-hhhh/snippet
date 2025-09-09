#!/usr/bin/env python3
"""
PQ量化参数M影响分析 - 优化版本

针对实际可行的参数范围进行测试
"""

import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

def pq_parameter_demo():
    """
    PQ参数M影响演示
    """
    print("🚀 PQ量化参数M影响分析Demo")
    print("=" * 60)
    
    # 配置参数
    dim = 128
    nb = 20000  # 减少数据量，避免训练时间过长
    nq = 500
    k = 10
    nbits = 4  # 使用4位，每个子量化器16个聚类中心
    
    print(f"配置: dim={dim}, nb={nb}, nq={nq}, k={k}, nbits={nbits}")
    print(f"每个子量化器聚类中心数: {2**nbits}")
    
    # 生成测试数据
    np.random.seed(42)
    database = np.random.randn(nb, dim).astype(np.float32)
    queries = np.random.randn(nq, dim).astype(np.float32)
    
    # 规范化
    faiss.normalize_L2(database)
    faiss.normalize_L2(queries)
    
    # 计算真实最近邻
    print("计算真实最近邻...")
    index_flat = faiss.IndexFlatL2(dim)
    index_flat.add(database)
    _, ground_truth = index_flat.search(queries, k)
    
    # 测试不同M值
    m_values = [4, 8, 16, 32, 64]  # 选择合理的M值范围
    results = []
    
    print(f"\n开始测试M值: {m_values}")
    print("-" * 60)
    
    for m in m_values:
        if dim % m != 0:
            print(f"跳过 M={m}: 维度{dim}不能被M整除")
            continue
            
        print(f"\n🔬 测试 M={m} (子向量维度: {dim//m})")
        
        try:
            # 1. 测试纯PQ索引
            print("  测试 IndexPQ...")
            pq_result = test_pq_index(database, queries, ground_truth, dim, m, nbits, k)
            
            # 2. 测试HNSW+PQ索引  
            print("  测试 IndexHNSWPQ...")
            hnswpq_result = test_hnswpq_index(database, queries, ground_truth, dim, m, nbits, k)
            
            # 合并结果
            result = {
                'M': m,
                'sub_dim': dim // m,
                'clusters_per_subvector': 2**nbits,
                **pq_result,
                **{f'hnsw_{k}': v for k, v in hnswpq_result.items()}
            }
            
            results.append(result)
            
            # 打印结果
            print(f"    PQ: 构建={pq_result['build_time']:.2f}s, "
                  f"搜索={pq_result['search_time_ms']:.2f}ms, "
                  f"召回率@10={pq_result['recall_10']:.3f}")
            print(f"    HNSW+PQ: 构建={hnswpq_result['build_time']:.2f}s, "
                  f"搜索={hnswpq_result['search_time_ms']:.2f}ms, "
                  f"召回率@10={hnswpq_result['recall_10']:.3f}")
            
        except Exception as e:
            print(f"  ❌ 测试M={m}失败: {e}")
            continue
    
    if not results:
        print("❌ 没有成功的测试结果")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(results)
    
    # 绘制结果
    plot_pq_results(df)
    
    # 保存结果
    df.to_csv('pq_analysis_results.csv', index=False)
    print(f"\n💾 结果已保存到: pq_analysis_results.csv")
    
    # 分析总结
    print_analysis(df)
    
    return df

def test_pq_index(database, queries, ground_truth, dim, m, nbits, k):
    """测试纯PQ索引"""
    index = faiss.IndexPQ(dim, m, nbits, faiss.METRIC_L2)
    
    # 训练
    start = time.time()
    index.train(database)
    index.add(database)
    build_time = time.time() - start
    
    # 搜索
    start = time.time()
    distances, labels = index.search(queries, k)
    search_time = (time.time() - start) * 1000 / len(queries)  # ms per query
    
    # 计算召回率
    recall_1 = compute_recall(labels[:, :1], ground_truth[:, :1])
    recall_5 = compute_recall(labels[:, :5], ground_truth[:, :5])
    recall_10 = compute_recall(labels[:, :k], ground_truth[:, :k])
    
    # 内存估算
    memory_mb = len(database) * m * nbits / 8 / (1024 * 1024)
    compression_ratio = (len(database) * dim * 4) / (len(database) * m * nbits / 8)
    
    return {
        'build_time': build_time,
        'search_time_ms': search_time,
        'recall_1': recall_1,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'memory_mb': memory_mb,
        'compression_ratio': compression_ratio
    }

def test_hnswpq_index(database, queries, ground_truth, dim, m, nbits, k, hnsw_m=16):
    """测试HNSW+PQ索引"""
    index = faiss.IndexHNSWPQ(dim, m, nbits, hnsw_m, faiss.METRIC_L2)
    index.hnsw.efConstruction = 100  # 降低构建参数加快速度
    
    # 训练和构建
    start = time.time()
    index.train(database)
    index.add(database)
    build_time = time.time() - start
    
    # 搜索测试
    index.hnsw.efSearch = 32
    start = time.time()
    distances, labels = index.search(queries, k)
    search_time = (time.time() - start) * 1000 / len(queries)  # ms per query
    
    # 计算召回率
    recall_1 = compute_recall(labels[:, :1], ground_truth[:, :1])
    recall_5 = compute_recall(labels[:, :5], ground_truth[:, :5])
    recall_10 = compute_recall(labels[:, :k], ground_truth[:, :k])
    
    # 内存估算
    pq_memory = len(database) * m * nbits / 8
    hnsw_memory = len(database) * hnsw_m * 8  # 近似
    total_memory_mb = (pq_memory + hnsw_memory) / (1024 * 1024)
    
    return {
        'build_time': build_time,
        'search_time_ms': search_time,
        'recall_1': recall_1,
        'recall_5': recall_5,
        'recall_10': recall_10,
        'memory_mb': total_memory_mb
    }

def compute_recall(pred_labels, true_labels):
    """计算召回率"""
    nq, k = pred_labels.shape
    recall_sum = 0.0
    
    for i in range(nq):
        true_set = set(true_labels[i])
        pred_set = set(pred_labels[i])
        if len(true_set) > 0:
            recall_sum += len(true_set.intersection(pred_set)) / len(true_set)
    
    return recall_sum / nq

def plot_pq_results(df):
    """绘制结果图表"""
    # 设置中文字体
    plt.rcParams['font.family'] = ['DejaVu Sans']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PQ Parameter M Impact Analysis', fontsize=14, fontweight='bold')
    
    # 1. 构建时间
    axes[0, 0].plot(df['M'], df['build_time'], 'o-', label='PQ', linewidth=2, markersize=6)
    axes[0, 0].plot(df['M'], df['hnsw_build_time'], 's-', label='HNSW+PQ', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('M (Number of Subquantizers)')
    axes[0, 0].set_ylabel('Build Time (seconds)')
    axes[0, 0].set_title('Index Build Time')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 搜索时间
    axes[0, 1].plot(df['M'], df['search_time_ms'], 'o-', label='PQ', linewidth=2, markersize=6)
    axes[0, 1].plot(df['M'], df['hnsw_search_time_ms'], 's-', label='HNSW+PQ', linewidth=2, markersize=6)
    axes[0, 1].set_xlabel('M (Number of Subquantizers)')
    axes[0, 1].set_ylabel('Search Time (ms/query)')
    axes[0, 1].set_title('Search Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 召回率@10
    axes[0, 2].plot(df['M'], df['recall_10'], 'o-', label='PQ', linewidth=2, markersize=6)
    axes[0, 2].plot(df['M'], df['hnsw_recall_10'], 's-', label='HNSW+PQ', linewidth=2, markersize=6)
    axes[0, 2].set_xlabel('M (Number of Subquantizers)')
    axes[0, 2].set_ylabel('Recall@10')
    axes[0, 2].set_title('Recall@10')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim(0, 1)
    
    # 4. 内存使用
    axes[1, 0].plot(df['M'], df['memory_mb'], 'o-', label='PQ', linewidth=2, markersize=6)
    axes[1, 0].plot(df['M'], df['hnsw_memory_mb'], 's-', label='HNSW+PQ', linewidth=2, markersize=6)
    axes[1, 0].set_xlabel('M (Number of Subquantizers)')
    axes[1, 0].set_ylabel('Memory Usage (MB)')
    axes[1, 0].set_title('Memory Usage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. 压缩比
    axes[1, 1].plot(df['M'], df['compression_ratio'], 'o-', color='purple', linewidth=2, markersize=6)
    axes[1, 1].set_xlabel('M (Number of Subquantizers)')
    axes[1, 1].set_ylabel('Compression Ratio')
    axes[1, 1].set_title('PQ Compression Ratio')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 子向量维度 vs 召回率
    axes[1, 2].plot(df['sub_dim'], df['recall_10'], 'o-', label='PQ', linewidth=2, markersize=6)
    axes[1, 2].plot(df['sub_dim'], df['hnsw_recall_10'], 's-', label='HNSW+PQ', linewidth=2, markersize=6)
    axes[1, 2].set_xlabel('Subvector Dimension (dim/M)')
    axes[1, 2].set_ylabel('Recall@10')
    axes[1, 2].set_title('Subvector Dimension vs Recall')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('pq_analysis_chart.png', dpi=300, bbox_inches='tight')
    print("📊 图表已保存到: pq_analysis_chart.png")
    plt.show()

def print_analysis(df):
    """打印分析结果"""
    print("\n" + "=" * 60)
    print("📈 PQ量化参数M影响分析总结")
    print("=" * 60)
    
    print("\n📊 主要发现:")
    
    # 最佳性能点
    best_pq_recall = df.loc[df['recall_10'].idxmax()]
    best_hnsw_recall = df.loc[df['hnsw_recall_10'].idxmax()]
    fastest_pq = df.loc[df['search_time_ms'].idxmin()]
    fastest_hnsw = df.loc[df['hnsw_search_time_ms'].idxmin()]
    
    print(f"🎯 PQ最佳召回率: M={best_pq_recall['M']}, 召回率={best_pq_recall['recall_10']:.3f}")
    print(f"🎯 HNSW+PQ最佳召回率: M={best_hnsw_recall['M']}, 召回率={best_hnsw_recall['hnsw_recall_10']:.3f}")
    print(f"⚡ PQ最快搜索: M={fastest_pq['M']}, 时间={fastest_pq['search_time_ms']:.2f}ms")
    print(f"⚡ HNSW+PQ最快搜索: M={fastest_hnsw['M']}, 时间={fastest_hnsw['hnsw_search_time_ms']:.2f}ms")
    
    print(f"\n📋 关键趋势:")
    print(f"• M增加 → 子向量维度减少 → 量化精度可能降低")
    print(f"• M增加 → 子量化器数量增加 → 训练时间增加")
    print(f"• M适中时通常有最佳的精度/速度平衡")
    
    print(f"\n📊 详细结果:")
    display_cols = ['M', 'sub_dim', 'recall_10', 'search_time_ms', 'hnsw_recall_10', 'hnsw_search_time_ms']
    print(df[display_cols].to_string(index=False, float_format='%.3f'))
    
    print(f"\n💡 推荐:")
    # 找到平衡点（召回率和速度的综合评分）
    df['pq_score'] = df['recall_10'] * 0.7 - (df['search_time_ms'] / df['search_time_ms'].max()) * 0.3
    df['hnsw_score'] = df['hnsw_recall_10'] * 0.7 - (df['hnsw_search_time_ms'] / df['hnsw_search_time_ms'].max()) * 0.3
    
    best_pq_balanced = df.loc[df['pq_score'].idxmax()]
    best_hnsw_balanced = df.loc[df['hnsw_score'].idxmax()]
    
    print(f"📈 PQ平衡配置: M={best_pq_balanced['M']} (综合评分最高)")
    print(f"📈 HNSW+PQ平衡配置: M={best_hnsw_balanced['M']} (综合评分最高)")

if __name__ == "__main__":
    df = pq_parameter_demo()
