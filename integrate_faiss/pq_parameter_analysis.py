#!/usr/bin/env python3
"""
PQ量化参数M对索引性能影响的分析Demo

这个脚本测试不同的M值（PQ子量化器数量）对以下指标的影响：
1. 索引构建时间
2. 内存使用量
3. 搜索速度
4. 召回率精度

作者: GitHub Copilot
日期: 2025-09-09
"""

import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class PQBenchmark:
    def __init__(self, dim: int = 128, nb: int = 50000, nq: int = 1000, k: int = 10):
        """
        初始化PQ基准测试
        
        Args:
            dim: 向量维度
            nb: 数据库向量数量
            nq: 查询向量数量  
            k: 搜索近邻数量
        """
        self.dim = dim
        self.nb = nb
        self.nq = nq
        self.k = k
        
        # 生成测试数据
        print(f"生成测试数据: dim={dim}, nb={nb}, nq={nq}")
        np.random.seed(42)
        self.database = np.random.randn(nb, dim).astype(np.float32)
        self.queries = np.random.randn(nq, dim).astype(np.float32)
        
        # 规范化向量以获得更好的性能
        faiss.normalize_L2(self.database)
        faiss.normalize_L2(self.queries)
        
        # 计算真实的最近邻（用于召回率计算）
        print("计算真实最近邻（用于召回率评估）...")
        self._compute_ground_truth()
        
    def _compute_ground_truth(self):
        """计算真实的最近邻作为召回率评估的基准"""
        index_flat = faiss.IndexFlatL2(self.dim)
        index_flat.add(self.database)
        
        _, self.ground_truth = index_flat.search(self.queries, self.k)
        
    def test_pq_parameter(self, m_values: List[int], nbits: int = 6, 
                         hnsw_m: int = 16, ef_construction: int = 200) -> pd.DataFrame:
        """
        测试不同M值对PQ性能的影响
        
        Args:
            m_values: 要测试的M值列表（子量化器数量）
            nbits: 每个PQ码的位数
            hnsw_m: HNSW图的连接数
            ef_construction: HNSW构建时的ef参数
            
        Returns:
            包含测试结果的DataFrame
        """
        results = []
        
        print(f"\n开始测试PQ参数，M值范围: {m_values}")
        print(f"其他参数: nbits={nbits}, hnsw_m={hnsw_m}, ef_construction={ef_construction}")
        print("=" * 80)
        
        for m in m_values:
            if self.dim % m != 0:
                print(f"跳过 M={m}: 维度{self.dim}不能被M整除")
                continue
                
            print(f"\n测试 M={m} (子向量维度: {self.dim//m})")
            print("-" * 40)
            
            try:
                # 测试结果字典
                result = {
                    'M': m,
                    'sub_dim': self.dim // m,
                    'nbits': nbits,
                    'codebook_size': 2**nbits,
                    'compression_ratio': None,
                    'build_time': None,
                    'memory_usage_mb': None,
                    'search_time_us': None,
                    'recall_at_1': None,
                    'recall_at_5': None,
                    'recall_at_10': None
                }
                
                # 1. 测试IndexPQ（纯PQ索引）
                pq_result = self._test_index_pq(m, nbits)
                result.update(pq_result)
                
                # 2. 测试IndexHNSWPQ（HNSW+PQ索引）
                hnswpq_result = self._test_index_hnswpq(m, nbits, hnsw_m, ef_construction)
                
                # 将HNSWPQ结果添加到result中，带前缀区分
                for key, value in hnswpq_result.items():
                    result[f'hnsw_{key}'] = value
                
                results.append(result)
                
                # 打印当前结果
                print(f"PQ构建时间: {result['build_time']:.2f}s")
                print(f"PQ内存使用: {result['memory_usage_mb']:.1f}MB")
                print(f"PQ召回率@10: {result['recall_at_10']:.3f}")
                print(f"HNSWPQ构建时间: {result['hnsw_build_time']:.2f}s")
                print(f"HNSWPQ搜索时间: {result['hnsw_search_time_us']:.1f}μs")
                print(f"HNSWPQ召回率@10: {result['hnsw_recall_at_10']:.3f}")
                
            except Exception as e:
                print(f"测试 M={m} 时出错: {e}")
                continue
                
        return pd.DataFrame(results)
    
    def _test_index_pq(self, m: int, nbits: int) -> Dict:
        """测试纯PQ索引"""
        # 创建PQ索引
        index = faiss.IndexPQ(self.dim, m, nbits, faiss.METRIC_L2)
        
        # 训练索引
        start_time = time.time()
        index.train(self.database)
        build_time = time.time() - start_time
        
        # 添加向量
        index.add(self.database)
        
        # 估算内存使用
        memory_usage = self.nb * m * nbits / 8 / (1024 * 1024)  # MB
        compression_ratio = (self.nb * self.dim * 4) / (self.nb * m * nbits / 8)
        
        # 搜索测试
        start_time = time.time()
        distances, labels = index.search(self.queries, self.k)
        search_time = (time.time() - start_time) * 1000000 / self.nq  # μs per query
        
        # 计算召回率
        recall_1 = self._compute_recall(labels[:, :1], self.ground_truth[:, :1])
        recall_5 = self._compute_recall(labels[:, :5], self.ground_truth[:, :5])
        recall_10 = self._compute_recall(labels[:, :10], self.ground_truth[:, :10])
        
        return {
            'compression_ratio': compression_ratio,
            'build_time': build_time,
            'memory_usage_mb': memory_usage,
            'search_time_us': search_time,
            'recall_at_1': recall_1,
            'recall_at_5': recall_5,
            'recall_at_10': recall_10
        }
    
    def _test_index_hnswpq(self, m: int, nbits: int, hnsw_m: int, ef_construction: int) -> Dict:
        """测试HNSW+PQ索引"""
        # 创建HNSWPQ索引
        index = faiss.IndexHNSWPQ(self.dim, m, nbits, hnsw_m, faiss.METRIC_L2)
        index.hnsw.efConstruction = ef_construction
        
        # 训练和构建索引
        start_time = time.time()
        index.train(self.database)
        index.add(self.database)
        build_time = time.time() - start_time
        
        # 估算内存使用（PQ码 + HNSW图）
        pq_memory = self.nb * m * nbits / 8
        hnsw_memory = self.nb * hnsw_m * 8  # 近似估算
        total_memory = (pq_memory + hnsw_memory) / (1024 * 1024)  # MB
        
        # 搜索测试（使用不同的efSearch值）
        ef_search_values = [16, 32, 64, 128]
        search_results = {}
        
        for ef_search in ef_search_values:
            index.hnsw.efSearch = ef_search
            
            start_time = time.time()
            distances, labels = index.search(self.queries, self.k)
            search_time = (time.time() - start_time) * 1000000 / self.nq  # μs per query
            
            # 计算召回率
            recall_1 = self._compute_recall(labels[:, :1], self.ground_truth[:, :1])
            recall_5 = self._compute_recall(labels[:, :5], self.ground_truth[:, :5])
            recall_10 = self._compute_recall(labels[:, :10], self.ground_truth[:, :10])
            
            search_results[ef_search] = {
                'search_time_us': search_time,
                'recall_at_1': recall_1,
                'recall_at_5': recall_5,
                'recall_at_10': recall_10
            }
        
        # 选择ef_search=64的结果作为主要结果
        main_result = search_results[64]
        
        return {
            'build_time': build_time,
            'memory_usage_mb': total_memory,
            'search_time_us': main_result['search_time_us'],
            'recall_at_1': main_result['recall_at_1'],
            'recall_at_5': main_result['recall_at_5'],
            'recall_at_10': main_result['recall_at_10'],
            'ef_search_results': search_results
        }
    
    def _compute_recall(self, pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
        """计算召回率"""
        assert pred_labels.shape == true_labels.shape
        
        nq, k = pred_labels.shape
        recall_sum = 0.0
        
        for i in range(nq):
            true_set = set(true_labels[i])
            pred_set = set(pred_labels[i])
            recall_sum += len(true_set.intersection(pred_set)) / len(true_set)
            
        return recall_sum / nq
    
    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """绘制测试结果图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PQ量化参数M对索引性能的影响分析', fontsize=16, fontweight='bold')
        
        # 1. 构建时间对比
        axes[0, 0].plot(results_df['M'], results_df['build_time'], 'o-', label='PQ', linewidth=2, markersize=8)
        axes[0, 0].plot(results_df['M'], results_df['hnsw_build_time'], 's-', label='HNSW+PQ', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('M (子量化器数量)')
        axes[0, 0].set_ylabel('构建时间 (秒)')
        axes[0, 0].set_title('索引构建时间')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 内存使用量
        axes[0, 1].plot(results_df['M'], results_df['memory_usage_mb'], 'o-', label='PQ', linewidth=2, markersize=8)
        axes[0, 1].plot(results_df['M'], results_df['hnsw_memory_usage_mb'], 's-', label='HNSW+PQ', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('M (子量化器数量)')
        axes[0, 1].set_ylabel('内存使用 (MB)')
        axes[0, 1].set_title('内存使用量')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 压缩比
        axes[0, 2].plot(results_df['M'], results_df['compression_ratio'], 'o-', color='green', linewidth=2, markersize=8)
        axes[0, 2].set_xlabel('M (子量化器数量)')
        axes[0, 2].set_ylabel('压缩比')
        axes[0, 2].set_title('PQ压缩比')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 搜索时间
        axes[1, 0].plot(results_df['M'], results_df['search_time_us'], 'o-', label='PQ', linewidth=2, markersize=8)
        axes[1, 0].plot(results_df['M'], results_df['hnsw_search_time_us'], 's-', label='HNSW+PQ', linewidth=2, markersize=8)
        axes[1, 0].set_xlabel('M (子量化器数量)')
        axes[1, 0].set_ylabel('搜索时间 (μs/query)')
        axes[1, 0].set_title('平均搜索时间')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 召回率@10
        axes[1, 1].plot(results_df['M'], results_df['recall_at_10'], 'o-', label='PQ', linewidth=2, markersize=8)
        axes[1, 1].plot(results_df['M'], results_df['hnsw_recall_at_10'], 's-', label='HNSW+PQ', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('M (子量化器数量)')
        axes[1, 1].set_ylabel('召回率@10')
        axes[1, 1].set_title('召回率@10')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(0, 1)
        
        # 6. 子向量维度 vs 召回率
        axes[1, 2].plot(results_df['sub_dim'], results_df['recall_at_10'], 'o-', label='PQ', linewidth=2, markersize=8)
        axes[1, 2].plot(results_df['sub_dim'], results_df['hnsw_recall_at_10'], 's-', label='HNSW+PQ', linewidth=2, markersize=8)
        axes[1, 2].set_xlabel('子向量维度 (dim/M)')
        axes[1, 2].set_ylabel('召回率@10')
        axes[1, 2].set_title('子向量维度 vs 召回率')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(0, 1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        
        plt.show()
    
    def print_analysis_summary(self, results_df: pd.DataFrame):
        """打印分析总结"""
        print("\n" + "="*80)
        print("PQ量化参数M影响分析总结")
        print("="*80)
        
        print(f"\n📊 测试配置:")
        print(f"   向量维度: {self.dim}")
        print(f"   数据库大小: {self.nb:,} 向量")
        print(f"   查询数量: {self.nq:,}")
        print(f"   搜索近邻数: {self.k}")
        
        print(f"\n📈 主要发现:")
        
        # 最佳性能点分析
        best_recall_pq = results_df.loc[results_df['recall_at_10'].idxmax()]
        best_recall_hnsw = results_df.loc[results_df['hnsw_recall_at_10'].idxmax()]
        fastest_search_pq = results_df.loc[results_df['search_time_us'].idxmin()]
        fastest_search_hnsw = results_df.loc[results_df['hnsw_search_time_us'].idxmin()]
        
        print(f"   🎯 PQ最佳召回率: M={best_recall_pq['M']}, 召回率={best_recall_pq['recall_at_10']:.3f}")
        print(f"   🎯 HNSW+PQ最佳召回率: M={best_recall_hnsw['M']}, 召回率={best_recall_hnsw['hnsw_recall_at_10']:.3f}")
        print(f"   ⚡ PQ最快搜索: M={fastest_search_pq['M']}, 时间={fastest_search_pq['search_time_us']:.1f}μs")
        print(f"   ⚡ HNSW+PQ最快搜索: M={fastest_search_hnsw['M']}, 时间={fastest_search_hnsw['hnsw_search_time_us']:.1f}μs")
        
        # M值趋势分析
        print(f"\n📋 M值影响趋势:")
        print(f"   • M值增加 → 子向量维度减少 → 量化精度降低")
        print(f"   • M值增加 → 码书数量增加 → 训练时间增加")
        print(f"   • M值增加 → 存储开销基本不变（M*nbits固定）")
        
        # 推荐配置
        print(f"\n💡 推荐配置:")
        balanced_idx = results_df.iloc[(results_df['recall_at_10'] * 0.7 + (1 - results_df['search_time_us'] / results_df['search_time_us'].max()) * 0.3).idxmax()]
        balanced_hnsw_idx = results_df.iloc[(results_df['hnsw_recall_at_10'] * 0.7 + (1 - results_df['hnsw_search_time_us'] / results_df['hnsw_search_time_us'].max()) * 0.3).idxmax()]
        
        print(f"   📈 PQ均衡配置: M={balanced_idx['M']} (召回率={balanced_idx['recall_at_10']:.3f}, 搜索时间={balanced_idx['search_time_us']:.1f}μs)")
        print(f"   📈 HNSW+PQ均衡配置: M={balanced_hnsw_idx['M']} (召回率={balanced_hnsw_idx['hnsw_recall_at_10']:.3f}, 搜索时间={balanced_hnsw_idx['hnsw_search_time_us']:.1f}μs)")
        
        print(f"\n详细结果表格:")
        print(results_df[['M', 'sub_dim', 'recall_at_10', 'search_time_us', 'hnsw_recall_at_10', 'hnsw_search_time_us']].to_string(index=False))

def main():
    """主函数"""
    print("🚀 PQ量化参数M影响分析Demo")
    print("=" * 80)
    
    # 初始化基准测试
    benchmark = PQBenchmark(
        dim=128,        # 128维向量
        nb=50000,       # 5万个数据库向量
        nq=1000,        # 1000个查询向量
        k=10            # 搜索前10个邻居
    )
    
    # 测试不同的M值
    # M值必须能整除维度，对于128维，可选择: 1, 2, 4, 8, 16, 32, 64, 128
    m_values = [1, 2, 4, 8, 16, 32, 64]
    
    print(f"\n🔬 开始测试M值: {m_values}")
    
    # 运行基准测试
    results_df = benchmark.test_pq_parameter(
        m_values=m_values,
        nbits=6,            # 6位量化 (64个聚类中心)
        hnsw_m=16,          # HNSW连接数
        ef_construction=200  # HNSW构建参数
    )
    
    # 保存结果
    results_df.to_csv('pq_benchmark_results.csv', index=False)
    print(f"\n💾 结果已保存到: pq_benchmark_results.csv")
    
    # 绘制图表
    benchmark.plot_results(results_df, 'pq_benchmark_analysis.png')
    
    # 打印分析总结
    benchmark.print_analysis_summary(results_df)
    
    print(f"\n✅ 分析完成！")

if __name__ == "__main__":
    main()
