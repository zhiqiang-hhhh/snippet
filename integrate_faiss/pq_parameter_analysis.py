#!/usr/bin/env python3
"""
量化参数对比分析 Demo：IndexFlat、IndexPQ、IndexSQ

对比内容：
1. 索引构建时间
2. 内存使用量（仅编码存储，忽略小量元数据）
3. 搜索速度
4. 召回率精度

作者: GitHub Copilot
日期: 2025-09-15
"""

import numpy as np
import faiss
import time
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')


class PQBenchmark:
    def __init__(self, dim: int = 128, nb: int = 65537, nq: int = 1000, k: int = 10):
        """
        初始化量化基准测试

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
        print(f"生成测试数据: dim={dim}, nb={nb:,}, nq={nq}")
        np.random.seed(42)
        self.database = np.random.randn(nb, dim).astype(np.float32)
        self.queries = np.random.randn(nq, dim).astype(np.float32)

        # 规范化向量（可选：为 L2 搜索带来更稳定的度量）
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

    # -------------------------------
    # 估算编码大小（bytes/向量）
    # -------------------------------
    def _pq_code_bytes_per_vec(self, m: int, nbits: int) -> float:
        return m * nbits / 8.0

    def _sq_code_bytes_per_vec(self, qtype: int) -> float:
        # 近似估算：按每维的码字大小
        QT = faiss.ScalarQuantizer
        if qtype == QT.QT_8bit or qtype == QT.QT_8bit_uniform or qtype == QT.QT_8bit_direct:
            return self.dim * 1.0
        if qtype == QT.QT_4bit:
            return self.dim * 0.5
        if qtype == QT.QT_fp16:
            return self.dim * 2.0
        # 兜底：按1字节/维
        return self.dim * 1.0

    def _flat_code_bytes_per_vec(self) -> float:
        return self.dim * 4.0

    # -------------------------------
    # 召回率计算
    # -------------------------------
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

    # -------------------------------
    # 各索引测试
    # -------------------------------
    def _test_index_flat(self) -> Dict:
        """IndexFlat（无量化）"""
        index = faiss.IndexFlatL2(self.dim)

        start_time = time.time()
        index.add(self.database)
        build_time = time.time() - start_time

        # 搜索
        start_time = time.time()
        distances, labels = index.search(self.queries, self.k)
        search_time = (time.time() - start_time) * 1e6 / self.nq  # μs per query

        # 召回率
        recall_1 = self._compute_recall(labels[:, :1], self.ground_truth[:, :1])
        recall_5 = self._compute_recall(labels[:, :5], self.ground_truth[:, :5])
        recall_10 = self._compute_recall(labels[:, :10], self.ground_truth[:, :10])

        code_bytes_per_vec = self._flat_code_bytes_per_vec()
        memory_usage_mb = self.nb * code_bytes_per_vec / (1024 * 1024)
        compression_ratio = (self.dim * 4.0) / code_bytes_per_vec  # 恒为 1

        return {
            'method': 'Flat',
            'params': 'Flat',
            'M': None,
            'nbits': None,
            'qtype': None,
            'code_bytes_per_vec': code_bytes_per_vec,
            'compression_ratio': compression_ratio,
            'build_time': build_time,
            'memory_usage_mb': memory_usage_mb,
            'search_time_us': search_time,
            'recall_at_1': recall_1,
            'recall_at_5': recall_5,
            'recall_at_10': recall_10
        }

    def _test_index_pq(self, m: int, nbits: int) -> Dict:
        """IndexPQ（产品量化）"""
        index = faiss.IndexPQ(self.dim, m, nbits, faiss.METRIC_L2)

        # 训练和添加
        start_time = time.time()
        index.train(self.database)
        build_train = time.time() - start_time

        start_time = time.time()
        index.add(self.database)
        build_add = time.time() - start_time

        build_time = build_train + build_add

        # 搜索
        start_time = time.time()
        distances, labels = index.search(self.queries, self.k)
        search_time = (time.time() - start_time) * 1e6 / self.nq  # μs per query

        # 召回率
        recall_1 = self._compute_recall(labels[:, :1], self.ground_truth[:, :1])
        recall_5 = self._compute_recall(labels[:, :5], self.ground_truth[:, :5])
        recall_10 = self._compute_recall(labels[:, :10], self.ground_truth[:, :10])

        code_bytes_per_vec = self._pq_code_bytes_per_vec(m, nbits)
        memory_usage_mb = self.nb * code_bytes_per_vec / (1024 * 1024)
        compression_ratio = (self.dim * 4.0) / code_bytes_per_vec

        return {
            'method': 'PQ',
            'params': f'M={m}, nbits={nbits}',
            'M': m,
            'nbits': nbits,
            'qtype': None,
            'code_bytes_per_vec': code_bytes_per_vec,
            'compression_ratio': compression_ratio,
            'build_time': build_time,
            'memory_usage_mb': memory_usage_mb,
            'search_time_us': search_time,
            'recall_at_1': recall_1,
            'recall_at_5': recall_5,
            'recall_at_10': recall_10
        }

    def _test_index_sq(self, qtype: int) -> Dict:
        """IndexSQ（标量量化）"""
        index = faiss.IndexScalarQuantizer(self.dim, qtype, faiss.METRIC_L2)

        # 训练（部分 qtype 需要训练，统一调用不影响）
        start_time = time.time()
        index.train(self.database)
        build_train = time.time() - start_time

        start_time = time.time()
        index.add(self.database)
        build_add = time.time() - start_time

        build_time = build_train + build_add

        # 搜索
        start_time = time.time()
        distances, labels = index.search(self.queries, self.k)
        search_time = (time.time() - start_time) * 1e6 / self.nq  # μs per query

        # 召回率
        recall_1 = self._compute_recall(labels[:, :1], self.ground_truth[:, :1])
        recall_5 = self._compute_recall(labels[:, :5], self.ground_truth[:, :5])
        recall_10 = self._compute_recall(labels[:, :10], self.ground_truth[:, :10])

        code_bytes_per_vec = self._sq_code_bytes_per_vec(qtype)
        memory_usage_mb = self.nb * code_bytes_per_vec / (1024 * 1024)
        compression_ratio = (self.dim * 4.0) / code_bytes_per_vec

        # qtype 名称
        QT = faiss.ScalarQuantizer
        qtype_name_map = {
            QT.QT_8bit: 'QT_8bit',
            QT.QT_4bit: 'QT_4bit',
            QT.QT_8bit_uniform: 'QT_8bit_uniform',
            QT.QT_fp16: 'QT_fp16',
            getattr(QT, 'QT_8bit_direct', -1): 'QT_8bit_direct'
        }
        qname = qtype_name_map.get(qtype, str(qtype))

        return {
            'method': 'SQ',
            'params': f'{qname}',
            'M': None,
            'nbits': None,
            'qtype': qname,
            'code_bytes_per_vec': code_bytes_per_vec,
            'compression_ratio': compression_ratio,
            'build_time': build_time,
            'memory_usage_mb': memory_usage_mb,
            'search_time_us': search_time,
            'recall_at_1': recall_1,
            'recall_at_5': recall_5,
            'recall_at_10': recall_10
        }

    # -------------------------------
    # 主测试入口
    # -------------------------------
    def test_quantizers(
        self,
        pq_m_values: List[int],
        pq_nbits_values: List[int],
        sq_qtypes: List[int]
    ) -> pd.DataFrame:
        """
        对比 IndexFlat、IndexPQ（多组 M/nbits）、IndexSQ（多种 qtype）
        """
        results = []

        print("\n开始测试 IndexFlat / IndexPQ / IndexSQ")
        print("=" * 80)

        # 1) IndexFlat
        try:
            print("\n测试 IndexFlat ...")
            res = self._test_index_flat()
            results.append(res)
            print(f"Flat 搜索时间: {res['search_time_us']:.1f}μs, 召回@10: {res['recall_at_10']:.3f}")
        except Exception as e:
            print(f"IndexFlat 测试失败: {e}")

        # 2) IndexPQ：网格测试
        print("\n测试 IndexPQ 参数网格 ...")
        for m in pq_m_values:
            if self.dim % m != 0:
                print(f"跳过 M={m}: 维度{self.dim}不能被M整除")
                continue
            for nbits in pq_nbits_values:
                try:
                    print(f"- PQ: M={m}, nbits={nbits} (子向量维度: {self.dim//m})")
                    res = self._test_index_pq(m, nbits)
                    results.append(res)
                    print(f"  搜索: {res['search_time_us']:.1f}μs, 召回@10: {res['recall_at_10']:.3f}, 压缩比: {res['compression_ratio']:.1f}x")
                except Exception as e:
                    print(f"  PQ(M={m}, nbits={nbits}) 测试失败: {e}")

        # 3) IndexSQ：多 qtype
        print("\n测试 IndexSQ 多种量化类型 ...")
        for qt in sq_qtypes:
            try:
                res = self._test_index_sq(qt)
                results.append(res)
                print(f"- SQ {res['params']}: 搜索 {res['search_time_us']:.1f}μs, 召回@10 {res['recall_at_10']:.3f}, 压缩比 {res['compression_ratio']:.1f}x")
            except Exception as e:
                print(f"  SQ(qtype={qt}) 测试失败: {e}")

        return pd.DataFrame(results)

    # -------------------------------
    # 绘图与输出
    # -------------------------------
    def plot_results(self, results_df: pd.DataFrame, save_path: str = None):
        """绘制结果图表"""
        methods = results_df['method'].unique()
        colors = {'Flat': 'black', 'PQ': 'tab:blue', 'SQ': 'tab:orange'}

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle('IndexFlat / IndexPQ / IndexSQ 量化对比', fontsize=16, fontweight='bold')

        # 1. 压缩比（越大越省存储）
        for m in methods:
            df = results_df[results_df['method'] == m]
            axes[0].scatter(df['compression_ratio'], df['recall_at_10'], label=m, s=60, c=colors.get(m, None))
        axes[0].set_xlabel('压缩比 (float32占用 / 码本占用)')
        axes[0].set_ylabel('召回率@10')
        axes[0].set_title('压缩比 vs 召回率@10')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # 2. 搜索时间
        for m in methods:
            df = results_df[results_df['method'] == m]
            axes[1].scatter(df['compression_ratio'], df['search_time_us'], label=m, s=60, c=colors.get(m, None))
        axes[1].set_xlabel('压缩比')
        axes[1].set_ylabel('搜索时间 (μs/query)')
        axes[1].set_title('压缩比 vs 搜索时间')
        axes[1].grid(True, alpha=0.3)

        # 3. 内存使用
        for m in methods:
            df = results_df[results_df['method'] == m]
            axes[2].scatter(df['compression_ratio'], df['memory_usage_mb'], label=m, s=60, c=colors.get(m, None))
        axes[2].set_xlabel('压缩比')
        axes[2].set_ylabel('内存使用 (MB)')
        axes[2].set_title('压缩比 vs 内存使用')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"图表已保存到: {save_path}")
        plt.show()

    def print_analysis_summary(self, results_df: pd.DataFrame):
        """打印分析总结"""
        print("\n" + "=" * 80)
        print("IndexFlat / IndexPQ / IndexSQ 量化对比分析总结")
        print("=" * 80)

        # 找到若干代表性最优点
        try:
            best_recall = results_df.loc[results_df['recall_at_10'].idxmax()]
            fastest = results_df.loc[results_df['search_time_us'].idxmin()]
            best_compress = results_df.loc[results_df['compression_ratio'].idxmax()]
            print(f"   🎯 最高召回: {best_recall['method']}({best_recall['params']}) 召回@10={best_recall['recall_at_10']:.3f}")
            print(f"   ⚡ 最快搜索: {fastest['method']}({fastest['params']}) 时间={fastest['search_time_us']:.1f}μs")
            print(f"   💾 最大压缩: {best_compress['method']}({best_compress['params']}) 压缩比={best_compress['compression_ratio']:.1f}x")
        except Exception:
            pass

        cols = ['method', 'params', 'compression_ratio', 'memory_usage_mb', 'build_time', 'search_time_us', 'recall_at_10']
        print("\n详细结果（部分列）：")
        print(results_df[cols].sort_values(['method', 'params']).to_string(index=False))


def main():
    """主函数"""
    print("🚀 量化参数对比分析（IndexFlat / IndexPQ / IndexSQ）")
    print("=" * 80)

    # 初始化基准测试
    benchmark = PQBenchmark(
        dim=128,        # 128维向量
        nb=50000,     # 2555904个数据库向量（注意：较大，运行时间较长）
        nq=1000,        # 1000个查询向量
        k=10            # 搜索前10个邻居
    )

    # PQ 参数网格（M 需整除 dim）
    pq_m_values = [1, 4, 16, 32]
    pq_nbits_values = [4, 8]

    # SQ 量化类型
    QT = faiss.ScalarQuantizer
    sq_qtypes = [
        QT.QT_8bit,
        QT.QT_4bit,
        getattr(QT, 'QT_8bit_direct', None),
    ]
    sq_qtypes = [qt for qt in sq_qtypes if qt is not None]

    print(f"\n🔬 开始测试 PQ(M×nbits)与 SQ(qtype):")
    print(f"PQ M 值: {pq_m_values}")
    print(f"PQ nbits 值: {pq_nbits_values}")
    print(f"SQ qtypes: {sq_qtypes}")

    # 运行基准测试
    results_df = benchmark.test_quantizers(
        pq_m_values=pq_m_values,
        pq_nbits_values=pq_nbits_values,
        sq_qtypes=sq_qtypes
    )

    # 保存结果
    results_df.to_csv('quantizer_benchmark_results.csv', index=False)
    print(f"\n💾 结果已保存到: quantizer_benchmark_results.csv")

    # 绘制图表
    benchmark.plot_results(results_df, 'quantizer_benchmark_analysis.png')

    # 打印分析总结
    benchmark.print_analysis_summary(results_df)

    print(f"\n✅ 分析完成！")


if __name__ == "__main__":
    main()
