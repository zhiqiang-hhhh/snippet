#!/usr/bin/env python3
"""
NSW (Navigable Small World) 单层小世界图 Demo
- 单层图（没有层级），构建时用 efConstruction 控制探索宽度，从候选中选 M 个最近邻连边
- 查询时用 efSearch 控制探索宽度，返回 top-k
- 带日志（包含文件名与行号）和 CLI，可直接运行
"""

import numpy as np
import heapq
import random
import argparse
import time
import logging
from typing import List


def setup_logger(level=logging.INFO):
    logger = logging.getLogger("NSW")
    if logger.handlers:
        logger.setLevel(level)
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s", datefmt="%H:%M:%S")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


class NSW:
    def __init__(self, dim: int, M: int = 8, efC: int = 100, efS: int = 50, logger: logging.Logger = None, verbose: bool = False):
        self.dim = dim
        self.M = M              # 每个节点的最大近邻数（针对新节点，老节点可能临时超过）
        self.efC = efC          # 构建阶段探索宽度（efConstruction）
        self.efS = efS          # 查询阶段探索宽度（efSearch）
        self.vectors: List[np.ndarray] = []
        self.graph: List[List[int]] = []  # graph[i] = 节点 i 的邻居列表（单层）
        self.enter_point: int | None = None
        self.logger = logger or setup_logger(logging.INFO)
        self.verbose = verbose

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    # -----------------
    # 构建（插入）
    # -----------------
    def add(self, vec: np.ndarray):
        idx = len(self.vectors)
        self.vectors.append(vec)

        # 初始化邻居表
        self.graph.append([])

        # 第一个点：成为入口
        if self.enter_point is None:
            self.enter_point = idx
            self.logger.info(f"add first node idx={idx} -> set as enter_point")
            return

        # 1) 从入口出发进行受限探索，收集候选区域
        start = self.enter_point
        # 候选堆（小根堆）：(距离, 节点)
        candidates: List[tuple[float, int]] = [(self._distance(vec, self.vectors[start]), start)]
        visited = {start}
        # 这里用 results 仅用于驱动弹出，visited 汇聚了最终的候选区域
        while candidates:
            d, node = heapq.heappop(candidates)
            # self.logger.debug(f"[add] neighbor of {node} is {self.graph[node]}")
            for nb in self.graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    dist = self._distance(vec, self.vectors[nb])
                    # 当堆未达 efC 时放宽接纳；否则仅接纳比当前弹出更近的点
                    if dist < d or len(candidates) < self.efC:
                        heapq.heappush(candidates, (dist, nb))

        # 2) 在候选区域内选出 M 个最近邻，建立双向连边
        neighbors = sorted([(self._distance(vec, self.vectors[x]), x) for x in visited])[: self.M]
        self.graph[idx] = [x for _, x in neighbors]
        for _, x in neighbors:
            self.graph[x].append(idx)
        if self.verbose:
            self.logger.debug(f"[add] idx={idx} connect {len(neighbors)} neighbors -> {self.graph[idx]}")

        # 3) 可选：更新入口为更“中心”的一个点（简单策略：保持不变或偶尔采样更新）
        # 这里保持不变，演示即可

    # -----------------
    # 查询
    # 首先把 entry_point 的邻居，全部添加到 candidates 里面
    # 从邻居里面找一个距离最近的，添加到 results
    # 然后再把这个节点的邻居加入 candidates
    # ef_search 控制 candidates 的最大长度
    # -----------------
    def search(self, query: np.ndarray, k: int = 10) -> List[int]:
        if self.enter_point is None:
            return []
        start = self.enter_point
        self.logger.info(f"[search] start from {start}, k={k}")

        candidates: List[tuple[float, int]] = [(self._distance(query, self.vectors[start]), start)]
        visited = {start}
        results: List[tuple[float, int]] = []

        while candidates:
            d, node = heapq.heappop(candidates)
            results.append((d, node))
            for nb in self.graph[node]:
                if nb not in visited:
                    visited.add(nb)
                    dist = self._distance(query, self.vectors[nb])
                    if dist < d or len(candidates) < self.efS:
                        # 这里把 nb 放入候选，那么下次 while candidates 的时候就可能在这个节点的邻居中继续探索
                        heapq.heappush(candidates, (dist, nb))

        topk = [nid for _, nid in sorted(results)[:k]]
        self.logger.info(f"[search] visited={len(visited)}, explored={len(results)}, topk={topk}")
        return topk


def plot_graph(graph: List[List[int]],
               out_file: str | None = None,
               show: bool = False,
               layout: str = "pca",
               dpi: int = 150,
               logger: logging.Logger | None = None,
               vectors: List[np.ndarray] | np.ndarray | None = None):
    """
    可视化 NSW 图：每个节点显示其 id，边用线连接。
    - 支持矩形布局，且尽量反映相对距离：pca（默认，基于向量的前两主成分）。
    - 也支持 spring/circular（依赖 networkx）以及 grid（规则网格）。
    - 可保存到文件（out_file）或显示（show）。
    """
    logger = logger or setup_logger(logging.INFO)
    n = len(graph)
    if n == 0:
        logger.warning("plot_graph: empty graph, skip")
        return

    # 构建边集合（去重无向边）
    edges = set()
    for i, nbrs in enumerate(graph):
        for j in nbrs:
            a, b = (i, j) if i <= j else (j, i)
            if a != b:
                edges.add((a, b))

    pos = {}
    used_layout = False

    # 1) PCA 布局：基于原始向量降到 2D，尽量保相对距离
    if layout == "pca" and vectors is not None:
        try:
            X = np.asarray(vectors, dtype=float)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            # 屏蔽长度不匹配的情况
            if X.shape[0] != n:
                logger.warning(f"plot_graph: vectors length {X.shape[0]} != graph size {n}, fallback to grid")
            else:
                # 中心化后做 SVD，取前两主成分
                Xc = X - X.mean(axis=0, keepdims=True)
                # 当特征维度或样本数较小时，SVD 也能稳定工作
                U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                # 投影到前两主成分
                PC = (Xc @ Vt[:2].T) if Vt.shape[0] >= 2 else np.hstack([Xc @ Vt[:1].T, np.zeros((n, 1))])
                # 不做等比例缩放，让坐标自适应填充矩形画布
                for i in range(n):
                    pos[i] = (float(PC[i, 0]), float(PC[i, 1]) if PC.shape[1] > 1 else 0.0)
                used_layout = True
        except Exception as e:
            logger.debug(f"PCA layout failed ({e}), will try other layouts")

    # 2) spring/circular（如可用）
    if not used_layout and layout in ("spring", "circular"):
        try:
            import networkx as nx  # type: ignore
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            if layout == "spring":
                pos = nx.spring_layout(G, seed=42)
            else:
                pos = nx.circular_layout(G)
            used_layout = True
        except Exception as e:
            logger.debug(f"networkx not available or failed to layout ({e}), will try grid")

    # 3) grid 布局：规则网格，避免圆形
    if not used_layout:
        import math
        cols = max(1, int(math.ceil(math.sqrt(n * 16 / 9))))  # 尝试接近 16:9 的矩形
        rows = max(1, int(math.ceil(n / cols)))
        sx, sy = 1.0, 1.0
        for i in range(n):
            r = i // cols
            c = i % cols
            pos[i] = (c * sx, -r * sy)

    # 画图
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:
        logger.error(f"matplotlib 未安装，无法绘图: {e}")
        return

    # 使用矩形画布；不强制等比，便于填满画布并体现相对距离
    fig, ax = plt.subplots(figsize=(10, 6), dpi=dpi)
    ax.set_aspect('auto')
    ax.axis('off')

    # 先画边
    for (u, v) in edges:
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color='#888', linewidth=0.8, zorder=1)

    # 再画点
    xs = [pos[i][0] for i in range(n)]
    ys = [pos[i][1] for i in range(n)]
    ax.scatter(xs, ys, s=100, c='#1f77b4', zorder=2)

    # 标注 id
    for i in range(n):
        ax.text(pos[i][0], pos[i][1], str(i), color='white', ha='center', va='center', fontsize=8, zorder=3)

    if out_file:
        fig.savefig(out_file, bbox_inches='tight')
        logger.info(f"graph saved to {out_file}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="NSW (Navigable Small World) single-layer demo")
    parser.add_argument("--dim", type=int, default=16, help="vector dimension")
    parser.add_argument("--N", type=int, default=1000, help="number of database vectors")
    parser.add_argument("--queries", type=int, default=3, help="number of query vectors")
    parser.add_argument("--k", type=int, default=3, help="top-k")
    parser.add_argument("--M", type=int, default=8, help="max neighbors per node")
    parser.add_argument("--efC", type=int, default=100, help="efConstruction (build breadth)")
    parser.add_argument("--efS", type=int, default=50, help="efSearch (query breadth)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--verbose", action="store_true", help="print step-by-step details (alias to DEBUG)")
    # 绘图相关
    parser.add_argument("--plot-graph", action="store_true", help="plot the NSW graph after build")
    parser.add_argument("--plot-file", type=str, default="", help="path to save the plotted graph (PNG/PDF/SVG)")
    parser.add_argument("--show-plot", action="store_true", help="show the plotted graph in a window")
    parser.add_argument("--layout", type=str, choices=["pca", "spring", "circular", "grid"], default="pca", help="graph layout algorithm")
    args = parser.parse_args()

    level = logging.getLevelName("DEBUG" if args.verbose else args.log_level)
    logger = setup_logger(level)

    logger.info(f"config dim={args.dim}, N={args.N}, Q={args.queries}, k={args.k}, M={args.M}, efC={args.efC}, efS={args.efS}, seed={args.seed}")
    rng = np.random.default_rng(args.seed)
    xb = rng.random((args.N, args.dim), dtype=np.float32)
    xq = rng.random((args.queries, args.dim), dtype=np.float32)

    nsw = NSW(dim=args.dim, M=args.M, efC=args.efC, efS=args.efS, logger=logger, verbose=args.verbose)

    t0 = time.perf_counter()
    for i in range(args.N):
        nsw.add(xb[i])
        if not args.verbose and (i + 1) % max(1, args.N // 10) == 0:
            logger.info(f"added {i+1}/{args.N}")
    t1 = time.perf_counter()

    # 简单统计：平均度
    if args.N > 0:
        deg = [len(nsw.graph[i]) for i in range(len(nsw.graph))]
        avg_deg = (sum(deg) / len(deg)) if deg else 0.0
    else:
        avg_deg = 0.0

    logger.info(f"build done: add_time={t1 - t0:.3f}s, avg_degree={avg_deg:.2f}, enter_point={nsw.enter_point}")

    # 如需绘图
    if args.plot_graph:
        out_path = args.plot_file if args.plot_file else None
    plot_graph(nsw.graph, out_file=out_path, show=args.show_plot, layout=args.layout, logger=logger, vectors=nsw.vectors)

    t2 = time.perf_counter()
    all_res = []
    for qi in range(args.queries):
        res = nsw.search(xq[qi], k=args.k)
        all_res.append(res)
    t3 = time.perf_counter()
    avg_search_ms = ((t3 - t2) / max(1, args.queries)) * 1000.0
    logger.info(f"search done: total_time={t3 - t2:.3f}s, avg_per_query={avg_search_ms:.3f} ms, k={args.k}")

    # 打印前几个查询结果
    show = min(5, args.queries)
    for i in range(show):
        logger.info(f"Q{i} top{args.k} -> {all_res[i]}")


if __name__ == "__main__":
    main()
