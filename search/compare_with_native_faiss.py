import faiss
import numpy as np
import logging
import os
import random
from query import get_conn
import mysql.connector
from typing import List, Tuple, Set, Optional, Dict, Any
import math
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def read_tsv_data(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Read data from TSV file and return IDs and embeddings"""
    ids = []
    embeddings = []
    
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                ids.append(int(parts[0]))
                vector = np.array([float(x) for x in parts[1].strip('[]').split(',')], dtype=np.float32)
                embeddings.append(vector)
    
    return np.array(ids, dtype=np.int64), np.array(embeddings, dtype=np.float32)

def build_faiss_index(index_type: str, metric_type: str, embeddings: np.ndarray, ids: np.ndarray, dimension: int) -> faiss.IndexIDMap:
    """
    Build a Faiss index from embeddings with explicit IDs.
    index_type: "flat" or "hnsw"
    metric_type: "l2" or "ip"
    """
    # Select metric
    if metric_type.lower() in ["l2", "l2 distance"]:
        metric = faiss.METRIC_L2
    elif metric_type.lower() in ["ip", "inner product"]:
        metric = faiss.METRIC_INNER_PRODUCT
    else:
        raise ValueError(f"Unsupported metric_type: {metric_type}")

    # Select index type
    if index_type.lower() == "flat":
        index = faiss.IndexFlat(dimension, metric)
    elif index_type.lower() == "hnsw":
        index = faiss.IndexHNSWFlat(dimension, 32, metric)
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")

    # Add the ID mapping layer
    id_map = faiss.IndexIDMap(index)

    # Add vectors with their IDs
    id_map.add_with_ids(embeddings, ids)

    return id_map

def faiss_range_search(index: faiss.IndexIDMap, query_vector: np.ndarray, radius: float) -> List[Tuple[int, float]]:
    """Perform range search with Faiss native range_search API"""
    # For Faiss, the radius needs to be squared since it works with squared distances
    squared_radius = radius * radius
    
    # Reshape to ensure query_vector is 2D
    query_vector_reshaped = np.array([query_vector], dtype=np.float32)
    
    # Perform range search - returns distances and indices for points within radius
    lims, D, I = index.range_search(query_vector_reshaped, squared_radius)
    
    # Process results
    results = []
    # We only have one query, so results are from lims[0] to lims[1]
    for i in range(int(lims[0]), int(lims[1])):
        # Apply square root to convert from squared distance to actual distance
        results.append((int(I[i]), math.sqrt(float(D[i]))))
    
    # Sort by distance
    results.sort(key=lambda x: x[1])
    return results

def faiss_range_search_greater(index: faiss.IndexIDMap, query_vector: np.ndarray, radius: float, all_ids: Optional[np.ndarray] = None) -> List[Tuple[int, float]]:
    """
    Perform range search with Faiss for distances greater than radius
    by finding the set difference between all IDs and IDs with distance < radius
    """
    # For Faiss, the radius needs to be squared since it works with squared distances
    squared_radius = radius * radius
    
    # Reshape to ensure query_vector is 2D
    query_vector_reshaped = np.array([query_vector], dtype=np.float32)
    
    # Get all IDs with distance < radius
    lims, D, I = index.range_search(query_vector_reshaped, squared_radius)
    
    # Create a set of IDs with distance < radius
    ids_within_radius = set()
    for i in range(int(lims[0]), int(lims[1])):
        ids_within_radius.add(int(I[i]))
    
    # IDs with distance > radius are the set difference between all_ids and ids_within_radius
    ids_beyond_radius = set(all_ids) - ids_within_radius
    
    # Create a list to store the results
    results = []
    
    # If we have no IDs beyond the radius, return empty results
    if not ids_beyond_radius:
        return results
    
    # We need to get distances for IDs with distance > radius
    # Use regular search for a large number of results
    large_k = min(1000, len(all_ids))  # Use a large k to get most vectors
    D, I = index.search(query_vector_reshaped, large_k)
    
    # Create a mapping from ID to distance
    id_to_dist = {}
    for i in range(len(I[0])):
        if I[0][i] != -1:
            # Apply square root to convert from squared distance to actual distance
            id_to_dist[int(I[0][i])] = math.sqrt(float(D[0][i]))
    
    # Add IDs beyond radius with their distances to results
    for id_val in ids_beyond_radius:
        if id_val in id_to_dist:
            results.append((id_val, id_to_dist[id_val]))
    
    # Sort by distance
    results.sort(key=lambda x: x[1])
    return results

def faiss_topn_search(index: faiss.IndexIDMap, query_vector: np.ndarray, limit: int = 5) -> List[Tuple[int, float]]:
    """Perform top-N search with Faiss"""
    D, I = index.search(np.array([query_vector]), limit)
    results = []
    
    for i in range(min(len(I[0]), limit)):
        if I[0][i] != -1:  # Skip invalid results
            # Apply square root to convert from squared distance to actual distance
            results.append((int(I[0][i]), math.sqrt(float(D[0][i]))))
    
    return results

def faiss_compound_search(index: faiss.IndexIDMap, query_vector: np.ndarray, radius: float, limit: int = 5) -> List[Tuple[int, float]]:
    """Perform compound search with Faiss"""
    # For Faiss, the radius needs to be squared since it works with squared distances
    squared_radius = radius * radius
    
    # Reshape to ensure query_vector is 2D
    query_vector_reshaped = np.array([query_vector], dtype=np.float32)
    
    # Step 1: Perform range search to get IDs within radius
    lims, D, I = index.range_search(query_vector_reshaped, squared_radius)
    
    # If no results found within radius, return empty list
    if lims[1] == lims[0]:
        return []
    
    # Step 2: Create an IdSelectorBatch with numpy array of int64
    range_ids = np.array([int(I[i]) for i in range(int(lims[0]), int(lims[1]))], dtype=np.int64)
    logger.info(f"Range search found {len(range_ids)} IDs within radius")

    id_selector = faiss.IDSelectorBatch(range_ids)
    
    # Step 3: Perform top-K search with the ID selector to filter results
    # The limit parameter is used directly as K for the top-K search
    params = faiss.SearchParametersHNSW()
    params.sel = id_selector
    D_topn, I_topn = index.search(query_vector_reshaped, limit, params=params)
    logger.info(f"Top-N search found {len(I_topn[0])} IDs")
    # Process the results
    results = []
    for i in range(len(I_topn[0])):
        if I_topn[0][i] != -1:  # Skip invalid results
            # Apply square root to convert from squared distance to actual distance
            results.append((int(I_topn[0][i]), math.sqrt(float(D_topn[0][i]))))
    
    return results

def get_radius(query_vector: np.ndarray, embeddings: np.ndarray, percentile: int) -> float:
    """
    Calculate the radius based on a percentile of distances from query_vector to all embeddings
    
    Args:
        query_vector: The query vector
        embeddings: All embedding vectors
        percentile: The percentile (0-100) to use for radius calculation
    
    Returns:
        The distance value at the specified percentile
    """
    # Calculate L2 distances from query vector to all embeddings (vectorized)
    distances = np.sqrt(np.sum((embeddings - query_vector) ** 2, axis=1))
    
    # Return the specified percentile
    return float(np.percentile(distances, percentile))

# Global dictionary to track comparison results across all tests
comparison_summary = {
    "Range Search": {
        "total_tests": 0,
        "avg_overlap_pct": 0,
        "avg_max_diff": 0,
        "avg_mean_diff": 0,
        "dimensions_tested": set(),
        "diff_count": 0  # Track tests with common IDs
    },
    "Top-N Search": {
        "total_tests": 0,
        "avg_overlap_pct": 0,
        "avg_max_diff": 0,
        "avg_mean_diff": 0,
        "dimensions_tested": set(),
        "diff_count": 0
    },
    "Compound Search": {
        "total_tests": 0,
        "avg_overlap_pct": 0,
        "avg_max_diff": 0,
        "avg_mean_diff": 0,
        "dimensions_tested": set(),
        "diff_count": 0
    }
}

_table_metrics = {}

def compute_overlap(faiss_ids: set, doris_ids: set) -> float:
    """计算重合率（交集/FAISS结果数）"""
    if not faiss_ids:
        return 0.0
    return len(faiss_ids & doris_ids) / len(faiss_ids)

def compute_precision(doris_ids: set, ground_truth_ids: set) -> float:
    """计算正确率（Doris命中ground truth的比例）"""
    if not doris_ids:
        return 0.0
    return len(doris_ids & ground_truth_ids) / len(doris_ids)

def compute_recall(doris_ids: set, ground_truth_ids: set) -> float:
    """计算召回率（ground truth被Doris命中的比例）"""
    if not ground_truth_ids:
        return 0.0
    return len(doris_ids & ground_truth_ids) / len(ground_truth_ids)

def brute_force_topn(query_vector: np.ndarray, embeddings: np.ndarray, ids: np.ndarray, topk: int) -> List[Tuple[int, float]]:
    """暴力计算topk ground truth"""
    dists = np.linalg.norm(embeddings - query_vector, axis=1)
    idxs = np.argsort(dists)[:topk]
    return [(int(ids[i]), float(dists[i])) for i in idxs]

def compare_and_log(
    faiss_results: List[Tuple[int, float]],
    doris_results: List[Tuple],
    test_name: str,
    dimension: int = 0,
    table_name: str = None,
    ground_truth: Optional[List[Tuple[int, float]]] = None
):
    """对比并输出重合率、正确率、召回率"""
    faiss_ids = {result[0] for result in faiss_results}
    doris_ids = {int(result[0]) for result in doris_results}
    overlap = compute_overlap(faiss_ids, doris_ids)
    logger.info(f"[{test_name}] Overlap: {overlap*100:.2f}%")

    # 统计指标
    faiss_precision = faiss_recall = doris_precision = doris_recall = None

    if ground_truth is not None:
        gt_ids = {result[0] for result in ground_truth}
        # Doris 准确率/召回率
        doris_precision = compute_precision(doris_ids, gt_ids)
        doris_recall = compute_recall(doris_ids, gt_ids)
        # Faiss 准确率/召回率
        faiss_precision = compute_precision(faiss_ids, gt_ids)
        faiss_recall = compute_recall(faiss_ids, gt_ids)
    # Range search: 只输出正确率
    if test_name == "Range Search":
        if ground_truth is not None:
            gt_ids = {result[0] for result in ground_truth}
            precision = compute_precision(doris_ids, gt_ids)
            logger.info(f"[{test_name}] Precision: {precision*100:.2f}%")
        else:
            logger.info(f"[{test_name}] Precision: N/A (no ground truth)")
        recall = None
    else:
        # topn/compound: 输出正确率和召回率
        if ground_truth is not None:
            gt_ids = {result[0] for result in ground_truth}
            precision = compute_precision(doris_ids, gt_ids)
            recall = compute_recall(doris_ids, gt_ids)
            logger.info(f"[{test_name}] Precision: {precision*100:.2f}%")
            logger.info(f"[{test_name}] Recall: {recall*100:.2f}%")
        else:
            logger.info(f"[{test_name}] Precision: N/A (no ground truth)")
            logger.info(f"[{test_name}] Recall: N/A (no ground truth)")

    # 兼容原有的表统计
    if table_name:
        if not hasattr(print_comparison_summary, "_table_overlaps"):
            print_comparison_summary._table_overlaps = {}
        table_overlaps = print_comparison_summary._table_overlaps
        if table_name not in table_overlaps:
            table_overlaps[table_name] = []
        table_overlaps[table_name].append(overlap*100)

        # 新增：记录详细指标
        if table_name not in _table_metrics:
            _table_metrics[table_name] = {}
        _table_metrics[table_name][test_name] = {
            "overlap": overlap,
            "faiss_precision": faiss_precision,
            "faiss_recall": faiss_recall,
            "doris_precision": doris_precision,
            "doris_recall": doris_recall,
        }

def compare_results(faiss_results: List[Tuple[int, float]], doris_results: List[Tuple], test_name: str, dimension: int = 0, table_name: str = None) -> None:
    """
    Compare results from Faiss and Doris searches and log the differences.
    
    Args:
        faiss_results: List of (id, distance) tuples from Faiss
        doris_results: List of (id, distance) tuples from Doris
        test_name: Name of the test for logging
        dimension: Vector dimension for the current test
    """
    global comparison_summary
    
    # Extract IDs from both results
    faiss_ids = {result[0] for result in faiss_results}
    doris_ids = {int(result[0]) for result in doris_results}
    
    # Calculate overlap of IDs
    common_ids = faiss_ids.intersection(doris_ids)
    only_in_faiss = faiss_ids - doris_ids
    only_in_doris = doris_ids - faiss_ids
    
    # Calculate overlap percentage
    overlap_pct = (len(common_ids) / max(1, len(faiss_ids))) * 100
    
    # Calculate distance differences for common IDs
    diffs = []
    faiss_dist_map = {result[0]: result[1] for result in faiss_results}
    doris_dist_map = {int(result[0]): float(result[1]) for result in doris_results}
    
    for id_val in common_ids:
        diff = abs(faiss_dist_map[id_val] - doris_dist_map[id_val])
        diffs.append(diff)
    
    # Log comparison results
    logger.info(f"\n--- {test_name} Comparison ---")
    logger.info(f"Total IDs in Faiss: {len(faiss_ids)}, Total IDs in Doris: {len(doris_ids)}")
    logger.info(f"Common IDs: {len(common_ids)} ({overlap_pct:.1f}% overlap)")
    
    if only_in_faiss:
        logger.info(f"IDs only in Faiss: {only_in_faiss}")
    if only_in_doris:
        logger.info(f"IDs only in Doris: {only_in_doris}")
    
    max_diff = 0
    avg_diff = 0
    if diffs:
        max_diff = max(diffs)
        avg_diff = sum(diffs) / len(diffs)
        logger.info(f"Max distance difference: {max_diff:.6f}")
        logger.info(f"Avg distance difference: {avg_diff:.6f}")
    else:
        logger.info("No common IDs to compare distances")
    
    # Update the global summary
    if test_name in comparison_summary:
        comparison_summary[test_name]["total_tests"] += 1
        comparison_summary[test_name]["avg_overlap_pct"] += overlap_pct
        comparison_summary[test_name]["dimensions_tested"].add(dimension)
        if diffs:
            comparison_summary[test_name]["avg_max_diff"] += max_diff
            comparison_summary[test_name]["avg_mean_diff"] += avg_diff
            comparison_summary[test_name]["diff_count"] += 1  # Count tests with common IDs

    # 收集每个表的 overlap
    if table_name:
        if not hasattr(print_comparison_summary, "_table_overlaps"):
            print_comparison_summary._table_overlaps = {}
        table_overlaps = print_comparison_summary._table_overlaps
        if table_name not in table_overlaps:
            table_overlaps[table_name] = []
        table_overlaps[table_name].append(overlap_pct)

def print_comparison_summary():
    """Print a summary of all comparison results at the end of the script"""
    logger.info("\n" + "="*80)
    logger.info("OVERALL TEST RESULTS SUMMARY")
    logger.info("="*80)
    
    # Overall summary
    all_tests_count = sum(data["total_tests"] for data in comparison_summary.values())
    all_dimensions = set()
    for data in comparison_summary.values():
        all_dimensions.update(data["dimensions_tested"])
    
    logger.info(f"Total tests performed: {all_tests_count} across {len(all_dimensions)} dimensions")
    logger.info(f"Dimensions tested: {sorted(all_dimensions)}")
    
    # Per-test type summary
    for test_name, data in comparison_summary.items():
        total_tests = data["total_tests"]
        diff_count = data.get("diff_count", 0)
        if total_tests > 0:
            avg_overlap = data["avg_overlap_pct"] / total_tests
            avg_max_diff = (data["avg_max_diff"] / diff_count) if diff_count > 0 else 0
            avg_mean_diff = (data["avg_mean_diff"] / diff_count) if diff_count > 0 else 0
            
            logger.info(f"\n{test_name} Summary (across {total_tests} tests):")
            logger.info(f"Average overlap percentage: {avg_overlap:.2f}%")
            logger.info(f"Average maximum distance difference: {avg_max_diff:.6f}")
            logger.info(f"Average mean distance difference: {avg_mean_diff:.6f}")
            logger.info(f"Dimensions tested: {sorted(data['dimensions_tested'])}")
    
    logger.info("\nCONCLUSION:")
    # Add overall conclusion based on results
    if all_tests_count > 0:
        avg_all_overlap = sum(
            data["avg_overlap_pct"] for data in comparison_summary.values()
        ) / all_tests_count
        logger.info(f"Overall average overlap: {avg_all_overlap:.2f}%")
        if avg_all_overlap > 90:
            logger.info("EXCELLENT MATCH: Doris and Faiss results are highly consistent (>90% overlap)")
        elif avg_all_overlap > 75:
            logger.info("GOOD MATCH: Doris and Faiss results are mostly consistent (>75% overlap)")
        elif avg_all_overlap > 50:
            logger.info("FAIR MATCH: Doris and Faiss results show moderate consistency (>50% overlap)")
        else:
            logger.info("POOR MATCH: Significant differences between Doris and Faiss results (<50% overlap)")
    else:
        logger.info("No tests performed.")

    # 打印测试结果较差的表（平均 overlap < 75%）
    bad_tables = []
    if hasattr(print_comparison_summary, "_table_overlaps"):
        table_overlaps = print_comparison_summary._table_overlaps
        for table, overlaps in table_overlaps.items():
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                if avg_overlap < 75:
                    bad_tables.append((table, avg_overlap))
    if bad_tables:
        logger.info("\nTables with poor test results (average overlap < 75%):")
        for table, avg_overlap in bad_tables:
            logger.info(f"  {table}: average overlap = {avg_overlap:.2f}%")
    else:
        logger.info("\nNo tables with average overlap < 75%.")
    
    logger.info("="*80)

    # 新增：打印详细表格
    if _table_metrics:
        import tabulate
        headers = [
            "Table",
            "Range Overlap", "Range Faiss P", "Range Faiss R", "Range Doris P", "Range Doris R",
            "TopN Overlap", "TopN Faiss P", "TopN Faiss R", "TopN Doris P", "TopN Doris R",
            "Compound Overlap", "Compound Faiss P", "Compound Faiss R", "Compound Doris P", "Compound Doris R"
        ]
        rows = []
        for table in sorted(_table_metrics.keys()):
            row = [table]
            # Range Search
            m = _table_metrics[table].get("Range Search", {})
            row.append(f"{(m.get('overlap', 0)*100):.2f}" if m.get('overlap') is not None else "")
            row.append(f"{(m.get('faiss_precision', 0)*100):.2f}" if m.get('faiss_precision') is not None else "")
            row.append(f"{(m.get('faiss_recall', 0)*100):.2f}" if m.get('faiss_recall') is not None else "")
            row.append(f"{(m.get('doris_precision', 0)*100):.2f}" if m.get('doris_precision') is not None else "")
            row.append(f"{(m.get('doris_recall', 0)*100):.2f}" if m.get('doris_recall') is not None else "")
            # Top-N Search
            m = _table_metrics[table].get("Top-N Search", {})
            row.append(f"{(m.get('overlap', 0)*100):.2f}" if m.get('overlap') is not None else "")
            row.append(f"{(m.get('faiss_precision', 0)*100):.2f}" if m.get('faiss_precision') is not None else "")
            row.append(f"{(m.get('faiss_recall', 0)*100):.2f}" if m.get('faiss_recall') is not None else "")
            row.append(f"{(m.get('doris_precision', 0)*100):.2f}" if m.get('doris_precision') is not None else "")
            row.append(f"{(m.get('doris_recall', 0)*100):.2f}" if m.get('doris_recall') is not None else "")
            # Compound Search
            m = _table_metrics[table].get("Compound Search", {})
            row.append(f"{(m.get('overlap', 0)*100):.2f}" if m.get('overlap') is not None else "")
            row.append(f"{(m.get('faiss_precision', 0)*100):.2f}" if m.get('faiss_precision') is not None else "")
            row.append(f"{(m.get('faiss_recall', 0)*100):.2f}" if m.get('faiss_recall') is not None else "")
            row.append(f"{(m.get('doris_precision', 0)*100):.2f}" if m.get('doris_precision') is not None else "")
            row.append(f"{(m.get('doris_recall', 0)*100):.2f}" if m.get('doris_recall') is not None else "")
            rows.append(row)
        try:
            table_str = tabulate.tabulate(rows, headers, tablefmt="github")
        except Exception:
            # fallback: simple print
            table_str = "\n".join(
                ["\t".join(headers)] +
                ["\t".join(row) for row in rows]
            )
        logger.info("\nDETAILED TABLE METRICS:\n" + table_str)

    # ===== 新增：绘制精度-召回率图像 =====
    # 只画 Top-N Search 的精度和召回率
    table_points = []
    for table, metrics in _table_metrics.items():
        m = metrics.get("Top-N Search", {})
        # 只画有数据的
        if m.get("doris_precision") is not None and m.get("doris_recall") is not None:
            # 取doris的精度和召回率
            table_points.append({
                "table": table,
                "precision": m["doris_precision"],
                "recall": m["doris_recall"],
            })

    if table_points:
        # 解析维度和行数
        for p in table_points:
            parts = p["table"].split("_")
            p["dim"] = int(parts[1])
            p["num"] = int(parts[3])

        # 第一张图：同一维度连线
        plt.figure(figsize=(10, 7))
        dims = sorted(set(p["dim"] for p in table_points))
        colors = plt.cm.get_cmap('tab20', len(dims))
        for idx, dim in enumerate(dims):
            group = [p for p in table_points if p["dim"] == dim]
            group = sorted(group, key=lambda x: x["num"])
            plt.plot(
                [g["recall"] for g in group],
                [g["precision"] for g in group],
                marker='o',
                label=f'dim={dim}',
                color=colors(idx)
            )
            for g in group:
                plt.text(g["recall"], g["precision"], g["table"], fontsize=7)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall by Dimension (Top-N Search)")
        plt.legend()
        plt.grid(True)
        img_path1 = "summary_precision_recall_by_dim.png"
        plt.savefig(img_path1)
        logger.info(f"Saved precision-recall plot by dimension to {img_path1}")
        plt.close()

        # 第二张图：同一行数连线
        plt.figure(figsize=(10, 7))
        nums = sorted(set(p["num"] for p in table_points))
        colors = plt.cm.get_cmap('tab20', len(nums))
        for idx, num in enumerate(nums):
            group = [p for p in table_points if p["num"] == num]
            group = sorted(group, key=lambda x: x["dim"])
            plt.plot(
                [g["recall"] for g in group],
                [g["precision"] for g in group],
                marker='o',
                label=f'num={num}',
                color=colors(idx)
            )
            for g in group:
                plt.text(g["recall"], g["precision"], g["table"], fontsize=7)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall by Row Count (Top-N Search)")
        plt.legend()
        plt.grid(True)
        img_path2 = "summary_precision_recall_by_num.png"
        plt.savefig(img_path2)
        logger.info(f"Saved precision-recall plot by row count to {img_path2}")
        plt.close()
def main() -> None:
    tables = [
        "dim_1_num_10",
        "dim_1_num_1000",
        "dim_1_num_2000",
        "dim_1_num_5000",
        "dim_1_num_10000",
        # "dim_4_num_10",
        # "dim_4_num_1000",
        # "dim_4_num_2000",
        # "dim_4_num_5000",
        # "dim_4_num_10000",
        # "dim_8_num_10",
        # "dim_8_num_1000",
        # "dim_8_num_2000",
        # "dim_8_num_5000",
        # "dim_8_num_10000",
        # "dim_16_num_10",
        # "dim_16_num_1000",
        # "dim_16_num_2000",
        # "dim_16_num_5000",
        # "dim_16_num_10000",
        # "dim_32_num_10",
        # "dim_32_num_1000",
        # "dim_32_num_2000",
        # "dim_32_num_5000",
        # "dim_32_num_10000",
        # "dim_1024_num_10",
        # "dim_1024_num_1000",
        # "dim_1024_num_2000",
        # "dim_1024_num_5000",
        # "dim_1024_num_10000",
        # "dim_2048_num_10",
        # "dim_2048_num_1000",
        # "dim_2048_num_2000",
        # "dim_2048_num_5000",
        # "dim_2048_num_10000",
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # Connect to Doris
    conn = get_conn()
    cursor = conn.cursor()
    
    for table in tables:
        logger.info(f"Processing table: {table}")
        
        # Extract dimension from table name
        dimension = int(table.split("_")[1])
        
        # Read data from TSV
        tsv_path = os.path.join(data_dir, f"{table}.tsv")
        ids, embeddings = read_tsv_data(tsv_path)
        
        if len(embeddings) == 0:
            logger.error(f"No data found in {tsv_path}")
            continue
            
        # Build Faiss index with explicit IDs
        index = build_faiss_index("hnsw", "l2", embeddings, ids, dimension)
        logger.info(f"Built Faiss index with {len(embeddings)} vectors of dimension {dimension}")
        
        # Select a random vector for querying
        random_idx = random.randint(0, len(embeddings) - 1)
        query_vector = embeddings[random_idx]
        query_vector_str = str(query_vector.tolist()).replace(' ', '')
        
        # Use Doris
        cursor.execute(f"USE vector_test;")

        # 1. Range search with percentile-based radius
        percentile_less = random.randint(30, 70)  # Random percentile between 30-70
        radius_less = get_radius(query_vector, embeddings, percentile_less)
        logger.info(f"--- Range Search (distance < {radius_less:.2f}, {percentile_less}th percentile) ---")
        
        # Faiss range search
        faiss_range_results = faiss_range_search(index, query_vector, radius_less)
        brute_force_index = build_faiss_index("flat","l2", embeddings, ids, dimension)
        ground_truth = faiss_range_search(brute_force_index, query_vector, radius_less)
        # Doris range search
        range_search_sql = f"""
            SELECT * FROM (SELECT id, l2_distance(embedding, {query_vector_str}) as distance FROM {table} WHERE l2_distance(embedding, {query_vector_str}) < {radius_less}) as SUB ORDER BY distance;"""
        if dimension <= 8:
            logger.info(f"SQL: {range_search_sql}")
        cursor.execute(range_search_sql)
        doris_range_results = cursor.fetchall()

        # 比较结果
        compare_and_log(
            faiss_range_results,
            doris_range_results,
            "Range Search",
            dimension,
            table_name=table,
            ground_truth=ground_truth
        )

        # 3. Top-N search
        logger.info("--- Top-N Search ---")
        
        # 随机生成topK
        topk = random.randint(3, 100)
        logger.info(f"Top-N search with K={topk}")

        # Faiss top-N search
        faiss_topn_results = faiss_topn_search(index, query_vector, limit=topk)
        # logger.info(f"Faiss top-N search results: {faiss_topn_results}")
        
        # Doris top-N search
        topn_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector_str}) FROM {table} ORDER BY l2_distance(embedding, {query_vector_str}) limit {topk};"""
        if dimension <= 8:
            logger.info(f"SQL: {topn_search_sql}")
        cursor.execute(topn_search_sql)
        doris_topn_results = cursor.fetchall()
        # logger.info(f"Doris top-N search results: {doris_topn_results}")
        
        # 计算topn ground truth（暴力法）
        gt_topn = brute_force_topn(query_vector, embeddings, ids, topk)
        compare_and_log(
            faiss_topn_results,
            doris_topn_results,
            "Top-N Search",
            dimension,
            table_name=table,
            ground_truth=gt_topn
        )

        # 4. Compound search with percentile-based radius
        percentile_compound = random.randint(10, 40)  # Random percentile between 10-40
        radius_compound = get_radius(query_vector, embeddings, percentile_compound)
        # 随机生成compound search的topK
        compound_topk = random.randint(3, 10)
        logger.info(f"--- Compound Search (distance < {radius_compound:.2f}, {percentile_compound}th percentile, and sorted, K={compound_topk}) ---")
        
        # Faiss implementation of compound search
        faiss_compound_results = faiss_compound_search(index, query_vector, radius_compound, compound_topk)
        faiss_compound_results.sort(key=lambda x: x[1])  # Sort by distance
        # logger.info(f"Faiss compound search results: {faiss_compound_results}")
        
        # Doris compound search
        compound_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector_str}) as distance FROM {table} WHERE l2_distance(embedding, {query_vector_str}) < {radius_compound} ORDER BY distance limit {compound_topk};"""
        if dimension <= 8:
            logger.info(f"SQL: {compound_search_sql}")
        cursor.execute(compound_search_sql)
        doris_compound_results = cursor.fetchall()
        
        # 计算compound ground truth（暴力法+距离过滤+排序+topk）
        dists = np.linalg.norm(embeddings - query_vector, axis=1)
        mask = dists < radius_compound
        filtered_ids = ids[mask]
        filtered_dists = dists[mask]
        idxs = np.argsort(filtered_dists)[:compound_topk]
        gt_compound = [(int(filtered_ids[i]), float(filtered_dists[i])) for i in idxs]

        compare_and_log(
            faiss_compound_results,
            doris_compound_results,
            "Compound Search",
            dimension,
            table_name=table,
            ground_truth=gt_compound
        )

        logger.info(f"Completed comparison for table {table}\n" + "-"*50)
    
    # Print summary of all test results
    print_comparison_summary()

if __name__ == "__main__":
    main()