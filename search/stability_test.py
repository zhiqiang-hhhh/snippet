import faiss
import numpy as np
import logging
import os
import random
import time
from query import get_conn
import mysql.connector
from typing import List, Tuple, Set, Optional, Dict, Any
import math
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
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
            if len(parts) >= 3:  # Now expecting id, category, embedding
                ids.append(int(parts[0]))
                # Skip category (parts[1]) and use embedding (parts[2])
                vector = np.array([float(x) for x in parts[2].strip('[]').split(',')], dtype=np.float32)
                embeddings.append(vector)
    
    return np.array(ids, dtype=np.int64), np.array(embeddings, dtype=np.float32)

def build_faiss_index(index_type: str, metric_type: str, embeddings: np.ndarray, ids: np.ndarray, dimension: int) -> faiss.IndexIDMap:
    """Build a Faiss index from embeddings with explicit IDs."""
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
    squared_radius = radius * radius
    query_vector_reshaped = np.array([query_vector], dtype=np.float32)
    
    lims, D, I = index.range_search(query_vector_reshaped, squared_radius)
    
    results = []
    for i in range(int(lims[0]), int(lims[1])):
        results.append((int(I[i]), math.sqrt(float(D[i]))))
    
    results.sort(key=lambda x: x[1])
    return results

def faiss_topn_search(index: faiss.IndexIDMap, query_vector: np.ndarray, limit: int = 5) -> List[Tuple[int, float]]:
    """Perform top-N search with Faiss"""
    D, I = index.search(np.array([query_vector]), limit)
    results = []
    
    for i in range(min(len(I[0]), limit)):
        if I[0][i] != -1:
            results.append((int(I[0][i]), math.sqrt(float(D[0][i]))))
    
    return results

def faiss_compound_search(index: faiss.IndexIDMap, query_vector: np.ndarray, radius: float, limit: int = 5) -> List[Tuple[int, float]]:
    """Perform compound search with Faiss"""
    squared_radius = radius * radius
    query_vector_reshaped = np.array([query_vector], dtype=np.float32)
    
    lims, D, I = index.range_search(query_vector_reshaped, squared_radius)
    
    if lims[1] == lims[0]:
        return []
    
    range_ids = np.array([int(I[i]) for i in range(int(lims[0]), int(lims[1]))], dtype=np.int64)
    id_selector = faiss.IDSelectorBatch(range_ids)
    
    params = faiss.SearchParametersHNSW()
    params.sel = id_selector
    D_topn, I_topn = index.search(query_vector_reshaped, limit, params=params)
    
    results = []
    for i in range(len(I_topn[0])):
        if I_topn[0][i] != -1:
            results.append((int(I_topn[0][i]), math.sqrt(float(D_topn[0][i]))))
    
    return results

def get_radius(query_vector: np.ndarray, embeddings: np.ndarray, percentile: int) -> float:
    """Calculate the radius based on a percentile of distances"""
    distances = np.sqrt(np.sum((embeddings - query_vector) ** 2, axis=1))
    return float(np.percentile(distances, percentile))

def test_doris_query(cursor, table: str, query_vector_str: str, test_type: str, **kwargs) -> Tuple[bool, List[Tuple], str]:
    """
    执行Doris查询并捕获任何错误
    返回: (成功标志, 结果列表, 错误信息)
    """
    try:
        if test_type == "range":
            radius = kwargs['radius']
            sql = f"""
                SELECT * FROM (SELECT id, l2_distance_approximate(embedding, {query_vector_str}) as distance, category 
                FROM {table} WHERE l2_distance_approximate(embedding, {query_vector_str}) < {radius}) as SUB ORDER BY distance;"""
        elif test_type == "topn":
            topk = kwargs['topk']
            sql = f"""SELECT id, l2_distance_approximate(embedding, {query_vector_str}), category 
                FROM {table} ORDER BY l2_distance_approximate(embedding, {query_vector_str}) limit {topk};"""
        elif test_type == "compound":
            radius = kwargs['radius']
            topk = kwargs['topk']
            sql = f"""SELECT id, l2_distance_approximate(embedding, {query_vector_str}) as distance, category 
                FROM {table} WHERE l2_distance_approximate(embedding, {query_vector_str}) < {radius} ORDER BY distance limit {topk};"""
        else:
            return False, [], f"Unknown test type: {test_type}"
        
        cursor.execute(sql)
        results = cursor.fetchall()
        return True, results, ""
    
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Doris query failed: {error_msg}")
        return False, [], error_msg

def run_stability_test(table: str, embeddings: np.ndarray, ids: np.ndarray, dimension: int, 
                      cursor, num_iterations: int = 50) -> Dict[str, Any]:
    """
    对单个表执行稳定性测试
    """
    logger.info(f"Starting stability test for table {table} with {num_iterations} iterations")
    
    # 构建Faiss索引
    index = build_faiss_index("hnsw", "l2", embeddings, ids, dimension)
    
    # 测试结果统计
    stats = {
        "table": table,
        "dimension": dimension,
        "total_iterations": num_iterations,
        "range_search": {"success": 0, "failed": 0, "errors": []},
        "topn_search": {"success": 0, "failed": 0, "errors": []},
        "compound_search": {"success": 0, "failed": 0, "errors": []},
        "query_vectors_used": [],
        "parameters_used": {
            "topk_values": [],
            "radius_values": [],
            "percentiles_used": []
        }
    }
    
    cursor.execute(f"USE vector_test;")
    
    for iteration in range(num_iterations):
        logger.info(f"Table {table} - Iteration {iteration + 1}/{num_iterations}")
        
        # 随机选择查询向量
        random_idx = random.randint(0, len(embeddings) - 1)
        query_vector = embeddings[random_idx]
        query_vector_str = str(query_vector.tolist()).replace(' ', '')
        stats["query_vectors_used"].append(random_idx)
        
        # 随机生成参数
        topk = random.randint(5, 50)
        percentile_range = random.randint(20, 80)
        percentile_compound = random.randint(10, 50)
        radius_range = get_radius(query_vector, embeddings, percentile_range)
        radius_compound = get_radius(query_vector, embeddings, percentile_compound)
        
        stats["parameters_used"]["topk_values"].append(topk)
        stats["parameters_used"]["radius_values"].extend([radius_range, radius_compound])
        stats["parameters_used"]["percentiles_used"].extend([percentile_range, percentile_compound])
        
        # 1. Range Search测试
        try:
            faiss_range_results = faiss_range_search(index, query_vector, radius_range)
            success, doris_range_results, error_msg = test_doris_query(
                cursor, table, query_vector_str, "range", radius=radius_range
            )
            
            if success:
                stats["range_search"]["success"] += 1
                logger.debug(f"Range search success: Faiss={len(faiss_range_results)}, Doris={len(doris_range_results)}")
            else:
                stats["range_search"]["failed"] += 1
                stats["range_search"]["errors"].append({
                    "iteration": iteration + 1,
                    "error": error_msg,
                    "parameters": {"radius": radius_range, "percentile": percentile_range}
                })
        except Exception as e:
            stats["range_search"]["failed"] += 1
            stats["range_search"]["errors"].append({
                "iteration": iteration + 1,
                "error": f"Faiss error: {str(e)}",
                "parameters": {"radius": radius_range, "percentile": percentile_range}
            })
        
        # 2. Top-N Search测试
        try:
            faiss_topn_results = faiss_topn_search(index, query_vector, limit=topk)
            success, doris_topn_results, error_msg = test_doris_query(
                cursor, table, query_vector_str, "topn", topk=topk
            )
            
            if success:
                stats["topn_search"]["success"] += 1
                logger.debug(f"TopN search success: Faiss={len(faiss_topn_results)}, Doris={len(doris_topn_results)}")
            else:
                stats["topn_search"]["failed"] += 1
                stats["topn_search"]["errors"].append({
                    "iteration": iteration + 1,
                    "error": error_msg,
                    "parameters": {"topk": topk}
                })
        except Exception as e:
            stats["topn_search"]["failed"] += 1
            stats["topn_search"]["errors"].append({
                "iteration": iteration + 1,
                "error": f"Faiss error: {str(e)}",
                "parameters": {"topk": topk}
            })
        
        # 3. Compound Search测试
        try:
            faiss_compound_results = faiss_compound_search(index, query_vector, radius_compound, topk)
            success, doris_compound_results, error_msg = test_doris_query(
                cursor, table, query_vector_str, "compound", radius=radius_compound, topk=topk
            )
            
            if success:
                stats["compound_search"]["success"] += 1
                logger.debug(f"Compound search success: Faiss={len(faiss_compound_results)}, Doris={len(doris_compound_results)}")
            else:
                stats["compound_search"]["failed"] += 1
                stats["compound_search"]["errors"].append({
                    "iteration": iteration + 1,
                    "error": error_msg,
                    "parameters": {"radius": radius_compound, "topk": topk, "percentile": percentile_compound}
                })
        except Exception as e:
            stats["compound_search"]["failed"] += 1
            stats["compound_search"]["errors"].append({
                "iteration": iteration + 1,
                "error": f"Faiss error: {str(e)}",
                "parameters": {"radius": radius_compound, "topk": topk, "percentile": percentile_compound}
            })
        
        # 每10次迭代输出一次进度
        if (iteration + 1) % 10 == 0:
            logger.info(f"Progress: {iteration + 1}/{num_iterations} completed for table {table}")
    
    # 计算成功率
    total_tests = num_iterations * 3  # 3种测试类型
    total_success = (stats["range_search"]["success"] + 
                    stats["topn_search"]["success"] + 
                    stats["compound_search"]["success"])
    stats["overall_success_rate"] = total_success / total_tests if total_tests > 0 else 0
    
    logger.info(f"Stability test completed for table {table}")
    logger.info(f"Success rates - Range: {stats['range_search']['success']}/{num_iterations}, "
               f"TopN: {stats['topn_search']['success']}/{num_iterations}, "
               f"Compound: {stats['compound_search']['success']}/{num_iterations}")
    logger.info(f"Overall success rate: {stats['overall_success_rate']*100:.2f}%")
    
    return stats

def save_stability_results(all_stats: List[Dict[str, Any]], output_file: str = "stability_test_results.json"):
    """保存稳定性测试结果到JSON文件"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_stats, f, indent=2, ensure_ascii=False)
        logger.info(f"Stability test results saved to {output_file}")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def print_stability_summary(all_stats: List[Dict[str, Any]]):
    """打印稳定性测试总结"""
    logger.info("\n" + "="*80)
    logger.info("STABILITY TEST SUMMARY")
    logger.info("="*80)
    
    total_tables = len(all_stats)
    total_errors = 0
    
    for stats in all_stats:
        table = stats["table"]
        dimension = stats["dimension"]
        iterations = stats["total_iterations"]
        success_rate = stats["overall_success_rate"] * 100
        
        range_errors = len(stats["range_search"]["errors"])
        topn_errors = len(stats["topn_search"]["errors"])
        compound_errors = len(stats["compound_search"]["errors"])
        table_total_errors = range_errors + topn_errors + compound_errors
        total_errors += table_total_errors
        
        logger.info(f"Table: {table} (dim={dimension})")
        logger.info(f"  Iterations: {iterations}")
        logger.info(f"  Success Rate: {success_rate:.2f}%")
        logger.info(f"  Errors: Range={range_errors}, TopN={topn_errors}, Compound={compound_errors}")
        
        # 如果有错误，显示错误详情
        if table_total_errors > 0:
            logger.info(f"  Error Details:")
            for error_info in (stats["range_search"]["errors"] + 
                             stats["topn_search"]["errors"] + 
                             stats["compound_search"]["errors"]):
                logger.info(f"    Iteration {error_info['iteration']}: {error_info['error']}")
    
    logger.info(f"\nOverall Summary:")
    logger.info(f"  Total Tables Tested: {total_tables}")
    logger.info(f"  Total Errors: {total_errors}")
    if total_errors == 0:
        logger.info("  🎉 ALL TESTS PASSED - Doris is stable!")
    else:
        logger.info(f"  ⚠️  Some tests failed - Review errors above")

def main():
    """主函数"""
    # 测试参数配置
    NUM_ITERATIONS = 30  # 每个表的测试次数
    
    # 选择要测试的表（可以根据需要调整）
    test_tables = [
        "dim_1_num_10",
        "dim_4_num_100", 
        "dim_8_num_1000",
        "dim_16_num_1000",
        "dim_32_num_1000",
        "dim_1024_num_100",
    ]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    
    # 连接Doris
    try:
        conn = get_conn()
        cursor = conn.cursor()
        logger.info("Connected to Doris successfully")
    except Exception as e:
        logger.error(f"Failed to connect to Doris: {e}")
        return
    
    all_stats = []
    
    for table in test_tables:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing table: {table}")
        logger.info(f"{'='*60}")
        
        # 提取维度信息
        dimension = int(table.split("_")[1])
        
        # 读取TSV数据
        tsv_path = os.path.join(data_dir, f"{table}.tsv")
        if not os.path.exists(tsv_path):
            logger.warning(f"TSV file not found: {tsv_path}, skipping table {table}")
            continue
        
        try:
            ids, embeddings = read_tsv_data(tsv_path)
            
            if len(embeddings) == 0:
                logger.error(f"No data found in {tsv_path}")
                continue
            
            logger.info(f"Loaded {len(embeddings)} vectors of dimension {dimension}")
            
            # 执行稳定性测试
            stats = run_stability_test(table, embeddings, ids, dimension, cursor, NUM_ITERATIONS)
            all_stats.append(stats)
            
        except Exception as e:
            logger.error(f"Error processing table {table}: {e}")
            continue
    
    # 关闭数据库连接
    cursor.close()
    conn.close()
    
    # 保存和打印结果
    save_stability_results(all_stats)
    print_stability_summary(all_stats)
    
    logger.info("\nStability test completed!")

if __name__ == "__main__":
    main()
