import faiss
import numpy as np
import logging
import os
import random
from query import get_conn
import mysql.connector
from typing import List, Tuple, Set, Optional, Dict, Any
import math

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

def build_faiss_index(embeddings: np.ndarray, ids: np.ndarray, dimension: int) -> faiss.IndexIDMap:
    """Build a Faiss L2 index from embeddings with explicit IDs"""
    # For IndexFlatL2, we need to use IndexIDMap to support custom IDs
    index = faiss.IndexHNSWFlat(dimension, 32)
    
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
    
    # Sort by distance and limit results
    results.sort(key=lambda x: x[1])
    return results

def faiss_range_search_greater(index: faiss.IndexIDMap, query_vector: np.ndarray, radius: float, all_ids: Optional[np.ndarray] = None, limit: int = 5) -> List[Tuple[int, float]]:
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
    
    # Sort by distance and limit results
    results.sort(key=lambda x: x[1])
    return results[:limit]

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
        "dimensions_tested": set()
    },
    "Top-N Search": {
        "total_tests": 0,
        "avg_overlap_pct": 0,
        "avg_max_diff": 0,
        "avg_mean_diff": 0,
        "dimensions_tested": set()
    },
    "Compound Search": {
        "total_tests": 0,
        "avg_overlap_pct": 0,
        "avg_max_diff": 0,
        "avg_mean_diff": 0,
        "dimensions_tested": set()
    }
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
        if total_tests > 0:
            avg_overlap = data["avg_overlap_pct"] / total_tests
            avg_max_diff = data["avg_max_diff"] / total_tests
            avg_mean_diff = data["avg_mean_diff"] / total_tests
            
            logger.info(f"\n{test_name} Summary (across {total_tests} tests):")
            logger.info(f"Average overlap percentage: {avg_overlap:.2f}%")
            logger.info(f"Average maximum distance difference: {avg_max_diff:.6f}")
            logger.info(f"Average mean distance difference: {avg_mean_diff:.6f}")
            logger.info(f"Dimensions tested: {sorted(data['dimensions_tested'])}")
    
    logger.info("\nCONCLUSION:")
    # Add overall conclusion based on results
    if all_tests_count > 0:
        avg_all_overlap = sum(data["avg_overlap_pct"] for data in comparison_summary.values()) / all_tests_count
        logger.info(f"Overall average overlap: {avg_all_overlap:.2f}%")
        if avg_all_overlap > 90:
            logger.info("EXCELLENT MATCH: Doris and Faiss results are highly consistent (>90% overlap)")
        elif avg_all_overlap > 75:
            logger.info("GOOD MATCH: Doris and Faiss results are mostly consistent (>75% overlap)")
        elif avg_all_overlap > 50:
            logger.info("FAIR MATCH: Doris and Faiss results show moderate consistency (>50% overlap)")
        else:
            logger.info("POOR MATCH: Significant differences between Doris and Faiss results (<50% overlap)")
    
    # 打印测试结果较差的表（平均 overlap < 75%）
    bad_tables = []
    if hasattr(print_comparison_summary, "_table_overlaps"):
        table_overlaps = print_comparison_summary._table_overlaps
        for table, overlaps in table_overlaps.items():
            if overlaps:
                avg_overlap = sum(overlaps) / len(overlaps)
                if avg_overlap < 85:
                    bad_tables.append((table, avg_overlap))
    if bad_tables:
        logger.info("\nTables with poor test results (average overlap < 75%):")
        for table, avg_overlap in bad_tables:
            logger.info(f"  {table}: average overlap = {avg_overlap:.2f}%")
    
    logger.info("="*80)

def main() -> None:
    tables = [
        "dim_1_num_1000",
        "dim_4_num_1000",
        "dim_8_num_1000",
        "dim_16_num_1000",
        "dim_32_num_1000",
        "dim_1024_num_1000",
        "dim_2048_num_1000",
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
        index = build_faiss_index(embeddings, ids, dimension)
        logger.info(f"Built Faiss index with {len(embeddings)} vectors of dimension {dimension}")
        
        # Select a random vector for querying
        random_idx = random.randint(0, len(embeddings) - 1)
        query_vector = embeddings[random_idx]
        query_vector_str = str(query_vector.tolist()).replace(' ', '')
        
        # Use Doris
        cursor.execute(f"USE vector_test;")
        
        # Add explanation about the square root applied to Faiss distances
        logger.info("Note: Faiss distances have been square-rooted to match Doris L2 distances")
        
        # 1. Range search with percentile-based radius
        percentile_less = random.randint(30, 70)  # Random percentile between 30-70
        radius_less = get_radius(query_vector, embeddings, percentile_less)
        logger.info(f"--- Range Search (distance < {radius_less:.2f}, {percentile_less}th percentile) ---")
        
        # Faiss range search
        faiss_range_results = faiss_range_search(index, query_vector, radius_less)
        # logger.info(f"Faiss range search results: {faiss_range_results}")
        
        # Doris range search
        range_search_sql = f"""
            SELECT * FROM (SELECT id, l2_distance(embedding, {query_vector_str}) as distance FROM {table} WHERE l2_distance(embedding, {query_vector_str}) < {radius_less}) as SUB ORDER BY distance;"""
        cursor.execute(range_search_sql)
        doris_range_results = cursor.fetchall()
        logger.info(f"SQL: {range_search_sql}")
        # logger.info(f"Doris range search results: {doris_range_results}")
        
        # Compare range search results
        compare_results(faiss_range_results, doris_range_results, "Range Search", dimension, table_name=table)
        
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
        cursor.execute(topn_search_sql)
        doris_topn_results = cursor.fetchall()
        # logger.info(f"Doris top-N search results: {doris_topn_results}")
        
        # Compare top-N search results
        compare_results(faiss_topn_results, doris_topn_results, "Top-N Search", dimension, table_name=table)
        
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
        compound_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector_str}) FROM {table} WHERE l2_distance(embedding, {query_vector_str}) < {radius_compound} ORDER BY l2_distance(embedding, {query_vector_str}) limit {compound_topk};"""
        logger.info(f"SQL: {compound_search_sql}")
        cursor.execute(compound_search_sql)
        doris_compound_results = cursor.fetchall()
        # logger.info(f"Doris compound search results: {doris_compound_results}")
        
        # Compare compound search results
        compare_results(faiss_compound_results, doris_compound_results, "Compound Search", dimension, table_name=table)
        
        logger.info(f"Completed comparison for table {table}\n" + "-"*50)
    
    # Print summary of all test results
    print_comparison_summary()

if __name__ == "__main__":
    main()