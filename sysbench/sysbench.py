import subprocess
import threading
import argparse
import time
import numpy as np

class QueryExecutionError(Exception):
    pass

def run_sql_query(sql, client_id, results, host, lock):
    start_time = time.time()
    command = f"""mysql -uroot -h{host} -P9130 -D test_db_30M -e \"{sql}\""""
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    end_time = time.time()
    elapsed_time = (end_time - start_time) * 1000  # Convert to milliseconds

    if result.returncode != 0:
        print(f"Client {client_id}: Error executing query: {result.stderr}")
        raise QueryExecutionError(f"Client {client_id}: Error executing query: {result.stderr}")

    with lock:
        results.append(elapsed_time)

def load_sql_queries(file_path):
    with open(file_path, 'r') as file:
        queries = file.readlines()
    return [query.strip() for query in queries]

def run_load_test(queries, iterations, clients, host):
    results = []
    lock = threading.Lock()
    try:
        threads = []
        for client_id in range(clients):
            thread = threading.Thread(target=run_client, args=(queries, iterations, client_id, results, host, lock))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Print final statistics
        min_cost, max_cost, p95_cost, qps = calculate_statistics(results)
        print(f"Final: Min Cost: {min_cost} ms, Max Cost: {max_cost} ms, P95 Cost: {p95_cost} ms, QPS: {qps}")

    except QueryExecutionError as e:
        print(e)
        for thread in threads:
            if thread.is_alive():
                thread.join()  # Ensure all threads are properly joined before exiting

    return results

def run_client(queries, iterations, client_id, results, host, lock):
    for i in range(iterations):
        for query in queries:
            run_sql_query(query, client_id, results, host, lock)

        # Print statistics after each iteration
        min_cost, max_cost, p95_cost, qps = calculate_statistics(results)
        print(f"Client {client_id} Iteration {i+1}: Min Cost: {min_cost} ms, Max Cost: {max_cost} ms, P95 Cost: {p95_cost} ms, QPS: {qps}")

        # Clear results for the next iteration
        with lock:
            results.clear()

def calculate_statistics(results):
    min_cost = np.min(results)
    max_cost = np.max(results)
    p95_cost = np.percentile(results, 95)
    qps = len(results) / (np.sum(results) / 1000)  # Queries per second
    return min_cost, max_cost, p95_cost, qps

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SQL Load Test Script")
    parser.add_argument('-i', type=int, required=True, help="Number of iterations")
    parser.add_argument('-c', type=int, required=True, help="Number of concurrent clients")
    parser.add_argument('-f', type=str, required=True, help="Path to the cleaned SQL file")
    parser.add_argument('--host', type=str, required=True, help="MySQL host")

    args = parser.parse_args()

    queries = load_sql_queries(args.f)
    results = run_load_test(queries, args.i, args.c, args.host)
    min_cost, max_cost, p95_cost, qps = calculate_statistics(results)

    print(f"Min Cost: {min_cost} ms")
    print(f"Max Cost: {max_cost} ms")
    print(f"P95 Cost: {p95_cost} ms")
    print(f"QPS: {qps}")