import numpy as np
import mysql.connector
import pandas as pd
import logging
import time
import os
import requests
from requests.auth import HTTPBasicAuth
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def create_data_directory():
    """Create a data directory to store CSV files if it doesn't exist"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created data directory at {data_dir}")
    return data_dir

def save_dataset_to_tsv(dataset, table_name, data_dir):
    """Save the dataset to a TSV file without headers"""
    tsv_path = os.path.join(data_dir, f"{table_name}.tsv")
    
    # Check if the TSV file already exists
    if os.path.exists(tsv_path):
        logger.info(f"TSV file already exists at {tsv_path}, skipping creation")
        return tsv_path
    
    logger.info(f"Saving dataset to {tsv_path}")
    
    # Convert embedding lists to strings for TSV storage but keep the square brackets
    dataset_copy = dataset.copy()
    dataset_copy['embedding'] = dataset_copy['embedding'].apply(lambda x: str(x))
    
    # Write to TSV without headers
    dataset_copy.to_csv(tsv_path, index=False, sep='\t', header=False)
    logger.info(f"Dataset saved to {tsv_path} without headers")
    return tsv_path

def stream_load_to_doris(table_name, tsv_path, host="127.0.0.1", port=5937, db="vector_test"):
    """Load data from TSV file to Doris using Stream Load"""
    logger.info(f"Loading data to table {table_name} using Stream Load")
    start_time = time.time()
    
    url = f"http://{host}:{port}/api/{db}/{table_name}/_stream_load"
    headers = {
        'Expect': '100-continue',
        'Content-Type': 'text/plain; charset=UTF-8',
        'label': f"load_{table_name}_{int(time.time())}",
        'format': 'csv',  # Changed back to csv format which works with TSV when specifying column_separator
        # 'column_separator': '\t',
        'columns': 'id,embedding',
    }
    
    with open(tsv_path, 'rb') as f:
        tsv_data = f.read()

    try:
        auth = HTTPBasicAuth("root", "")
        session = requests.sessions.Session()
        session.should_strip_auth = lambda old_url, new_url: False  # Don't strip auth
        resp = session.request(
            'PUT', url=url,
            data=tsv_data,
            headers=headers, auth=auth
        )
        
        # Log detailed response
        logger.info(f"Stream load status code: {resp.status_code}")
        logger.info(f"Raw response text: {resp.text}")
        
        # Try to parse JSON response for detailed error information
        try:
            response_json = resp.json()
            logger.info(f"Response JSON: {json.dumps(response_json, indent=2)}")
            
            # Log specific Doris error details
            if 'status' in response_json:
                logger.info(f"Load status: {response_json['status']}")
            if 'msg' in response_json:
                logger.info(f"Message: {response_json['msg']}")
            if 'errorURL' in response_json:
                logger.info(f"Error URL: {response_json['errorURL']}")
            if 'errorMsg' in response_json:
                logger.info(f"Error message: {response_json['errorMsg']}")
                
        except json.JSONDecodeError:
            logger.warning("Could not parse response as JSON")

    except Exception as e:
        logger.error(f"Error during stream load: {e}", exc_info=True)

def generate_dataset(dim, num, low=1, high=100):
    """
    生成包含 ID 的向量表：DataFrame，列为 id 和 embedding（嵌套 list）
    """
    logger.info(f"Generating dataset with dim={dim}, num={num}")
    start_time = time.time()
    embeddings = np.random.uniform(low, high, size=(num, dim))
    df = pd.DataFrame({
        "id": np.arange(num),
        "embedding": embeddings.tolist()
    })
    logger.info(f"Dataset generated in {time.time() - start_time:.2f} seconds")
    return df


def compute_median_l2_distance(query_vector, dataset):
    """
    dataset: DataFrame，embedding 列是 List[float]
    """
    logger.info(f"Computing median L2 distance for {len(dataset)} vectors")
    start_time = time.time()
    X = np.array(dataset["embedding"].tolist())
    diff = X - query_vector
    distances = np.sum(diff ** 2, axis=1)
    # Remove adding l2_distance to the dataset
    median = np.median(distances)
    logger.info(f"Median L2 distance computed: {median:.6f} in {time.time() - start_time:.2f} seconds")
    return median, dataset


def generate_create_table_sql(dim, num):
    logger.info(f"Generating SQL for table with dim={dim}, num={num}")
    table_name = f"dim_{dim}_num_{num}"
    return f"""
CREATE TABLE `{table_name}` (
  `id` int NOT NULL COMMENT "",
  `embedding` array<float>  NOT NULL  COMMENT "",
  INDEX idx_test_ann (`embedding`) USING ANN PROPERTIES(
      "index_type"="hnsw",
      "metric_type"="l2_distance",
      "dim"="{dim}"
  )
) ENGINE=OLAP
DUPLICATE KEY(`id`) COMMENT "OLAP"
DISTRIBUTED BY HASH(`id`) BUCKETS 1
PROPERTIES (
  "replication_num" = "1"
);
"""

def get_conn():
    logger.info("Establishing database connection")
    try:
        conn = mysql.connector.connect(
            user="root",
            password="",
            host="127.0.0.1",
            port=6937
        )
        logger.info("Database connection successful")
        return conn
    except mysql.connector.Error as err:
        logger.error(f"Database connection failed: {err}")
        raise

def main():
    logger.info("Starting main process")
    try:
        # Create data directory
        data_dir = create_data_directory()
        
        conn = get_conn()
        cursor = conn.cursor()
        
        logger.info("Dropping database if exists")
        cursor.execute("DROP DATABASE IF EXISTS vector_test")
        
        logger.info("Creating database vector_test")
        cursor.execute("CREATE DATABASE vector_test")
        
        logger.info("Switching to vector_test database")
        cursor.execute("USE vector_test")

        # dims = [1, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        # counts = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000]
        # dims = [1024, 2048, 4096, 8192]
        dims = [1, 4, 8, 16, 32,  1024, 2048]
        counts = [10, 1000, 2000, 5000, 10000]

        logger.info(f"Testing dimensions: {dims}")
        logger.info(f"Testing counts: {counts}")

        for dim in dims:
            for count in counts:
                logger.info(f"=== Processing dim={dim}, count={count} ===")
                
                # Generate dataset
                dataset = generate_dataset(dim, count)
                
                # Compute median distance
                median, dataset = compute_median_l2_distance(dataset.iloc[0]["embedding"], dataset)
                print(f"Median L2 distance: {median}")

                # Create table
                table_name = f"dim_{dim}_num_{count}"
                logger.info(f"Creating table {table_name}")
                create_table_sql = generate_create_table_sql(dim, count)
                cursor.execute(create_table_sql)
                
                # Save dataset to TSV instead of CSV
                tsv_path = save_dataset_to_tsv(dataset, table_name, data_dir)
                
                # Load data using stream load with the TSV file
                logger.info(f"Loading {count} rows into table {table_name} using stream load")
                stream_load_to_doris(table_name, tsv_path)
                
                logger.info(f"Completed processing for dim={dim}, count={count}")
                
            logger.info(f"Finished processing dimension {dim}")

        logger.info("Closing database connection")
        cursor.close()
        conn.close()
        logger.info("Process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main process: {e}", exc_info=True)
    
    
if __name__ == "__main__":
    main()