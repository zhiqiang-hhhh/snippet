import logging
import faiss
from faiss.contrib.datasets import DatasetSIFT1M

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_sift():
    """Load SIFT-1M dataset to Doris"""
    logger.info("Loading SIFT 1M dataset")
    
    # 1. Load SIFT-1M dataset from data directory.
    # User should link the pre-downloaded SIFT-1M dataset to the data directory.
    # http://corpus-texmex.irisa.fr/
    dataSet = DatasetSIFT1M()
    query = dataSet.get_queries()
    database = dataSet.get_database()
    groundtruth = dataSet.get_groundtruth()
    
    logger.info(f"Database shape: {database.shape}")
    logger.info(f"Query shape: {query.shape}")
    logger.info(f"Groundtruth shape: {groundtruth.shape}")
    
    # 2. Do transformation. We need a tsv file, so that we can use Doris stream load.
    import os
    import numpy as np
    
    # Create data directory if it doesn't exist
    data_dir = "data_sift"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        logger.info(f"Created directory: {data_dir}")
    
    # Transform database vectors to TSV format
    database_file = os.path.join(data_dir, "sift_database.tsv")
    logger.info(f"Writing database vectors to {database_file}")
    
    with open(database_file, 'w') as f:
        for i, vector in enumerate(database):
            # Convert vector to string format suitable for Doris array column
            vector_str = '[' + ','.join(map(str, vector.astype(int))) + ']'
            f.write(f"{i}\t{vector_str}\n")
    
    logger.info(f"Successfully wrote {len(database)} database vectors to {database_file}")
    
    # Transform query vectors to TSV format
    query_file = os.path.join(data_dir, "sift_queries.tsv")
    logger.info(f"Writing query vectors to {query_file}")
    
    with open(query_file, 'w') as f:
        for i, vector in enumerate(query):
            # Convert vector to string format suitable for Doris array column
            vector_str = '[' + ','.join(map(str, vector.astype(int))) + ']'
            f.write(f"{i}\t{vector_str}\n")
    
    logger.info(f"Successfully wrote {len(query)} query vectors to {query_file}")
    
    # Transform groundtruth to TSV format
    groundtruth_file = os.path.join(data_dir, "sift_groundtruth.tsv")
    logger.info(f"Writing groundtruth to {groundtruth_file}")
    
    with open(groundtruth_file, 'w') as f:
        for i, gt_row in enumerate(groundtruth):
            # Convert groundtruth to string format (top-k nearest neighbor indices)
            gt_str = '[' + ','.join(map(str, gt_row)) + ']'
            f.write(f"{i}\t{gt_str}\n")
    
    logger.info(f"Successfully wrote {len(groundtruth)} groundtruth entries to {groundtruth_file}")
    
    logger.info("SIFT-1M dataset transformation completed successfully!")
    
    return {
        'database_file': database_file,
        'query_file': query_file,
        'groundtruth_file': groundtruth_file,
        'database_shape': database.shape,
        'query_shape': query.shape,
        'groundtruth_shape': groundtruth.shape
    }

if __name__ == "__main__":
    try:
        result = load_sift()
        logger.info("Dataset loading completed successfully!")
        logger.info(f"Results: {result}")
    except Exception as e:
        logger.error(f"Error loading SIFT dataset: {e}")
        raise