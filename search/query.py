import mysql.connector
import logging
import os
import random
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def get_conn() -> mysql.connector.connection.MySQLConnection:
    """Establish a connection to the MySQL database and return the connection object"""
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

def main() -> None:
    tables = [
        "dim_4_num_1000",
    ]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, 'data')
    conn = get_conn()
    cursor = conn.cursor()
    
    iterations = 1
    for i in range(iterations):
        for table in tables:
            # 随机选择一行作为查询向量
            query_vector = None
            with open(f"{data_dir}/{table}.tsv", "r") as f:
                # 读取所有行
                lines = f.readlines()
                if not lines:
                    logger.error(f"Empty file: {data_dir}/{table}.tsv")
                    continue
                    
                # 随机选择一行
                random_line = random.choice(lines)
                query_vector = random_line.strip().split("\t")[1]

            cursor.execute(f"USE vector_test;") 
            # Step1 do with-in range search
            range_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector})
                FROM {table} WHERE l2_distance(embedding, {query_vector}) < 10 limit 5;"""
            logger.info(f"Executing range search SQL:\n{range_search_sql}")
            cursor.execute(range_search_sql)
            range_results = cursor.fetchall()
            logger.info(f"Range search results: {range_results}")
            # Step2 do with-out range search
            range_search_sql_2 = f"""SELECT id, l2_distance(embedding, {query_vector})
                FROM {table} WHERE l2_distance(embedding, {query_vector}) > 10 limit 5;"""
            # logger.info(f"Executing range search SQL:\n{range_search_sql_2}")
            cursor.execute(range_search_sql_2)
            range_results_2 = cursor.fetchall()
            logger.info(f"Range search results: {range_results_2}")
            
            
            # Step2 do topn search
            topn_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector})
                FROM {table} ORDER BY l2_distance(embedding, {query_vector}) limit 5;"""
            # logger.info(f"Executing topn search SQL:\n{topn_search_sql}")
            cursor.execute(topn_search_sql)
            topn_results = cursor.fetchall()
            logger.info(f"TopN search results: {topn_results}")
            
            
            # TODO: order by desc

            # Step3 do compound search
            compound_search_sql = f"""SELECT id, l2_distance(embedding, {query_vector})
                FROM {table} WHERE l2_distance(embedding, {query_vector}) < 4.0
                ORDER BY l2_distance(embedding, {query_vector}) limit 5;"""
            # logger.info(f"Executing compound search SQL:\n{compound_search_sql}")
            cursor.execute(compound_search_sql)
            compound_results = cursor.fetchall()
            logger.info(f"Compound search results: {compound_results}")

            # Step3 do compound search2
            compound_search_sql_2 = f"""SELECT id, l2_distance(embedding, {query_vector})
                FROM {table} WHERE l2_distance(embedding, {query_vector}) > 5.0
                ORDER BY l2_distance(embedding, {query_vector}) limit 5;"""
            # logger.info(f"Executing compound search SQL:\n{compound_search_sql_2}")
            cursor.execute(compound_search_sql_2)
            compound_results_2 = cursor.fetchall()
            logger.info(f"Compound search results: {compound_results_2}")

if __name__ == "__main__":
    main()