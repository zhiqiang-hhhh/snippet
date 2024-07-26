import sys
import os
import traceback
import uuid
import mysql.connector
from mysql.connector import Error
import myprofile as profile
import logging
import database

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s][%(filename)s:%(lineno)d] - %(message)s')


# List of valid suite names
VALID_SUITES = ['clickbench', 'tpch']

def validate_arguments(args):
    if len(args) != 2:
        logging.error("Invalid number of arguments.")
        raise ValueError("Usage: python multi-process-poc.py [clickbench|tpch]")
    
    suits_name = args[1]
    
    if suits_name not in VALID_SUITES:
        logging.error(f"Invalid suite name: {suits_name}")
        raise ValueError(f"Invalid suite name: {suits_name}. Valid suite names are: {', '.join(VALID_SUITES)}")
    
    logging.info(f"Validated arguments. Suite name: {suits_name}")
    return suits_name

def generate_uuid():
    return str(uuid.uuid4())

def get_queries_path(suits_name, modified=False):
    sub_dir = 'modified' if modified else 'origin'
    queries_path = os.path.join('sql', suits_name, sub_dir)
    logging.info(f"Queries path: {queries_path}")
    return queries_path

def read_query_file(query_file_path):
    with open(query_file_path, 'r') as file:
        queries = file.read()
    # logging.info(f"Read queries from file: {query_file_path}, content: {queries}")
    return queries

def execute_query_file(queries, connection):
    cursor = connection.cursor()
    # for query in queries.striplines():
    #     query = query.strip()
    #     if query and query.startswith('SELECT') or query.startswith('USE') or query.startswith('SET'):
    cursor.execute(queries)
    while cursor.nextset():
        # Fetch all results to avoid 'Unread result found' error
        cursor.fetchall()
    connection.commit()

def connect_to_mysql():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',
            user='root',
            port=6937,
            password=''
        )
        if connection.is_connected():
            logging.info("Connected to MySQL database")
            return connection
    except Error as e:
        logging.error(f"Error connecting to MySQL: {e}")
        raise ConnectionError(f"Error connecting to MySQL: {e}")

def main():
    try:
        database.init_database()
        suits_name = validate_arguments(sys.argv)
        query_file_dir = get_queries_path(suits_name, modified=True)
        
        # list all files in the directory
        query_files_list = os.listdir(query_file_dir)
        logging.info(f"List of files in the directory: {query_files_list}")
        
        connection = connect_to_mysql()
        cursor = connection.cursor()
        cursor.execute("SET enable_profile=true;")
        begin_uuid = generate_uuid()
        end_uuid = generate_uuid()
        logging.info("Begin UUID: %s, end UUID: %s", begin_uuid, end_uuid)
        cursor.execute("USE demo")
        cursor.execute(f"SELECT '{begin_uuid}'")  # Generate a unique identifier
        cursor.fetchall()
        
        if (suits_name == 'clickbench'):
            cursor.execute("USE clickbench")
        elif (suits_name == 'tpch'):
            cursor.execute("USE tpch")
        for file in query_files_list:
            abs_path = os.path.join(query_file_dir, file)
            logging.info(abs_path)
            query_content = read_query_file(abs_path)
            execute_query_file(query_content, connection)
            cursor.fetchall()

        cursor.execute(f"USE demo")
        cursor.execute(f"SELECT '{end_uuid}'")  # Generate a unique identifier
        cursor.fetchall()
        
        profile_ids = profile.get_profile_list_by_range("root", "", begin_uuid, end_uuid)

        query_idx = 1
        for profile_id in profile_ids:
            profile_content = profile.get_profile_content("root", "", profile_id)
            profile.ayalyze_profile(query_idx, profile_content['data']['profile'])
            query_idx += 1

        database.pretty_print_results("SELECT * FROM res_table ORDER BY CAST(REPLACE(query_idx, 'q', '') AS INTEGER)")
    except Exception as e:
        logging.error(f"An error occurred: {traceback.format_exc()}")
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            logging.info("MySQL connection closed")

if __name__ == '__main__':
    main()
