import mysql.connector
from tabulate import tabulate

def get_analyze_cluster_conn() :
    conn = mysql.connector.connect(
        user="root", password="", host='127.0.0.1', port=6937)
    return conn

def get_poc_cluster_conn():
    conn = mysql.connector.connect(
        user="root", password="", host='62.234.39.208', port=9030)
    return conn

def init_database():
    conn = get_analyze_cluster_conn()
    cursor = conn.cursor()
    cursor.execute("DROP DATABASE IF EXISTS poc")
    cursor.execute("CREATE DATABASE IF NOT EXISTS poc")
    cursor.execute("USE poc")

    # column_names = [
    #     'idx BIGINT', 'QueryID TEXT', 'FE TEXT', 'Type TEXT', 'Begin DATETIME',
    #     'End DATETIME', 'Total TEXT', 'Status TEXT', 'User TEXT', 'Catalog TEXT', 'DB TEXT', 'Sql TEXT']

    column_names = [
        'idx BIGINT', 'QueryID TEXT', 'FE TEXT', 'Type TEXT', 'Begin DATETIME',
        'End DATETIME', 'Total TEXT', 'Status TEXT', 'User TEXT', 'DB TEXT', 'Sql TEXT']
    
    # Create a table with the extracted column names
    cursor.execute(f"CREATE TABLE query_info ({', '.join(column_names)}) " +
                   "duplicate key (idx)" +
                   " distributed by hash(idx) buckets 3 " +
                   " properties(\"replication_num\" = \"1\");")
    
    cursor.execute("CREATE TABLE query_base (idx INT, sql TEXT, db TEXT) DUPLICATE KEY (idx) PROPERTIES (\"replication_num\"=\"1\")")

    cursor.execute("CREATE TABLE IF NOT EXISTS res_table " +
                   "(query_idx VARCHAR , query_id TEXT, total TEXT, " +
                   "avg_rows_read DOUBLE, avg_block_mb DOUBLE, " +
                   "avg_sered_mb DOUBLE, avg_ser_ms DOUBLE, avg_deser_ms DOUBLE," +
                   " avg_ser_deser_sum DOUBLE) duplicate key (query_idx)" +
                   " distributed by hash(query_idx) buckets 3 " +
                   " properties(\"replication_num\" = \"1\");")
    
    cursor.fetchall()
    cursor.close()
    conn.commit()

def pretty_print_results(query):
    conn = get_analyze_cluster_conn()
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    headers = [description[0] for description in cursor.description]
    print(tabulate(result, headers=headers, tablefmt='pretty'))
    return result

