import os
import sqlite3
import logging
from tabulate import tabulate

def init_database():
    if os.path.exists('datail.db'):
            os.remove('datail.db')

    conn = sqlite3.connect('datail.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS res_table " +
                   "(query_idx TEXT, query_id TEXT, avg_rows_read DOUBLE, avg_block_mb DOUBLE, avg_sered_mb DOUBLE, avg_ser_ms DOUBLE, avg_deser_ms DOUBLE, avg_ser_deser_sum DOUBLE)")

    column_names = [
        'Query ID TEXT', 'FE节点 TEXT', '查询类型 TEXT', '开始时间 DATETIME', '结束时间 DATETIME', '执行时长 TEXT', '状态 TEXT',
        '查询用户 TEXT', 'Catalog TEXT', '执行数据库 TEXT', 'Sql TEXT']
    
    # Create a table with the extracted column names
    cursor.execute(f"CREATE TABLE query_info ({', '.join(column_names)})")
    cursor.fetchall()
    cursor.close()
    conn.commit()

def pretty_print_results(query):
    conn = sqlite3.connect('datail.db')
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    conn.close()
    headers = [description[0] for description in cursor.description]
    print(tabulate(result, headers=headers, tablefmt='pretty'))
    return result
