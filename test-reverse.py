#!/usr/bin/env python
import mysql.connector

doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = doris_conn.cursor()

database_name = "test-reverse"
cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
cursor.execute(f"USE {database_name};")

table_name = "test_table_reverse"
# drop_table_query = f"DROP TABLE IF EXISTS {database_name}.{table_name};"
# cursor.execute(drop_table_query)

create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (col_0 INT,"
for i in range(0, 1000):
    create_table_query += f"col_{i+1} DOUBLE"
    if i != 1000:
        create_table_query += ","
create_table_query += ") ENGINE=OLAP DUPLICATE KEY(`col_0`) DISTRIBUTED BY HASH(`col_0`) BUCKETS 1 "
create_table_query += "PROPERTIES(\"replication_allocation\" = \"tag.location.default: 1\");"
# create_table_query += ") ENGINE=MergeTree ORDER BY col_0"
cursor.execute(create_table_query)
