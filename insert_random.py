#!/usr/bin/env python

import datetime
import random
import mysql.connector
import random
import string

def random_string(length):
    letters = string.ascii_letters + string.digits
    return ''.join(random.choice(letters) for _ in range(length))

# conn = mysql.connector.connect(user="default", password="", host='127.0.0.1', port=9999)
# conn = mysql.connector.connect(user="root", password="123", port=3307)
doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = doris_conn.cursor()
# cursor = conn.cursor()

database_name = "demo"
# database_name = "demo"
# cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database_name};")
# cursor.execute(f"USE {database_name};")

# table_name1 = "T_DC1713778622201"
table_name2 = "T_DC1713779161996"
table_name1 = "perf_like"
# drop_table_query = f"DROP TABLE IF EXISTS {database_name}.{table_name};"
# cursor.execute(drop_table_query)

# create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} (col_0 INT,"
# for i in range(0, 1000):
#     create_table_query += f"col_{i+1} DOUBLE"
#     if i != 1000:
#         create_table_query += ","
# create_table_query += ") ENGINE=OLAP DUPLICATE KEY(`col_0`) DISTRIBUTED BY HASH(`col_0`) BUCKETS 1 "
# create_table_query += "PROPERTIES(\"replication_allocation\" = \"tag.location.default: 1\");"
# # create_table_query += ") ENGINE=MergeTree ORDER BY col_0"
# cursor.execute(create_table_query)

# for clickhouse
# for r in range(1000):
#     insert_query = f"INSERT INTO {database_name}.{table_name1} VALUES ("
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',"
#     insert_query += "\'2020-02-03\',"
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',"
#     insert_query += "\'2020-02-04\', \'2020-02-05\', \'2020-02-06\',"
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',"
#     insert_query += "\'2020-02-07\',"
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\'"
#     insert_query += ")"
#     cursor.execute(insert_query)
# print("Inserted table 1, 1000 rows, ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

for r in range(10000000):
    insert_query = f"INSERT INTO {database_name}.{table_name1} VALUES ("
    insert_query += f"{r},"
    insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\""
    insert_query += ")"
    cursor.execute(insert_query)
conn.commit()
print("Inserted table 1, 10000000 rows, ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# for r in range(2000):
#     insert_query = f"INSERT INTO {database_name}.{table_name1} VALUES ("
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\","
#     insert_query += "\"2020-02-03\","
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\","
#     insert_query += "\"2020-02-04\", \"2020-02-05\", \"2020-02-06\","
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\","
#     insert_query += "\"2020-02-07\","
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\""
#     insert_query += ")"
#     cursor.execute(insert_query)
# conn.commit()
# print("Inserted table 1, 1000 rows, ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# for r in range(1000):
#     insert_query = f"INSERT INTO {database_name}.{table_name2} VALUES ("
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\',"
#     insert_query += f"{r},"
#     insert_query += "\'2020-02-08\',"
#     insert_query += f"\'{random_string(5)}\',\'{random_string(5)}\',\'{random_string(5)}\'"
#     insert_query += ")"
#     cursor.execute(insert_query)
# conn.commit()
# print("Inserted table 2, 1000 rows, ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))





# for r in range(500):
#     insert_query = f"INSERT INTO {database_name}.{table_name2} VALUES ("
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\","
#     insert_query += f"{r},"
#     insert_query += "\"2020-02-08\","
#     insert_query += f"\"{random_string(5)}\",\"{random_string(5)}\",\"{random_string(5)}\""
#     insert_query += ")"
#     cursor.execute(insert_query)
# print("Inserted table 2, 500 rows, ", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# query = f"SELECT count(*) FROM {database_name}.{table_name};"
# cursor.execute(query)

# while True:
#     result = cursor.fetchmany()
#     if not result:
#         break

#     for row in result:
#         print(row)

cursor.close()
# conn.commit()
# conn.close()