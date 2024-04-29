#!/usr/bin/env python
import time
import mysql.connector

doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
# ch_conn = mysql.connector.connect(user="default", password="", host='10.16.10.8', port=9999)

cursor = doris_conn.cursor()
database_name = "demo"
cursor.execute(f"USE {database_name};")
table_name = "bigstring_tbl"

ddl = "create table if not exists {} (id int, str string) engine=olap distributed by hash(id) properties(\"replication_num\"=\"1\")".format(table_name)
cursor.execute(ddl)

baseSize = 1000 * 1000 # 1MB

for i in range(1):
    dml = "insert into {} values ({}, \"{}\")".format(table_name, i, "a"*baseSize*50) # 50MB
    cursor.execute(dml)

# for i in range(110):
#     if i <= 95:
#         continue
#     else:
#         dml = "insert into {} values ({}, \"{}\")".format(table_name, i, "a"*baseSize*i)
#         cursor.execute(dml)


# query = "select sum(length(str)) from {}".format(table_name)
# cursor.execute(query)
# res = cursor.fetchall()
# print("total size: {}".format(res[0][0])) # 25 * 10 = 250MB

# query = "select id, length(str) from {} order by id".format(table_name)
# cursor.execute(query)
# res = cursor.fetchall()
# for r in res:
#     print("row id: {} size: {}".format(r[0], r[1]))

# query = "select /*+SET_VAR(max_msg_size_of_result_receiver=204857600)*/ id, str from {} where id=105 order by id".format(table_name)
# cursor.execute(query)
# res = cursor.fetchall()
# for r in res:
#     print("in python row id: {} size: {}".format(r[0], len(r[1])))

# query = "select id, str from {} where id=105 order by id".format(table_name)
# cursor.execute(query)
# res = cursor.fetchall()
# for r in res:
#     print("in python row id: {} size: {}".format(r[0], len(r[1])))


# for i in range(96, 110):    
#     query = "select id, str from {} where id={}".format(table_name, i)
#     cursor.execute(query)
#     res = cursor.fetchall()
#     for r in res:
#         print("in python row id: {} size: {}".format(r[0], len(r[1])))
