#!/usr/bin/env python
import mysql.connector
# import pymysql
import datetime

# doris_conn = pymysql.connect(user="root", password="", host='10.16.10.8', port=6937)
doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = doris_conn.cursor()

t1 = datetime.datetime.now()
cursor.execute("select * from ODS.test_table_double")
t2 = datetime.datetime.now()
print("Without faster float convert execute costs: ", (t2 - t1).total_seconds())

t1 = datetime.datetime.now()
cursor.fetchall()
t2 = datetime.datetime.now()
print("Without faster float convert fetch all costs: ", (t2 - t1).total_seconds())

t3 = datetime.datetime.now()
cursor.execute("select /*+SET_VAR(faster_float_convert=true)*/ * from ODS.test_table_double")
t4 = datetime.datetime.now()
print("With faster float convert costs: ", (t4 - t3).total_seconds())

t3 = datetime.datetime.now()
cursor.fetchall()
t4 = datetime.datetime.now()
print("With faster float convert fetch all costs: ", (t4 - t3).total_seconds())


# df = DataFrame(results)
# print("to pandas cost:" + str(datetime.now() - currentTime))
# print("pandas.size: ", df.info(memory_usage='deep'))