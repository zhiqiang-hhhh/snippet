import sys
import time
import mysql.connector

import random

def random_float(seed, lower_bound, upper_bound):
    """
    Generate a random floating-point number within the specified range.

    Args:
        seed (int): The seed value for random number generation.
        lower_bound (float): The lower bound of the range (inclusive).
        upper_bound (float): The upper bound of the range (inclusive).

    Returns:
        float: A random floating-point number within the specified range [lower_bound, upper_bound].
    """
    random.seed(seed)  # 设置随机数种子
    return random.uniform(lower_bound, upper_bound)  # 返回指定范围内的随机浮点数
    

doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = doris_conn.cursor()

db = "demo"
tbl = "float_convert"

ddl = f"CREATE TABLE IF NOT EXISTS {db}.{tbl}(rowid int"
for r in range(500):
    ddl += f", f{r} double"
for r in range(500):
    ddl += f", d{r} double"
ddl += ") DISTRIBUTED BY HASH (rowid) PROPERTIES(\"replication_num\"=\"1\");"
cursor.execute(ddl)

do_insert = True

if do_insert:
    row_num = 5000
    start_time = time.time()

    for r in range(row_num):
        dml = f"INSERT INTO {db}.{tbl} VALUES ({r}"
        for c in range(500):
            f = random_float(r, sys.float_info.min, sys.float_info.max)
            d = random_float(r, sys.float_info.min, sys.float_info.max)
            dml += f",{f},{d}"
        dml += ");"
        cursor.execute(dml)
        
    end_time = time.time()
    execution_time = end_time - start_time

    print(f"INSERT INTO VALUES COSTS: {execution_time:.2f}")

cursor.execute("SET enable_profile=true;")

for _ in range(3):
    cursor.execute("SELECT /*+SET_VAR(faster_float_convert=true)*/ * FROM demo.float_convert;")
    cursor.fetchall()

for _ in range(3):
    cursor.execute("SELECT /*+SET_VAR(faster_float_convert=false)*/ * FROM demo.float_convert;")
    cursor.fetchall()
