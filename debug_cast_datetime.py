import mysql.connector

def get_conn() :
    conn = mysql.connector.connect(
        user="root", password="", host='127.0.0.1', port=6937)
    return conn


conn = get_conn()
cursor = conn.cursor()

sql = 'use demo;'
cursor.execute(sql)
sql = 'drop table if exists t1;'
cursor.execute(sql)
sql = ' create table t1(a int, b datetime) properties(\'replication_num\'= \'1\');'
cursor.execute(sql)
sql = 'insert into t1 values (1, \'2003-04-22 03:16:42\')'
cursor.execute(sql)
# sql = 'select cast(date_ceil(b, interval 0 day) as datetime) from t1;'
sql = 'select b, date_ceil(b, interval 0 day), cast(date_ceil(b, interval 0 day) as datetime) from t1;'
# cursor.execute("set parallel_pipeline_task_num=2")
idx = 1
while True:
    cursor.execute(sql)
    res = cursor.fetchall()
    if (res[0][2] != None):
        print(f"idx{idx}\n{res}")
        break
    idx += 1
