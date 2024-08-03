import mysql.connector

def get_conn() :
    conn = mysql.connector.connect(
        user="root", password="", host='127.0.0.1', port=6937)
    return conn

sql = 'SELECT histogram(k7, 5) FROM agg_func_db.baseall'
# sql = 'select histogram(k5, 5) from agg_func_db.baseall where k5 is null or k5 = 243.325'
conn = get_conn()
cursor = conn.cursor()

# cursor.execute("set parallel_pipeline_task_num=2")
idx = 1
while True:
    cursor.execute(sql)
    res = cursor.fetchall()
    if "\"num_buckets\":5" not in res[0][0]:
    # if """"num_buckets":1""" not in res[0][0]:
        print(f"idx{idx}\n{res}")
        break
    idx += 1
