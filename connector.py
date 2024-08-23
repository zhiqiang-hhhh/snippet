import airflow.operators
import mysql.connector
import airflow

# 创建数据库连接
conn = mysql.connector.connect(
    host="127.0.0.1",      # 数据库主机地址
    user="root",  # 数据库用户名
    password="",  # 数据库密码
    database="demo"  # 目标数据库
)

# 创建游标对象
cursor = conn.cursor()

# 读取 SQL 文件
sql_file = "/mnt/disk1/hezhiqiang/t.sql"

with open(sql_file, 'r') as file:
    sql_script = file.read()

# 分割 SQL 文件中的语句
sql_commands = sql_script.split(';')

# 执行 SQL 文件中的每个语句
for command in sql_commands:
    # 如果命令不为空
    if command.strip():
        try:
            cursor.execute(command)
            conn.commit()  # 提交更改
        except mysql.connector.Error as err:
            print(f"Error: {err}")
            conn.rollback()  # 回滚更改

# 关闭游标和连接
cursor.close()
conn.close()

print("SQL script executed successfully.")
