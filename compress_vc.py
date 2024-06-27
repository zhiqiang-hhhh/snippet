import mysql.connector

connection = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = connection.cursor()


cursor.execute("use demo;")
cursor.execute("set dry_run_query=true;")
cursor.execute("set enable_profile=true;")
cursor.execute("select * from compress_2 order by vc;")
cursor.fetchall()
cursor.execute("select * from compress_2 order by vc;")
cursor.fetchall()
cursor.execute("select * from compress_2 order by vc;")
cursor.fetchall()
cursor.execute("select * from compress_2 order by compress_varchar(vc, 7);")
cursor.fetchall()
cursor.execute("select * from compress_2 order by compress_varchar(vc, 7);")
cursor.fetchall()
cursor.execute("select * from compress_2 order by compress_varchar(vc, 7);")
cursor.fetchall()
