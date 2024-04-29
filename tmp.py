#!/usr/bin/env python

import mysql.connector

doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)

cursor = doris_conn.cursor()

cursor.execute("select (CAST(1.0 AS DOUBLE));")

while True:
    result = cursor.fetchmany()
    if not result:
        break

    for row in result:
        print(row)

cursor.execute("set global faster_float_convert=true")
cursor.execute("select (CAST(1 AS DOUBLE));")

while True:
    result = cursor.fetchmany()
    if not result:
        break

    for row in result:
        print(row)

cursor.close()
doris_conn.commit()
doris_conn.close()