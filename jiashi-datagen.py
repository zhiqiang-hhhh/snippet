#!/usr/bin/env python
import mysql.connector
# import pymysql
import datetime

# doris_conn = pymysql.connect(user="root", password="", host='10.16.10.8', port=6937)
doris_conn = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
cursor = doris_conn.cursor()
db_name = "dmec"
# ds_cust_profile
# ds_ast_ret_tran_cust_prod_agent
table_name = "ds_cust_profile" 
cursor.execute("desc {}.{}".format(db_name, table_name))
schema_info = cursor.fetchall()
print(schema_info)

typename_to_typeid = {
    "DATE" : 0,
    "INTEGER" : 1,
    "DECIMAL" : 2,
    "CHARACTER" : 3
}

def generator_date() :
    return "\"2023-10-15\""

def generator_integer() :
    return "uuid_numeric()"

def generator_decimal() :
    return "10.0"

def generator_character() :
    return "\"AAA\""

typeid_to_generator = {
    0 : generator_date,
    1 : generator_integer,
    2 : generator_decimal,
    3 : generator_character,
}

col_types = []

for col_info in schema_info:
    col_type = col_info[1]
    if col_type == "DATE" or col_type == "DATETIME":
        col_types.append("DATE")
        continue
    if col_type.startswith("VARCHAR") or col_type.startswith("STRING") or col_type.startswith("CHAR"):
        col_types.append("CHARACTER")
        continue
    if col_type.startswith("DECIMAL"):
        col_types.append("DECIMAL")
        continue
    if col_type.startswith("INT"):
        col_types.append("INTEGER")
        continue
    print("Unknown type " + col_type)

values = ""
for col_type in col_types:
    values = values + typeid_to_generator[typename_to_typeid[col_type]]() + ','
values = values[:-1]

print(values)

cursor.execute("insert into {}.{} values ({})".format(db_name, table_name, values))

