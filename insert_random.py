#!/usr/bin/env python

import random
import mysql.connector
from prettytable import PrettyTable
from faker import Faker

def execute_ddl(connection, ddl):
    cursor = connection.cursor()
    cursor.execute(ddl)
    connection.commit()



def get_table_structure(cursor, table_name):
    cursor.execute(f"DESCRIBE {table_name}")
    columns = cursor.fetchall()
    
    table = PrettyTable(['Field', 'Type', 'Null', 'Key', 'Default', 'Extra'])
    for column in columns:
        table.add_row([column[0], column[1], column[2], column[3], column[4], column[5]])
    
    print(table)
    return columns

def generate_random_data(column, faker):
    col_type = column[1]
    col_name = column[0]
    if 'INT' in col_type:
        return random.randint(0, 1000)
    elif 'VARCHAR' in col_type or 'TEXT' in col_type:
        if col_name == 'KPI_CODE' or col_name == 'KPI_ID' or col_name == 'FLIGHT_DATE':
            return f"""{random.choice(['A', 'B', 'C'])}"""
        else:
            return faker.text(max_nb_chars=int(col_type.split('(')[1][:-1]))
    elif 'DATE' in col_type:
        return faker.date()
    elif 'DATETIME' in col_type:
        return faker.date_time()
    elif 'FLOAT' in col_type or 'DOUBLE' in col_type:
        return random.uniform(0, 1000)
    else:
        return None

def insert_random_data(connection, table_name, columns, faker, num_rows=100):
    cursor = connection.cursor()
    for _ in range(num_rows):
        row_data = [generate_random_data(col, faker) for col in columns]
        placeholders = ', '.join(['%s'] * len(row_data))
        columns_names = ', '.join([col[0] for col in columns])
        sql = f"INSERT INTO {table_name} ({columns_names}) VALUES ({placeholders})"
        cursor.execute(sql, row_data)
    connection.commit()
    
def generate_ddl(table_name, col_name, col_type):
    columns_def = ", ".join([f"{name} {type}" for name, type in zip(col_name, col_type)])
    ddl = f"""CREATE TABLE IF NOT EXISTS {table_name} ({columns_def}) UNIQUE KEY(`KPI_CODE`)
              distributed by hash({col_name[0]}) properties("replication_num"="1")"""
    return ddl

def create_table_and_insert_random_data(table_name, col_name, col_type, ddl="", num_rows=100):
    if not ddl:
        ddl = generate_ddl(table_name, col_name, col_type)

    connection = mysql.connector.connect(user="root", password="", host='10.16.10.8', port=6937)
    cursor = connection.cursor()
    faker = Faker()
    try:
        execute_ddl(connection, "USE demo;")
        execute_ddl(connection, f"DROP TABLE IF EXISTS {table_name};")
        # Execute the DDL statement
        execute_ddl(connection, ddl)
        # Get the table structure
        columns = get_table_structure(cursor, table_name)

        # Insert random data
        insert_random_data(connection, table_name, columns, faker, num_rows)

        print(f"Successfully inserted random data into {table_name}")
    finally:
        cursor.close()
        connection.close()
    
def main():
    table_name = "DIM_KPI_CODE"
    ddl1 = """
        CREATE TABLE `DIM_KPI_CODE` (
        `KPI_CODE` varchar(10) NULL COMMENT '指标系列代码',
        `KPI_PART` varchar(10) NULL COMMENT '所属板块',
        `KPI_PART_NAME` varchar(40) NULL COMMENT '所属板块名称',
        `KPI_LVL` varchar(10) NULL COMMENT '指标层级',
        `KPI_FORMAT` varchar(20) NULL COMMENT '展示格式',
        `KPI_NAME` varchar(80) NULL COMMENT '指标名称',
        `KPI_LVL_DESC` varchar(40) NULL COMMENT '指标层级',
        `KPI_EXPLAN` varchar(2000) NULL COMMENT '指标解释'
        ) ENGINE=OLAP
        UNIQUE KEY(`KPI_CODE`)
        COMMENT '分析指标基础表'
        DISTRIBUTED BY HASH(`KPI_CODE`) BUCKETS 10
        PROPERTIES (
        "replication_allocation" = "tag.location.default: 1",
        "is_being_synced" = "false",
        "storage_format" = "V2",
        "disable_auto_compaction" = "false",
        "enable_single_replica_compaction" = "false");
    """
    create_table_and_insert_random_data(table_name, [], [], ddl1, 100)
    table_name = "DWA_CHECK_KPI"
    ddl2 = """
                CREATE TABLE `DWA_CHECK_KPI` (
        `FLIGHT_DATE` date NULL,
        `KPI_ID` varchar(30) NULL COMMENT '指标ID',
        `P_OR_C` varchar(10) NULL COMMENT '客货标识',
        `D_OR_I` varchar(10) NULL COMMENT '国际国内',
        `ROUTE_AREA` varchar(10) NULL COMMENT '航线区域',
        `AIRCRAFT_BODY` varchar(10) NULL COMMENT '宽体窄体',
        `KPI_LVL` varchar(10) NULL COMMENT '指标层级',
        `KPI_FORMAT` varchar(20) NULL COMMENT '展示格式',
        `KPI_CODE` varchar(10) NULL COMMENT '指标系列代码',
        `KPI_NAME` varchar(80) NULL COMMENT '指标名称',
        `KPI_T` decimal(18, 8) NULL COMMENT '指标分子值',
        `KPI_B` decimal(18, 8) NULL COMMENT '指标分母值',
        `KPI_ID_T` varchar(30) NULL COMMENT '分子指标ID',
        `KPI_ID_B` varchar(30) NULL COMMENT '分母指标ID',
        `KPI_T_LM` decimal(18, 8) NULL COMMENT '上月同期指标分子值',
        `KPI_B_LM` decimal(18, 8) NULL COMMENT '上月同期指标分子值',
        `KPI_T_LY` decimal(18, 8) NULL COMMENT '去年同期指标分子值',
        `KPI_B_LY` decimal(18, 8) NULL COMMENT '去年同期指标分子值'
        ) ENGINE=OLAP
        UNIQUE KEY(`FLIGHT_DATE`, `KPI_ID`)
        COMMENT '分析指标基础表'
        DISTRIBUTED BY HASH(`FLIGHT_DATE`) BUCKETS 10
        PROPERTIES (
        "replication_allocation" = "tag.location.default: 1",
        "is_being_synced" = "false",
        "storage_format" = "V2",
        "disable_auto_compaction" = "false",
        "enable_single_replica_compaction" = "false"
        );
    """
    create_table_and_insert_random_data(table_name, [], [], ddl2, 100)
    table_name = "DIM_KPI_CUBE"
    ddl3 = """
    CREATE TABLE `DIM_KPI_CUBE` (
  `KPI_ID` varchar(30) NULL COMMENT '指标ID',
  `KPI_CODE` varchar(10) NULL COMMENT '指标系列代码',
  `KPI_LVL` varchar(10) NULL COMMENT '指标层级',
  `KPI_FORMAT` varchar(20) NULL COMMENT '展示格式',
  `KPI_NAME` varchar(80) NULL COMMENT '指标名称',
  `P_OR_C` varchar(10) NULL COMMENT '客货标识',
  `D_OR_I` varchar(10) NULL COMMENT '国际国内',
  `ROUTE_AREA` varchar(10) NULL COMMENT '航线区域',
  `AIRCRAFT_BODY` varchar(10) NULL COMMENT '宽体窄体',
  `D_OR_I_DESC` varchar(40) NULL COMMENT '国际国内',
  `P_OR_C_DESC` varchar(40) NULL COMMENT '客货标识',
  `ROUTE_AREA_DESC` varchar(40) NULL COMMENT '航线区域',
  `AIRCRAFT_BODY_DESC` varchar(40) NULL COMMENT '宽体窄体',
  `KPI_LVL_DESC` varchar(40) NULL COMMENT '指标层级',
  `KPI_DEGREE` varchar(80) NULL COMMENT '指标维度',
  `KPI_FULL_NAME` varchar(256) NULL,
  `SERIES_1` varchar(10) NULL COMMENT '国际化2.0指标'
) ENGINE=OLAP
UNIQUE KEY(`KPI_ID`)
COMMENT '分析指标基础表'
DISTRIBUTED BY HASH(`KPI_ID`) BUCKETS 10
PROPERTIES (
"replication_allocation" = "tag.location.default: 1",
"is_being_synced" = "false",
"storage_format" = "V2",
"disable_auto_compaction" = "false",
"enable_single_replica_compaction" = "false"
);
    """
    create_table_and_insert_random_data(table_name, [], [], ddl3, 100)

if __name__ == "__main__":
    main()