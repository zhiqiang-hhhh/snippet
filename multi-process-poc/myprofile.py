# api_client.py
import re
import time
import requests
from requests.auth import HTTPBasicAuth
import sqlite3
import logging
import database

def fetch_query_info(username, password):
    url = "http://127.0.0.1:5937/rest/v2/manager/query/query_info"  # Replace with the actual URL
    try:
        response = requests.get(url, auth=HTTPBasicAuth(username, password))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        query_info = response.json()  # Parse the JSON response
        return query_info
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    return None

def get_profile_content(username, password, profile_id):
    url = f"http://127.0.0.1:5937/rest/v2/manager/query/profile/text/{profile_id}"
    try:
        time.sleep(2)
        response = requests.get(url, auth=HTTPBasicAuth(username, password))
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx and 5xx)
        # return response.text()
        profile_content = response.json()  # Parse the JSON response
        return profile_content
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except requests.exceptions.ConnectionError as conn_err:
        print(f"Connection error occurred: {conn_err}")
    except requests.exceptions.Timeout as timeout_err:
        print(f"Timeout error occurred: {timeout_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"An error occurred: {req_err}")
    return None

def store_data_in_sqlite(data):
    # Connect to an in-memory SQLite database
    conn = sqlite3.connect('datail.db')
    cursor = conn.cursor()
    
    # Extract column names and rows
    # column_names = data['columnNames']
    rows = data['rows']
    logging.info(f"Query info sample {rows[0]}")
    for row in rows:
        row = row[:-2]  # Remove the last two element
        placeholders = ', '.join('?' * len(row))
        cursor.execute(f"INSERT INTO query_info VALUES ({placeholders})", row)
    
    conn.commit()
    logging.info("Data stored in in-memory database successfully.")
    return conn

def get_profile_list_by_range(username, password, begin_uuid, end_uuid):
    query_info = fetch_query_info(username, password)
    if query_info and 'data' not in query_info:
        # Throw an exception
        raise Exception("Data not found in query_info")
        
    data = query_info['data']
    conn = store_data_in_sqlite(data)
    cursor = conn.cursor()
    cursor.execute(
        f"SELECT Query ID, 开始时间, Sql FROM query_info where Sql like '%{begin_uuid}%' or Sql like '%{end_uuid}%' order by 开始时间")
    res = cursor.fetchall()
    begin_query_id, begin_time = res[0][0], res[0][1]
    end_query_id, end_time = res[1][0], res[1][1]
    logging.info(f"begin_query_uuid\t{begin_query_id}\tbegin_time:\t{begin_time}")
    logging.info(f"end_query_uuid  \t{end_query_id}\tend_time:    \t{end_time}")
    res = database.pretty_print_results(f"SELECT Query ID, 开始时间, 执行数据库 FROM query_info where 开始时间 >= '{begin_time}' AND 开始时间 <= '{end_time}' AND (执行数据库 = 'clickbench' OR 执行数据库 = 'tpch') order by 开始时间")
    logging.info(f"Profile list fetched successfully. Count {len(res)}")
    final_res = []
    for row in res:
        final_res.append(row[0])
    return final_res

def parse_to_m_bytes(bytes_str):
    pattern = re.compile(r'(\d+(\.\d+)?)\s*(\w+)')
    match = pattern.fullmatch(bytes_str)
    if not match:
        raise ValueError(f"Invalid bytes string {bytes_str}")

    value = float(match.group(1))
    unit = match.group(3).upper()

    # 根据单位转换为MB
    if unit == 'B':
        return value / (1024 * 1024)
    elif unit == 'KB':
        return value / 1024
    elif unit == 'MB':
        return value
    elif unit == 'GB':
        return value * 1024
    elif unit == 'TB':
        return value * 1024 * 1024
    else:
        raise ValueError(f"Invalid unit {unit}")

def parse_to_milliseconds(duration_str):
    pattern = re.compile(r'(?:(\d+(?:\.\d+)?)hour)?(?:(\d+(?:\.\d+)?)min)?(?:(\d+(?:\.\d+)?)sec)?(?:(\d+(?:\.\d+)?)ms)?(?:(\d+(?:\.\d+)?)us)?')
    match = pattern.fullmatch(duration_str)
    if not match:
        raise ValueError(f"Invalid duration string {duration_str}")

    hours = float(match.group(1)) if match.group(1) else 0
    minutes = float(match.group(2)) if match.group(2) else 0
    seconds = float(match.group(3)) if match.group(3) else 0
    milliseconds = float(match.group(4)) if match.group(4) else 0
    microseconds = float(match.group(5)) if match.group(5) else 0

    total_milliseconds = (hours * 3600 * 1000 +
                          minutes * 60 * 1000 +
                          seconds * 1000 +
                          milliseconds +
                          microseconds / 1000)
    
    return total_milliseconds

def ayalyze_profile(query_idx, profile_content):
    execution_profile_pattern = re.compile(r'(?=.*Execution)(?=.*Profile)')
    scan_operator_pattern = re.compile(r'OLAP_SCAN_OPERATOR \(id=') 
    vscanner_pattern = re.compile(r'VScanner:')
    # 定义正则表达式以匹配包含 RowsRead 并且有 (数字) 的行
    rowsread_pattern = re.compile(r'- RowsRead:.*\(\d+\)')
    # 定义正则表达式以匹配以 - ScanNodeDeserializationBlockTime: 和数字开头的行
    deser_pattern = re.compile(r'- ScanNodeDeserializationBlockTime: \d+(\.\d+)?')
    ser_pattern = re.compile(r'- ScannerSerializationBlockTime: \d+(\.\d+)?')
    col_count_pattern = re.compile(r'- ColumnCount: \d+(\.\d+)?')
    sered_binary_bytes_pattern = re.compile(r'- SerializedBinarySize: \d+(\.\d+)?')
    block_size_pattern = re.compile(r'- BlockSize: \d+(\.\d+)?')
    total_pattern = re.compile(r'- Total: \d+(\.\d+)?')  
    
    lines = profile_content.split('\n')
    total_time_ms = 0
    query_id = ''
    sql = ''
    scan_operator_offset = []
    scanner_offset = []

    for i, line in enumerate(lines):
        if line.strip().startswith("- Profile ID"):
            query_id = line.split(' ')[-1]
        if line.strip().startswith("- Sql Statement"):
            sql = line.split('Statement: ')[-1]
        if total_pattern.search(line):
            total_time_ms = parse_to_milliseconds(line.split(' ')[-1])
        if execution_profile_pattern.search(line):
            for j, line in enumerate(lines[i:]):
                if scan_operator_pattern.search(line.strip()):
                    scan_operator_offset.append(i+j)
                if vscanner_pattern.search(line):
                    scanner_offset.append(i+j)
            break

    if len(scan_operator_offset) != len(scanner_offset):
        raise ValueError("Scan operator and scanner count mismatch")

    logging.info("Processing profile %s", query_id)
    
    total = len(scan_operator_offset)
    # 确保不越界
    scan_operator_offset.append(len(lines))
    
    analyze_result = []
    for i in range(total):
        segment_b = scan_operator_offset[i]
        segment_e = scan_operator_offset[i+1]

        rows_read = 0
        col_count = 0
        block_mb = 0
        serialized_mb = 0
        ser_time = 0
        deser_time = 0

        valid = True
        for i, line in enumerate(lines[segment_b:segment_e]):
            match_rowsread = rowsread_pattern.search(line)
            if match_rowsread:
                rows_read = int(line.split('(')[1].split(')')[0])
                if (rows_read == 0):
                    valid = False
                    break
                continue
            match_deser = deser_pattern.search(line)
            if match_deser:
                deser_time = parse_to_milliseconds(line.split(' ')[-1])
                continue
            match_ser = ser_pattern.search(line)
            if match_ser:
                ser_time = parse_to_milliseconds(line.split(' ')[-1])
                continue
            match_col_count = col_count_pattern.search(line)
            if match_col_count:
                col_count = int(line.split(' ')[-1])
                continue
            match_sered_binary = sered_binary_bytes_pattern.search(line)
            if match_sered_binary:
                str = line.split(' ')[-2] + line.split(' ')[-1]
                try:
                    serialized_mb = parse_to_m_bytes(str)
                    continue
                except ValueError as e:
                    valid = False
                    break
            match_block_size = block_size_pattern.search(line)
            if match_block_size:
                str = line.split(' ')[-2] + line.split(' ')[-1]
                try:
                    block_mb = parse_to_m_bytes(str)
                    continue
                except ValueError as e:
                    valid = False
                    break
        if valid or rows_read > 0:
            analyze_result.append([query_id, total_time_ms, sql, rows_read, col_count, block_mb, serialized_mb, ser_time, deser_time])        

    # logging.info("ScanOperator in total %d,", len(analyze_result))

    conn = sqlite3.connect('datail.db')
    cursor = conn.cursor()
    cursor.execute(f"CREATE TABLE q{query_idx} (query_id TEXT, total INT, sql_content TEXT, rows_read INT, column_count INT, block_mb INT, serialized_mb INT, serialization_ms DOUBLE, deserialization_ms DOUBLE)")
    # logging.info("Analyze result sample %s", analyze_result[:10])
    for row in analyze_result:
        placeholders = ', '.join('?' * len(row))
        cursor.execute(f"INSERT INTO q{query_idx} VALUES ({placeholders})", row)
        cursor.execute("COMMIT")

    cursor.execute(
        "INSERT INTO res_table " +
        f"SELECT \'q{query_idx}\',\'{query_id}\', avg(TMP.rows_read) as avg_rows_read, avg(TMP.block_mb) as avg_block_mb, avg(TMP.serialized_mb) as avg_sered_mb, " +
        "avg(TMP.avg1) as avg_ser_ms, avg(TMP.avg2) as avg_deser_ms, avg(TMP.avg3) as avg_ser_deser_sum FROM (" +
        "SELECT rows_read, block_mb, serialized_mb," +
        " avg(serialization_ms) as avg1, avg(deserialization_ms) as avg2," +
        " avg(serialization_ms + deserialization_ms) as avg3" +
        f" FROM q{query_idx} WHERE block_mb > 0 AND serialized_mb > 0" +
        " GROUP BY rows_read, block_mb, serialized_mb order by rows_read ) AS TMP")
    cursor.execute("COMMIT")
