import mysql.connector

def get_conn() :
    conn = mysql.connector.connect(
        user="root", password="", host='127.0.0.1', port=6937)
    return conn

conn = get_conn()
cursor = conn.cursor()
cursor.execute("set enable_profile=true")
sql = "select split_by_string('null',',')"
cursor.execute(sql)

profile_ids = profile.get_profile_list_by_range("root", "", begin_uuid, end_uuid, suits_name)


"build  merged  simple  profile  failed"