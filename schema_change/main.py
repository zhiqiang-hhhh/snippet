import logging
import pymysql
import random
from datetime import datetime, timedelta
import time

# 配置日志 - 精简输出
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# 数据库配置
DB_CONFIG = {
    'host': 'localhost',
    'port': 6937,
    'user': 'root',
    'password': '',
    'database': 'demo',
    'charset': 'utf8mb4'
}

create_test_table_sql = """
CREATE TABLE `dm_fila_pcbi_holiday_store_good_retail_online_tmp0` (
  `holiday_dt` date NULL COMMENT "节假日日期",
  `cms_code` varchar(50) NULL COMMENT "cms编码",
  `sku_id` varchar(20) NULL COMMENT "sku_id",
  `holiday_cd` text NULL COMMENT "节假日代码",
  `holiday_nm` text NULL COMMENT "节假日名称",
  `art_no` text NULL COMMENT "货号",
  `size_name` text NULL COMMENT "尺码",
  `dccy` text NULL COMMENT "币种",
  `pub_art_flag` text NULL COMMENT "广宣用品标识",
  `oper_staff_id` text NULL COMMENT "营业员id",
  `oper_staff_name` text NULL COMMENT "营业员姓名",
  `brand_code` text NULL COMMENT "集团品牌编码",
  `brand_name` text NULL COMMENT "集团品牌名称",
  `achi_blng_store_code` text NULL COMMENT "业绩归属门店编码",
  `achi_blng_store_name` text NULL COMMENT "业绩归属门店名称",
  `order_hour` text NULL COMMENT "订单小时段",
  `is_valid_store` text NULL COMMENT "是否有效店",
  `same_store_flag` text NULL COMMENT "同店标识",
  `td_type_cd` text NULL COMMENT "同店类型代码",
  `td_type_desc` text NULL COMMENT "同店类型描述",
  `sale_channel_cd` text NULL COMMENT "销售渠道代码",
  `sale_channel_desc` text NULL COMMENT "销售渠道描述",
  `sale_mode_cd` text NULL COMMENT "销售模式代码",
  `sale_mode_desc` text NULL COMMENT "销售模式描述",
  `settle_amt` decimal(20,4) NULL COMMENT "结算金额",
  `rmb_settle_amt` decimal(20,4) NULL COMMENT "人民币结算金额",
  `good_qty` int NULL COMMENT "商品数量",
  `tag_amt` decimal(20,4) NULL COMMENT "吊牌金额",
  `rmb_tag_amt` decimal(20,4) NULL COMMENT "人民币吊牌金额",
  `no_gift_settle_amt` decimal(20,4) NULL COMMENT "不含赠品结算金额",
  `no_gift_rmb_settle_amt` decimal(20,4) NULL COMMENT "不含赠品人民币结算金额",
  `no_gift_good_qty` int NULL COMMENT "不含赠品商品数量",
  `no_gift_tag_amt` decimal(20,4) NULL COMMENT "不含赠品吊牌金额",
  `no_gift_rmb_tag_amt` decimal(20,4) NULL COMMENT "不含赠品人民币吊牌金额",
  `same_store_settle_amt` decimal(20,4) NULL COMMENT "同店结算金额",
  `same_store_rmb_settle_amt` decimal(20,4) NULL COMMENT "同店人民币结算金额",
  `same_store_good_qty` int NULL COMMENT "同店商品数量",
  `same_store_tag_amt` decimal(20,4) NULL COMMENT "同店吊牌金额",
  `same_store_rmb_tag_amt` decimal(20,4) NULL COMMENT "同店人民币吊牌金额",
  `td_no_gift_settle_amt` decimal(20,4) NULL COMMENT "同店不含赠品结算金额",
  `td_no_gift_rmb_settle_amt` decimal(20,4) NULL COMMENT "同店不含赠品人民币结算金额",
  `td_no_gift_good_qty` int NULL COMMENT "同店不含赠品商品数量",
  `td_no_gift_tag_amt` decimal(20,4) NULL COMMENT "同店不含赠品吊牌金额",
  `td_no_gift_rmb_tag_amt` decimal(20,4) NULL COMMENT "同店不含赠品人民币吊牌金额",
  `tq_settle_amt` decimal(20,4) NULL COMMENT "同期结算金额",
  `tq_rmb_settle_amt` decimal(20,4) NULL COMMENT "同期人民币结算金额",
  `tq_good_qty` int NULL COMMENT "同期商品数量",
  `tq_tag_amt` decimal(20,4) NULL COMMENT "同期吊牌金额",
  `tq_rmb_tag_amt` decimal(20,4) NULL COMMENT "同期人民币吊牌金额",
  `tq_no_gift_settle_amt` decimal(20,4) NULL COMMENT "同期不含赠品结算金额",
  `tq_no_gift_rmb_settle_amt` decimal(20,4) NULL COMMENT "同期不含赠品人民币结算金额",
  `tq_no_gift_good_qty` int NULL COMMENT "同期不含赠品商品数量",
  `tq_no_gift_tag_amt` decimal(20,4) NULL COMMENT "同期不含赠品吊牌金额",
  `tq_no_gift_rmb_tag_amt` decimal(20,4) NULL COMMENT "同期不含赠品人民币吊牌金额",
  `tq_same_store_settle_amt` decimal(20,4) NULL COMMENT "同期同店结算金额",
  `tq_same_store_rmb_settle_amt` decimal(20,4) NULL COMMENT "同期同店人民币结算金额",
  `tq_same_store_good_qty` int NULL COMMENT "同期同店商品数量",
  `tq_same_store_tag_amt` decimal(20,4) NULL COMMENT "同期同店吊牌金额",
  `tq_same_store_rmb_tag_amt` decimal(20,4) NULL COMMENT "同期同店人民币吊牌金额",
  `tq_td_no_gift_settle_amt` decimal(20,4) NULL COMMENT "同期同店不含赠品结算金额",
  `tq_td_no_gift_rmb_settle_amt` decimal(20,4) NULL COMMENT "同期同店不含赠品人民币结算金额",
  `tq_td_no_gift_good_qty` int NULL COMMENT "同期同店不含赠品商品数量",
  `tq_td_no_gift_tag_amt` decimal(20,4) NULL COMMENT "同期同店不含赠品吊牌金额",
  `tq_td_no_gift_rmb_tag_amt` decimal(20,4) NULL COMMENT "同期同店不含赠品人民币吊牌金额",
  `sys_src` text NULL COMMENT "数据源"
) ENGINE=OLAP
DUPLICATE KEY(`holiday_dt`, `cms_code`, `sku_id`)
COMMENT 'DM斐乐PCBI-节假日店铺商品零售'
PARTITION BY RANGE(`holiday_dt`)
(PARTITION p2019 VALUES [('0000-01-01'), ('2020-01-01')),
PARTITION p2020 VALUES [('2020-01-01'), ('2021-01-01')),
PARTITION p202106 VALUES [('2021-01-01'), ('2021-07-01')),
PARTITION p202107 VALUES [('2021-07-01'), ('2021-08-01')),
PARTITION p202108 VALUES [('2021-08-01'), ('2021-09-01')),
PARTITION p202109 VALUES [('2021-09-01'), ('2021-10-01')),
PARTITION p202110 VALUES [('2021-10-01'), ('2021-11-01')),
PARTITION p202111 VALUES [('2021-11-01'), ('2021-12-01')),
PARTITION p202112 VALUES [('2021-12-01'), ('2022-01-01')),
PARTITION p202201 VALUES [('2022-01-01'), ('2022-02-01')),
PARTITION p202202 VALUES [('2022-02-01'), ('2022-03-01')),
PARTITION p202203 VALUES [('2022-03-01'), ('2022-04-01')),
PARTITION p202204 VALUES [('2022-04-01'), ('2022-05-01')),
PARTITION p202205 VALUES [('2022-05-01'), ('2022-06-01')),
PARTITION p202206 VALUES [('2022-06-01'), ('2022-07-01')),
PARTITION p202207 VALUES [('2022-07-01'), ('2022-08-01')),
PARTITION p202208 VALUES [('2022-08-01'), ('2022-09-01')),
PARTITION p202209 VALUES [('2022-09-01'), ('2022-10-01')),
PARTITION p202210 VALUES [('2022-10-01'), ('2022-11-01')),
PARTITION p202211 VALUES [('2022-11-01'), ('2022-12-01')),
PARTITION p202212 VALUES [('2022-12-01'), ('2023-01-01')),
PARTITION p202301 VALUES [('2023-01-01'), ('2023-02-01')),
PARTITION p202302 VALUES [('2023-02-01'), ('2023-03-01')),
PARTITION p202303 VALUES [('2023-03-01'), ('2023-04-01')),
PARTITION p202304 VALUES [('2023-04-01'), ('2023-05-01')),
PARTITION p202305 VALUES [('2023-05-01'), ('2023-06-01')),
PARTITION p202306 VALUES [('2023-06-01'), ('2023-07-01')),
PARTITION p202307 VALUES [('2023-07-01'), ('2023-08-01')),
PARTITION p202308 VALUES [('2023-08-01'), ('2023-09-01')),
PARTITION p202309 VALUES [('2023-09-01'), ('2023-10-01')),
PARTITION p202310 VALUES [('2023-10-01'), ('2023-11-01')),
PARTITION p202311 VALUES [('2023-11-01'), ('2023-12-01')),
PARTITION p202312 VALUES [('2023-12-01'), ('2024-01-01')),
PARTITION p202401 VALUES [('2024-01-01'), ('2024-02-01')),
PARTITION p202402 VALUES [('2024-02-01'), ('2024-03-01')),
PARTITION p202403 VALUES [('2024-03-01'), ('2024-04-01')),
PARTITION p202404 VALUES [('2024-04-01'), ('2024-05-01')),
PARTITION p202405 VALUES [('2024-05-01'), ('2024-06-01')),
PARTITION p202406 VALUES [('2024-06-01'), ('2024-07-01')),
PARTITION p202407 VALUES [('2024-07-01'), ('2024-08-01')),
PARTITION p202408 VALUES [('2024-08-01'), ('2024-09-01')),
PARTITION p202409 VALUES [('2024-09-01'), ('2024-10-01')),
PARTITION p202410 VALUES [('2024-10-01'), ('2024-11-01')),
PARTITION p202411 VALUES [('2024-11-01'), ('2024-12-01')),
PARTITION p202412 VALUES [('2024-12-01'), ('2025-01-01')),
PARTITION p202501 VALUES [('2025-01-01'), ('2025-02-01')),
PARTITION p202502 VALUES [('2025-02-01'), ('2025-03-01')),
PARTITION p202503 VALUES [('2025-03-01'), ('2025-04-01')),
PARTITION p202504 VALUES [('2025-04-01'), ('2025-05-01')),
PARTITION p202505 VALUES [('2025-05-01'), ('2025-06-01')),
PARTITION p202506 VALUES [('2025-06-01'), ('2025-07-01')),
PARTITION p202507 VALUES [('2025-07-01'), ('2025-08-01')),
PARTITION p202508 VALUES [('2025-08-01'), ('2025-09-01')),
PARTITION p202509 VALUES [('2025-09-01'), ('2025-10-01')))
DISTRIBUTED BY HASH(`holiday_dt`, `cms_code`) BUCKETS 4
PROPERTIES (
"replication_num" = "1",
"min_load_replica_num" = "-1",
"is_being_synced" = "false",
"dynamic_partition.enable" = "true",
"dynamic_partition.time_unit" = "MONTH",
"dynamic_partition.time_zone" = "America/New_York",
"dynamic_partition.start" = "-2147483648",
"dynamic_partition.end" = "3",
"dynamic_partition.prefix" = "p",
"dynamic_partition.buckets" = "5",
"dynamic_partition.create_history_partition" = "false",
"dynamic_partition.history_partition_num" = "-1",
"dynamic_partition.hot_partition_num" = "0",
"dynamic_partition.reserved_history_periods" = "NULL",
"dynamic_partition.storage_policy" = "",
"dynamic_partition.start_day_of_month" = "1",
"storage_format" = "V2",
"inverted_index_storage_format" = "DEFAULT",
"light_schema_change" = "false",
"disable_auto_compaction" = "false",
"enable_single_replica_compaction" = "false",
"group_commit_interval_ms" = "10000",
"group_commit_data_bytes" = "134217728"
);
"""

alter_1 = """
alter table demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0 add column `order_type_cd` TEXT AFTER `sys_src`;
"""

alter_2 = """
alter table demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0 set ("light_schema_change" = "true");
"""

alter_3 = """
alter table demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0 add column `order_type_desc` TEXT AFTER `order_type_cd`;
"""

select_sql = """
SELECT * FROM demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0;
"""

def get_connection():
    """获取数据库连接"""
    try:
        connection = pymysql.connect(**DB_CONFIG)
        return connection
    except Exception as e:
        logger.error(f"数据库连接失败: {e}")
        raise

def execute_sql(connection, sql, description=""):
    """执行SQL语句"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(sql)
            connection.commit()
    except Exception as e:
        logger.error(f"SQL执行失败 {description}: {e}")
        connection.rollback()
        raise

def generate_test_data(num_rows=5000):
    """生成测试数据"""
    # 基础数据
    holiday_codes = ['NYD', 'CNY', 'VD', 'WD', 'LD', 'ND', 'CD', 'TG']
    holiday_names = ['元旦', '春节', '情人节', '妇女节', '劳动节', '国庆节', '圣诞节', '感恩节']
    cms_codes = [f'CMS{i:04d}' for i in range(1, 101)]
    sizes = ['XS', 'S', 'M', 'L', 'XL', 'XXL']
    currencies = ['CNY', 'USD', 'EUR', 'JPY']
    sys_sources = ['POS', 'ONLINE', 'MOBILE', 'WECHAT']
    brands = ['FILA', 'NIKE', 'ADIDAS']
    channels = ['ONLINE', 'OFFLINE', 'HYBRID']
    modes = ['RETAIL', 'WHOLESALE', 'DISCOUNT']
    
    data_rows = []
    base_date = datetime(2024, 1, 1)
    
    for i in range(num_rows):
        # 生成随机日期
        random_days = random.randint(0, 365)
        holiday_dt = (base_date + timedelta(days=random_days)).strftime('%Y-%m-%d')
        
        # 基础字段
        cms_code = random.choice(cms_codes)
        sku_id = f'SKU{i+1:06d}'
        holiday_cd = random.choice(holiday_codes)
        holiday_nm = random.choice(holiday_names)
        art_no = f'ART{random.randint(100000, 999999)}'
        size_name = random.choice(sizes)
        dccy = random.choice(currencies)
        
        # 新增字段
        pub_art_flag = random.choice(['Y', 'N'])
        oper_staff_id = f'STAFF{random.randint(1000, 9999)}'
        oper_staff_name = f'员工{random.randint(1, 100)}'
        brand_code = random.choice(brands)
        brand_name = brand_code
        achi_blng_store_code = f'STORE{random.randint(100, 999)}'
        achi_blng_store_name = f'门店{random.randint(1, 100)}'
        order_hour = f'{random.randint(0, 23):02d}'
        is_valid_store = random.choice(['Y', 'N'])
        same_store_flag = random.choice(['Y', 'N'])
        td_type_cd = random.choice(['TD1', 'TD2', 'TD3'])
        td_type_desc = random.choice(['类型1', '类型2', '类型3'])
        sale_channel_cd = random.choice(channels)
        sale_channel_desc = sale_channel_cd
        sale_mode_cd = random.choice(modes)
        sale_mode_desc = sale_mode_cd
        
        # 金额和数量字段
        settle_amt = round(random.uniform(10.0, 1000.0), 4)
        rmb_settle_amt = round(settle_amt * random.uniform(6.0, 7.5), 4)
        good_qty = random.randint(1, 10)
        tag_amt = round(settle_amt * random.uniform(1.1, 2.0), 4)
        rmb_tag_amt = round(tag_amt * random.uniform(6.0, 7.5), 4)
        
        # 其他衍生字段 (设置为0或NULL，简化数据生成)
        no_gift_settle_amt = settle_amt * 0.9
        no_gift_rmb_settle_amt = rmb_settle_amt * 0.9
        no_gift_good_qty = good_qty
        no_gift_tag_amt = tag_amt * 0.9
        no_gift_rmb_tag_amt = rmb_tag_amt * 0.9
        
        # 同店相关字段
        same_store_settle_amt = settle_amt if same_store_flag == 'Y' else 0
        same_store_rmb_settle_amt = rmb_settle_amt if same_store_flag == 'Y' else 0
        same_store_good_qty = good_qty if same_store_flag == 'Y' else 0
        same_store_tag_amt = tag_amt if same_store_flag == 'Y' else 0
        same_store_rmb_tag_amt = rmb_tag_amt if same_store_flag == 'Y' else 0
        
        # 其他字段设为0简化
        td_no_gift_settle_amt = 0
        td_no_gift_rmb_settle_amt = 0
        td_no_gift_good_qty = 0
        td_no_gift_tag_amt = 0
        td_no_gift_rmb_tag_amt = 0
        
        tq_settle_amt = 0
        tq_rmb_settle_amt = 0
        tq_good_qty = 0
        tq_tag_amt = 0
        tq_rmb_tag_amt = 0
        tq_no_gift_settle_amt = 0
        tq_no_gift_rmb_settle_amt = 0
        tq_no_gift_good_qty = 0
        tq_no_gift_tag_amt = 0
        tq_no_gift_rmb_tag_amt = 0
        
        tq_same_store_settle_amt = 0
        tq_same_store_rmb_settle_amt = 0
        tq_same_store_good_qty = 0
        tq_same_store_tag_amt = 0
        tq_same_store_rmb_tag_amt = 0
        tq_td_no_gift_settle_amt = 0
        tq_td_no_gift_rmb_settle_amt = 0
        tq_td_no_gift_good_qty = 0
        tq_td_no_gift_tag_amt = 0
        tq_td_no_gift_rmb_tag_amt = 0
        
        sys_src = random.choice(sys_sources)
        
        row = (
            holiday_dt, cms_code, sku_id, holiday_cd, holiday_nm, art_no, size_name, dccy,
            pub_art_flag, oper_staff_id, oper_staff_name, brand_code, brand_name,
            achi_blng_store_code, achi_blng_store_name, order_hour, is_valid_store,
            same_store_flag, td_type_cd, td_type_desc, sale_channel_cd, sale_channel_desc,
            sale_mode_cd, sale_mode_desc, settle_amt, rmb_settle_amt, good_qty, tag_amt,
            rmb_tag_amt, no_gift_settle_amt, no_gift_rmb_settle_amt, no_gift_good_qty,
            no_gift_tag_amt, no_gift_rmb_tag_amt, same_store_settle_amt, same_store_rmb_settle_amt,
            same_store_good_qty, same_store_tag_amt, same_store_rmb_tag_amt, td_no_gift_settle_amt,
            td_no_gift_rmb_settle_amt, td_no_gift_good_qty, td_no_gift_tag_amt, td_no_gift_rmb_tag_amt,
            tq_settle_amt, tq_rmb_settle_amt, tq_good_qty, tq_tag_amt, tq_rmb_tag_amt,
            tq_no_gift_settle_amt, tq_no_gift_rmb_settle_amt, tq_no_gift_good_qty, tq_no_gift_tag_amt,
            tq_no_gift_rmb_tag_amt, tq_same_store_settle_amt, tq_same_store_rmb_settle_amt,
            tq_same_store_good_qty, tq_same_store_tag_amt, tq_same_store_rmb_tag_amt,
            tq_td_no_gift_settle_amt, tq_td_no_gift_rmb_settle_amt, tq_td_no_gift_good_qty,
            tq_td_no_gift_tag_amt, tq_td_no_gift_rmb_tag_amt, sys_src
        )
        data_rows.append(row)
    
    return data_rows

def insert_data(connection, data_rows):
    """批量插入数据"""
    # 检查第一行数据的字段数量
    if not data_rows:
        logger.error("没有数据要插入")
        return
        
    actual_fields = len(data_rows[0])
    logger.info(f"实际数据字段数量: {actual_fields}")
    
    # 动态生成占位符
    insert_sql = f"""
    INSERT INTO demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0 
    (holiday_dt, cms_code, sku_id, holiday_cd, holiday_nm, art_no, size_name, dccy,
     pub_art_flag, oper_staff_id, oper_staff_name, brand_code, brand_name,
     achi_blng_store_code, achi_blng_store_name, order_hour, is_valid_store,
     same_store_flag, td_type_cd, td_type_desc, sale_channel_cd, sale_channel_desc,
     sale_mode_cd, sale_mode_desc, settle_amt, rmb_settle_amt, good_qty, tag_amt,
     rmb_tag_amt, no_gift_settle_amt, no_gift_rmb_settle_amt, no_gift_good_qty,
     no_gift_tag_amt, no_gift_rmb_tag_amt, same_store_settle_amt, same_store_rmb_settle_amt,
     same_store_good_qty, same_store_tag_amt, same_store_rmb_tag_amt, td_no_gift_settle_amt,
     td_no_gift_rmb_settle_amt, td_no_gift_good_qty, td_no_gift_tag_amt, td_no_gift_rmb_tag_amt,
     tq_settle_amt, tq_rmb_settle_amt, tq_good_qty, tq_tag_amt, tq_rmb_tag_amt,
     tq_no_gift_settle_amt, tq_no_gift_rmb_settle_amt, tq_no_gift_good_qty, tq_no_gift_tag_amt,
     tq_no_gift_rmb_tag_amt, tq_same_store_settle_amt, tq_same_store_rmb_settle_amt,
     tq_same_store_good_qty, tq_same_store_tag_amt, tq_same_store_rmb_tag_amt,
     tq_td_no_gift_settle_amt, tq_td_no_gift_rmb_settle_amt, tq_td_no_gift_good_qty,
     tq_td_no_gift_tag_amt, tq_td_no_gift_rmb_tag_amt, sys_src)
    VALUES ({",".join(["%s"] * actual_fields)})
    """
    
    try:
        with connection.cursor() as cursor:
            cursor.executemany(insert_sql, data_rows)
            connection.commit()
    except Exception as e:
        logger.error(f"数据插入失败: {e}")
        logger.error(f"第一行数据样例: {data_rows[0]}")
        logger.error(f"生成的SQL: {insert_sql}")
        connection.rollback()
        raise

def query_data(connection):
    """查询数据"""
    try:
        with connection.cursor() as cursor:
            cursor.execute(select_sql)
            results = cursor.fetchall()
            return results
    except Exception as e:
        logger.error(f"查询失败: {e}")
        raise

def run_single_test(connection, iteration):
    """执行单次完整测试"""
    try:
        # 步骤1: 建表
        try:
            execute_sql(connection, "DROP TABLE IF EXISTS demo.dm_fila_pcbi_holiday_store_good_retail_online_tmp0", "删除已存在的表")
        except:
            pass
            
        execute_sql(connection, create_test_table_sql, "创建测试表")
        
        # 步骤2: 生成并插入5000行数据
        test_data = generate_test_data(3632)
        insert_data(connection, test_data)
        
        # 等待数据加载完成
        # time.sleep(2)
        
        execute_sql(connection, select_sql, "执行初始查询以验证数据插入")
        
        # 步骤3: 执行 alter_1
        execute_sql(connection, alter_1, "执行 ALTER 1 - 添加 order_type_cd 列")
        
        # 步骤4: 执行 alter_2
        time.sleep(10)
        execute_sql(connection, alter_2, "执行 ALTER 2 - 启用 light_schema_change")
        
        # 步骤5: 执行 alter_3
        time.sleep(2)
        execute_sql(connection, alter_3, "执行 ALTER 3 - 添加 order_type_desc 列")
        
        # 步骤6: 执行 select_sql
        results = query_data(connection)
        
        logger.info(f"第 {iteration} 次测试完成，查询返回 {len(results)} 行数据")
        return True
        
    except Exception as e:
        logger.error(f"第 {iteration} 次测试失败: {e}")
        raise

def main():
    """主函数，循环执行测试直到出错"""
    connection = None
    iteration = 1
    
    try:
        # 获取数据库连接
        connection = get_connection()
        logger.info("开始循环测试，按 Ctrl+C 停止")
        
        while True:
            logger.info(f"开始第 {iteration} 次测试...")
            
            try:
                run_single_test(connection, iteration)
                iteration += 1
                
                # 每次测试之间稍作等待
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"测试在第 {iteration} 次迭代时失败: {e}")
                break
                
    except KeyboardInterrupt:
        logger.info(f"用户中断，总共完成了 {iteration-1} 次测试")
        
    except Exception as e:
        logger.error(f"程序异常退出: {e}")
        
    finally:
        if connection:
            connection.close()
            logger.info("数据库连接已关闭")

if __name__ == "__main__":
    main()