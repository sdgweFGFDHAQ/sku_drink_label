# -*- coding:utf-8 -*-
######################################################
# 描述：用于读取算法服务器csv文件写入di_store_dedupe_labeling
# 修改记录：
# 日期           版本       修改人    修改原因说明
# 2023/02/20     V1.00      lrz      新建代码
######################################################

import sys
import pandas as pd
import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from recall_tools.pyspark_common import pySparkCm
import csv
from recall_tools.ssh import SSH
import gc

# 清空目标表数据
# table_name = 'di_store_drink_label_predict'
# cm = pySparkCm(table_name)
# spark = cm.sparkenv()
# spark.sql("truncate table standard_db.di_store_drink_label_predict")
# print("table truncate complete")

# 远程连接服务器读取文件
ssh = SSH()
connect = ssh.connect()
sftp_client = connect.open_sftp()
print("=======start=========")
# 读取csv文件输出字典
data = []
try:
    with sftp_client.open(
            "/home/data/temp/zhouzx/workplace/ensemble_learning/datasets/di_store_drink_label_predict.csv") as f:
        for row in csv.DictReader(f, skipinitialspace=True):
            dict_line = dict(row)
            data.append(dict_line)
except Exception as ex:
    print("==========error=========")
    sys.exit(ex)

sftp_client.close()
ssh.close()
print("read data success: " + str(len(data)))

# 定义变量
table_name = 'di_store_drink_label_predict'
cm = pySparkCm(table_name)
spark = cm.sparkenv()
print('==========df===============')
df = spark.createDataFrame(data) \
    .selectExpr('id', 'name', 'storetype', 'cast(pred_plant as int)', 'cast(pred_juice as int)',
                'cast(pred_protein as int)', 'cast(pred_flavored as int)', 'cast(pred_tea as int)',
                'cast(pred_carbonated as int)', 'cast(pred_coffee as int)', 'cast(pred_water as int)',
                'cast(pred_special as int)')

df.printSchema()

# 写入表
print("write data count=" + str(df.count()))
cm.write_to_hudi(df, 'standard_db', 'di_store_drink_label_predict', 'id', '', 'id', 'append', 'insert')
print("Complete!")
