#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/8 14:42
# @Author  : zzx
# @File    : down_predict_sku_data.py
# @Software: PyCharm
# from impala.dbapi import connect
from pyhive import hive
import pandas as pd

prefix_path = "/home/DI/zhouzx/code/sku_drink_label"


# 预测集 预测融合数据的品类
def count_matching_number(fetch_size=1000000):
    # conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
    #                        database='standard_db', auth='CUSTOM')
    conn = hive.Connection(host='124.71.220.115', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    try:
        sql = "WITH sku AS (" \
              "SELECT DISTINCT ds.store_id,dssdl.drink_label " \
              "from standard_db.di_store_sku_drink_label dssdl " \
              "inner join standard_db.di_sku as ds " \
              "on dssdl.sku_code = ds.sku_code WHERE dssdl.sku_name is not null), " \
              "dedupe as (" \
              "SELECT d.id,d.name,d.appcode,d.category1_new,d.state,d.city,ds.predict_category " \
              "FROM standard_db.di_store_classify_dedupe d " \
              "LEFT JOIN standard_db.di_store_dedupe_labeling ds on d.id=ds.store_id " \
              "WHERE d.appcode like '%,%' and (d.appcode like '%高德%' or d.appcode like '%腾讯%' or d.appcode like '%百度%')) " \
              "SELECT dedupe.*,sku.drink_label as drink_label " \
              "FROM dedupe LEFT JOIN sku ON sku.store_id = dedupe.id"
        cursor.execute(sql)

        count = 0
        while True:
            results = cursor.fetchmany(fetch_size)
            if not results:
                break
            di_sku_log_data = pd.DataFrame(results, columns=["id", "name", "appcode", "category1_new", "state", "city",
                                                             "predict_category", "drink_label"])
            di_sku_log_data.to_csv(prefix_path + '/datasets/di_sku_drink_data_{}.csv'.format(count))
            print("待打标数据集(预测集)数据量{}:".format(count))
            count += 1
        print("SQL执行完成！")
    except Exception as e:
        print("出错了！")
        print(e)
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


def sku_data_pretreatment():
    # 整理数据：去重、解析label标签、转化为0/1标签
    use_columns = ['id', 'name', 'appcode', 'category1_new', 'predict_category', 'drink_label']
    keep_columns = ['id', 'name', 'appcode', 'storetype']
    merge_column = ['drink_label']

    # 集成学习预测品类标签，对相应数据集进行处理
    dfs = []
    for i in range(6):
        read_path = prefix_path + "/datasets/di_sku_drink_data_{}.csv".format(i)
        df = pd.read_csv(read_path, usecols=use_columns, keep_default_na=False)
        print("======第{}个csv文件，数据量为{}=======".format(i, df.shape[0]))
        dfs.append(df)

    csv_data = pd.concat(dfs, axis=0, ignore_index=True)

    save_path = prefix_path + "/datasets/di_sku_log_drink_labeling_zzx.csv"
    print("======合并同store_id的sku商品=======")
    print("dedupe 原始sku数据集：", csv_data.shape[0])
    csv_data.drop_duplicates(keep='first', inplace=True)
    print("dedupe 原始sku去重：", csv_data.shape[0])
    # 合并原始类别和预测类别
    csv_data['storetype'] = csv_data['category1_new'].fillna(csv_data['predict_category'])

    csv_result = csv_data[keep_columns + merge_column]
    csv_result.to_csv(save_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    # 关联sku表，下载预测集
    count_matching_number()
    # 处理数据集格式
    sku_data_pretreatment()
