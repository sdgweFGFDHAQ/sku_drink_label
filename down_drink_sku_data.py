# from impala.dbapi import connect
from pyhive import hive
import pandas as pd

prefix_path = "/home/DI/zhouzx/code/sku_drink_label"
source_dataset_path = '/datasets/di_sku_log_drink_data.csv'
standard_dataset_path = '/datasets/di_sku_log_drink_data_clean.csv'


# 对查询表数据量进行统计
def count_matching_number():
    # conn = hive.Connection(host='192.168.0.150', port=10015, username='hive', password='xwbigdata2022',
    #                        database='standard_db', auth='CUSTOM')
    conn = hive.Connection(host='124.71.220.115', port=10015, username='hive', password='xwbigdata2022',
                           database='standard_db', auth='CUSTOM')
    cursor = conn.cursor()
    try:
        sql = "WITH sku AS (" \
              "select DISTINCT ds.store_id, dssdl.brand_name,dssdl.series_name,dssdl.sku_name,dssdl.sku_code,dssdl.drink_label " \
              "from standard_db.di_store_sku_drink_label dssdl " \
              "inner join standard_db.di_sku as ds " \
              "on dssdl.sku_name is not null and dssdl.sku_code = ds.sku_code) " \
              "select DISTINCT dsd.id as id,dsd.name as name,dsd.appcode as appcode,dsd.channeltype_new as channeltype_new," \
              "dsd.category1_new as category1_new,dsd.state as state,dsd.city as city, sku.brand_name as brand_name," \
              "sku.series_name as series_name,sku.sku_name as sku_name,sku.sku_code as sku_code,sku.drink_label as drink_label " \
              "from sku " \
              "inner join standard_db.di_store_dedupe as dsd on sku.store_id = dsd.id"

        cursor.execute(sql)
        di_sku_log = cursor.fetchall()
        di_sku_log_data = pd.DataFrame(di_sku_log,
                                       columns=["id", "name", "appcode", "channeltype_new", "category1_new", "state",
                                                "city", "brand_name", "series_name", "sku_name",
                                                "sku_code", "drink_label"]).set_index("id")
        di_sku_log_data.to_csv(prefix_path + source_dataset_path)
    except Exception as e:
        print("出错了！")
        print(e)
    finally:
        # 关闭连接
        cursor.close()
        conn.close()


def sku_data_pretreatment():
    # 整理数据：去重、解析label标签、转化为0/1标签
    use_columns = ['id', 'name', 'appcode', 'category1_new', 'drink_label']
    keep_columns = ['id', 'name', 'appcode', 'storetype']
    merge_column = ['drink_label']

    # 集成学习预测品类标签，对相应数据集进行处理
    csv_data = pd.read_csv(prefix_path + source_dataset_path, usecols=use_columns, keep_default_na=False)

    print("======合并同store_id的sku商品=======")
    print("dedupe 原始sku数据集：", csv_data.shape[0])
    csv_data.drop_duplicates(keep='first', inplace=True)
    print("dedupe 原始sku去重：", csv_data.shape[0])
    # 合并原始类别和预测类别
    csv_data['storetype'] = csv_data['category1_new']

    # 筛选指定列
    csv_result = csv_data[keep_columns + merge_column]
    csv_result = csv_result.set_index(keep_columns)

    # 合并售卖的商品
    grouped = csv_result.groupby(by=keep_columns)
    result = grouped.agg({merge_column: lambda x: list(set(item for item in x if item != ''))})
    result.reset_index(inplace=True)
    print("合并饮料商品：", result.shape[0])

    print("========提取0-1饮料标签=========")
    exist_df = result
    print("合并饮料sku数据集：", exist_df.shape[0])
    # 去除storetype为空的数据
    exist_df = exist_df.dropna(subset=['storetype'])
    exist_df = exist_df[exist_df['storetype'].notnull() & (exist_df['storetype'] != '')].reset_index(drop=True)
    print("合并饮料sku删除空值：", exist_df.shape[0])

    # 将'drinkTypes'列的列表元素提取为新的列
    new_columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
                   '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    extracted_columns = exist_df[merge_column].apply(
        lambda x: [1 if item in x else 0 for item in new_columns]).tolist()
    extracted_df = pd.DataFrame(extracted_columns, columns=new_columns, dtype=int).reset_index(drop=True)
    # 将提取的列添加到原始DataFrame中
    for c in new_columns:
        exist_df[c] = extracted_df[c]
    exist_df['labels_token'] = extracted_columns

    exist_df.to_csv(standard_dataset_path, index=False, encoding='utf-8')
    # 统计新增列中值为1的数量
    column_counts = exist_df[new_columns].sum()
    print("饮料标签列中值为1的数量", column_counts)


def split_ttv():
    file_path = prefix_path + "/datasets"
    data_df = pd.read_excel(file_path + "/fs_drink_sku_data.xlsx")
    vt_df = data_df.sample(frac=0.2, random_state=2023, axis=0)
    # 训练集
    train_df = data_df[~data_df.index.isin(vt_df.index)]
    train_df.to_csv(file_path + "/fs_sku_drink_data_train.csv", index=True)
    # 验证集
    validation_df = vt_df.sample(frac=0.5, random_state=2023, axis=0)
    validation_df.to_csv(file_path + "/fs_sku_drink_data_valid.csv", index=True)
    # 测试集
    test_df = vt_df[~vt_df.index.isin(validation_df.index)]
    test_df.to_csv(file_path + "/fs_sku_drink_data_test.csv", index=True)


if __name__ == '__main__':
    # 下载有饮料标签的数据集-现通过人工清洗的方式获得模型训练集、测试集
    count_matching_number()
    # 处理训练集、测试集数据格式
    sku_data_pretreatment()
    # 把人工清洗的小样本数据集划分为训练集、验证集、测试集给集成学习使用
    # split_ttv()

# nohup python -u down_drink_sku_data.py> down_di_sku_data.log 2>&1 &
