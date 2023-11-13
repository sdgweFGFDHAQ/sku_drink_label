#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/10/23 14:11
# @Author  : zzx
# @File    : di_store_drink_label_predict_CK.py
# @Software: PyCharm
# -*- encoding: utf-8 -*-
import math
from clickhouse_sqlalchemy import select, make_session, get_declarative_base, engines
from sqlalchemy import create_engine, Column, MetaData, types, text
import logging
import pandas as pd

logging.basicConfig(filename="logging.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s",
                    datefmt="%d-%M-%Y %H:%M:%S", level=logging.INFO)

predict_path = '/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/'
upload_table_name = 'di_store_drink_label_predict'
upload_batch_size = 100000
result_columns = ['id', 'name', 'storetype', 'pred_plant', 'pred_juice', 'pred_protein', 'pred_flavored', 'pred_tea',
                  'pred_carbonated', 'pred_coffee', 'pred_water', 'pred_special']


# 创建ClickhouseClient类
class ClickhouseClient:
    def __init__(self, conf):
        self._connection = 'clickhouse://{user}:{password}@{server_host}:{port}/{db}'.format(**conf)
        self._engine = create_engine(self._connection, pool_size=100, pool_recycle=3600, pool_timeout=20)
        self._session = make_session(self._engine)
        self._metadata = MetaData(bind=self._engine)
        self._base = get_declarative_base(metadata=self._metadata)

    def create_table(self, table_name):
        class Rate(self._base):
            pk = Column(types.Integer, primary_key=True, autoincrement=True)
            id = Column(types.String, primary_key=False)
            name = Column(types.String)
            state = Column(types.String)
            predict_category = Column(types.String)

            __tablename__ = table_name
            __table_args__ = (
                engines.Memory(),
            )

        if not self._engine.dialect.has_table(self._engine, Rate.__tablename__):
            Rate.__table__.create()
        return Rate

    def insert_data(self, table, data, batch_size=10000):
        session = self._session
        total_rows = len(data)
        num_batches = math.floor(total_rows / batch_size)  # 计算总批次数
        try:
            for i in range(num_batches):
                batch_data = data[i * batch_size:(i + 1) * batch_size]
                session.bulk_insert_mappings(table, batch_data)
            if total_rows % batch_size != 0:
                batch_data = data[num_batches * batch_size:total_rows]
                session.bulk_insert_mappings(table, batch_data)
            session.commit()
            logging.info("Data inserted successfully.")
        except Exception as e:
            session.rollback()
            logging.info(f"Error inserting data: {str(e)}")
        finally:
            session.close()

    def clear_data(self, table):
        session = self._session
        try:
            # 清空数据表
            session.query(table).filter(True).delete()
            session.commit()
        except Exception as e:
            session.rollback()
            logging.info(f"Error inserting data: {str(e)}")
        finally:
            # 关闭会话
            session.close()

    def query_data_with_raw_sql(self, sql):
        try:
            # 使用 text() 函数构建原生 SQL 查询
            query = text(sql)

            # 执行查询并获取结果
            result = self._session.execute(query).fetchall()  # 可以使用.fetchmany(size=50000)优化

            return result
        except Exception as e:
            logging.info(f"Error querying data with raw SQL: {str(e)}")
            return []
        finally:
            self._session.close()


conf = {
    'user': 'default',
    'password': 'xwclickhouse2022',
    'server_host': '139.9.51.13',
    'port': 9090,
    'db': 'ai_db'
}

# 创建clickhouse客户端
clickhouse_client = ClickhouseClient(conf)


# 分类算法预测类别，建表并上传数据
def upload_predict_data():
    logging.info("预测数据集上传到数据库")
    # 创建表
    table = clickhouse_client.create_table(table_name=upload_table_name)
    # 清空数据表
    clickhouse_client.clear_data(table=table)

    data = pd.read_csv(
        predict_path + 'di_store_drink_label_predict.csv',
        usecols=result_columns,
        index_col=False)
    # 调整列名
    data_dict = data.to_dict(orient='records')

    # 插入数据
    clickhouse_client.insert_data(table=table, data=data_dict, batch_size=upload_batch_size)

    logging.info("写入数据库完成!")


if __name__ == '__main__':
    # # 分类算法预测类别，建表并上传数据
    upload_predict_data()
# nohup python -u readCK.py > /dev/null 2>&1 &

# # 编写 ClickHouse 支持的 SQL 查询
# sql = """
#       SELECT * FROM store_tags_statistics WHERE tag_id = 2004 AND num = 70
#       """
# # 执行查询并打印结果
# result = clickhouse_client.query_data_with_raw_sql(sql)
# logging.info(result)
