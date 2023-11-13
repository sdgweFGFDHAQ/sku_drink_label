import argparse
import numpy as np
import pandas as pd
from ydata_profiling import ProfileReport


csv_path = {0: '/home/DI/zhouzx/code/workplace/fewsamples/data/di_sku_proto_predict_result.csv',
            1: '/home/DI/luxb/PU-learning-tagging/PUlearning_tagging/code/pred_tag_result.csv',
            2: '/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/predict_result_data_0918.csv'}

csv_col = {
    0: ['pred_植物饮料', 'pred_果蔬汁类及其饮料', 'pred_蛋白饮料', 'pred_风味饮料', 'pred_茶（类）饮料',
        'pred_碳酸饮料',
        'pred_咖啡（类）饮料', 'pred_包装饮用水', 'pred_特殊用途饮料'],
    1: ['植物饮料score', '果蔬汁类及其饮料score', '蛋白饮料score', '风味饮料score', '茶（类）饮料score',
        '碳酸饮料score',
        '咖啡（类）饮料score', '包装饮用水score', '特殊用途饮料score'],
    2: ['plant', 'fruit', 'protein', 'flavored', 'tea', 'carbonated',
        'coffee', 'water', 'special']}

columns = ['pred_植物饮料', 'pred_果蔬汁类及其饮料', 'pred_蛋白饮料', 'pred_风味饮料',
           'pred_茶（类）饮料', 'pred_碳酸饮料', 'pred_咖啡（类）饮料', 'pred_包装饮用水', 'pred_特殊用途饮料']
result_columns = ['id', 'name', 'storetype', 'pred_plant', 'pred_juice', 'pred_protein', 'pred_flavored', 'pred_tea',
                  'pred_carbonated', 'pred_coffee', 'pred_water', 'pred_special']

pd.set_option('display.max_columns', None)


def get_data():
    proto_df = pd.read_csv(csv_path[0])
    print(proto_df.head())
    sim_df = pd.read_csv(csv_path[1])
    print(sim_df.head())
    textcnn_df = pd.read_csv(csv_path[2])
    print(textcnn_df.head())

    profile = ProfileReport(proto_df, title="Profiling Report")
    profile.to_file("proto_report.html")

    profile = ProfileReport(sim_df, title="Profiling Report")
    profile.to_file("sim_report.html")

    profile = ProfileReport(textcnn_df, title="Profiling Report")
    profile.to_file("textcnn_df_report.html")


def alter_columns():
    proto_df = pd.read_csv(csv_path[0], usecols=['id', 'name', 'storeType'] + csv_col[0])
    proto_df.columns = result_columns
    print(proto_df.head())
    proto_df.to_csv('/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/proto_predict_result.csv', index=False)

    sim_df = pd.read_csv(csv_path[1], usecols=['id', 'name', 'storeType'] + csv_col[1])
    sim_df.columns = result_columns
    print(sim_df.head())
    sim_df.to_csv('/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/sim_predict_result.csv', index=False)

    textcnn_df = pd.read_csv(csv_path[2], usecols=['id', 'comment_text'] + csv_col[2])
    textcnn_df.insert(2, 'comment_text1', textcnn_df['comment_text'])
    textcnn_df.columns = result_columns
    print(textcnn_df.head())
    textcnn_df.to_csv('/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/textcnn_predict_result.csv',
                      index=False)


def get_vote_result():
    prefix = '/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/'

    p_df = pd.read_csv(prefix + 'proto_predict_result.csv')
    p_df1 = p_df[['id'] + columns]
    p_df1.set_index('id', inplace=True)

    s_df = pd.read_csv(prefix + 'sim_predict_result.csv')
    s_df = s_df[['id'] + columns]
    s_df.set_index('id', inplace=True)

    t_df = pd.read_csv(prefix + 'textcnn_predict_result.csv')
    t_df = t_df[['id'] + columns]
    t_df.set_index('id', inplace=True)

    sum_df = p_df1 + 1.5 * s_df + t_df
    sum_df.reset_index(inplace=True)
    print(sum_df.head())

    vote_df = pd.DataFrame(columns=columns)
    vote_df['id'] = sum_df['id']
    vote_df['name'] = p_df['name']
    vote_df['storetype'] = p_df['storetype']
    for column in columns:
        vote_series = sum_df[column].apply(lambda x: 1 if x > 1.5 else 0)
        vote_df[column] = vote_series
    vote_df.columns = result_columns
    print(vote_df.head())
    vote_df.to_csv(prefix + 'di_store_drink_label_predict.csv', index=False)


def verify_data():
    path = {0: '/home/DI/zhouzx/code/workplace/fewsamples/data/di_sku_proto_predict_result.csv',
            1: '/home/DI/luxb/PU-learning-tagging/PUlearning_tagging/code/pred_tag_result.csv',
            2: '/home/DI/zhouzx/code/workplace/ensemble_learning/datasets/predict_result_data_0918.csv',
            3: '/home/DI/zhouzx/code/workplace/fewsamples/data/di_sku_log_chain_drink_labels_clean_dgl.csv'}

    str_re = '大卖场'
    proto_df = pd.read_csv(path[0], usecols=['id', 'name', 'storeType'] + csv_col[0])
    filter0 = proto_df[proto_df['storeType'].str.contains(str_re)]
    print("======proto_df=======")
    print(filter0.head(10))
    sim_df = pd.read_csv(path[1], usecols=['id', 'name', 'storeType'] + csv_col[1])
    filter1 = sim_df[sim_df['storeType'].str.contains(str_re)]
    print("======sim_df=======")
    print(filter1.head(10))
    textcnn_df = pd.read_csv(path[2], usecols=['id', 'comment_text'] + csv_col[2])
    filter2 = textcnn_df[textcnn_df['comment_text'].str.contains(str_re)]
    print("======textcnn_df=======")
    print(filter2.head(10))
    train_df = pd.read_csv(path[3])
    train_df['storetype'] = train_df['storetype'].fillna('')
    filter3 = train_df[train_df['storetype'].str.contains(str_re)]
    print("======train_df=======")
    print(filter3.head(10))


if __name__ == '__main__':
    # 查看数据集情况
    # get_data()
    # 用于基模型预测及元模型验证
    alter_columns()  # 统一标准
    get_vote_result()  # 投票法
    # 查看训练集标签情况
    # verify_data()

# /home/DI/luxb/PU-learning-tagging/PUlearning_tagging/code/pred_tag_result.csv xb
# nohup python -u main.py> log.log 2>&1 &
