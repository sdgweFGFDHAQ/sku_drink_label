import pandas as pd
from ydata_profiling import ProfileReport

csv_path = {
    0: '/home/DI/zhouzx/code/sku_drink_label/datasets/di_sku_proto_predict_result.csv',
    1: '/home/DI/zhouzx/code/sku_drink_label/datasets/pred_tag_result.csv'
}

base_col = ['id', 'name', 'storetype']
csv_col = {
    0: ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
        '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料'],
    1: ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
        '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
}

columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
           '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
result_columns = ['id', 'name', 'storetype', 'plant_pred', 'juice_pred', 'protein_pred', 'flavored_pred', 'tea_pred',
                  'carbonated_pred', 'coffee_pred', 'water_pred', 'special_pred']

pd.set_option('display.max_columns', None)


# 分析各个模型预测出来的结果
def get_data():
    proto_df = pd.read_csv(csv_path[0])
    print(proto_df.head())
    sim_df = pd.read_csv(csv_path[1])
    print(sim_df.head())

    profile = ProfileReport(proto_df, title="Profiling Report")
    profile.to_file("proto_report.html")

    profile = ProfileReport(sim_df, title="Profiling Report")
    profile.to_file("sim_report.html")


# 防止各个模型预测的文件列名不统一
def alter_columns():
    proto_df = pd.read_csv(csv_path[0], usecols=base_col + csv_col[0])
    proto_df.columns = columns
    print(proto_df.head())
    proto_df.to_csv('/home/DI/zhouzx/code/sku_drink_label/datasets/proto_predict_result.csv', index=False)

    sim_df = pd.read_csv(csv_path[1], usecols=base_col + csv_col[1])
    sim_df.columns = columns
    print(sim_df.head())
    sim_df.to_csv('/home/DI/zhouzx/code/sku_drink_label/datasets/sim_predict_result.csv', index=False)


# 计算投票结果
def get_vote_result():
    prefix = '/home/DI/zhouzx/code/sku_drink_label/datasets'

    p_df = pd.read_csv(prefix + 'proto_predict_result.csv')
    p_df1 = p_df[['id'] + columns]
    p_df1.set_index('id', inplace=True)

    s_df = pd.read_csv(prefix + 'sim_predict_result.csv')
    s_df = s_df[['id'] + columns]
    s_df.set_index('id', inplace=True)

    # t_df = pd.read_csv(prefix + 'textcnn_predict_result.csv')
    # t_df = t_df[['id'] + columns]
    # t_df.set_index('id', inplace=True)

    sum_df = p_df1 + s_df
    sum_df.reset_index(inplace=True)
    print(sum_df.head())

    vote_df = pd.DataFrame(columns=columns)
    for col_name in base_col:
        vote_df[col_name] = sum_df[col_name]
    for column in columns:
        vote_series = sum_df[column].apply(lambda x: 1 if x > 1 else 0)
        vote_df[column] = vote_series
    # 把中文列名改为英文，防止出错。也可以不改
    vote_df.columns = result_columns
    print(vote_df.head())
    vote_df.to_csv(prefix + 'di_store_drink_label_predict.csv', index=False)


# 简单观察个数据集情况
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
