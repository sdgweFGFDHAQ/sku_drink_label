import warnings

import numpy as np
import pandas as pd

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader

from models.proto_model import ProtoTypicalNet

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter('./logs/v1')

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
pretrian_bert_url0 = "IDEA-CCNL/Erlangshen-DeBERTa-v2-320M-Chinese"
# 旧版本
labeled_update_path = '/datasets/is_7t1.csv'
labeled_di_sku_path = '/datasets/di_sku_log_drink_labels.csv'
labeled_di_sku_path2 = '/datasets/di_sku_log_chain_drink_labels_clean_dgl.csv'
# 当前版本
prefix_path = "/home/DI/zhouzx/code/sku_drink_label"
train_path = '/datasets/fs_sku_drink_data_train.csv'
valid_path = '/datasets/fs_sku_drink_data_valid.csv'
test_path = '/datasets/fs_sku_drink_data_test.csv'

token_max_length = 12
batch_size = 16
epochs = 20


def get_Support_Query(train_df, label_list, k=10):
    def minimize_set(df_i, count_list, excessive_list):
        tally = count_list.copy()
        tally = [i - j for i, j in zip(tally, df_i['multi_label'])]
        if (min(count_list) >= k) and (k - 1 not in count_list):
            excessive_list.append(df_i['index_col'])
            count_list.clear()
            count_list.extend(tally)

    df = train_df.copy()
    df['multi_label'] = df[label_list].values.tolist()
    df['index_col'] = df.index

    # 随机抽样
    support_df, add_list = pd.DataFrame(), []
    label_number = len(label_list)
    count = [0] * label_number
    for label_i in range(label_number):
        # 取出当前标签的df
        label_i_df = df[df[label_list[label_i]] == 1]
        # 满足每个标签最少出现K次，如果原始数据集df不足K条则结束
        while count[label_i] < k and label_i_df.shape[0] > 0:
            add_series = label_i_df.sample(n=1)
            drop = label_i_df.drop(add_series.index, inplace=False)
            add_list.append(add_series)
            count = [i + j for i, j in zip(count, add_series['multi_label'].values[0])]
            df.drop(add_series.index, inplace=True)
            label_i_df = drop
    support_df = pd.concat([support_df] + add_list)

    # 删除多余的数据
    delete_list = []
    support_df.apply(minimize_set, args=(count, delete_list), axis=1)
    support_df.drop(delete_list, inplace=True)
    return support_df


# 线下跑店(店铺存在且售品) 1399条
def get_labeled_dataloader(df, bert_tokenizer, label_list):
    # 创建输入数据的空列表
    input_ids = []
    attention_masks = []
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理特征
        encoded_dict = bert_tokenizer.encode_plus(
            row['name'],
            row['storetype'],
            add_special_tokens=True,
            max_length=14,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

        # 处理类别
        labels_tensor = torch.tensor([row[label] for label in label_list])
        label2id_list.append(labels_tensor)

    dataset = TensorDataset(torch.stack(input_ids), torch.stack(label2id_list))
    return dataset


def threshold_EVA(y_pred, y_true):
    acc, pre, rec, f1 = torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0])
    # 设置阈值
    y_pred = (y_pred > 0.5).int()  # 使用0.5作为阈值，大于阈值的为预测为正类
    try:
        # 准确率
        correct = (y_pred == y_true).int()
        acc = correct.sum() / (correct.shape[0] * correct.shape[1])
        # acc = accuracy_score(y_pred, y_true)
        y_pred = y_pred.cpu().numpy()
        y_true = y_true.cpu().numpy()
        # 精确率
        pre = precision_score(y_pred, y_true, average='weighted')
        # 召回率
        rec = recall_score(y_pred, y_true, average='weighted')
        # F1
        f1 = f1_score(y_pred, y_true, average='weighted')
    except Exception as e:
        print(str(e))
    return acc, pre, rec, f1


def training(dataset, model, rt):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=rt)
    # 使用Adam优化器
    optimizer = optim.Adam(model.parameters(), lr=0.0002)

    model.train()
    epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
    for i, support_input in enumerate(dataloader):
        # 1. 放到GPU上
        feature = support_input[0].to(device, dtype=torch.long)
        label = support_input[1].to(device, dtype=torch.long)
        # 2. 清空梯度
        optimizer.zero_grad()
        # 3. 计算输出
        output = model(feature)
        # 4. 计算损失
        loss = criterion(output, label.float())
        # 5. 反向传播
        loss.backward()
        # 6. 更新梯度
        optimizer.step()
        # 7.预测结果
        epoch_los += loss.item()
        accu, precision, recall, f1s = threshold_EVA(output, label)
        epoch_acc += accu.item()
        epoch_prec += precision.item()
        epoch_recall += recall.item()
        epoch_f1s += f1s.item()
    num_batches = len(dataloader)
    loss_value = epoch_los / num_batches
    acc_value, prec_value = epoch_acc / num_batches, epoch_prec / num_batches
    rec_value, f1_value = epoch_recall / num_batches, epoch_f1s / num_batches
    return acc_value, loss_value, prec_value, rec_value, f1_value


def evaluating(dataset, model, rt):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    criterion = nn.BCEWithLogitsLoss(reduction='mean', weight=rt)

    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0, 0.0
        for i, support_input in enumerate(dataloader):
            # 1. 放到GPU上
            feature = support_input[0].to(device, dtype=torch.long)
            label = support_input[1].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(feature)
            # 3. 计算损失
            loss = criterion(output, label.float())
            epoch_los += loss.item()
            # 4.预测结果
            accu, precision, recall, f1s = threshold_EVA(output, label)
            epoch_acc += accu.item()
            epoch_prec += precision.item()
            epoch_recall += recall.item()
            epoch_f1s += f1s.item()
        num_batches = len(dataloader)
        loss_value = epoch_los / num_batches
        acc_value, prec_value = epoch_acc / num_batches, epoch_prec / num_batches
        rec_value, f1_value = epoch_recall / num_batches, epoch_f1s / num_batches
    return acc_value, loss_value, prec_value, rec_value, f1_value


def predicting(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    with torch.no_grad():
        output_list = []
        epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0.0, 0.0, 0.0, 0.0
        for i, support_input in enumerate(dataloader):
            # 1. 放到GPU上
            feature = support_input[0].to(device, dtype=torch.long)
            label = support_input[1].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(feature)
            # 3. 计算评估指标
            accu, precision, recall, f1s = threshold_EVA(output, label)
            epoch_acc += accu.item()
            epoch_prec += precision.item()
            epoch_recall += recall.item()
            epoch_f1s += f1s.item()
            # 4. 输出预测值
            output_list.extend([tensor.cpu().numpy() for tensor in output])

        num_batches = len(dataloader)
        acc_value, prec_value = epoch_acc / num_batches, epoch_prec / num_batches
        rec_value, f1_value = epoch_recall / num_batches, epoch_f1s / num_batches
        label_list = [[np.where(arr > 0.5, 1, 0) for arr in row] for row in output_list]
    return output_list, label_list, acc_value, prec_value, rec_value, f1_value


# 训练 测试 分析
def train_and_test(support_dataset, test_dataset, proto_model, ratio, save_path):
    max_accuracy = 0.0
    for step in range(epochs):
        train_acc_value, train_loss_value, train_prec_value, \
            train_rec_value, train_f1_value = training(support_dataset, proto_model, ratio)
        test_acc_value, test_loss_value, test_prec_value, \
            test_rec_value, test_f1_value = evaluating(test_dataset, proto_model, ratio)
        print("epochs:{} 训练集 accuracy: {:.2%},loss:{:.4f} "
              "| 验证集 accuracy: {:.2%},loss:{:.4f}"
              .format(step, train_acc_value, train_loss_value, test_acc_value, test_loss_value))
        print("         -- 训练集 precision: {:.2%},recall: {:.2%},F1:{:.2%} "
              "| 验证集 precision: {:.2%},recall: {:.2%},F1:{:.2%}"
              .format(train_prec_value, train_rec_value, train_f1_value, test_prec_value, test_rec_value,
                      test_f1_value))
        # writer.add_scalars('acc', {'train_acc': train_acc_value, 'test_acc': test_acc_value}, global_step=step)
        # writer.add_scalars('loss', {'train_loss': train_loss_value, 'test_loss': test_loss_value}, global_step=step)

        # 保存最佳模型
        if test_acc_value > max_accuracy:
            max_accuracy = test_acc_value
            torch.save(proto_model.state_dict(), save_path)


# bert模型
def run_proto_bert():
    features = ['id', 'name', 'storetype']
    # labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
    #           '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    labels = ['plant_clean', 'fruit_vegetable_clean', 'protein_clean', 'flavored_clean', 'tea_clean',
              'carbonated_clean', 'coffee_clean', 'water_clean', 'special_uses_clean']
    columns = []
    columns.extend(features)
    columns.extend(labels)

    # 读取并清洗训练集
    train_df = pd.read_csv(prefix_path + train_path, usecols=columns)
    train_df = train_df[train_df['name'].notnull() & (train_df['name'] != '')]
    train_df = train_df[train_df['storetype'].notnull() & (train_df['storetype'] != '')]
    print('sq_set len:{}'.format(train_df.shape[0]))
    # 读取并清洗训练集
    valid_df = pd.read_csv(prefix_path + valid_path, usecols=columns)
    valid_df = valid_df[valid_df['name'].notnull() & (valid_df['name'] != '')]
    valid_df = valid_df[valid_df['storetype'].notnull() & (valid_df['storetype'] != '')]
    print('test_set len:{}'.format(valid_df.shape[0]))
    # SQ方法划分训练集、验证集
    # sq_set = get_Support_Query(labeled_df, labels, k=2000)
    # test_set = labeled_df.drop(sq_set.index)

    # dataloader
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    support_dataset = get_labeled_dataloader(train_df, tokenizer, labels)
    test_dataset = get_labeled_dataloader(valid_df, tokenizer, labels)

    # 计算标签为1的占比,作为阈值
    num_ones = torch.tensor((train_df[labels] == 1).sum(axis=0))
    ratio = (num_ones / valid_df.shape[0]).to(device)

    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)

    print("==============模型训练==================")
    # 训练 测试 分析
    train_and_test(support_dataset, test_dataset, proto_model, ratio, prefix_path + '/models/proto_model.pth')
    print("==============训练完成==================")


def predict():
    features = ['name', 'storetype']
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    labels = ['plant_clean', 'fruit_vegetable_clean', 'protein_clean', 'flavored_clean', 'tea_clean',
              'carbonated_clean', 'coffee_clean', 'water_clean', 'special_uses_clean']
    columns = []
    columns.extend(features)
    columns.extend(labels)

    test_df = pd.read_csv(prefix_path + test_path, usecols=columns)
    test_df = test_df[test_df['name'].notnull() & (test_df['name'] != '')]
    test_df = test_df[test_df['storetype'].notnull() & (test_df['storetype'] != '')]

    # 加载模型做预测
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    test_dataset = get_labeled_dataloader(test_df, tokenizer, labels)

    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)
    # 加载模型做预测
    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)
    proto_model.load_state_dict(torch.load(prefix_path + '/models/proto_model.pth'))

    print("==============开始预测==============")
    output_result, lable_result, test_acc, test_prec, test_rec, test_f1 \
        = predicting(test_dataset, proto_model)
    print("测试集 accuracy: {:.2%},precision:{:.4f} recall: {:.2%},F1:{:.4f}"
          .format(test_acc, test_prec, test_rec, test_f1))
    print("=============返回预测结果============")
    drink_values = pd.DataFrame(output_result, columns=labels)  # 模型输出值
    drink_df = pd.DataFrame(lable_result, columns=labels)  # 数值转标签
    source_df = test_df[['id', 'name', 'storetype']].reset_index(drop=True)

    predict_result = pd.concat([source_df, drink_df], axis=1)

    pd.options.display.float_format = '{:.6f}'.format
    predict_result.to_csv(prefix_path + '/datasets/test_predict_result.csv', index=False)
    print("预测完成")


if __name__ == '__main__':
    # 训练模型
    run_proto_bert()
    # 测试集评估泛化能力
    predict()

# nohup python -u prototypical.py> log.log 2>&1 &
# tensorboard --logdir=E:\pyProjects\pycharm_project\workplace\fewsamples\logs\v1 --port 8123
# $ python script.py --method b --path '/data'
