import pandas as pd
from sklearn.model_selection import train_test_split, KFold
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sku_path = './datasets/di_sku_log_drink_labels.csv'
csv_path = {0: './datasets/sku_predict_result2.csv',
            1: './datasets/sku_predict_result2.csv',
            2: './datasets/sku_predict_result2.csv'}
csv_columns = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料', '碳酸饮料', '咖啡（类）饮料',
               '包装饮用水', '特殊用途饮料']


# 定义元模型的类
class MetaModel(nn.Module):
    def __init__(self):
        super(MetaModel, self).__init__()
        self.meta = nn.Sequential(nn.Linear(3, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        output = self.meta(x)
        return output


def threshold_EVA(y_pred, y_true):
    # 设置阈值
    y_pred = (y_pred > 0.5).int()  # 使用0.5作为阈值，大于阈值的为预测为正类

    TP = ((y_pred == y_true) & (y_true == 1)).sum()
    TN = ((y_pred == y_true) & (y_true == 0)).sum()
    FN = ((y_pred != y_true) & (y_true == 1)).sum()
    FP = ((y_pred != y_true) & (y_true == 0)).sum()
    # print("TP, FN, FP, TN", TP, FN, FP, TN)
    acc = (TP + TN) / (TP + TN + FN + FP)
    pre = TP / (TP + FP)
    rec = TP / (TP + FN)
    f1 = 2 * TP / (2 * TP + FP + FN)
    return acc, pre, rec, f1


def get_train_eval():
    # base_csv:用于基模型预测 meta_csv:元模型验证
    drink_df = pd.read_csv(sku_path, usecols=csv_columns)
    print("输出第一步划分数据集的数量")
    # ...
    # 暂时跳过K折步骤
    # data_x, data_y = base_csv['feature'], base_csv['label']
    # kf_5 = KFold(n_splits=5)
    # k, epochs = 0, 5
    # best_accuracy = 0.
    # for t_train, t_test in kf_5.split(data_x, data_y):
    #     print('==================第{}折================'.format(k + 1))
    #     k += 1
    #     model = Model() #.......
    #     train_ds = DefineDataset(data_x[t_train], data_y[t_train])
    #     train_ip = DataLoader(dataset=train_ds, batch_size=32, shuffle=True, drop_last=True)
    #     test_ds = DefineDataset(data_x[t_test], data_y[t_test])
    #     test_ip = DataLoader(dataset=test_ds, batch_size=32, shuffle=False, drop_last=True)
    #     accuracy_list = list()
    #     # run epochs
    #     for ep in range(epochs):
    #         training(train_ip, model)
    #         _, pre_av = predicting(test_ip, model)
    #         accuracy_list.append(round(pre_av, 3))
    #     mean_accuracy = np.mean(accuracy_list)
    #     if mean_accuracy > best_accuracy:
    #         best_accuracy = mean_accuracy
    print("输出每个分类模型中每折训练的准确率")


def training(dataloader, model):
    train_len = len(dataloader)
    # 2、训练元模型
    criterion_meta = nn.BCEWithLogitsLoss()
    optimizer_meta = optim.Adam(model.parameters(), lr=0.01)

    model.train()
    epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0, 0, 0, 0, 0
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)

        optimizer_meta.zero_grad()
        outputs_meta = model(inputs)
        outputs_meta = outputs_meta.squeeze(1)

        loss = criterion_meta(outputs_meta, labels)
        loss.backward()
        optimizer_meta.step()
        # 评估结果
        epoch_los += loss.item()
        accu, precision, recall, f1s = threshold_EVA(outputs_meta, labels)
        epoch_acc += accu.item()
        epoch_prec += precision.item()
        epoch_recall += recall.item()
        epoch_f1s += f1s.item()

    loss_value = epoch_los / train_len
    acc_value = epoch_acc / train_len
    prec_value = epoch_prec / train_len
    rec_value = epoch_recall / train_len
    f1_value = epoch_f1s / train_len
    return acc_value, loss_value, prec_value, rec_value, f1_value


def evaluating(dataloader, model):
    eval_len = len(dataloader)
    # 3、获取测试集，同样拼接各分类模型输出，使用元模型进行预测
    criterion_meta = nn.BCEWithLogitsLoss()

    model.eval()
    with torch.no_grad():
        epoch_los, epoch_acc, epoch_prec, epoch_recall, epoch_f1s = 0, 0, 0, 0, 0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.float32)

            outputs_meta = model(inputs)
            outputs_meta = outputs_meta.squeeze(1)

            loss = criterion_meta(outputs_meta, labels)
            epoch_los += loss.item()
            # 评估结果
            accu, precision, recall, f1s = threshold_EVA(outputs_meta, labels)
            epoch_acc += accu.item()
            epoch_prec += precision.item()
            epoch_recall += recall.item()
            epoch_f1s += f1s.item()
        loss_value = epoch_los / eval_len
        acc_value = epoch_acc / eval_len
        prec_value = epoch_prec / eval_len
        rec_value = epoch_recall / eval_len
        f1_value = epoch_f1s / eval_len
        return acc_value, loss_value, prec_value, rec_value, f1_value


def train_meta_model(path, columns):
    # 预测完成后，读取结果
    proto_csv = pd.read_csv(path[0], usecols=columns)
    sim_csv = pd.read_csv(path[1], usecols=columns)
    textcnn_csv = pd.read_csv(path[2], usecols=columns)

    # 1、获取训练集，拼接各分类模型输出，及真实标签
    features_labels = pd.concat([proto_csv['碳酸饮料'], sim_csv['植物饮料'], textcnn_csv['蛋白饮料']],
                                axis=1)
    features_labels['label'] = proto_csv['包装饮用水']
    base_df, meta_df = train_test_split(features_labels, test_size=0.2, random_state=23)  # 划分成基数据、元数据

    meta_model = MetaModel().to(device)
    # 将元特征和标签转换为张量
    train_dataset = TensorDataset(torch.tensor(base_df[['碳酸饮料', '植物饮料', '蛋白饮料']].values),
                                  torch.tensor(base_df['label'].tolist()))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=128, shuffle=True, drop_last=True)
    eval_dataset = TensorDataset(torch.tensor(meta_df[['碳酸饮料', '植物饮料', '蛋白饮料']].values),
                                 torch.tensor(meta_df['label'].tolist()))
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=128, shuffle=False, drop_last=False)

    num_epochs_meta = 10
    for epoch in range(num_epochs_meta):
        best_accuracy = 0.0
        train_acc_value, train_loss_value, train_prec_value, train_rec_value, train_f1_value \
            = training(train_dataloader, meta_model)
        test_acc_value, test_loss_value, test_prec_value, test_rec_value, test_f1_value \
            = evaluating(eval_dataloader, meta_model)
        print("epochs:{} 训练集 loss:{:.4f}, accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}, F1:{:.2%}"
              .format(epoch, train_loss_value, train_acc_value, train_prec_value, train_rec_value, train_f1_value))
        print("         验证集 loss:{:.4f}, accuracy: {:.2%}, precision: {:.2%}, recall: {:.2%}, F1:{:.2%}"
              .format(test_loss_value, test_acc_value, test_prec_value, test_rec_value, test_f1_value))

        if test_acc_value > best_accuracy:
            torch.save(meta_model, "best_lstm_bert.model")


# 4、后续处理，如计算准确率、保存预测结果等


if __name__ == '__main__':
    # 用于基模型预测及元模型验证
    get_train_eval()
    # 2、训练元模型
    train_meta_model(path=csv_path, columns=csv_columns)
    # 使用元模型进行预测
