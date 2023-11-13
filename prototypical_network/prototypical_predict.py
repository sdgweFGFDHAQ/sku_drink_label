#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/9/11 15:13
# @Author  : zzx
# @File    : prototypical_predict.py
# @Software: PyCharm
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
import torch
from torch.utils.data import TensorDataset, DataLoader

from models.proto_model import ProtoTypicalNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

pretrian_bert_url = "IDEA-CCNL/Erlangshen-DeBERTa-v2-97M-Chinese"
labeled_di_sku_path = "/home/DI/zhouzx/code/workbranch/di_sku_log/data/di_sku_log_drink_labeling_zzx.csv"
batch_size = 512


def get_labeled_dataloader(df, bert_tokenizer):
    # 创建输入数据的空列表
    input_ids = []
    attention_masks = []
    label2id_list = []
    # 遍历数据集的每一行
    for index, row in df.iterrows():
        # 处理特征
        encoded_dict = bert_tokenizer.encode_plus(
            row['name'],
            row['storeType'],
            add_special_tokens=True,
            max_length=14,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids.append(encoded_dict['input_ids'].squeeze())
        attention_masks.append(encoded_dict['attention_mask'].squeeze())

    dataset = TensorDataset(torch.stack(input_ids), )
    return dataset


def predicting(dataset, model):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    model.eval()
    with torch.no_grad():
        output_list = []
        for i, support_input in enumerate(dataloader):
            # 1. 放到GPU上
            support_input0 = support_input[0].to(device, dtype=torch.long)
            # 2. 计算输出
            output = model(support_input0)
            output_list.extend([tensor.cpu().numpy() for tensor in output])
        label_list = [[np.where(arr > 0.5, 1, 0) for arr in row] for row in output_list]
    return output_list, label_list


def proto_bert_predict():
    features = ['id', 'name', 'storeType']
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    columns = ['drink_label', 'labels_token']
    labeled_df = pd.read_csv(labeled_di_sku_path, usecols=features + columns)
    print('labeled_df', labeled_df.shape[0])

    # 加载模型做预测
    tokenizer = AutoTokenizer.from_pretrained(pretrian_bert_url)
    bert_layer = AutoModel.from_pretrained(pretrian_bert_url)

    proto_model = ProtoTypicalNet(
        bert_layer=bert_layer,
        input_dim=768,
        hidden_dim=128,
        num_class=len(labels)
    ).to(device)
    proto_model.load_state_dict(torch.load('./models/proto_model.pth'))

    labeled_dataset = get_labeled_dataloader(labeled_df, tokenizer)
    print("==========开始做预测=========", time.strftime('%H:%M:%S', time.localtime(time.time())))
    output_result, lable_result = predicting(labeled_dataset, proto_model)

    print("========预测完成，生成df对象=========", time.strftime('%H:%M:%S', time.localtime(time.time())))
    drink_df = pd.DataFrame(output_result, columns=['pred_' + label for label in labels])
    source_df = labeled_df[features + columns].reset_index(drop=True)
    predict_result = pd.concat([source_df, drink_df], axis=1)
    predict_result.to_csv('./data/di_sku_proto_predict_result2.csv')
    print("========保存到csv文件=========", time.strftime('%H:%M:%S', time.localtime(time.time())))


def analysis():
    df = pd.read_csv("./data/di_sku_proto_predict_result.csv")
    print(df.shape[0])
    print(df.head())
    have_label_df = df[df['drink_label'].apply(lambda x: isinstance(x, list) and len(x) > 0)]
    print('have_label_df number: ', have_label_df.shape[0])

    # 计算准确率
    labels = ['植物饮料', '果蔬汁类及其饮料', '蛋白饮料', '风味饮料', '茶（类）饮料',
              '碳酸饮料', '咖啡（类）饮料', '包装饮用水', '特殊用途饮料']
    for label in labels:
        # 计算准确率
        accuracy = accuracy_score(df[label], df['pred_' + label])
        # 计算查准率
        precision = precision_score(df[label], df['pred_' + label])
        # 计算召回率
        recall = recall_score(df[label], df['pred_' + label])
        # 计算 F1 分数
        f1 = f1_score(df[label], df['pred_' + label])
        print('accuracy: {}'.format(accuracy))
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('f1: {}'.format(f1))


if __name__ == '__main__':
    # 预测
    proto_bert_predict()

    # 设置列宽度，将其设置为 None 表示不限制
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_colwidth', None)
    df = pd.read_csv("./data/di_sku_proto_predict_result2.csv")
    print(df.shape[0])
    print(df.head())

    # 效果验证
    # analysis()
# nohup python -u prototypical.py > /dev/null 2>&1 &
