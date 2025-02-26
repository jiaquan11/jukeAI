# -*- coding: utf-8 -*-
import sys  
import os  

# 获取当前脚本所在目录的父目录  
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  
# 将父目录添加到 sys.path  
sys.path.append(parent_dir)  

# 获取当前脚本所在目录  
current_dir = os.path.dirname(os.path.abspath(__file__))  
# 将当前目录添加到 sys.path  
sys.path.append(current_dir) 

import torch.nn.utils.rnn as rnn_utils  # 导入rnn_utils模块，用于处理可变长度序列的填充和排序
from torch.utils.data import Dataset, DataLoader  # 导入Dataset和DataLoader模块，用于加载和处理数据集
import torch  # 导入torch模块，用于处理张量和构建神经网络
import pickle  # 导入pickle模块，用于序列化和反序列化Python对象
from dataset import *  # 导入自定义的数据集类
from parameter_config import *

params = ParameterConfig()

def load_dataset(train_path, valid_path):
    # print('进入函数')
    """
    加载训练集和验证集
    :param train_path: 训练数据集路径
    :return: 训练数据集和验证数据集
    """
    with open(train_path, "rb") as f:
        train_input_list = pickle.load(f)  # 从文件中加载输入列表

    with open(valid_path, "rb") as f:
        valid_input_list = pickle.load(f)  # 从文件中加载输入列表
    # 划分训练集与验证集
    # print(len(input_list))  # 打印输入列表的长度
    # print(input_list[0])
    #
    train_dataset = MyDataset(train_input_list, 300)  # 创建训练数据集对象
    val_dataset = MyDataset(valid_input_list, 300)  # 创建验证数据集对象
    return train_dataset, val_dataset  # 返回训练数据集和验证数据集

def collate_fn(batch):
    """
    自定义的collate_fn函数，用于将数据集中的样本进行批处理
    :param batch: 样本列表
    :return: 经过填充的输入序列张量和标签序列张量
    """
    # print(f'batch-->{batch}')
    # print(f'batch的长度-->{len(batch)}')
    # print(f'batch的第一个样本的长度--》{batch[0].shape}')
    # print(f'batch的第二个样本的长度--》{batch[1].shape}')
    # print(f'batch的第三个样本的长度--》{batch[2].shape}')
    # print(f'batch的第四个样本的长度--》{batch[3].shape}')
    # print(f'*'*80)
    #rnn_utils.pad_sequence：将根据一个batch中，最大句子长度，进行补齐
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)  # 对输入序列进行填充，使其长度一致
    # print(f'input_ids-->{input_ids}')
    # print(f'batch的第一个样本的长度--》{input_ids[0].shape}')
    # print(f'batch的第二个样本的长度--》{input_ids[1].shape}')
    # print(f'batch的第三个样本的长度--》{input_ids[2].shape}')
    # print(f'batch的第四个样本的长度--》{input_ids[3].shape}')
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)  # 对标签序列进行填充，使其长度一致
    # print(f'labels-->{labels}')
    return input_ids, labels  # 返回经过填充的输入序列张量和标签序列张量


def get_dataloader(train_path, valid_path):
    """
    获取训练数据集和验证数据集的DataLoader对象
    :param train_path: 训练数据集路径
    :return: 训练数据集的DataLoader对象和验证数据集的DataLoader对象
    """
    train_dataset, val_dataset = load_dataset(train_path, valid_path)  # 加载训练数据集和验证数据集
    # print(f'train_dataset-->{len(train_dataset)}')
    # print(f'val_dataset-->{len(val_dataset)}')
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params.batch_size,
                                  shuffle=True,
                                  collate_fn=collate_fn, #批处理函数，将某个批次里面的每个样本进行填充处理，保证序列等长
                                  drop_last=True)  # 创建训练数据集的DataLoader对象
    validate_dataloader = DataLoader(val_dataset,
                                     batch_size=params.batch_size,
                                     shuffle=True,
                                     collate_fn=collate_fn,
                                     drop_last=True)  # 创建验证数据集的DataLoader对象
    return train_dataloader, validate_dataloader  # 返回训练数据集的DataLoader对象和验证数据集的DataLoader对象

if __name__ == '__main__':
    train_path = '/root/autodl-tmp/aipro/gpt2-chatbot/data/medical_train.pkl'
    valid_path = '/root/autodl-tmp/aipro/gpt2-chatbot/data/medical_valid.pkl'
    # load_dataset(train_path)
    train_dataloader, validate_dataloader = get_dataloader(train_path, valid_path)
    for input_ids, labels in train_dataloader:
        print('你好')
        print(f'input_ids--->{input_ids.shape}')
        # print(f'input_ids--->{input_ids}')
        print(f'labels--->{labels.shape}')
        print('*'*80)
        break
    #     break