#-*- coding: utf-8 -*-
import torch

class ParameterConfig():
    def __init__(self):
        # 判断是否使用GPU（1.电脑里必须有显卡；2.必须安装cuda版本的pytorch）
        # 下载cuda版本的pytorch链接：https://pytorch.org/get-started/previous-versions/
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 词典路径：在vocab文件夹里面
        self.vocab_path = '/root/autodl-tmp/aipro/gpt2-chatbot/vocab/vocab.txt'
        # 训练文件路径
        self.train_path = '/root/autodl-tmp/aipro/gpt2-chatbot/data/medical_train.pkl'
        # 验证数据文件路径
        self.valid_path = '/root/autodl-tmp/aipro/gpt2-chatbot/data/medical_valid.pkl'
        # 模型配置文件
        self.config_json = '/root/autodl-tmp/aipro/gpt2-chatbot/config/config.json'
        # 模型保存路径
        self.save_model_path = '/root/autodl-tmp/aipro/gpt2-chatbot/save_model1'
        # 如果你有预训练模型就写上路径（我们本次没有直接运用GPT2它预训练好的模型，而是仅只用了该模型的框架）
        self.pretrained_model = ''
        # 保存对话语料
        self.save_samples_path = 'sample'
        # 忽略一些字符：句子需要长度补齐，针对补的部分，没有意义，所以一般不进行梯度更新
        self.ignore_index = -100
        # 历史对话句子的长度
        self.max_history_len = 3# "dialogue history的最大长度"
        # 每一个完整对话的句子最大长度
        self.max_len = 300  # '每个utterance的最大长度,超过指定长度则进行截断,默认25'
        self.repetition_penalty = 10.0 # "重复惩罚参数，若生成的对话重复性较高，可适当提高该参数"
        self.topk = 4 #'最高k选1。默认8'
        self.batch_size = 8 #一个批次几个样本
        self.epochs = 4 # 训练几轮
        self.loss_step = 1 # 多少步汇报一次loss
        self.lr = 2.6e-5
        # eps，为了增加数值计算的稳定性而加到分母里的项，其为了防止在实现中除以零
        self.eps = 1.0e-09
        self.max_grad_norm = 2.0
        self.gradient_accumulation_steps = 4
        # 默认.warmup_steps = 4000
        self.warmup_steps = 100 # 使用Warmup预热学习率的方式,即先用最初的小学习率训练，然后每个step增大一点点，直到达到最初设置的比较大的学习率时（注：此时预热学习率完成），采用最初设置的学习率进行训练（注：预热学习率完成后的训练过程，学习率是衰减的），有助于使模型收敛速度变快，效果更佳。

if __name__ == '__main__':
    pc = ParameterConfig()
    print(pc.train_path)
    print(pc.device)
    print(torch.cuda.device_count())