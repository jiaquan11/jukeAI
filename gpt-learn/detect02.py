#定制化生成内容
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")
model = AutoModelForCausalLM.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\gpt-learn\model\models--uer--gpt2-chinese-cluecorpussmall\snapshots\c2c0249d8a2731f269414cc3b22dff021f8e07a3")

#加载我们自己训练的权重(中文古诗词)
model.load_state_dict(torch.load("params/net.pt"))

#定义函数，用于生成5言绝句 text是提示词，row是生成文本的行数，col是每行的字符数。
def generate(text,row,col):
    #定义一个内部递归函数，用于生成文本
    def generate_loop(data):
        #禁用梯度计算
        with torch.no_grad():
            #使用data中的字典的数据作为模型的输入，并获取输出
            out = model(**data)
        #获取最后一个字(logits未归一化的概率输出)
        out = out["logits"]
        #选择每个序列的最后一个logits，对应于下一个词的预测
        out = out[:,-1]

        #找到概率排名前50的值，以此为分界线，小于该值的全部舍去
        topk_value = torch.topk(out,50).values
        #获取每个输出序列中前50个最大的logits（为保持原维度不变，需要对结果增加一个维度，因为索引操作会降维）
        topk_value = topk_value[:,-1].unsqueeze(dim=1)
        #将所有小于第50大的值的logits设置为负无穷大，减少低概率词被选中的可能性
        out = out.masked_fill(out< topk_value,-float("inf"))

        #将特殊符号的logits值设置为负无穷，防止模型生成这些符号。
        for i in ",.()《》[]「」{}（）":
            out[:,tokenizer.get_vocab()[i]] = -float("inf")
        for i in ['[UNK]','[PAD]']:
            out[:,tokenizer.get_vocab()[i]] = -float("inf")

        #根据概率采样，无放回，避免生成重复的内容
        out = out.softmax(dim=1)
        #从概率分布中进行随机采样，选择下一个词的ID
        out = out.multinomial(num_samples=1)

        #强制添加标点符号
        #计算当前生成的文本长度与预期的长度的比例
        c = data["input_ids"].shape[1] / (col+1)

        #如果当前的长度是预期长度的整数倍，则添加标点符号
        if c %1 ==0:
            if c%2==0:
                #在偶数位添加句号
                out[:,0] = tokenizer.get_vocab()["."]
            else:
                #在奇数位添加逗号
                out[:,0] = tokenizer.get_vocab()[","]

        #将生成的新词ID添加到输入序列的末尾
        data["input_ids"] = torch.cat([data["input_ids"],out],dim=1)
        #更新注意力掩码，标记所有有效位置
        data["attention_mask"] = torch.ones_like(data["input_ids"])
        #更新token的ID类型，通常在BERTm模型中使用，但是在GPT中是不用的
        data["token_type_ids"] = torch.ones_like(data["input_ids"])
        #更新标签，这里将输入ID复制到标签中，在语言生成模型中通常用于预测下一个词
        data["labels"] = data["input_ids"].clone()

        #检查生成的文本长度是否达到或者超过指定的函数和列数
        if data["input_ids"].shape[1] >= row*col + row+1:
            #如果达到长度要求，则返回最终的data字典
            return data
        #如果长度未达到要求，递归调用generate_loop函数继续生成文本
        return generate_loop(data)

    #生成3首诗词
    #使用tokenizer对输入的文本进行编码，并重复3次生成3个样本
    data = tokenizer.batch_encode_plus([text]*3,return_tensors="pt")
    #移除编码后的序列中的最后一个token（结束符号）
    data["input_ids"] = data["input_ids"][:,:-1]
    #创建一个与input_ids形状相同的全1张量，用于注意力掩码
    data["attention_mask"] = torch.ones_like(data["input_ids"])
    #创建一个与input_ids形状相同的全0张量，用于token的ID类型
    data["token_type_ids"] = torch.zeros_like(data["input_ids"])
    #复制input到labels，用于模型目标
    data["labels"] = data["input_ids"].clone()

    #调用generate_loop函数开始生成文本
    data = generate_loop(data)

    #遍历生成的3个样本
    for i in range(3):
        print(i,tokenizer.decode(data["input_ids"][i]))

if __name__ == '__main__':
    generate("白",row=4,col=5)