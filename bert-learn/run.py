import torch
from net import Model
from transformers import BertTokenizer

#模型使用接口(主观评估)

#定义设备信息
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#从模型中加载分词器
token = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\bert-learn\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
names = ["负向评价", "正向评价"]

#加载模型
model = Model().to(DEVICE)

def collate_fn(data):
    '''
    数据集的处理函数
    '''
    #获取数据集中的文本和标签
    sents = [] #由控制台输入的文本
    sents.append(data)
    #对文本进行编码
    data = token.batch_encode_plus(
        batch_text_or_text_pairs=sents,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt",
        return_length=True,
    )
    input_ids = data["input_ids"]
    attention_mask = data["attention_mask"]
    token_type_ids = data["token_type_ids"]
    return input_ids, attention_mask, token_type_ids

#测试函数
#输入测试数据，与已训练好的模型进行对比，输出模型判定结果
#需要加载训练好的模型参数
def test():
    #加载训练好的参数
    model.load_state_dict(torch.load("params/0_bert.pth", map_location=DEVICE))
    #开启测试评估模式
    model.eval()

    while True:
        data = input("请输入测试数据(输入'q'退出): ")
        if data == "q":
            print("退出测试")
            break
        input_ids, attention_mask, token_type_ids = collate_fn(data) #对输入文本进行分词编码
        #将数据传入设备中
        input_ids, attention_mask, token_type_ids = input_ids.to(DEVICE), attention_mask.to(DEVICE), token_type_ids.to(DEVICE)
        #将数据输入到模型，得到输出
        with torch.no_grad():
            out = model(input_ids, attention_mask, token_type_ids)
            out = out.argmax(dim=1)
            print("模型判定: ", names[out], "\n")

if __name__ == "__main__":
    test()



