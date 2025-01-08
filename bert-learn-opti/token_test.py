from transformers import AutoTokenizer, BertTokenizer

#从模型中加载字典和分词器
token = BertTokenizer.from_pretrained(r"D:\study\computerStudy\personcode\jukeAI\bert-learn-opti\model\bert-base-chinese\models--bert-base-chinese\snapshots\c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f")
print(token)

sents = ["价格在这个地段属于适中, 附近有早餐店,小饭店, 比较方便,无早也无所",
         "房间不错,只是上网速度慢得无法忍受,打开一个网页要等半小时,连邮件都无法收。另前台工作人员服务态度是很好，只是效率有得改善。"]
#批量编码句子
out = token.batch_encode_plus(
    batch_text_or_text_pairs=[sents[0], sents[1]],
    add_special_tokens=True,
    #当句子长度超过max_length时，截断
    truncation=True,
    max_length=35,
    #一律补0到max_length长度
    padding="max_length",
    #可取值为tf,pt,np,默认为List
    return_tensors=None,
    #返回attention_mask
    return_attention_mask=True,#返回attention_mask，用于区分padding和非padding
    return_token_type_ids=True,
    return_special_tokens_mask=True,
    #返回length长度
    return_length=True,
)

#input_ids:就是位置索引，每个词对应的位置索引
#token_type_ids第一个句子和特殊符号的位置是0，第二个句子的位置是1  ()只针对上下文编码
#special_tokens_mask:特殊符号的位置是1，其余是0
#print(out)
for k, v in out.items():
    print(k, ":", v)

#解码文本数据
print(token.decode(out['input_ids'][0]), token.decode(out['input_ids'][1]))
