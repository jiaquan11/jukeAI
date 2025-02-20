'''
1、. 匹配任意除换行符“\n”外的字符；
2、* 前面的字符出现0次或以上；
3、? 表示前边字符的0次或1次重复
4、.* 贪婪，匹配从.*前面为开始到后面为结束的所有内容
5、*? 非贪婪，遇到开始和结束就进行截取，因此截取多次符合的结果，中间没有字符也会被截取。
6. (.*?) 非贪婪，与上面一样，只是与上面的相比多了一个括号，只保留括号的内容
'''
import re
str1 = 'aabbabaabbaa'
#. 匹配任意除换行符“\n”外的字符；
print(re.findall(r'a.b', str1))
# * 前面的字符出现0次或以上
print(re.findall(r'a*b', str1))
# ? 表示前边字符的0次或1次重复
print(re.findall(r'a?b', str1))
# # .* 贪婪，匹配从.*前面为开始到后面为结束的所有内容
print(re.findall(r'a.*b',str1))
# # *? 非贪婪，遇到开始和结束就进行截取，因此截取多次符合的结果，中间没有字符也会被截取
print(re.findall(r'a*?b',str1))
# # (.*?) 非贪婪，与上面一样，只是与上面的相比多了一个括号，只保留括号的内容
print(re.findall(r'a(.*?)b',str1))
# 如果response代表模型输出结果：
response = '```json{"subject": "夜半歌声"}、{"subject": "夜半歌声"}``````json{"subject1": "夜半您"}```'
if '```json' in response:
    res = re.findall(r'```json(.*?)```', response)
    print(res)
    if len(res) and res[0]:
        response = res[0]
    print(response)
    str1 = response.replace('、', ',')
    print(str1)

# import json
# # #
# list1= [{"name": "张三"}, {"age":18}]
# print(type(list1))
# # #
# str1 = json.dumps(list1, ensure_ascii=False)
# print(str1)
# print(type(str1))
# # # print(type(str1))
# # #
# a = json.loads(str1)
# print(a)
# print(type(a))
#
# #
# # dict1 = {"信息":{"name": "张三"},"班级":{"class":9}}
# # print(type(dict1))
# # a = json.dumps(dict1, ensure_ascii=False)
# # print(a)
# # print(type(a))
# # b = json.loads(a)
# # print(b)
# # print(b["信息"])

# from rich import print
# # rich包：可以对输出进行美化
#
# pre_history = [
#     (
#         f'现在你是一个文本分类器，你需要按照要求将我给你的句子分类到：类别中。',
#         f'好的。'
#     )
# ]
# # print(pre_history)
# print(f'>>> [bold bright_red]sentence: {pre_history}')
#

# def fun(a, b):
#     return a+b
# # print(fun(2, 3))
# print(fun('a', 'b'))


# def fun(a: int, b: int)-> int:
#     return a+b
#
# print(fun(1, 2))
