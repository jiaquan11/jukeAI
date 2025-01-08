import pandas as pd
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

'''
文本分类数据集的不均衡问题
需要将数据集中的类别标签进行均衡，
这样才能保证模型训练的效果，不会因为数据集的不均衡而导致模型的偏向
'''
#读取csv文件
csv_file_path = "data/Weibo/validation.csv"
df = pd.read_csv(csv_file_path)

#定义重采样策略
#如果想要过采样，使用RandomOverSampler  由少往多采样
#如果想要欠采样，使用RandomUnderSampler 由多往少采样
#我们在这里使用RandomUnderSampler进行欠采样
#random_state是随机种子，控制随机数生成器的种子
rus = RandomUnderSampler(sampling_strategy="auto", random_state=42)

#将特征和标签分开
X = df[["text"]]
Y = df[["label"]]
print(Y)

#应用重采样
X_resampled, Y_resampled = rus.fit_resample(X, Y)
print(Y_resampled)

#合并特征和标签，创建新的DataFrame
df_resampled = pd.concat([X_resampled, Y_resampled], axis=1)
print(df_resampled)

#保存均衡数据到新的csv文件
df_resampled.to_csv("new_validation.csv", index=False)