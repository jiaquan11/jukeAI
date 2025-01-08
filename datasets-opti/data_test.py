import pandas as pd
from sympy.integrals.meijerint_doc import category

#数据集分布不均衡
df = pd.read_csv("data/Weibo/train.csv")
#统计每个类别的数据量
category_counts = df["label"].value_counts()

#统计每个类别的比值
total_data = len(df)
category_ratios = (category_counts / total_data) * 100
print(category_counts)
print(category_ratios)