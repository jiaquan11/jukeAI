import os
import shutil
from sklearn.model_selection import train_test_split

# 数据集根目录
dataset_root = 'dataset/train'

# 新创建的目录用于存放划分后的数据集
new_dataset_root = 'dataset/split_dataset'

# 划分比例
train_ratio = 0.8

# 创建新目录下的train和val目录
os.makedirs(os.path.join(new_dataset_root, 'train', 'wake'), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, 'train', 'not_wake'), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, 'val', 'wake'), exist_ok=True)
os.makedirs(os.path.join(new_dataset_root, 'val', 'not_wake'), exist_ok=True)


# 划分数据集
def split_dataset(class_dir):
    # 获取所有.wav文件的完整路径
    files = [os.path.join(dataset_root, class_dir, f) for f in os.listdir(os.path.join(dataset_root, class_dir)) if
             f.endswith('.wav')]

    # 划分训练集和验证集
    train_files, val_files = train_test_split(files, train_size=train_ratio, random_state=42)

    # 复制文件到新目录下的train和val文件夹
    for file in train_files:
        shutil.copy(file, os.path.join(new_dataset_root, 'train', class_dir))

    for file in val_files:
        shutil.copy(file, os.path.join(new_dataset_root, 'val', class_dir))


# 对wake和notwake两个类别分别进行划分
split_dataset('wake')
split_dataset('not_wake')

print("数据集划分完成！")
