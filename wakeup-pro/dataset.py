import os
import torch
from torch.utils.data import Dataset
from audio_processor import AudioProcessor


class WakeWordDataset(Dataset):
    def __init__(self, data_dir, sr=16000, n_mfcc=13, max_len=100, augment=True):
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.max_len = max_len
        self.augment = augment
        self.samples = []

        # 遍历数据目录
        for label, cls in enumerate(['not_wake', 'wake']):
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.exists(cls_dir):
                continue

            for fname in os.listdir(cls_dir):
                path = os.path.join(cls_dir, fname)
                self.samples.append((path, label))

        # 平衡类别
        wake_count = sum(1 for _, label in self.samples if label == 1)
        not_wake_count = len(self.samples) - wake_count
        self.weights = [
            1 / wake_count if label == 1 else 1 / not_wake_count
            for _, label in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """ 获取单个样本，包含错误处理和样本过滤 """
        max_retry = 3  # 最大重试次数
        for _ in range(max_retry):
            try:
                path, label = self.samples[idx]

                # 加载音频
                y, sr = AudioProcessor.load_audio(path, self.sr)

                # 校验音频长度
                if len(y) < 0.3 * sr:  # 短于0.3秒视为无效
                    raise ValueError(f"Audio too short: {len(y) / sr:.2f}s < 0.3s")

                # 数据增强（仅对正样本）
                if self.augment and label == 1:
                    y = AudioProcessor.augment_audio(y, sr)

                # 特征提取
                features = AudioProcessor.extract_features(
                    y, sr, self.n_mfcc, self.max_len
                )

                # 最终维度校验
                if features.shape[1] != self.max_len:
                    raise ValueError(
                        f"Feature length mismatch: {features.shape[1]} vs {self.max_len}"
                    )

                return (
                    torch.FloatTensor(features),  # (MFCC, Time)
                    torch.tensor(label, dtype=torch.long),
                    path  # 保留路径用于调试
                )

            except Exception as e:
                print(f"Error processing {path} (attempt {_ + 1}/{max_retry}): {str(e)}")
                # 选择下一个样本
                idx = (idx + 1) % len(self)

        # 重试多次失败后返回空数据
        print(f"Failed after {max_retry} retries, returning zero data")
        return (
            torch.zeros((self.n_mfcc * 3, self.max_len)),  # 39x100
            torch.tensor(-1, dtype=torch.long),
            "invalid_sample"
        )

    def get_loader(self, batch_size=32, shuffle=True, num_workers=4):
        """获取DataLoader"""
        sampler = torch.utils.data.WeightedRandomSampler(self.weights, len(self.weights)) if shuffle else None
        return torch.utils.data.DataLoader(
            self,
            batch_size=batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=num_workers
        )