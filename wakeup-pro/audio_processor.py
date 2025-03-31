import librosa
import numpy as np
import soundfile as sf


class AudioProcessor:
    @staticmethod
    def load_audio(path, sr=16000):
        """加载音频并统一采样率"""
        try:
            y, orig_sr = sf.read(path)
            if y.ndim > 1:  # 转为单声道
                y = y.mean(axis=1)
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
            return y, sr
        except Exception as e:
            print(f"Error loading {path}: {str(e)}")
            return np.zeros(sr * 1), sr  # 返回1秒静音

    @staticmethod
    def extract_features(y, sr=16000, n_mfcc=13, max_len=100):
        """完整的特征提取流程"""
        # 预加重
        y = librosa.effects.preemphasis(y)

        # VAD语音活性检测
        trimmed, _ = librosa.effects.trim(y, top_db=20)
        if len(trimmed) < int(0.3 * sr):
            trimmed = y  # 保留过短音频

        # 提取MFCC特征
        mfcc = librosa.feature.mfcc(
            y=trimmed,
            sr=sr,
            n_mfcc=n_mfcc,
            n_fft=int(0.025 * sr),
            hop_length=int(0.01 * sr)
        )

        # 计算差分特征
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.vstack([mfcc, delta, delta2])  # (39, T)

        # 标准化
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        # 长度标准化
        if features.shape[1] < max_len:
            # 使用边缘填充替代零填充
            pad_width = max_len - features.shape[1]
            features = np.pad(features, ((0, 0), (0, pad_width)), mode='edge')
        elif features.shape[1] > max_len:
            # 确保有足够的长度进行截取
            start = np.random.randint(0, features.shape[1] - max_len)
            features = features[:, start:start + max_len]
        # 如果正好等于max_len则不做处理

        return features

    @staticmethod
    def augment_audio(y, sr):

        # 限制增强幅度
        y = y * np.random.uniform(0.9, 1.1)  # 缩小音量变化范围

        # 添加适度的噪声
        if np.random.rand() < 0.5:
            noise = np.random.normal(0, 0.002, len(y))
            y += noise

        # 控制音高变化范围
        if np.random.rand() < 0.3:
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=np.random.randint(-1, 2))

        return y