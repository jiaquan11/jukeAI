import sounddevice as sd
import numpy as np
from pathlib import Path
import time
from datetime import datetime
from scipy.io.wavfile import write

class AudioCollector:
    def __init__(self, root_dir="dataset"):
        self.root_dir = Path(root_dir)
        self.sample_rate = 16000
        self.duration = 3  # 每个样本的时长（秒）

        # 创建目录结构
        (self.root_dir / "train/wake").mkdir(parents=True, exist_ok=True)
        (self.root_dir / "train/not_wake").mkdir(parents=True, exist_ok=True)

    def _record(self, save_path):
        """执行单次录音"""
        print("开始录音...（说话！）")
        audio = sd.rec(int(self.duration * self.sample_rate),
                       samplerate=self.sample_rate,
                       channels=1,
                       blocking=True)
        sd.wait()
        # 保存为16bit PCM格式
        write(save_path, self.sample_rate, (audio * 32767).astype(np.int16))
        print(f"保存到 {save_path}")

    def collect_wake_word(self, count=100):
        """采集唤醒词样本"""
        for i in range(count):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.root_dir / f"train/wake/{timestamp}_{i}.wav"
            self._record(save_path)
            time.sleep(1)  # 采集间隔

    def collect_background(self, count=300):
        """采集背景音样本"""
        input("请保持环境安静，准备采集背景音...（按回车开始）")
        for i in range(count):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = self.root_dir / f"train/not_wake/{timestamp}_{i}.wav"
            self._record(save_path)
            time.sleep(0.3)


if __name__ == "__main__":
    collector = AudioCollector()

    print("1. 采集唤醒词（需要说唤醒词）")
    collector.collect_wake_word(count=50)  # 采集50个正样本

    print("\n2. 采集背景音（保持安静或日常环境声）")
    collector.collect_background(count=150)