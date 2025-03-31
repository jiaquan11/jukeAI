import argparse
import queue
import threading
import time
import sounddevice as sd
import numpy as np
import torch
from crnn import WakeWordCRNN
from audio_processor import AudioProcessor


class RealTimeDetector:
    def __init__(self, args):
        self.device = torch.device(args.device)
        self.model = WakeWordCRNN().to(self.device)
        self.model.load_state_dict(torch.load(args.model_path, map_location=self.device))
        self.model.eval()

        # 音频参数
        self.sr = 16000
        self.window = 3  # 检测窗口长度（秒）
        self.stride = 1  # 检测间隔（秒）

        # 环形缓冲区
        self.buffer = np.zeros(int(self.window * self.sr), dtype=np.float32)
        self.q = queue.Queue()
        self.stop_event = threading.Event()

        # 检测参数
        self.threshold = args.threshold
        self.min_trigger = 2  # 连续触发次数

    def audio_callback(self, indata, frames, time, status):
        """音频输入回调"""
        self.buffer = np.roll(self.buffer, -frames)
        self.buffer[-frames:] = indata[:, 0]
        self.q.put(self.buffer.copy())

    def predict_worker(self):
        """预测线程"""
        trigger_count = 0
        while not self.stop_event.is_set():
            try:
                audio = self.q.get(timeout=1)
                # 预处理
                features = AudioProcessor.extract_features(
                    audio, self.sr, max_len=100
                )
                tensor = torch.FloatTensor(features).unsqueeze(0).to(self.device)

                # 推理
                with torch.no_grad():
                    output = self.model(tensor)
                    prob = torch.softmax(output, dim=1)[0][1].item()

                # 触发逻辑
                if prob > self.threshold:
                    trigger_count += 1
                    if trigger_count >= self.min_trigger:
                        print(f"\033[92m! 主人我在！ ({prob:.1%})\033[0m")
                        trigger_count = 0
                else:
                    trigger_count = max(0, trigger_count - 1)

                print(f"Current score: {prob:.1%}", end='\r')

            except queue.Empty:
                continue

    def start(self):
        """启动检测"""
        print(f"Starting detection (threshold={self.threshold})...")
        try:
            # 启动预测线程
            predict_thread = threading.Thread(target=self.predict_worker)
            predict_thread.start()

            # 启动音频流
            with sd.InputStream(
                    samplerate=self.sr,
                    channels=1,
                    callback=self.audio_callback,
                    blocksize=int(self.stride * self.sr)
            ):
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop_event.set()
            predict_thread.join()
            print("\nStopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--threshold", type=float, default=0.85)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    detector = RealTimeDetector(args)
    detector.start()