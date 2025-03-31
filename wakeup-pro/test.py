import argparse
import numpy as np
from pathlib import Path
import torch
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from crnn import WakeWordCRNN
from dataset import WakeWordDataset


def evaluate(args):
    device = torch.device(args.device)

    # 加载模型
    model = WakeWordCRNN().to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 加载数据集
    test_dataset = WakeWordDataset(args.test_dir, augment=False)
    test_loader = test_dataset.get_loader(args.batch_size, shuffle=False)

    # 推理
    all_labels = []
    all_preds = []
    all_paths = []

    with torch.no_grad():
        for features, labels, paths in test_loader:
            features = features.to(device)
            outputs = model(features)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_paths.extend(paths)

    # 生成报告
    print(classification_report(all_labels, all_preds, target_names=['not_wake', 'wake']))

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        all_labels, all_preds,
        display_labels=['Not Wake', 'Wake'],
        cmap='Blues', ax=ax
    )
    plt.savefig(Path(args.save_dir) / "confusion_matrix.png")

    # 保存错误样本
    with open(Path(args.save_dir) / "errors.txt", 'w') as f:
        for path, true, pred in zip(all_paths, all_labels, all_preds):
            if true != pred:
                f.write(f"{path}\tTrue: {true}\tPred: {pred}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", default="dataset/train", required=True)
    parser.add_argument("--model_path", default="checkpoints/best_model.pth", required=True)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_dir", default="./results")
    args = parser.parse_args()

    Path(args.save_dir).mkdir(exist_ok=True)
    evaluate(args)