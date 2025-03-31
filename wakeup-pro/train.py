import argparse
import torch
from torch import optim
from pathlib import Path
from tqdm import tqdm
from crnn import WakeWordCRNN
from dataset import WakeWordDataset
from torch import nn


def train(args):
    # 初始化设备
    device = torch.device(args.device)
    print(device)

    # 加载数据集
    train_dataset = WakeWordDataset(args.train_dir, augment=True)
    valid_dataset = WakeWordDataset(args.valid_dir, augment=False)

    train_loader = train_dataset.get_loader(args.batch_size, num_workers=4)
    valid_loader = valid_dataset.get_loader(args.batch_size, shuffle=False)

    # 初始化模型
    model = WakeWordCRNN(input_dim=39).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.3
    )
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, args.pos_weight]).to(device))

    # 在训练循环开始前添加
    print("正在验证数据集...")
    for f, l, p in tqdm(train_loader):
        if f.shape[-1] != 100:  # 检查特征长度是否符合预期
            print(f"发现异常特征：{p}，形状：{f.shape}")
            raise ValueError("特征长度不一致，请检查音频预处理逻辑")

    # 训练循环
    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        # 训练阶段
        for features, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for features, labels, _ in valid_loader:
                features = features.to(device)
                labels = labels.to(device)

                outputs = model(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        # 计算指标
        train_acc = 100. * correct / total
        val_acc = 100. * val_correct / val_total
        scheduler.step(val_acc)

        # 保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Path(args.save_dir) / "best_model.pth")

        # 打印日志
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss / len(valid_loader):.4f} | Acc: {val_acc:.2f}%")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", default="dataset/split_dataset/train")
    parser.add_argument("--valid_dir", default="dataset/split_dataset/val")
    parser.add_argument("--save_dir", default="./checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--pos_weight", type=float, default=3.0)
    args = parser.parse_args()

    # 创建保存目录
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    train(args)