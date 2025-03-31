
import torch.nn as nn

class WakeWordCRNN(nn.Module):

    def __init__(self, input_dim=39):
        super().__init__()
        # 改进的CNN部分
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(32),
            nn.GELU(),  # 改用GELU激活
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d((2, 2)),

            nn.AdaptiveAvgPool2d((None, 25))  # 动态调整时间维度
        )

        # 改进的RNN部分
        self.gru = nn.GRU(
            input_size=64 * (input_dim // 4),
            hidden_size=128,
            bidirectional=True,
            num_layers=2,  # 增加层数
            dropout=0.3
        )

        # 改进分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.cnn(x)  # (B, C, F, T)

        # 调整维度
        B, C, F, T = x.size()
        x = x.permute(0, 3, 1, 2)  # (B, T, C, F)
        x = x.reshape(B, T, -1)  # (B, T, C*F)

        # 使用全部时间步输出
        x, _ = self.gru(x)  # (B, T, 256)
        x = x.mean(dim=1)  # 使用平均池化代替最后时间步

        return self.classifier(x)