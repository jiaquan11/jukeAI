import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AdamW
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F


# ========== 配置参数 ==========
class Config:
    # 模型设置
    teacher_model_name = "Qwen/Qwen-7B"
    student_model_name = "Qwen/Qwen-1.8B"

    # 训练参数
    batch_size = 16
    num_epochs = 3
    learning_rate = 2e-5
    max_seq_length = 512
    temperature = 5.0
    alpha = 0.7  # 蒸馏损失权重

    # 设备设置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    grad_accum_steps = 4  # 梯度累积步数


config = Config()


# ========== 数据加载 ==========
class DistillationDataset(Dataset):
    def __init__(self, tokenizer, sample_texts):
        self.tokenizer = tokenizer
        self.examples = []

        # 示例数据（实际需替换为真实数据集）
        sample_texts = [
            "人工智能的核心理念是",
            "大语言模型蒸馏的关键在于",
            "深度学习模型的压缩方法包括"
        ]

        for text in sample_texts:
            encoding = tokenizer(
                text,
                max_length=config.max_seq_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            self.examples.append(encoding)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return {
            "input_ids": self.examples[idx]["input_ids"].squeeze(),
            "attention_mask": self.examples[idx]["attention_mask"].squeeze()
        }


# ========== 模型初始化 ==========
def load_models():
    # 加载教师模型（冻结参数）
    teacher = AutoModelForCausalLM.from_pretrained(
        config.teacher_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).eval()

    # 加载学生模型
    student = AutoModelForCausalLM.from_pretrained(
        config.student_model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16
    ).train()

    return teacher, student


# ========== 蒸馏损失函数 ==========
class DistillationLoss:
    @staticmethod
    def calculate(
            teacher_logits,  # 教师模型logits [batch, seq_len, vocab]
            student_logits,  # 学生模型logits [batch, seq_len, vocab]
            temperature=config.temperature,
            alpha=config.alpha
    ):
        # 软目标蒸馏损失
        soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        soft_student = F.log_softmax(student_logits / temperature, dim=-1)

        kl_loss = F.kl_div(
            soft_student,
            soft_teacher,
            reduction="batchmean",
            log_target=False
        ) * (temperature ** 2)

        # 学生自训练损失（交叉熵）
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = teacher_logits.argmax(-1)[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        return alpha * kl_loss + (1 - alpha) * ce_loss


# ========== 训练流程 ==========
def train():
    # 初始化组件
    tokenizer = AutoTokenizer.from_pretrained(config.teacher_model_name)
    teacher, student = load_models()

    # 数据集示例
    dataset = DistillationDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=config.batch_size)

    # 优化器设置
    optimizer = AdamW(student.parameters(), lr=config.learning_rate)

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler()

    # 训练循环
    step_count = 0
    student.to(config.device)

    for epoch in range(config.num_epochs):
        for batch_idx, batch in enumerate(dataloader):
            inputs = {k: v.to(config.device) for k, v in batch.items()}

            # 教师模型前向（不计算梯度）
            with torch.no_grad(), torch.cuda.amp.autocast():
                teacher_outputs = teacher(**inputs)

            # 学生模型前向
            with torch.cuda.amp.autocast():
                student_outputs = student(**inputs)
                loss = DistillationLoss.calculate(
                    teacher_outputs.logits,
                    student_outputs.logits
                )

            # 反向传播（带梯度累积）
            scaler.scale(loss / config.grad_accum_steps).backward()

            if (batch_idx + 1) % config.grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

                # 参数更新
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                step_count += 1

                # 学习率调整（示例）
                lr = config.learning_rate * min(step_count ** -0.5, step_count * (300 ** -1.5))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # 打印训练信息
                if step_count % 10 == 0:
                    print(f"Epoch {epoch + 1} | Step {step_count} | Loss: {loss.item():.4f}")

    # 保存蒸馏后的模型
    student.save_pretrained("./distilled_qwen")
    tokenizer.save_pretrained("./distilled_qwen")


if __name__ == "__main__":
    train()