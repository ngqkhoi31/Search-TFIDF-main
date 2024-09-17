import torch.nn as nn

class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # Định nghĩa các lớp của mô hình, ví dụ:
        self.fc = nn.Linear(256, 1)  # Thay đổi kích thước phù hợp với mô hình của bạn

    def forward(self, x):
        # Định nghĩa forward pass
        out = self.fc(x)
        return out
