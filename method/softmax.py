import torch


def softmax_stable(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    数值稳定的 Softmax 实现（面试标准答案）
    核心技巧：x_max = max(x) → exp(x - x_max) 避免溢出
    """
    # 关键1：沿指定维度取最大值（保持维度便于广播）
    x_max = x.max(dim=dim, keepdim=True).values  # [B, L, 1] 若 dim=-1
    
    # 关键2：减最大值（数值稳定核心！）
    x_shifted = x - x_max
    
    # 关键3：指数 + 归一化
    exp_x = torch.exp(x_shifted)
    sum_exp = exp_x.sum(dim=dim, keepdim=True)
    
    return exp_x / sum_exp