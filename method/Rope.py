from typing import Optional

import torch

x = torch.tensor([1,2,3,4,5])
y = torch.tensor([10,20,30,40,50])

condition = x > 3
# where函数根据condition的值选择x或y中的元素,x满足条件则选择x中的元素保留,否则选择y中的元素
result = torch.where(condition, x, y)
print(result)

# 生成一个从0到9的等差数列，步长为2
t = torch.arange(0,10,2)
print(t)
# 生成一个从5到1的等差数列，步长为-1
t2 = torch.arange(5,0,-1)
print(t2)

v1 = torch.tensor([1,2,3])
v2 = torch.tensor([4,5,6])
# outer函数计算两个向量的外积，结果是一个矩阵，其中每个元素是v1中的一个元素与v2中的一个元素的乘积
result = torch.outer(v1,v2)
print(result)

t1 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])
t2 = torch.tensor([[[13,14,15],[16,17,18]],[[19,20,21],[22,23,24]]])
print(t1.shape)
# cat函数将t1和t2在指定维度上进行拼接，dim=0表示在第0维上拼接，结果是一个新的张量，其中t1和t2的元素按照指定的维度进行连接
result = torch.cat((t1,t2),dim=0)
print(result.shape)

t1 = torch.tensor([[1,2],[3,4]])
t2 = t1.unsqueeze(0)
print(t1.shape)
print(t2.shape)





import torch
import torch.nn.functional as F

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    将张量后半部分取负并交换位置，用于高效实现旋转。
    Example: [a, b, c, d] -> [-c, -d, a, b]
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(
    x: torch.Tensor,          # [batch, seq_len, num_heads, head_dim]
    position_ids: torch.Tensor,  # [batch, seq_len] 或 [seq_len]
    base: int = 10000,
    max_seq_len: int = None   # 预留扩展接口（如线性缩放）
) -> torch.Tensor:
    """
    对输入张量应用 Rotary Positional Embedding。
    
    关键原理：
    - 将每个 head_dim 拆分为复数对 (x1, x2)
    - 通过 sin/cos 实现旋转：[x1*cos - x2*sin, x1*sin + x2*cos]
    - 旋转矩阵满足 R_m^T R_n = R_{n-m}，天然编码相对位置
    
    注意：
    1. head_dim 必须为偶数（标准实现要求）
    2. 仅应用于 Query 和 Key，Value 无需处理
    3. position_ids 支持变长序列/滑动窗口等场景
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert x.dtype in [torch.float16, torch.float32, torch.bfloat16], "Unsupported dtype"
    
    # 标准化 position_ids 形状 -> [batch, seq_len]
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # 1. 计算频率基底 inv_freq [head_dim//2]
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (base ** (dim_indices / head_dim))  # [head_dim//2]
    
    # 2. 生成位置频率 [batch, seq_len, head_dim//2]
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    
    # 3. 计算 sin/cos 并扩展维度以匹配输入 [batch, seq_len, 1, head_dim//2]
    sin = torch.sin(freqs).unsqueeze(2)
    cos = torch.cos(freqs).unsqueeze(2)
    
    # 4. 应用旋转（两种等效实现，此处使用 rotate_half 技巧）
    # 方案A（推荐）：利用 rotate_half 避免显式拆分，更高效
    x_rotated = (x * cos) + (rotate_half(x) * sin)
    
    # 方案B（直观但稍慢）：显式拆分计算（注释供理解）
    # x1, x2 = x.chunk(2, dim=-1)
    # x_rotated = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    
    return x_rotated

# ==================== 使用示例 ====================
if __name__ == "__main__":
    # 模拟输入（LLaMA 风格）
    batch, seq_len, heads, head_dim = 2, 8, 4, 64
    q = torch.randn(batch, seq_len, heads, head_dim)
    k = torch.randn(batch, seq_len, heads, head_dim)
    
    # 位置索引（支持非连续：如 [0,1,2,5,6,7,10,11] 用于滑动窗口）
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, -1)  # [2, 8]
    
    # 应用 RoPE
    q_rope = apply_rope(q, position_ids)
    k_rope = apply_rope(k, position_ids)
    
    print("✅ RoPE applied successfully!")
    print(f"Query shape: {q_rope.shape} | Key shape: {k_rope.shape}")
    print(f"First token rotation check (non-zero): {not torch.allclose(q[0,0], q_rope[0,0])}")
    
    # 验证相对位置性质（可选）
    # 内积 <R_m q, R_n k> 应仅依赖 n-m（此处省略完整验证）


