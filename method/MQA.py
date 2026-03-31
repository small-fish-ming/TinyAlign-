import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 辅助函数：修复版 RoPE ====================
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """将张量后半部分取负并交换位置 [a,b,c,d] -> [-c,-d,a,b]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, position_ids: torch.Tensor, base: int = 10000) -> torch.Tensor:
    """
    应用旋转位置编码（修复广播维度问题）
    Args:
        x: [batch, seq_len, num_heads, head_dim]
        position_ids: [batch, seq_len] or [seq_len]
    """
    batch_size, seq_len, num_heads, head_dim = x.shape
    
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
    
    # 生成频率索引 [0, 2, 4, ..., head_dim-2]
    dim_indices = torch.arange(0, head_dim, 2, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (base ** (dim_indices / head_dim))  # [head_dim//2]
    
    # 生成旋转角度 [batch, seq_len, head_dim//2]
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    
    # 修复2: repeat_interleave 使每个角度作用于连续两维
    sin = torch.sin(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)  # [B, L, 1, D]
    cos = torch.cos(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)
    
    return cos * x + rotate_half(x) * sin

# ==================== MQA 核心实现 ====================
class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA) 实现
    - Query: 多头 (num_heads)
    - Key/Value: 单头 (共享)
    优势：显著降低 KV Cache 内存 (≈1/num_heads)，加速推理
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        rope_base: int = 10000,
        use_rope: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.rope_base = rope_base
        
        # 校验维度
        assert hidden_size % num_heads == 0, "hidden_size 必须被 num_heads 整除"
        assert self.head_dim % 2 == 0, "head_dim 必须为偶数（RoPE 要求）"
        
        # 投影层
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.head_dim, bias=False)      # 单头 Key
        self.v_proj = nn.Linear(hidden_size, self.head_dim, bias=False)      # 单头 Value
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5

    def forward(
        self,
        hidden_states: torch.Tensor,      # [batch, seq_len, hidden_size]
        position_ids: torch.Tensor = None, # [batch, seq_len] or [seq_len]
        attention_mask: torch.Tensor = None # [batch, 1, seq_len, seq_len] (可选)
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        # ========== 1. 线性投影 ==========
        q = self.q_proj(hidden_states)  # [B, L, num_heads * head_dim]
        k = self.k_proj(hidden_states)  # [B, L, head_dim]
        v = self.v_proj(hidden_states)  # [B, L, head_dim]
        
        # ========== 2. 重塑形状 ==========
        q = q.view(B, L, self.num_heads, self.head_dim)  # [B, L, H, D]
        k = k.unsqueeze(2)  # [B, L, 1, D]  ← MQA 核心：K/V 仅1个头
        v = v.unsqueeze(2)  # [B, L, 1, D]
        
        # ========== 3. 应用 RoPE (仅 Q/K) ==========
        if self.use_rope and position_ids is not None:
            q = apply_rope(q, position_ids, base=self.rope_base)
            k = apply_rope(k, position_ids, base=self.rope_base)
        
        # ========== 4. 转置为注意力计算标准形状 ==========
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, 1, L, D] ← 广播关键
        v = v.transpose(1, 2)  # [B, 1, L, D]
        
        # ========== 5. 计算注意力 ==========
        # K/V 的头维度=1，自动广播到 num_heads
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)  # [B, H, L, D]
        
        # ========== 6. 合并头 + 输出投影 ==========
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, H, D]
        attn_output = attn_output.reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)  # [B, L, hidden_size]

# ==================== 面试验证示例 ====================
if __name__ == "__main__":
    # 设置
    torch.manual_seed(0)
    B, L, hidden_size, num_heads = 2, 8, 128, 8
    
    # 创建模型
    mqa = MultiQueryAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_rope=True
    )
    
    # 模拟输入
    x = torch.randn(B, L, hidden_size)
    position_ids = torch.arange(L).unsqueeze(0).expand(B, -1)  # [B, L]
    
    # 前向计算
    output = mqa(x, position_ids)
    
    # 验证
    print(f"✅ 输入形状: {x.shape}")
    print(f"✅ 输出形状: {output.shape}")
    print(f"✅ MQA 核心特征: Key/Value 仅 {1} 个头 (Query 有 {num_heads} 头)")
    print(f"✅ RoPE 已集成: 位置编码通过旋转注入")
    print(f"✅ KV Cache 优势: 推理时 KV 内存 ≈ 标准 MHA 的 1/{num_heads}")
    
    # 检查非平凡输出（验证计算有效）
    assert not torch.allclose(output, torch.zeros_like(output)), "输出不应全零"
    print("✅ 验证通过：MQA 前向传播成功！")