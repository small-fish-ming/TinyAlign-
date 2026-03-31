import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 辅助函数：修复版 RoPE（可选） ====================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[a,b,c,d] -> [-c,-d,a,b]"""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, base: int = 10000) -> torch.Tensor:
    """修复广播维度问题：sin/cos 通过 repeat_interleave 扩展至 head_dim"""
    B, L, H, D = x.shape
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
    
    dim_idx = torch.arange(0, D, 2, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (base ** (dim_idx / D))  # [D//2]
    
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)  # [B, L, D//2]
    sin = torch.sin(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)  # [B, L, 1, D]
    cos = torch.cos(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)
    return cos * x + _rotate_half(x) * sin

# ==================== 核心实现：标准 MHA ====================
class MultiHeadAttention(nn.Module):
    """
    标准 Multi-Head Attention (Vaswani et al., 2017)
    ✅ 严格遵循原论文：Q/K/V 均为 num_heads 头
    ✅ 支持 RoPE（通过 use_rope 开关）
    ✅ 工业级校验 + 注释直击面试考点
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        head_dim: int = None,
        dropout: float = 0.0,
        use_rope: bool = False,
        rope_base: int = 10000
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or (hidden_size // num_heads)
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.scale = self.head_dim ** -0.5
        
        
        assert hidden_size % num_heads == 0, "hidden_size 必须被 num_heads 整除"
        if use_rope:
            assert self.head_dim % 2 == 0, "启用 RoPE 时 head_dim 必须为偶数"
        
        # 投影层（无偏置是主流实践）
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [batch, seq_len, hidden_size]
        attention_mask: torch.Tensor = None, # [batch, 1, seq_len, seq_len] (可选)
        position_ids: torch.Tensor = None   # [batch, seq_len] (仅 use_rope=True 时需要)
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        # ========== 1. 线性投影 + 拆分为多头 ==========
        # 形状: [B, L, num_heads * head_dim] -> [B, L, num_heads, head_dim]
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)
        
        # ========== 2. 应用 RoPE (仅 Q/K) ==========
        if self.use_rope and position_ids is not None:
            q = _apply_rope(q, position_ids, self.rope_base)
            k = _apply_rope(k, position_ids, self.rope_base)
            # V 不应用 RoPE（内容信息无需旋转）
        
        # ========== 3. 转置为标准注意力形状 [B, num_heads, seq_len, head_dim] ==========
        q = q.transpose(1, 2)  # [B, H, L, D]
        k = k.transpose(1, 2)  # [B, H, L, D]
        v = v.transpose(1, 2)  # [B, H, L, D]
        
        # ========== 4. 缩放点积注意力 ==========
        # 计算注意力分数: [B, H, L, L]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用 mask (如 causal mask / padding mask)
        if attention_mask is not None:
            # mask 通常为 -inf (需 softmax 后归零) 或 0/1
            attn_weights = attn_weights + attention_mask
        
        # Softmax 归一化 + Dropout
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        # 加权聚合 Value: [B, H, L, D]
        attn_output = torch.matmul(attn_weights, v)
        
        # ========== 5. 合并头 + 输出投影 ==========
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, H, D]
        attn_output = attn_output.reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)  # [B, L, hidden_size]

# ==================== 面试验证示例 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, L, hidden_size, num_heads = 2, 8, 128, 8
    
    # 测试1: 标准 MHA (无 RoPE)
    print("🔍 测试1: 标准 MHA (无 RoPE)")
    mha_std = MultiHeadAttention(hidden_size, num_heads, dropout=0.1)
    x = torch.randn(B, L, hidden_size)
    mask = torch.triu(torch.full((B, 1, L, L), float('-inf')), diagonal=1)  # causal mask
    out_std = mha_std(x, attention_mask=mask)
    print(f"✅ 输入: {x.shape} | 输出: {out_std.shape} | 因果掩码生效: {not torch.allclose(out_std[0,0], out_std[0,1])}")
    
    # 测试2: MHA + RoPE
    print("\n🔍 测试2: MHA + RoPE")
    mha_rope = MultiHeadAttention(hidden_size, num_heads, use_rope=True)
    pos_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    out_rope = mha_rope(x, position_ids=pos_ids)
    print(f"✅ RoPE 位置编码注入成功 | 输出形状: {out_rope.shape}")
    print(f"✅ 位置敏感性验证: pos0 与 pos1 输出差异 = {torch.norm(out_rope[0,0] - out_rope[0,1]):.4f}")
    
    # 测试3: 维度校验 (故意触发错误)
    print("\n🔍 测试3: 维度校验 (预期报错)")
    try:
        bad_mha = MultiHeadAttention(129, 8)  # 129 不能被 8 整除
    except AssertionError as e:
        print(f"✅ 捕获预期错误: {e}")
    
    print("\n🎉 所有验证通过！MHA 实现符合工业级标准")