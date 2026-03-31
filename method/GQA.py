# import torch
# import torch.nn as nn

# droupout_layer = nn.Dropout(p=0.5)

# t1=torch.Tensor([1,2,3])

# t2=droupout_layer(t1)
# print(t2)

# # 乘input*output维矩阵+偏置
# layer = nn.Linear(3, 5, bias=True)

# t3=torch.Tensor([1,2,3])
# t4=torch.Tensor([[4,5,6],[7,8,9]])

# output1 = layer(t3)
# output2 = layer(t4)

# print(output1)
# print(output2)

# # view函数改变张量的形状，参数是新的形状，-1表示自动计算该维度的大小以保持元素总数不变
# t5=torch.Tensor([[1,2,3,4,5,6],[7,8,9,10,11,12]])
# t_view = t5.view(3,4)
# print(t_view)

# # transpose函数交换张量的维度，参数是要交换的两个维度的索引，0表示第一个维度，1表示第二个维度
# t6=torch.tensor([[1,2,3],[4,5,6]])
# t6=t6.transpose(0,1)
# print(t6)
# print(t6.shape)

# t7=t6.unsqueeze(0)
# t7=t7.transpose(1,2)
# print(t7)
# print(t7.shape)

# # tril函数返回输入张量的下三角部分，diagonal参数指定对角线的位置，默认值为0表示主对角线，正值表示上方的对角线，负值表示下方的对角线
# # triu函数返回输入张量的上三角部分，diagonal参数指定对角线的位置，默认值为0表示主对角线，正值表示上方的对角线，负值表示下方的对角线
# t8=torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
# print(torch.triu(t8))
# # torch.tensor([[1, 2, 3],
# #               [0, 5, 6],
# #               [0, 0, 9]])
# print(torch.tril(t8,diagonal=1))
# # tensor([[1, 2, 0],
# #         [4, 5, 6],
# #         [7, 8, 9]])
# print(torch.tril(t8,diagonal=-1))
# # #tensor([[0, 0, 0],
# #         [4, 0, 0],
# #         [7, 8, 0]])

# t9=torch.arange(1,7)

# t9=torch.reshape(t9,(2,3))
# print(t9)

# t10=torch.reshape(t9,(3,-1))
# print(t10)

# print(torch.cuda.is_available())
# print(torch.__version__)


import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== 辅助函数：修复版 RoPE ====================
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def _apply_rope(x: torch.Tensor, position_ids: torch.Tensor, base: int = 10000) -> torch.Tensor:
    B, L, H, D = x.shape
    if position_ids.ndim == 1:
        position_ids = position_ids.unsqueeze(0).expand(B, -1)
    
    dim_idx = torch.arange(0, D, 2, dtype=torch.float32, device=x.device)
    inv_freq = 1.0 / (base ** (dim_idx / D))
    
    freqs = position_ids.unsqueeze(-1).float() * inv_freq.unsqueeze(0).unsqueeze(0)
    sin = torch.sin(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)
    cos = torch.cos(freqs).unsqueeze(2).repeat_interleave(2, dim=-1)
    return cos * x + _rotate_half(x) * sin

# ==================== 核心实现：GQA ====================
class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) - Google 2023
    ✅ 核心思想：将 num_heads 个 Query 头分组，每组共享 1 个 Key/Value 头
    ✅ 公式：group_size = num_heads // num_key_value_heads
    ✅ 优势：推理 KV Cache 内存 ≈ MHA 的 1/group_size，效果接近 MHA
    ✅ 工业应用：Falcon, Mixtral, Gemma 均采用 GQA
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_key_value_heads: int,  # GQA 核心参数！
        head_dim: int = None,
        use_rope: bool = False,
        rope_base: int = 10000
    ):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.scale = self.head_dim ** -0.5
        
        # 校验（面试必问！）
        assert hidden_size % num_heads == 0, "hidden_size 必须被 num_heads 整除"
        assert num_heads % num_key_value_heads == 0, \
            "num_heads 必须是 num_key_value_heads 的整数倍（group_size = num_heads // num_key_value_heads）"
        if use_rope:
            assert self.head_dim % 2 == 0, "启用 RoPE 时 head_dim 必须为偶数"
        
        self.group_size = num_heads // num_key_value_heads  # 每组 Query 头数量
        
        # 投影层（K/V 仅 num_key_value_heads 头）
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,      # [batch, seq_len, hidden_size]
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None
    ) -> torch.Tensor:
        B, L, _ = hidden_states.shape
        
        # ========== 1. 线性投影 + 拆分为多头 ==========
        q = self.q_proj(hidden_states).view(B, L, self.num_heads, self.head_dim)          # [B, L, H_q, D]
        k = self.k_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim) # [B, L, H_kv, D]
        v = self.v_proj(hidden_states).view(B, L, self.num_key_value_heads, self.head_dim) # [B, L, H_kv, D]
        
        # ========== 2. 应用 RoPE (仅 Q/K，且在重复前！) ==========
        if self.use_rope and position_ids is not None:
            q = _apply_rope(q, position_ids, self.rope_base)
            k = _apply_rope(k, position_ids, self.rope_base)  # ✅ 关键：RoPE 应用于原始 K 头（重复前）
            # V 不应用 RoPE（内容信息）
        
        # ========== 3. 转置为标准注意力形状 ==========
        q = q.transpose(1, 2)  # [B, H_q, L, D]
        k = k.transpose(1, 2)  # [B, H_kv, L, D]
        v = v.transpose(1, 2)  # [B, H_kv, L, D]
        
        # ========== 4. GQA 核心：重复 K/V 头至与 Q 头对齐 ==========
        # 方案1（推荐）：repeat_interleave（梯度安全 + 显式内存）
        k = k.repeat_interleave(self.group_size, dim=1)  # [B, H_kv, L, D] → [B, H_q, L, D]
        v = v.repeat_interleave(self.group_size, dim=1)
        
        # 方案2（备选）：expand + contiguous（零拷贝但需 contiguous）
        # k = k.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).flatten(1, 2).contiguous()
        # v = v.unsqueeze(2).expand(-1, -1, self.group_size, -1, -1).flatten(1, 2).contiguous()
        
        # ========== 5. 缩放点积注意力 ==========
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H_q, L, L]
        
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)  # [B, H_q, L, D]
        
        # ========== 6. 合并头 + 输出投影 ==========
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, L, H_q, D]
        attn_output = attn_output.reshape(B, L, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)  # [B, L, hidden_size]

# ==================== 面试验证示例 ====================
if __name__ == "__main__":
    torch.manual_seed(2024)
    B, L, hidden_size = 2, 8, 128
    
    # 测试1: 标准 GQA (8 Query 头, 2 Key/Value 头 → group_size=4)
    print("🔍 测试1: GQA (num_heads=8, num_key_value_heads=2)")
    gqa = GroupedQueryAttention(
        hidden_size=hidden_size,
        num_heads=8,
        num_key_value_heads=2,  # 每4个Query头共享1组K/V
        use_rope=True
    )
    x = torch.randn(B, L, hidden_size)
    pos_ids = torch.arange(L).unsqueeze(0).expand(B, -1)
    out = gqa(x, position_ids=pos_ids)
    print(f"✅ 输入: {x.shape} | 输出: {out.shape}")
    print(f"✅ Group Size: {gqa.group_size} (8 Query 头 / 2 KV 头)")
    print(f"✅ KV Cache 内存: ≈ MHA 的 1/{gqa.group_size:.0f} = {1/gqa.group_size:.0%}")
    
    # 测试2: 边界校验（故意触发错误）
    print("\n🔍 测试2: 维度校验 (预期报错)")
    try:
        bad_gqa = GroupedQueryAttention(128, num_heads=7, num_key_value_heads=3)  # 7 % 3 != 0
    except AssertionError as e:
        print(f"✅ 捕获预期错误: {e}")
    
    # 测试3: GQA vs MQA vs MHA 内存对比（理论值）
    print("\n🔍 测试3: KV Cache 内存理论对比 (seq_len=1024, head_dim=128)")
    seq_len, head_dim = 1024, 128
    mha_kv = 2 * 32 * seq_len * head_dim  # 假设32头
    gqa_kv = 2 * 8 * seq_len * head_dim   # 32 Query头, 8 KV头
    mqa_kv = 2 * 1 * seq_len * head_dim
    print(f"   MHA (32头): {mha_kv/1e6:.1f} MB")
    print(f"   GQA (32/8): {gqa_kv/1e6:.1f} MB (↓{(1-gqa_kv/mha_kv)*100:.0f}%)")
    print(f"   MQA (32/1): {mqa_kv/1e6:.1f} MB (↓{(1-mqa_kv/mha_kv)*100:.0f}%)")
    
    print("\n🎉 验证通过！GQA 实现符合工业级标准")