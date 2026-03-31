"""
LoRA (Low-Rank Adaptation) 核心实现模块
================================================================
功能说明：
1. 定义LoRA低秩适配层（替代全参数微调）
2. 将LoRA注入预训练模型的指定线性层
3. 提供LoRA权重的保存/加载接口（仅保存增量参数）

设计原则：
✅ 参数高效：仅训练低秩矩阵A/B（原始权重冻结）
✅ 非侵入式：通过forward hook注入，不修改原始模型结构   
✅ 模块化：独立保存/加载LoRA权重，便于复用与共享
⚠️ 注意：本实现为教学示例，生产环境建议使用PEFT库（Hugging Face）
"""

import torch
from torch import optim, nn
from typing import Dict, Any


# ==================== LoRA核心层定义 ====================
class LoRA(nn.Module):
    """
    低秩适配层：通过低秩分解模拟全参数更新 ΔW ≈ B×A
    
    数学原理：
      原始权重更新：W' = W + ΔW
      LoRA近似：ΔW = B × A （其中 rank << min(in_features, out_features)）
      前向计算：output = W·x + B(A(x))
    
    初始化策略（关键！）：
      - A: 高斯初始化（std=0.02）→ 提供初始梯度信号
      - B: 全零初始化 → 保证训练起点 ΔW=0，不破坏预训练知识
    """
    def __init__(self, in_features: int, out_features: int, rank: int):
        super().__init__()
        self.rank = rank  # 低秩维度（典型值：4/8/16，越小参数越少）
        
        # 低秩分解：W (d×k) → A (d×r) + B (r×k)
        # 注意：bias=False（LoRA仅适配权重，偏置通常不微调）
        self.A = nn.Linear(in_features, rank, bias=False)  # 下投影：高维→低维
        self.B = nn.Linear(rank, out_features, bias=False) # 上投影：低维→高维
        
        # === 初始化策略（遵循LoRA原论文）===
        # A矩阵：小方差高斯分布（提供初始学习信号）
        self.A.weight.data.normal_(mean=0.0, std=0.02)  
        # B矩阵：全零初始化（关键！确保初始ΔW=0，保护预训练知识）
        self.B.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向计算：仅计算增量部分 ΔW·x = B(A(x))
        实际使用时：output = original_linear(x) + lora(x)
        """
        return self.B(self.A(x))


# ==================== 模型注入函数 ====================
def apply_lora(model: nn.Module, rank: int = 8) -> None:
    """
    将LoRA层注入预训练模型的指定线性层
    
    工作流程：
    1. 遍历模型所有模块
    2. 识别目标线性层（当前示例：仅方阵权重层）
    3. 为该层附加LoRA子模块
    4. 重写forward方法：原始输出 + LoRA增量
    
    ⚠️ 重要说明：
    - 条件 `module.weight.shape[0] == module.weight.shape[1]` 仅为示例！
      实际应用中通常：
        * 注入所有注意力层（q_proj, k_proj, v_proj, o_proj）
        * 或通过层名白名单指定（如 "qkv" in name）
      方阵条件会漏掉多数关键层（如768→3072的FFN层）
    - 修改原始模型forward：需谨慎处理（见闭包设计）
    """
    for name, module in model.named_modules():
        # === 目标层筛选（示例条件，需按实际调整）===
        # 注意：此条件过于严格！仅匹配输入输出维度相同的线性层（如某些残差连接层）
        # 实战建议：替换为 if "attn" in name and isinstance(module, nn.Linear):
        if isinstance(module, nn.Linear) and module.weight.shape[0] == module.weight.shape[1]:
            # 创建LoRA层（设备与原始模型对齐）
            lora = LoRA(
                in_features=module.weight.shape[1],  # 注意：Linear权重形状为(out, in)
                out_features=module.weight.shape[0],
                rank=rank
            ).to(model.device)
            
            # 将LoRA作为子模块附加到原线性层（便于后续保存/加载）
            setattr(module, "lora", lora)
            
            # 保存原始forward引用（关键！避免递归调用）
            original_forward = module.forward
            
            # === 闭包设计：安全绑定当前层的引用 ===
            # 使用默认参数捕获当前作用域的original_forward和lora（避免Python闭包延迟绑定问题）
            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                """
                重写后的前向逻辑：
                  output = 原始线性层输出 + LoRA增量
                优势：
                  - 无需修改原始权重
                  - 推理时可动态开关LoRA（设lora=None）
                """
                return layer1(x) + layer2(x)
            
            # 替换模块的forward方法（注入LoRA计算）
            module.forward = forward_with_lora
            print(f"✓ LoRA注入: {name} | 原始形状: {list(module.weight.shape)} | 秩: {rank}")


# ==================== 权重加载函数 ====================
def load_lora(model: nn.Module, path: str) -> None:
    """
    从文件加载LoRA权重到已注入LoRA的模型
    
    处理细节：
    1. 自动处理DataParallel保存的"module."前缀
    2. 按模块名精准匹配LoRA子模块
    3. 仅加载LoRA参数，原始模型权重保持冻结
    
    安全机制：
    - 通过hasattr检查确保目标模块已注入LoRA
    - 字典推导式精准提取对应模块的LoRA参数
    """
    # 加载状态字典（自动映射到模型所在设备）
    state_dict = torch.load(path, map_location=model.device)
    
    # 处理DataParallel保存的权重前缀（"module."）
    state_dict = {
        (k[7:] if k.startswith('module.') else k): v 
        for k, v in state_dict.items()
    }
    
    # 遍历模型，为每个含LoRA的模块加载对应权重
    for name, module in model.named_modules():
        if hasattr(module, 'lora'):  # 确保该模块已注入LoRA
            # 提取当前模块对应的LoRA参数（键名匹配："{name}.lora.{weight/bias}"）
            lora_state = {
                k.replace(f'{name}.lora.', ''): v 
                for k, v in state_dict.items() 
                if f'{name}.lora.' in k
            }
            if lora_state:
                module.lora.load_state_dict(lora_state)
                print(f"✓ 加载LoRA权重: {name} | 参数量: {sum(p.numel() for p in lora_state.values())}")


# ==================== 权重保存函数 ====================
def save_lora(model: nn.Module, path: str) -> None:
    """
    仅保存LoRA增量参数（非全模型！）
    
    优势：
    - 文件极小（通常<100MB，vs 全模型GB级）
    - 保护原始模型版权
    - 便于分享/复用适配器
    
    处理逻辑：
    1. 解包FSDP/CompiledModel包装（_orig_mod）
    2. 遍历所有含LoRA的模块
    3. 构建标准化键名（移除DataParallel前缀）
    4. 合并所有LoRA参数为单一状态字典
    """
    # 处理torch.compile()包装的模型（获取原始模型）
    raw_model = getattr(model, '_orig_mod', model)
    
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            # 清理DataParallel前缀（保存时统一格式）
            clean_name = name[7:] if name.startswith("module.") else name
            
            # 提取LoRA子模块状态字典，并重命名键（添加模块路径前缀）
            lora_state = {
                f'{clean_name}.lora.{k}': v 
                for k, v in module.lora.state_dict().items()
            }
            state_dict.update(lora_state)
            print(f"✓ 保存LoRA: {clean_name} | 形状: A{list(module.lora.A.weight.shape)}, B{list(module.lora.B.weight.shape)}")
    
    # 保存为标准PyTorch checkpoint
    torch.save(state_dict, path)
    print(f"\n✨ LoRA权重已保存至: {path} | 总参数量: {sum(p.numel() for p in state_dict.values()):,}")


# ==================== 使用示例（注释形式） ====================
"""
# 训练流程示例
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
apply_lora(model, rank=8)  # 注入LoRA

# 冻结原始权重（仅训练LoRA参数）
for name, param in model.named_parameters():
    if "lora" not in name:
        param.requires_grad = False

optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
# ... 正常训练循环 ...

# 保存仅LoRA权重
save_lora(model, "lora_adapter.pt")

# 推理时加载
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8B")
apply_lora(model, rank=8)
load_lora(model, "lora_adapter.pt")
"""