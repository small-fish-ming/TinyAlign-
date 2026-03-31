import os
import sys

# 设置包路径，确保模块导入正常工作
__package__ = "trainer"
# 将父目录添加到系统路径，解决模块导入问题
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse  # 命令行参数解析
import time  # 时间相关操作
import warnings  # 警告控制
import torch  # 深度学习框架
import torch.nn.functional as F  # 神经网络函数
import torch.distributed as dist  # 分布式训练
from contextlib import nullcontext  # 上下文管理器
from torch import optim  # 优化器
from torch.nn.parallel import DistributedDataParallel  # 分布式数据并行
from torch.utils.data import DataLoader, DistributedSampler  # 数据加载
from model.model import MiniMindConfig  # MiniMind模型配置
from dataset.lm_dataset import DPODataset  # DPO数据集
# 训练工具函数
from train.trainer_utils import (
    get_lr, Logger, is_main_process, lm_checkpoint, 
    init_distributed_mode, setup_seed, init_model, SkipBatchSampler
)

# 忽略所有警告（生产环境慎用）
warnings.filterwarnings('ignore')


def logits_to_log_probs(logits, labels):
    """
    将模型输出的logits转换为每个token的对数概率
    
    参数:
    logits: 模型输出的原始分数，形状为(batch_size, seq_len, vocab_size)
    labels: 真实token ID，形状为(batch_size, seq_len)
    
    返回:
    log_probs_per_token: 每个位置上对应真实token的对数概率，形状为(batch_size, seq_len)
    
    原理:
    1. 先对logits进行log_softmax，得到每个位置上所有token的对数概率分布
    2. 使用gather操作根据labels提取对应位置的真实token的对数概率
    """
    # 对logits进行log_softmax，得到每个位置上所有token的对数概率
    # dim=2 表示在vocab_size维度进行softmax
    log_probs = F.log_softmax(logits, dim=2)
    
    # 使用gather从log_probs中提取对应labels位置的对数概率
    # labels.unsqueeze(2) 将labels形状从(batch_size, seq_len)变为(batch_size, seq_len, 1)
    # torch.gather根据这个索引从log_probs中获取对应位置的值
    # squeeze(-1) 移除最后一个维度，恢复为(batch_size, seq_len)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    
    return log_probs_per_token


def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    """
    计算DPO (Direct Preference Optimization) 损失函数
    
    参数:
    ref_log_probs: 参考模型(通常是SFT模型)生成的对数概率，形状为(batch_size, seq_len)
    policy_log_probs: 当前策略模型生成的对数概率，形状为(batch_size, seq_len)
    mask: 注意力掩码，用于忽略padding部分，形状为(batch_size, seq_len)
    beta: 温度参数，控制优化强度，值越大对齐越强但可能损害性能
    
    返回:
    loss: 平均DPO损失值
    
    DPO损失原理:
    1. 首先计算序列级别的平均对数概率（考虑mask）
    2. 将batch分为两半：前半是chosen样本，后半是rejected样本
    3. 计算策略模型和参考模型的对数概率比率差异
    4. 通过sigmoid函数将差异转换为偏好概率，最小化负对数概率
    
    公式:
    loss = -log(sigmoid(beta * (pi_logratios - ref_logratios)))
    其中:
        pi_logratios = log_policy(chosen) - log_policy(rejected)
        ref_logratios = log_ref(chosen) - log_ref(rejected)
    
    注意: 序列级平均处理(而非求和)是为了防止长序列主导梯度更新
    """
    # 计算每个序列的有效长度（非padding部分）
    # clamp_min(1e-8) 防止除以零
    seq_lengths = mask.sum(dim=1, keepdim=True).clamp_min(1e-8)
    
    # 计算序列级别的平均对数概率:
    # 1. 用mask过滤padding部分 (ref_log_probs * mask)
    # 2. 对序列维度求和 (sum(dim=1))
    # 3. 除以有效序列长度，得到平均对数概率
    # 这样处理可以防止长序列主导梯度更新
    ref_log_probs = (ref_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()
    policy_log_probs = (policy_log_probs * mask).sum(dim=1) / seq_lengths.squeeze()

    # 将batch分为两半：前半是chosen样本，后半是rejected样本
    batch_size = ref_log_probs.shape[0]
    # 选择样本的参考模型对数概率
    chosen_ref_log_probs = ref_log_probs[:batch_size // 2]
    # 拒绝样本的参考模型对数概率
    reject_ref_log_probs = ref_log_probs[batch_size // 2:]
    # 选择样本的策略模型对数概率
    chosen_policy_log_probs = policy_log_probs[:batch_size // 2]
    # 拒绝样本的策略模型对数概率
    reject_policy_log_probs = policy_log_probs[batch_size // 2:]

    # 计算策略模型的对数概率比率: log π(y_w|x) - log π(y_l|x)
    pi_logratios = chosen_policy_log_probs - reject_policy_log_probs
    # 计算参考模型的对数概率比率: log π_ref(y_w|x) - log π_ref(y_l|x)
    ref_logratios = chosen_ref_log_probs - reject_ref_log_probs
    # 计算DPO的核心项: (log π(y_w|x) - log π(y_l|x)) - (log π_ref(y_w|x) - log π_ref(y_l|x))
    # 这相当于隐式地添加了KL散度约束
    logits = pi_logratios - ref_logratios
    
    # 应用beta参数缩放，通过logsigmoid计算损失
    # -log(sigmoid(beta * logits)) = -log(1/(1+exp(-beta*logits))) = log(1+exp(-beta*logits))
    loss = -F.logsigmoid(beta * logits)
    
    # 返回平均损失
    return loss.mean()


def train_epoch(epoch, loader, iters, ref_model, lm_config, start_step=0, wandb=None, beta=0.1):
    """
    训练一个epoch
    
    参数:
    epoch: 当前epoch索引
    loader: 数据加载器
    iters: 本epoch的总迭代次数
    ref_model: 参考模型（冻结，不更新）
    lm_config: 语言模型配置
    start_step: 起始step（用于恢复训练）
    wandb: wandb日志记录器
    beta: DPO损失中的温度参数
    """
    start_time = time.time()
    
    # 遍历数据加载器中的每个batch
    for step, batch in enumerate(loader, start=start_step + 1):
        # 将数据移动到指定设备
        x_chosen = batch['x_chosen'].to(args.device)  # 选择的输入序列
        x_rejected = batch['x_rejected'].to(args.device)  # 拒绝的输入序列
        y_chosen = batch['y_chosen'].to(args.device)  # 选择的输出序列
        y_rejected = batch['y_rejected'].to(args.device)  # 拒绝的输出序列
        mask_chosen = batch['mask_chosen'].to(args.device)  # 选择的掩码
        mask_rejected = batch['mask_rejected'].to(args.device)  # 拒绝的掩码
        
        # 将chosen和rejected数据拼接在一起，前半部分是chosen，后半是rejected
        x = torch.cat([x_chosen, x_rejected], dim=0)
        y = torch.cat([y_chosen, y_rejected], dim=0)
        mask = torch.cat([mask_chosen, mask_rejected], dim=0)

        # 计算当前学习率（使用余弦退火或线性预热+衰减）
        lr = get_lr(epoch * iters + step, args.epochs * iters, args.learning_rate)
        # 更新优化器中的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 在混合精度上下文中执行前向传播
        with autocast_ctx:
            # 参考模型推理（不计算梯度）
            with torch.no_grad():
                ref_outputs = ref_model(x)
                ref_logits = ref_outputs.logits
            # 计算参考模型的对数概率
            ref_log_probs = logits_to_log_probs(ref_logits, y)
            
            # 当前策略模型推理
            outputs = model(x)
            logits = outputs.logits
            # 计算策略模型的对数概率
            policy_log_probs = logits_to_log_probs(logits, y)
            
            # 计算DPO损失
            dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
            # 总损失 = DPO损失 + 辅助损失（如MoE路由损失）
            loss = dpo_loss_val + outputs.aux_loss
            # 梯度累积：除以累积步数，保持梯度规模一致
            loss = loss / args.accumulation_steps

        # 混合精度训练：缩放梯度防止下溢
        scaler.scale(loss).backward()

        # 梯度累积：仅当累积到指定步数后才更新参数
        if (step + 1) % args.accumulation_steps == 0:
            # 反缩放梯度，为优化器做准备
            scaler.unscale_(optimizer)
            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            # 优化器步骤（应用梯度更新）
            scaler.step(optimizer)
            # 更新缩放器状态
            scaler.update()
            # 清空梯度
            optimizer.zero_grad(set_to_none=True)

        # 定期记录训练日志
        if step % args.log_interval == 0 or step == iters - 1:
            spend_time = time.time() - start_time
            # 由于之前损失除以了累积步数，这里乘回去得到真实损失值
            current_loss = loss.item() * args.accumulation_steps
            current_dpo_loss = dpo_loss_val.item()
            current_aux_loss = outputs.aux_loss.item()
            current_lr = optimizer.param_groups[-1]['lr']
            # 估计剩余时间（分钟）
            eta_min = spend_time / (step + 1) * iters // 60 - spend_time // 60
            
            # 打印日志
            Logger(f'Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), loss: {current_loss:.4f}, dpo_loss: {current_dpo_loss:.4f}, aux_loss: {current_aux_loss:.4f}, learning_rate: {current_lr:.8f}, epoch_time: {eta_min:.3f}min')
            
            # 如果使用wandb，记录指标
            if wandb: 
                wandb.log({
                    "loss": current_loss, 
                    "dpo_loss": current_dpo_loss, 
                    "aux_loss": current_aux_loss, 
                    "learning_rate": current_lr, 
                    "epoch_time": eta_min
                })

        # 定期保存模型检查点
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            model.eval()  # 切换到评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            # 保存路径
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            # 获取原始模型（如果是DDP包装的）
            raw_model = model.module if isinstance(model, DistributedDataParallel) else model
            # 获取原始模型（如果使用了torch.compile）
            raw_model = getattr(raw_model, '_orig_mod', raw_model)
            # 获取模型状态字典
            state_dict = raw_model.state_dict()
            # 保存为半精度以节省空间
            torch.save({k: v.half().cpu() for k, v in state_dict.items()}, ckp)
            # 保存完整检查点（包括优化器状态等）
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=model, 
                optimizer=optimizer, scaler=scaler, epoch=epoch, 
                step=step, wandb=wandb, save_dir='../checkpoints'
            )
            model.train()  # 切换回训练模式
            del state_dict  # 释放内存

        # 显式删除变量，释放GPU内存
        del x_chosen, x_rejected, y_chosen, y_rejected, mask_chosen, mask_rejected, x, y, mask
        del ref_outputs, ref_logits, ref_log_probs, outputs, logits, policy_log_probs, loss


if __name__ == "__main__":
    # =============== 命令行参数解析 ===============
    parser = argparse.ArgumentParser(description="MiniMind DPO (Direct Preference Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='dpo', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--learning_rate", type=float, default=4e-8, help="初始学习率（建议<=5e-8避免遗忘）")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=100, help="日志打印间隔")
    parser.add_argument("--save_interval", type=int, default=100, help="模型保存间隔")
    parser.add_argument('--hidden_size', default=512, type=int, help="隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="隐藏层数量")
    parser.add_argument('--max_seq_len', default=1024, type=int, help="训练的最大截断长度（中文1token≈1.5~1.7字符）")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument("--data_path", type=str, default="../dataset/dpo.jsonl", help="DPO训练数据路径")
    parser.add_argument('--from_weight', default='full_sft', type=str, help="基于哪个权重训练")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测&续训（0=否，1=是）")
    parser.add_argument('--beta', default=0.1, type=float, help="DPO中的beta参数")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-DPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 1. 初始化环境和随机种子 ==========
    # 初始化分布式训练环境，返回本地rank
    local_rank = init_distributed_mode()
    # 如果已初始化分布式训练，设置对应的GPU设备
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"
    # 设置随机种子（每个进程不同，保证数据shuffle的随机性）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 2. 配置目录、模型参数、检查ckp ==========
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    # 创建模型配置
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)  # 转换为布尔值
    )
    # 如果需要从检查点恢复，加载检查点数据
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 3. 设置混合精度 ==========
    # 确定设备类型（CPU或CUDA）
    device_type = "cuda" if "cuda" in args.device else "cpu"
    # 设置数据类型
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # 设置自动混合精度上下文
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 4. 配置wandb ==========
    wandb = None
    # 仅在主进程上初始化wandb
    if args.use_wandb and is_main_process():
        import swanlab as wandb
        # 获取wandb ID（如果从检查点恢复）
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        # 设置是否恢复运行
        resume = 'must' if wandb_id else None
        # 生成运行名称（包含关键参数）
        wandb_run_name = f"MiniMind-DPO-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LR-{args.learning_rate}"
        # 初始化wandb
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 5. 定义模型和参考模型 ==========
    # 初始化策略模型和tokenizer
    model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)
    
    # 如果启用torch.compile，编译模型提升性能
    if args.use_compile == 1:
        model = torch.compile(model)
        Logger('torch.compile enabled')
    
    # 记录策略模型参数量
    Logger(f'策略模型总参数量：{sum(p.numel() for p in model.parameters()) / 1e6:.3f} M')
    
    # 初始化参考模型（通常是从SFT模型加载）
    ref_model, _ = init_model(lm_config, args.from_weight, device=args.device)
    # 设置为评估模式
    ref_model.eval()
    # 冻结参考模型参数（不计算梯度）
    ref_model.requires_grad_(False)
    # 记录参考模型参数量
    Logger(f'参考模型总参数量：{sum(p.numel() for p in ref_model.parameters()) / 1e6:.3f} M')
    
    # ========== 6. 准备数据集和优化器 ==========
    # 创建DPO数据集
    train_ds = DPODataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    # 创建分布式采样器（如果使用分布式训练）
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None
    # 创建混合精度训练的梯度缩放器
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype == 'float16'))
    # 创建AdamW优化器
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # ========== 7. 从检查点恢复状态 ==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        # 加载模型状态
        model.load_state_dict(ckp_data['model'])
        # 加载优化器状态
        optimizer.load_state_dict(ckp_data['optimizer'])
        # 加载梯度缩放器状态
        scaler.load_state_dict(ckp_data['scaler'])
        # 设置起始epoch和step
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)
    
    # ========== 8. DDP包装模型（分布式训练） ==========
    if dist.is_initialized():
        # 指定不参与DDP同步的参数（如RoPE位置编码）
        model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        # 使用DDP包装模型
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        # 设置分布式采样器的epoch（保证每个epoch数据shuffle不同）
        if train_sampler: 
            train_sampler.set_epoch(epoch)
        
        # 每个epoch使用不同的随机种子
        setup_seed(42 + epoch)
        # 生成随机索引
        indices = torch.randperm(len(train_ds)).tolist()
        
        # 计算需要跳过的step数（用于恢复训练）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        
        # 创建批处理采样器（支持跳过部分数据）
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        
        # 创建数据加载器
        loader = DataLoader(
            train_ds, 
            batch_sampler=batch_sampler,  # 自定义批处理采样器
            num_workers=args.num_workers,  # 数据加载线程数
            pin_memory=True  # 将数据固定在CUDA内存中，加速传输
        )
        
        # 如果需要跳过部分step，记录信息
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            # 训练一个epoch
            train_epoch(epoch, loader, len(loader) + skip, ref_model, lm_config, start_step, wandb, args.beta)
        else:
            train_epoch(epoch, loader, len(loader), ref_model, lm_config, 0, wandb, args.beta)
    
    # ========== 10. 清理分布式进程 ==========
    if dist.is_initialized(): 
        dist.destroy_process_group()