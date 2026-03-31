"""
MiniMind PPO (Proximal Policy Optimization) 强化学习训练主脚本
实现完整的RLHF（人类反馈强化学习）训练流程，包含：
- Actor-Critic架构：策略网络（Actor）与价值网络（Critic）
- 参考模型约束：防止策略崩坏的关键机制
- 多源奖励融合：Reward Model评分 + 推理格式奖励
- 分布式训练支持：DDP多卡训练
- 混合精度与梯度累积：优化显存与训练稳定性
- 完整的断点续训能力

核心创新点：
1. 双KL约束机制：同时监控与旧策略(old_actor)和参考模型(ref_model)的KL散度
2. 推理模型专用奖励：针对<think>/<answer>结构的格式与内容奖励
3. 答案内容加权评分：对推理模型的<answer>部分单独强化评估
"""

import os
import sys

# 设置包路径以便正确导入父目录模块（解决相对导入问题）
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import re
import warnings
import torch
import torch.distributed as dist
import torch.nn.functional as F
from transformers import AutoTokenizer
from contextlib import nullcontext  # 用于条件上下文管理（如混合精度训练开关）
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel  # PyTorch分布式数据并行封装
from torch.utils.data import DataLoader, DistributedSampler  # 分布式数据加载器
from torch.nn.utils import clip_grad_norm_  # 梯度裁剪防止爆炸
from torch.optim.lr_scheduler import CosineAnnealingLR  # 余弦退火学习率调度器
from transformers import AutoModel  # Hugging Face模型加载
from model.model import MiniMindConfig, MiniMindForCausalLM  # 自定义MiniMind模型
from dataset.lm_dataset import RLAIFDataset  # RLAIF格式偏好数据集
from train.trainer_utils import (  # 训练工具函数集合
    Logger, is_main_process, lm_checkpoint, init_distributed_mode, 
    setup_seed, SkipBatchSampler, init_model
)

warnings.filterwarnings('ignore')  # 忽略非关键警告（如transformers内部警告）


# ==================== Critic价值网络定义 ====================
class CriticModel(MiniMindForCausalLM):
    """
    价值函数网络：在MiniMind语言模型基础上添加价值预测头
    - 输入：完整序列（prompt + 生成的response）
    - 输出：每个token位置的价值估计（最终取response末尾位置作为序列整体价值）
    - 关键设计：复用语言模型的编码能力，仅替换输出层
    """
    def __init__(self, params):
        super().__init__(params)
        # 替换语言模型头为价值预测头（输出维度1：标量价值）
        self.value_head = nn.Linear(params.hidden_size, 1)
        # 保留基础Transformer结构，仅修改输出层

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # 获取Transformer最后一层隐藏状态 [batch_size, seq_len, hidden_size]
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])  # 应用最终层归一化
        
        # 通过价值头预测每个token位置的价值 [batch_size, seq_len, 1] -> [batch_size, seq_len]
        values = self.value_head(hidden_states).squeeze(-1)
        return values  # 返回序列中每个位置的价值估计张量


# ==================== 奖励计算函数 ====================
def calculate_rewards(prompts, responses, reward_model, reward_tokenizer):
    """
    综合计算多源奖励：格式奖励 + 标记完整性奖励 + Reward Model语义奖励
    Args:
        prompts: 原始prompt列表（含对话历史标记），长度为B
        responses: 模型生成的response文本列表，长度为B
        reward_model: 奖励模型实例（如InternLM2-Reward）
        reward_tokenizer: 奖励模型对应的tokenizer
    Returns:
        rewards: 形状为[B]的奖励张量（已裁剪到合理范围）
    """
    def reasoning_model_reward(rewards):
        """
        推理模型专用奖励函数（仅当args.reasoning=1时启用）
        包含两部分：
        1. 格式合规性奖励：检查是否符合指定XML结构
        2. 标记完整性奖励：检查关键标签出现次数
        """
        # 定义两种合法格式（允许单/双换行）
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>\n.*?\n</answer>$"

        # 检查每个response是否匹配任一格式
        matches_pattern = [re.match(pattern, response, re.S) for response in responses]
        matches_pattern2 = [re.match(pattern2, response, re.S) for response in responses]

        format_rewards = []
        for match_pattern, match_pattern2 in zip(matches_pattern, matches_pattern2):
            if match_pattern or match_pattern2:
                format_rewards.append(0.5)  # 格式正确奖励0.5
            else:
                format_rewards.append(0.0)  # 格式错误无奖励
        rewards += torch.tensor(format_rewards, device=args.device)  # 累加到总奖励

        # 标记完整性奖励：每个必需标签出现一次奖励0.25（满分1.0）
        def mark_num(text):
            reward = 0
            reward += 0.25 if text.count("<think>") == 1 else 0
            reward += 0.25 if text.count("</think>") == 1 else 0
            reward += 0.25 if text.count("<answer>") == 1 else 0
            reward += 0.25 if text.count("</answer>") == 1 else 0
            return reward

        mark_rewards = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_rewards, device=args.device)
        return rewards

    # 初始化奖励张量（全0，设备与计算一致）
    rewards = torch.zeros(len(responses), device=args.device)
    
    # 条件添加推理模型专用奖励（仅当训练推理模型时）
    if args.reasoning == 1:
        rewards = reasoning_model_reward(rewards)
    
    # 使用外部Reward Model计算语义奖励（核心人类偏好信号）
    with torch.no_grad():  # 禁用梯度计算（Reward Model固定不训练）
        reward_model_scores = []
        for prompt, response in zip(prompts, responses):
            # 从prompt中解析对话历史（提取system/user/assistant角色内容）
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            matches = re.findall(pattern, prompt, re.DOTALL)
            messages = [{"role": role, "content": content.strip()} for role, content in matches]
            
            # 构建完整对话：历史消息 + 当前生成的response
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            score = reward_model.get_score(reward_tokenizer, tmp_chat)  # 获取原始奖励分数(标量)
            
            # 裁剪分数到[-3.0, 3.0]防止异常值影响训练稳定性
            scale = 3.0
            score = max(min(score, scale), -scale)
            
            # 推理模型特殊处理：对<answer>内容单独加权评分（强化答案质量）
            if args.reasoning == 1:
                answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
                if answer_match:
                    answer_content = answer_match.group(1).strip()
                    # 仅用answer内容构建对话计算奖励（聚焦答案质量）
                    tmp_chat = messages + [{"role": "assistant", "content": answer_content}]
                    answer_score = reward_model.get_score(reward_tokenizer, tmp_chat)
                    answer_score = max(min(answer_score, scale), -scale)
                    # 混合策略：40%完整response + 60%answer内容（突出答案重要性）
                    score = score * 0.4 + answer_score * 0.6
            
            reward_model_scores.append(score)
        
        # 转换为张量并累加到总奖励
        reward_model_scores = torch.tensor(reward_model_scores, device=args.device)
        rewards += reward_model_scores
    
    return rewards  # 返回综合奖励张量 [B]


# ==================== PPO单轮训练函数 ====================
def ppo_train_epoch(epoch, loader, iters, old_actor_model, ref_model, actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, start_step=0, wandb=None):
    """
    执行一个epoch的PPO训练，核心流程：
    1. 采样：用当前Actor生成responses
    2. 评估：计算多源奖励 + Critic价值估计
    3. 计算优势函数：A = R - V
    4. 计算PPO损失：策略损失 + 价值损失 + KL约束
    5. 反向传播与优化（含梯度累积）
    
    Args:
        epoch: 当前训练轮次
        loader: 数据加载器
        iters: 总迭代步数
        old_actor_model: 旧策略模型（用于PPO比率计算）
        ref_model: 参考模型（SFT初始化，用于KL约束防止策略崩坏）
        actor_scheduler/critic_scheduler: 学习率调度器
        reward_model/reward_tokenizer: 外部奖励模型及tokenizer
        start_step: 起始step（用于续训）
        wandb: wandb日志对象
    """
    actor_model.train()  # 设置Actor为训练模式
    critic_model.train() # 设置Critic为训练模式

    for step, batch in enumerate(loader, start=start_step + 1):
        prompts = batch["prompt"]  # list[str], 长度B（原始prompt文本）
        
        # ========== 1. 编码prompt（左填充确保生成起始位置对齐）==========
        enc = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=args.max_seq_len, 
            padding_side="left"  # 左填充：保证所有样本生成起始位置相同
        ).to(args.device)  # input_ids: [B, P], attention_mask: [B, P]
        prompt_length = enc.input_ids.shape[1]  # prompt实际序列长度（含padding）

        # ========== 2. 生成responses（采样过程，不计算梯度）==========
        with torch.no_grad():
            # 处理DDP封装：访问原始模型进行generate
            model_for_gen = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            gen_out = model_for_gen.generate(
                input_ids=enc.input_ids,
                attention_mask=enc.attention_mask,
                max_new_tokens=args.max_gen_len,  # 限制生成长度
                do_sample=True,       # 启用随机采样（非贪婪搜索）
                temperature=0.8,      # 采样温度（控制多样性）
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )  # 输出形状: [B, prompt_len + response_len]

        # 解码生成的response（跳过prompt部分，移除特殊token）
        responses_text = [
            tokenizer.decode(gen_out[i, prompt_length:], skip_special_tokens=True) 
            for i in range(len(prompts))
        ]
        
        # ========== 3. 计算综合奖励（调用多源奖励函数）==========
        rewards = calculate_rewards(prompts, responses_text, reward_model, reward_tokenizer)  # [B]

        # ========== 4. Critic价值估计（计算序列末尾位置的价值）==========
        full_mask = (gen_out != tokenizer.pad_token_id).long()  # 有效token掩码 [B, L]
        values_seq = critic_model(input_ids=gen_out, attention_mask=full_mask)  # [B, L] 每个位置的价值
        
        # 定位response末尾位置（最后一个非padding token）
        last_indices = (full_mask * torch.arange(full_mask.size(1), device=gen_out.device)).argmax(dim=1)
        values = values_seq[torch.arange(values_seq.size(0), device=values_seq.device), last_indices]  # [B]
        
        # 计算优势函数：A = R - V (detach避免critic梯度影响actor更新)
        advantages = rewards - values.detach()  # [B]

        # ========== 5. Actor策略梯度计算（混合精度上下文）==========
        with autocast_ctx:  # 根据设备自动启用混合精度
            res = actor_model(input_ids=gen_out, attention_mask=full_mask)
            logits = res.logits  # [B, L, Vocab] 语言模型logits
            # MoE模型特有：辅助负载均衡损失（非MoE时为0）
            aux_loss = res.aux_loss if lm_config.use_moe else torch.tensor(0.0, device=args.device)
        
        # 准备训练标签（右移一位：预测下一个token）
        labels = gen_out[:, 1:].clone()  # [B, L-1]
        # 计算每个token的对数概率（仅关注response部分）
        logp_tokens = F.log_softmax(logits[:, :-1], dim=-1).gather(
            2, labels.unsqueeze(-1)
        ).squeeze(-1)  # [B, L-1]
        
        # 构建response部分的掩码（排除prompt和padding）
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len, device=gen_out.device).unsqueeze(0) >= (prompt_length - 1)
        final_mask = resp_mask & (~labels.eq(tokenizer.pad_token_id))  # [B, L-1]
        actor_logp = (logp_tokens * final_mask).sum(dim=1)  # 对response部分求和 [B]

        # ========== 6. 计算参考策略概率（用于KL约束）==========
        with torch.no_grad():
            # 旧策略（上一次更新的actor）概率 - 用于PPO比率计算
            old_logits = old_actor_model(input_ids=gen_out, attention_mask=full_mask).logits
            old_logp_tokens = F.log_softmax(old_logits[:, :-1], dim=-1).gather(
                2, labels.unsqueeze(-1)
            ).squeeze(-1)
            old_logp = (old_logp_tokens * final_mask).sum(dim=1)  # [B]
            
            # 参考模型（SFT初始化模型）概率 - 关键！防止策略偏离初始模型过远
            ref_logits = ref_model(input_ids=gen_out, attention_mask=full_mask).logits
            ref_logp_tokens = F.log_softmax(ref_logits[:, :-1], dim=-1).gather(
                2, labels.unsqueeze(-1)
            ).squeeze(-1)
            ref_logp = (ref_logp_tokens * final_mask).sum(dim=1)  # [B]

        # ========== 7. 损失计算（PPO核心）==========
        # 策略更新幅度监控（KL(old||current)）
        kl = (actor_logp - old_logp).mean()
        # 关键约束：KL(current||reference) - 防止模型"崩坏"的核心机制
        kl_ref = (actor_logp - ref_logp).mean()
        
        # PPO-Clip目标函数：限制策略更新幅度
        ratio = torch.exp(actor_logp - old_logp)  # 重要性采样比率 [B]
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()  # 最小化负目标（最大化优势）
        
        # Critic损失：MSE回归（预测价值与实际奖励的差距）
        value_loss = F.mse_loss(values, rewards)
        
        # 总损失 = 策略损失 + 价值损失系数 * 价值损失 + KL约束系数 * KL_ref + MoE辅助损失
        # 除以accumulation_steps实现梯度累积
        loss = (
            policy_loss + 
            args.vf_coef * value_loss + 
            args.kl_coef * kl_ref +  # 核心：约束策略与参考模型的KL散度
            aux_loss
        ) / args.accumulation_steps
        
        loss.backward()  # 反向传播（梯度累积）

        # ========== 8. 优化器步进（梯度累积后）==========
        if (step + 1) % args.accumulation_steps == 0:
            # 梯度裁剪防止爆炸（分别对Actor和Critic）
            clip_grad_norm_(actor_model.parameters(), args.grad_clip)
            clip_grad_norm_(critic_model.parameters(), args.grad_clip)
            
            actor_optimizer.step()
            critic_optimizer.step()
            actor_scheduler.step()  # 更新学习率
            critic_scheduler.step()
            
            # 清空梯度（为下一轮累积准备）
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

        # ========== 9. 日志与监控（仅主进程执行）==========
        if is_main_process():
            # 计算生成文本平均长度（监控生成质量）
            response_ids = gen_out[:, enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(), dim=1)  # 找到首个EOS位置
            has_eos = is_eos.any(dim=1)
            lengths = torch.where(has_eos, eos_indices + 1, torch.tensor(response_ids.shape[1], device=is_eos.device))
            avg_len = lengths.float().mean()
            
            # 收集标量指标
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = rewards.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()  # 关键监控指标！
            avg_len_val = avg_len.item()
            actor_lr = actor_optimizer.param_groups[0]['lr']
            critic_lr = critic_optimizer.param_groups[0]['lr']

            # WandB日志记录
            if wandb is not None:
                wandb.log({
                    "actor_loss": actor_loss_val,
                    "critic_loss": critic_loss_val,
                    "aux_loss": current_aux_loss,
                    "reward": reward_val,
                    "kl": kl_val,          # 与旧策略的KL（PPO稳定性）
                    "kl_ref": kl_ref_val,  # 与参考模型的KL（防崩坏关键！）
                    "avg_response_len": avg_len_val,
                    "actor_lr": actor_lr,
                })

            # 控制台日志（格式化输出）
            Logger(
                f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iters}), "
                f"Actor Loss: {actor_loss_val:.4f}, Critic Loss: {critic_loss_val:.4f}, Aux Loss: {current_aux_loss:.4f}, "
                f"Reward: {reward_val:.4f}, KL: {kl_val:.4f}, KL_ref: {kl_ref_val:.4f}, "
                f"Avg Response Len: {avg_len_val:.2f}, Actor LR: {actor_lr:.8f}, Critic LR: {critic_lr:.8f}"
            )

        # ========== 10. 定期更新旧策略模型（用于PPO比率计算）==========
        if (step + 1) % args.update_old_actor_freq == 0:
            # 获取原始模型（处理DDP和torch.compile封装）
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)  # 处理torch.compile封装
            state_dict = raw_actor.state_dict()
            # 加载到CPU避免GPU内存压力，使用时再移至device
            old_actor_model.load_state_dict({k: v.detach().cpu() for k, v in state_dict.items()})
            old_actor_model.to(args.device)

        # ========== 11. 定期保存检查点（仅主进程）==========
        if (step % args.save_interval == 0 or step == iters - 1) and is_main_process():
            actor_model.eval()  # 临时切换评估模式
            moe_suffix = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/{args.save_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
            
            # 保存Actor模型（半精度节省空间）
            raw_actor = actor_model.module if isinstance(actor_model, DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor, '_orig_mod', raw_actor)
            actor_state = raw_actor.state_dict()
            torch.save({k: v.half().cpu() for k, v in actor_state.items()}, ckp)
            
            # 保存完整训练状态（含critic、优化器、调度器等）
            lm_checkpoint(
                lm_config, weight=args.save_weight, model=actor_model, optimizer=actor_optimizer, 
                epoch=epoch, step=step, wandb=wandb, save_dir='../checkpoints',
                scheduler=actor_scheduler, critic_model=critic_model, 
                critic_optimizer=critic_optimizer, critic_scheduler=critic_scheduler
            )
            actor_model.train()  # 恢复训练模式
            del actor_state  # 释放内存

        # ========== 12. 显式内存清理（加速垃圾回收）==========
        # 删除中间变量避免显存累积（尤其在长序列生成时）
        del enc, gen_out, responses_text, rewards, full_mask, values_seq, values, advantages
        del logits, labels, logp_tokens, final_mask, actor_logp, old_logits, old_logp, ref_logits, ref_logp
        del kl, kl_ref, ratio, surr1, surr2, policy_loss, value_loss, loss


# ==================== 主训练流程 ====================
if __name__ == "__main__":
    # ========== 1. 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="MiniMind PPO (Proximal Policy Optimization)")
    parser.add_argument("--save_dir", type=str, default="../out", help="模型保存目录")
    parser.add_argument('--save_weight', default='ppo_actor', type=str, help="保存权重的前缀名")
    parser.add_argument("--epochs", type=int, default=1, help="训练总轮数")
    parser.add_argument("--batch_size", type=int, default=2, help="单卡batch size")
    parser.add_argument("--learning_rate", type=float, default=8e-8, help="Actor学习率")
    parser.add_argument("--critic_learning_rate", type=float, default=8e-8, help="Critic学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="混合精度类型（bfloat16/float16）")
    parser.add_argument("--num_workers", type=int, default=8, help="数据加载线程数")
    parser.add_argument("--accumulation_steps", type=int, default=1, help="梯度累积步数")
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--log_interval", type=int, default=1, help="日志打印间隔（已弃用，由is_main_process控制）")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔（步数）")
    parser.add_argument('--hidden_size', default=512, type=int, help="模型隐藏层维度")
    parser.add_argument('--num_hidden_layers', default=8, type=int, help="Transformer隐藏层数量")
    parser.add_argument('--use_moe', default=0, type=int, choices=[0, 1], help="是否使用MoE架构（0=否，1=是）")
    parser.add_argument('--max_seq_len', default=66, type=int, help="Prompt最大长度（含特殊token）")
    parser.add_argument("--max_gen_len", type=int, default=1536, help="生成response的最大长度")
    parser.add_argument("--data_path", type=str, default="../dataset/rlaif-mini.jsonl", help="RLAIF数据路径")
    parser.add_argument("--clip_epsilon", type=float, default=0.1, help="PPO裁剪参数epsilon")
    parser.add_argument("--vf_coef", type=float, default=0.5, help="Value function损失系数")
    parser.add_argument("--kl_coef", type=float, default=0.02, help="KL散度惩罚系数（针对ref_model）")
    parser.add_argument("--reasoning", type=int, default=1, choices=[0, 1], help='推理模型类型（0=普通模型，1=推理模型）')
    parser.add_argument("--update_old_actor_freq", type=int, default=4, help="更新old_actor_model的频率（步数）")
    parser.add_argument("--reward_model_path", type=str, default="../../internlm2-1_8b-reward", help="Reward模型路径")
    parser.add_argument('--from_resume', default=0, type=int, choices=[0, 1], help="是否自动检测并续训（0=否，1=是）")
    parser.add_argument("--use_wandb", action="store_true", help="是否使用wandb/swanlab记录实验")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-PPO", help="wandb项目名")
    parser.add_argument("--use_compile", default=0, type=int, choices=[0, 1], help="是否使用torch.compile加速（0=否，1=是）")
    args = parser.parse_args()

    # ========== 2. 初始化环境和随机种子 ==========
    local_rank = init_distributed_mode()  # 初始化分布式训练环境（单机/多机）
    if dist.is_initialized(): 
        args.device = f"cuda:{local_rank}"  # DDP模式下设置设备
    # 设置随机种子（分布式下每个进程种子不同，确保数据打乱一致性）
    setup_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))
    
    # ========== 3. 配置目录、模型参数、检查续训点 ==========
    os.makedirs(args.save_dir, exist_ok=True)  # 创建保存目录
    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        use_moe=bool(args.use_moe)
    )
    # 检查是否存在续训检查点
    ckp_data = lm_checkpoint(lm_config, weight=args.save_weight, save_dir='../checkpoints') if args.from_resume==1 else None
    
    # ========== 4. 设置混合精度上下文 ==========
    device_type = "cuda" if "cuda" in args.device else "cpu"
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    # CPU不支持autocast，使用nullcontext作为占位符
    autocast_ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast(dtype=dtype)
    
    # ========== 5. 初始化WandB/SwanLab日志 ==========
    wandb = None
    if args.use_wandb and is_main_process():  # 仅主进程初始化
        import swanlab as wandb  # 注：此处使用swanlab库（兼容wandb API）
        wandb_id = ckp_data.get('wandb_id') if ckp_data else None
        resume = 'must' if wandb_id else None  # 存在wandb_id则续传
        wandb_run_name = f"MiniMind-PPO-Epoch-{args.epochs}-BS-{args.batch_size}-LR-{args.learning_rate}"
        wandb.init(project=args.wandb_project, name=wandb_run_name, id=wandb_id, resume=resume)
    
    # ========== 6. 初始化模型和数据 ==========
    base_weight = "reason" if args.reasoning == 1 else "full_sft"  # 选择基础模型权重前缀
    
    # --- Actor模型（待优化策略）---
    actor_model, tokenizer = init_model(lm_config, base_weight, device=args.device)
    if args.use_compile == 1:
        actor_model = torch.compile(actor_model)  # Torch 2.0编译加速（需PyTorch>=2.0）
        Logger('torch.compile enabled')
    
    # --- Old Actor模型（用于PPO比率计算）---
    old_actor_model, _ = init_model(lm_config, base_weight, device=args.device)
    old_actor_model = old_actor_model.eval().requires_grad_(False)  # 固定不训练
    
    # --- Reference Model（关键！用于KL约束防止策略崩坏）---
    ref_model, _ = init_model(lm_config, base_weight, device=args.device)
    ref_model = ref_model.eval().requires_grad_(False)  # 固定不训练（SFT初始化模型）
    
    # --- Critic模型（价值网络）---
    moe_suffix = '_moe' if lm_config.use_moe else ''
    ckp = f'{args.save_dir}/{base_weight}_{lm_config.hidden_size}{moe_suffix}.pth'
    state_dict = torch.load(ckp, map_location=args.device)  # 加载SFT权重初始化
    critic_model = CriticModel(lm_config)
    critic_model.load_state_dict(state_dict, strict=False)  # strict=False跳过lm_head不匹配
    critic_model = critic_model.to(args.device)
    
    # --- Reward Model（人类偏好信号源）---
    reward_model = AutoModel.from_pretrained(
        args.reward_model_path, torch_dtype=torch.float16, trust_remote_code=True
    )
    reward_model = reward_model.to(args.device).eval().requires_grad_(False)  # 固定不训练
    reward_tokenizer = AutoTokenizer.from_pretrained(args.reward_model_path, trust_remote_code=True)
    
    # --- 数据集与优化器 ---
    train_ds = RLAIFDataset(args.data_path, tokenizer, max_length=(args.max_seq_len + args.max_gen_len))
    train_sampler = DistributedSampler(train_ds) if dist.is_initialized() else None  # DDP数据采样器
    
    actor_optimizer = optim.AdamW(actor_model.parameters(), lr=args.learning_rate)
    critic_optimizer = optim.AdamW(critic_model.parameters(), lr=args.critic_learning_rate)
    
    # 计算总训练步数（用于学习率调度）
    loader_for_count = DataLoader(train_ds, batch_size=args.batch_size, sampler=train_sampler)
    iters = len(loader_for_count)  # 单epoch步数
    total_optimizer_steps = (iters // args.accumulation_steps) * args.epochs  # 总优化器步数
    
    # 余弦退火学习率调度（最终学习率降至初始的1/10）
    actor_scheduler = CosineAnnealingLR(actor_optimizer, T_max=total_optimizer_steps, eta_min=args.learning_rate / 10)
    critic_scheduler = CosineAnnealingLR(critic_optimizer, T_max=total_optimizer_steps, eta_min=args.critic_learning_rate / 10)
    
    # ========== 7. 从检查点恢复训练状态（如需）==========
    start_epoch, start_step = 0, 0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)  # 兼容旧检查点
    
    # ========== 8. DDP封装模型（分布式训练）==========
    if dist.is_initialized():
        # 忽略RoPE频率缓冲区（非可训练参数，避免DDP同步错误）
        actor_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        critic_model._ddp_params_and_buffers_to_ignore = {"freqs_cos", "freqs_sin"}
        actor_model = DistributedDataParallel(actor_model, device_ids=[local_rank])
        critic_model = DistributedDataParallel(critic_model, device_ids=[local_rank])
        old_actor_model.to(args.device)  # 确保在正确设备上（非DDP模型）
    
    # ========== 9. 开始训练循环 ==========
    for epoch in range(start_epoch, args.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)  # DDP下每个epoch重置采样器确保数据打乱
        
        # 设置随机种子确保数据顺序一致（分布式下每个进程相同）
        setup_seed(42 + epoch)
        indices = torch.randperm(len(train_ds)).tolist()
        
        # 跳过已训练的step（续训支持）
        skip = start_step if (epoch == start_epoch and start_step > 0) else 0
        batch_sampler = SkipBatchSampler(train_sampler or indices, args.batch_size, skip)
        loader = DataLoader(
            train_ds, 
            batch_sampler=batch_sampler, 
            num_workers=args.num_workers, 
            pin_memory=True  # 加速数据传输到GPU
        )
        
        if skip > 0: 
            Logger(f'Epoch [{epoch + 1}/{args.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
        
        # 执行单轮训练
        if skip > 0:
            ppo_train_epoch(
                epoch, loader, len(loader) + skip, old_actor_model, ref_model, 
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 
                start_step, wandb
            )
        else:
            ppo_train_epoch(
                epoch, loader, len(loader), old_actor_model, ref_model, 
                actor_scheduler, critic_scheduler, reward_model, reward_tokenizer, 
                0, wandb
            )
    
    # ========== 10. 清理分布式进程组 ==========
    if dist.is_initialized():
        dist.destroy_process_group()  # 释放分布式资源
    
    Logger("PPO训练流程执行完毕！")