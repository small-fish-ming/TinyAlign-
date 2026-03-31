from logging import config
from sentry_sdk import init
from transformers import PretrainedConfig


class MokioMindConfig(PretrainedConfig):
    model_type = "mokiomind"

    def __init__(
        self,
        dropout: float = 0.0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        hidden_act: str = "silu",
        hidden_size: int = 512,
        intermediate_size: int = None,
        max_position_embeddings: int = 32768,
        num_attention_heads: int = 8,
        num_hidden_layers: int = 8,
        num_key_value_heads: int = 2,
        vocab_size: int = 6400,
        rms_norm_eps: float = 1e-05,
        rope_theta: int = 1000000,
        inference_rope_scaling: bool = False,
        flash_attention: bool = True,
        ############ MoE ############
        use_moe: bool = False,
        num_experts_per_tok: int = 2,
        n_routed_experts: int = 4,
        n_shared_experts: int = 1,
        scoring_func: str = "softmax",
        aux_loss_alpha: float = 0.1,
        seq_aux: bool = True,
        norm_topk_prob: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.dropout = dropout
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.flash_attention = flash_attention
        self.use_moe = use_moe
        self.num_experts_per_tok = num_experts_per_tok
        self.n_routed_experts = n_routed_experts
        self.n_shared_experts = n_shared_experts
        self.seq_aux = seq_aux
        self.norm_topk_prob = norm_topk_prob
        self.aux_loss_alpha = aux_loss_alpha
        self.scoring_func = scoring_func

        self.rope_scaling = (
            {
                "beta_fast": 4,
                "beta_slow": 1,
                "factor": 4,
                "original_max_position_embeddings": 2048,
                "type": "yarn",
            }
            if self.inference_rope_scaling
            else None
        )

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple, Union
from torch.nn import functional as F
from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
import torch.nn.init as nn_init


# 继承nn.Model类
class RMSNorm(nn.Module):
# __init__初始化
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

# _norm
    def _norm(self, x):
        return x*torch.rsqrt(x.pow(2).mean(-1, keepdim=True)+self.eps)
# forward
    def forward(self, x):
        return self.weight*self._norm(x.float()).type_as(x)
                                                        
# 写出最初RoPE式子
def precomput_freqs_cis(dim:int,end:int=int(32*1024),rope_base:float=1e6,rope_scaling:Optional[dict]=None):
    freqs=1.0/(rope_base**(torch.arange(0,dim,2)[:dim//2].float()/dim))

    if rope_scaling is not None:
        orig_max,factor,beta_fast,beta_slow=(
            rope_scaling.get("original_max_position_embeddings",2048),
            rope_scaling.get("factor",4),
            rope_scaling.get("beta_fast",4),
            rope_scaling.get("beta_slow",1),
        )
        if end / orig_max > 1.0:
            corr_dim=next((i for i in range(dim//2) if 2*math.pi/freqs[i]>orig_max),dim//2)
        # 计算power
            power=torch.arange(0,dim//2,device=freqs.device)/max(dim//2-1,1)
        # 计算beta
            beta=beta_slow+(beta_fast-beta_slow)*(power)
        # 计算scale
            scale=torch.where(
                torch.arange(0,dim//2,device=freqs.device)<corr_dim,
                (beta*factor-beta+1)/(beta*factor),
                1.0/factor
            )
        # 应用scale
            freqs= freqs*scale
    t=torch.arange(end,device=freqs.device)
    freqs=torch.outer(t,freqs).float()
    # 返回一个cos和sin
    freqs_cos=torch.cat([torch.cos(freqs),torch.cos(freqs)],dim=-1)
    freqs_sin=torch.cat([torch.sin(freqs),torch.sin(freqs)],dim=-1)
    return freqs_cos,freqs_sin

def apply_rotary_pos_emb(q,k,cos,sin,unsqueze_dim=1):
    # [a,b]->[-b,a]
    def rotate_half(x):
        return torch.cat([-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]],dim=-1)
    q_embed=(q*cos.unsqueeze(unsqueze_dim))+(rotate_half(q)*sin.unsqueeze(unsqueze_dim))
    k_embed=(k*cos.unsqueeze(unsqueze_dim))+(rotate_half(k)*sin.unsqueeze(unsqueze_dim))
    return q_embed,k_embed

def repeat_kv(x:torch.tensor,n_rep:int)->torch.tensor:
    bs,slen,num_key_value_heads,head_dim=x.shape
    if n_rep==1:
        return x
    #expand 利用 PyTorch 的广播机制，只对大小为 1 的维度或缺失维度进行“扩展”
    # x = torch.tensor([[1], [2]])  # shape: (2, 1)
    # y = x.expand(2, 3)            # shape: (2, 3)

    # print(y)
    # # tensor([[1, 1, 1],
    # #         [2, 2, 2]])
    return x.unsqueeze(3).expand(bs,slen,num_key_value_heads,n_rep,head_dim).reshape(bs,slen,num_key_value_heads*n_rep,head_dim)

class Attention(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        # 处理GQA：如果没有指定kv头数，则使用与query相同的头数
        # 三元运算符：condition ? value1 : value2
        self.num_key_value_heads=args.num_attention_heads if args.num_key_value_heads is None else args.num_key_value_heads

        # assert语句：断言检查，如果条件为False则抛出AssertionError
        # 确保query头数能被kv头数整除（GQA的基本要求
        assert args.num_attention_heads%self.num_key_value_heads==0,f"num_attention_heads must be divisible by num_key_value_heads, but got {args.num_attention_heads} and {self.num_key_value_heads}"
        
        self.n_local_heads=args.num_attention_heads
        self.n_rep=self.n_local_heads//self.num_key_value_heads
        self.head_dim=args.hidden_size//args.num_attention_heads

        self.q_proj=nn.Linear(args.hidden_size,args.num_attention_heads*self.head_dim,bias=False)
        self.k_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.v_proj=nn.Linear(args.hidden_size,self.num_key_value_heads*self.head_dim,bias=False)
        self.o_proj=nn.Linear(args.num_attention_heads*self.head_dim,args.hidden_size,bias=False)

        self.atten_dropout=nn.Dropout(args.dropout)
        self.resid_dropout=nn.Dropout(args.dropout)
        self.dropout=args.dropout

        self.flash=hasattr(torch.nn.functional,"scaled_dot_product_attention") and args.flash_attention
            
    def forward(self,x:torch.tensor,position_embeddings:tuple[torch.tensor,torch.tensor],past_key_value:Optional[Tuple[torch.tensor,torch.tensor]]=None,use_cache:bool=False,attention_mask:Optional[torch.tensor]=None)->torch.tensor:
        # 投影，得到q,k,v
        bsz,seq_len,_=x.shape
        xq,xk,xv=self.q_proj(x),self.k_proj(x),self.v_proj(x)
        # 把输入拆分成多个头，用view
        xq=xq.view(bsz,seq_len,self.n_local_heads,self.head_dim)
        xk=xk.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)
        xv=xv.view(bsz,seq_len,self.num_key_value_heads,self.head_dim)

        # position_embeddings是预计算的(cos, sin)，按序列位置切片并应用RoPE
        cos,sin=position_embeddings
        xq,xk=apply_rotary_pos_emb(xq,xk,cos[:seq_len],sin[:seq_len])
        # -------------------- KV cache 处理 --------------------
        # past_key_value: (past_k, past_v) 或 None
        # 当存在past时，将past拼接到当前k,v的时间维度上，便于自回归推理
        if past_key_value is not None:
            # past_key_value[0] 的shape为 [bsz, past_seq_len, n_local_kv_heads, 
            xk=torch.cat([past_key_value[0],xk],dim=1)
            xv=torch.cat([past_key_value[1],xv],dim=1)
        # 如果需要缓存，返回拼接后的(k,v)，否则past_kv置为None
        past_kv=(xk,xv) if use_cache else None
        # -------------------- GQA: 对KV重复以匹配Q头 --------------------
        # transpose到形状 [bsz, n_heads, seq_len, head_dim] 以便矩阵乘法
        xq=xq.transpose(1,2)
        # repeat_kv会把k/v的头数从 n_local_kv_heads -> n_local_kv_heads * n_rep (即等于n_local_heads)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)
        # -------------------- Attention计算 --------------------
        # 优先使用PyTorch 2.0+的scaled_dot_product_atten      tion（Flash Attention实现）
        if self.flash and seq_len > 1 and (attention_mask is None or torch.all(attention_mask == 1)):
            # 如果没有显式的attention_mask，直接传None让底层高效实现
            attn_mask = None if attention_mask is None else attention_mask.view(bsz, 1, 1, -1).expand(bsz, self.n_local_heads, seq_len, -1).bool()
            # F.scaled_dot_product_attention是PyTorch在新版本中提供的高效实现
            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True  # 自回归（因果）注意力
            )
        else:
            # 标准实现：scores = Q @ K^T / sqrt(d)
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dim)

            # causal mask: 上三角（对角线以上）置为 -inf，防止看到未来信息
            causal_mask = torch.triu(torch.full((seq_len, seq_len), float("-inf"), device=scores.device), diagonal=1)
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)  # 扩展batch和head维度·

            # 如果有attention_mask(0/1)，将其扩展后转为 -1e9 的加性mask（掩掉pad位置）
            #将输入的 attention_mask（通常用于标记有效 token 和 padding token）转换为一个“加性掩码”（additive mask），并加到注意力分数（scores）上，使得 padding 位置在 softmax 后的注意力权重趋近于 0。
            if attention_mask is not None:
                extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
                extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
                scores = scores + extended_attention_mask

            # softmax得到注意力权重
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.atten_dropout(scores)
            # 加权求和得到输出
            output = scores @ xv

        
        # 恢复形状并做输出投影 + 残差dropout
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)  # [bsz, seq_len, hidden]
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv


class MoEGate(nn.Module):
    def __init__(self, config: MokioMindConfig):
        super().__init__()
        self.config = config
        # 每个token分配到的专家数量
        self.top_k = config.num_experts_per_tok
        # 路由专家数量
        self.n_routed_experts = config.n_routed_experts
				# 计算分数函数
        self.scoring_func = config.scoring_func
        # 辅助损失的权重系数，用于判断是否需要负载均衡
        self.alpha = config.aux_loss_alpha
        # aux_loss 是如何统计的，按seq序列还是token批次
        self.seq_aux = config.seq_aux
				
				# 局部归一化
        self.norm_topk_prob = config.norm_topk_prob
        # 门控维度
        self.gating_dim = config.hidden_size
        # 参数，维度为路由专家数*门控维度
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()
        
    def reset_parameters(self) -> None:
		    # Kaiming初始化，也叫He初始化，高效初始化权重
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        # hidden_states: [bsz, seq_len, hidden]
        bsz, seq_len, h = hidden_states.shape

        # 将序列和batch合并，方便做同一个线性投影，计算简便
        # view(-1, h) -> [bsz * seq_len, hidden]
        hidden_states = hidden_states.view(-1, h)

        # 使用线性变换计算每个token对每个专家的logits
        # F.linear(input, weight, bias) 等价于 input @ weight.T + bias
        logits = F.linear(hidden_states, self.weight, None)  # [bsz*seq_len, n_routed_experts]

        # 得分函数：softmax是常用选择
        if self.scoring_func == 'softmax':
            scores = logits.softmax(dim=-1)
        else:
            raise NotImplementedError(f'insupportable scoring function for MoE gating: {self.scoring_func}')

        # top-k选择: topk_weight是概率，topk_idx是对应的专家索引
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)

        # 如果top-k>1且需要归一化（数值稳定性），对topk概率做局部归一化
        if self.top_k > 1 and self.norm_topk_prob:
		        # 计算这k个概率综合，1e-20是防止除以0
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            # 执行归一化
            topk_weight = topk_weight / denominator

        # 计算辅助负载均衡损失（aux loss），仅在训练且alpha>0时计算
        if self.training and self.alpha > 0.0:
            scores_for_aux = scores
            aux_topk = self.top_k
            # 用于aux loss的索引变形为 [bsz, seq_len*topk]
            topk_idx_for_aux_loss = topk_idx.view(bsz, -1)

            if self.seq_aux:
                # 序列级别的负载统计：统计每个专家被选到的次数,score_for_seq_aux的形状是[bsz, seq_len, n_routed_experts]
                scores_for_seq_aux = scores_for_aux.view(bsz, seq_len, -1)
                ce = torch.zeros(bsz, self.n_routed_experts, device=hidden_states.device)
                # scatter_add_用于累加计数：把1累加到对应专家的位置上
                ce.scatter_add_(dim=1, index=topk_idx_for_aux_loss, src=torch.ones(bsz, seq_len * aux_topk, device=hidden_states.device))
                # 归一化计数，使得期望值可比较
                ce = ce.div(seq_len * aux_topk / self.n_routed_experts)
                # aux_loss = alpha * mean_over_batch( sum( ce * mean(scores_for_token_over_seq) ) )
                aux_loss = (ce * scores_for_seq_aux.mean(dim=1)).sum(dim=1).mean() * self.alpha
            else:
                # token级别的负载统计
                mask_ce = F.one_hot(topk_idx_for_aux_loss.view(-1), num_classes=self.n_routed_experts)
                # 实际被选中次数/总次数，代表专家 j 在所有 Top-K 选择中出现的频率
                ce = mask_ce.float().mean(0)
                # 平均好感度，Gate多偏向某个专家
                Pi = scores_for_aux.mean(0)
                # fi 是一个归一化后的“负载因子”。
								# fi[j] == 1.0：完美平衡
								# fi[j] > 1.0：过载（被选中的次数超过了平均值）
								# fi[j] < 1.0：负载不足
                fi = ce * self.n_routed_experts
                # 计算辅助损失，用于下一次调整
                aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0

        return topk_idx, topk_weight, aux_loss
    

class MoEFeedForaward(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        self.args=args
        # 专家层
        self.experts=nn.ModuleList(
            [FeedForward(args)
             for _ in range(args.n_routed_experts)]
        )
        # 门控层
        self.gate=MoEGate(args)
        if args.n_shared_experts>0:
            self.shared_experts=nn.ModuleList(
                [FeedForward(args)
                 for _ in range(args.n_shared_experts)]
            )
    def forward(self,x):
        identity=x
        orig_shape=x.shape
        bsz,seq_len,h=orig_shape
        
        # 使用门控机制旋转专家
        topk_weight, topk_idx, aux_loss = self.gate(x)
        # 展开x以便处理
        x=x.view(-1,x.shape[-1])
        
        flat_topk_idx=topk_idx.view(-1)
        if self.training:
            # 按照定义的num_experts_per_tok重复输入token
            # 每个token安排num_experts_per_tok个专家处理
            x=x.repeat_interleave(self.args.num_experts_per_tok,dim=0)
            # y是空张量，和x形状相同
            y=torch.empty_like(x,dtype=torch.float32)
            # 遍历所有专家
            for i,expert in enumerate(self.experts):
                # 找到所有指向专家i的token
                # 然后将这些token输入专家i进行处理
                # 最后将结果放回y对应位置
                y[flat_topk_idx==i]=expert(x[flat_topk_idx==i]).to(y.dtype)
            # 加权求和
            # 最后的y意义是每个token经过专家处理后的加权结果
            y=(y.view(*topk_weight.shape,-1)*topk_weight.unsqueeze(-1)).sum(dim=1)
            y=y.view(*orig_shape)
        # 如果是推理阶段
        else:
            y = self.moe_infer(x, flat_topk_idx, topk_weight.view(-1, 1)).view(*orig_shape)
        if self.args.n_shared_experts > 0:
            for expert in self.shared_experts:
                y = y + expert(identity)
        self.aux_loss = aux_loss
        return y
    
    @torch.no_grad()
    # MoE推理方法
    def moe_infer(self, x, flat_expert_indices, flat_expert_weights):
        # 使用cache，创建一个和x形状相同的零张量
        expert_cache = torch.zeros_like(x)
        # 对专家索引进行排序，最后是[0,0,0,1,1,2,2,2,...]这样的顺序
        # 分拣
        idxs = flat_expert_indices.argsort()
        # 统计每个专家被分配到的token数量
        # 打包
        tokens_per_expert = flat_expert_indices.bincount().cpu().numpy().cumsum(0)
        # 计算每个token对应的专家索引
        token_idxs = idxs // self.config.num_experts_per_tok
        # 对每个打包好的包进行处理
        for i, end_idx in enumerate(tokens_per_expert):
            # 计算当前包的起始位置
            start_idx = 0 if i == 0 else tokens_per_expert[i - 1]
            if start_idx == end_idx:
                continue
            # 取出当前包对应的专家
            expert = self.experts[i]
            # 取出token对应的原始id
            exp_token_idx = token_idxs[start_idx:end_idx]
            # 取出token对应的数据
            expert_tokens = x[exp_token_idx]
            # 计算专家输出，一次性处理当前包的所有token
            expert_out = expert(expert_tokens).to(expert_cache.dtype)
            # 加权
            expert_out.mul_(flat_expert_weights[idxs[start_idx:end_idx]])
            # 将结果散点加到缓存中对应位置
            expert_cache.scatter_add_(0, exp_token_idx.view(-1, 1).repeat(1, x.shape[-1]), expert_out)

        return expert_cache


class FeedForward(nn.Module):
    def __init__(self,args:MokioMindConfig):
        super().__init__()
        if args.intermediate_size is None:
            intermediate_size=int(args.hidden_size*8/3)
            args.intermediate_size=64*((intermediate_size+64-1)//64)  # 向上取整到64的倍数
        # SwiGLU类似于Gated Linear Unit变体：act(gate(x)) * up(x)
        # gate_proj: hidden -> intermediate (用于计算gate部分)
        # up_proj: hidden -> intermediate (用于被gate的部分)
        # down_proj: intermediate -> hidden (用于投影回hidden维度)
        self.up_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.down_proj=nn.Linear(args.intermediate_size,args.hidden_size,bias=False)
        self.gate_proj=nn.Linear(args.hidden_size,args.intermediate_size,bias=False)
        self.dropout=nn.Dropout(args.dropout)
        # ACT2FN是transformers里激活函数的映射表，支持'silu','gelu'等
        self.act_fn=ACT2FN[args.hidden_act]

    def forward(self,x:torch.tensor)->torch.tensor:
        # forward实现使用SwiGLU风格的门控激活：
        # output = down_proj( act_fn(gate_proj(x)) * up_proj(x) )
        # 并在输出前应用dropout
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x)))
        
class MokioMindBlock(nn.Module):
    def __init__(self,Layer_id:int,args:MokioMindConfig):
        super().__init__()
        self.num_attention_heads=args.num_attention_heads
        self.hidden_size=args.hidden_size
        self.head_dim=args.hidden_size//args.num_attention_heads
        self.self_attn=Attention(args)

        self.layer_id=Layer_id
        self.input_layernorm=RMSNorm(args.hidden_size,eps=args.rms_norm_eps)
        self.post_attention_layernorm=RMSNorm(args.hidden_size,eps=args.rms_norm_eps)
        self.mlp=(FeedForward(args) if not args.use_moe else MoEFeedForaward(args))
    


    def forward(self,hidden_states,position_embeddings,past_key_value=None,use_cache=False,attention_mask=None):
        # 残差连接模式：先做LayerNorm -> Attention -> 残差相加 -> LayerNorm -> FFN -> 残差相加
        # 保存残差以供后续相加
        residual=hidden_states
         # 注意力子层：输入先归一化（RMSNorm），返回hidden_states和present_key_value（用于cache）
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),  # pre-norm
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask
        )

        # 注意力输出与残差相加
        hidden_states = hidden_states + residual
        # 前馈子层（post-attention layernorm）并相加
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present_key_value
    
class MokioMindModel(nn.Module):
    def __init__(self, args:MokioMindConfig):
        super().__init__()
        self.args=args
        self.vocab_size,self.num_hidden_layers=(
            args.vocab_size,args.num_hidden_layers
        )

        #         # 创建一个小型 embedding 层
        # emb = nn.Embedding(num_embeddings=5, embedding_dim=3)

        # # 假设 weight 随机初始化为：
        # # [[0.1, 0.2, 0.3],
        # #  [0.4, 0.5, 0.6],
        # #  [0.7, 0.8, 0.9],
        # #  [1.0, 1.1, 1.2],
        # #  [1.3, 1.4, 1.5]]

        # input_ids = torch.tensor([1, 3, 0])  # token IDs

        # output = emb(input_ids)
        # # output =
        # # [[0.4, 0.5, 0.6],   # 第1行（ID=1）
        # #  [1.0, 1.1, 1.2],   # 第3行（ID=3）
        # #  [0.1, 0.2, 0.3]]   # 第0行（ID=0）

        self.embed_tokens=nn.Embedding(args.vocab_size,args.hidden_size)

        self.dropout=nn.Dropout(args.dropout)
        self.layers=nn.ModuleList([MokioMindBlock(i,args) for i in range(args.num_hidden_layers)])
        
        self.norm=RMSNorm(args.hidden_size,eps=args.rms_norm_eps)

        # RoPE预计算
        freqs_cos, freqs_sin = precomput_freqs_cis(
            dim=args.hidden_size//args.num_attention_heads,
            end=args.max_position_embeddings,
            rope_base=args.rope_theta,
            rope_scaling=args.rope_scaling
        )

        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self,input_ids:torch.tensor,attention_mask:Optional[torch.tensor]=None,past_key_values:Optional[Tuple[Tuple[torch.tensor,torch.tensor],...]]=None,use_cache:bool=False,**kwargs)->torch.tensor:  
        batch_size, seq_len = input_ids.shape
        # 兼容性检查：某些框架会传入包含.layers属性的对象，视为不携带past信息
        if hasattr(past_key_values, "layers"):
            past_key_values=None
        # past_key_values为每层的(past_k, past_v)列表，如果为None则创建与层数相同的None列表
        past_key_values=past_key_values or [None]*self.num_hidden_layers

        # 计算start_pos：如果存在past，则start_pos为已有past序列长度
        # past_key_values[0] 形如 (k, v)，k.shape = [bsz, past_seq_len, n_kv_heads, head_dim]
        start_pos=(
            past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
        )            
        
        hidden_states=self.dropout(self.embed_tokens(input_ids))
        # 从注册的buffer中取出对应位置范围的cos/sin作为position_embeddings
        # self.freqs_cos/freqs_sin的shape为 [max_pos, head_dim]
        position_embddings=(self.freqs_cos[start_pos:start_pos+seq_len],self.freqs_sin[start_pos:start_pos+seq_len])
        # 逐层前向，通过zip把layer和对应的past_key_value配对
        presents=[]

        for layer_idx,(layer,past_key_value) in enumerate(zip(self.layers,past_key_values)):
            hidden_states,present=layer(
                hidden_states,
                position_embddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask
            )
            presents.append(present)
    
        hidden_states=self.norm(hidden_states)

        # 如果使用MoE，收集每层的aux_loss并求和返回以便训练使用
        aux_loss = sum(
            layer.mlp.aux_loss
            for layer in self.layers
            if isinstance(layer.mlp, MoEFeedForaward)
        )

        return hidden_states, presents, aux_loss
    

class MokioMindForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MokioMindConfig

    def __init__(self, args:MokioMindConfig):
        self.config = args
        super().__init__(args)
        self.model = MokioMindModel(args)  

        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

        # 权重共享
        # 输出层的权重和输入嵌入层的权重共享，这是一种常见的做法，可以减少模型参数并提升性能
        self.model.embed_tokens.weight = self.lm_head.weight

    def forward(self, input_ids:Optional[torch.tensor]=None, attention_mask:Optional[torch.tensor]=None, 
                past_key_values:Optional[Tuple[Tuple[torch.tensor,torch.tensor],...]]=None, 
                use_cache:bool=False, 
                logits_to_keep:Union[int,torch.Tensor]=None,**args):
        
        hidden_states, presents, aux_loss = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **args
        )

        # logits_to_keep用于在序列末尾保留一部分logits（用于截断或微调策略）
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

        # 通过lm_head将hidden states投影到词表logits
        # h: [bsz, seq_len, hidden]
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        output = CausalLMOutputWithPast(logits=logits, past_key_values=past_key_values, hidden_states=hidden_states)
        output.aux_loss = aux_loss
        return output

            


       