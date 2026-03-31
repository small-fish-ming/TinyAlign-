import torch
import torch.nn.functional as F
import torch.nn as nn

def half_rotate(x:torch.Tensor) ->torch.Tensor:
    """
    将张量后半部分取负并交换位置，用于高效实现旋转。
    Example: [a, b, c, d] -> [-c, -d, a, b]
    """
    x1,x2 = x.chunk(2,dim=-1)
    return torch.cat((-x2,x1),dim=-1)

def apply_rope(x:torch.Tensor,position_ids:torch.Tensor,base:int=10000) ->torch.Tensor:
    
    batch_size,seq_len,num_head,dim = x.shape

    if position_ids.ndim==1:
        position_ids=position_ids.unsqueeze(0).expand(batch_size,-1) #[b,s]

    
    dim_indices=torch.arange(0,dim,2,dtype=torch.float32,device=x.device)

    inv_fre=1/(base**(dim_indices/dim)) #[dim/2]

    freqs=position_ids.unsqueeze(-1).float()*inv_fre.unsqueeze(0).unsqueeze(0)

    sin=torch.sin(freqs).unsqueeze(2) #[b,s,1,d]
    cos=torch.cos(freqs).unsqueeze(2)

    sin = sin.repeat_interleave(2, dim=-1)  # [θ0,θ1,θ2] → [θ0,θ0,θ1,θ1,θ2,θ2]
    cos = cos.repeat_interleave(2, dim=-1)

    rotate_x=cos*x+half_rotate(x)*sin

    return rotate_x


class MultiQueryAttention(nn.Model):
    
    def __int__(self,hidden_size:int,num_head:int,head_dim:int,rope_base:int=10000,use_rope:bool=True):
        super().__init__()
        self.hidden_size=hidden_size
        self.num_head=num_head
        self.head_dim=head_dim
        self.rope_base=rope_base
        self.use_rope=use_rope
        #投影层
        self.q_proj=nn.Linear(self.hidden_size,self.num_head*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.hidden_size,self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.hidden_size,self.head_dim,bias=False)
        self.o_proj=nn.Linear(self.head_dim*self.num_head,self.hidden_size,bias=False)
        #缩放因子
        self.scale=self.head_dim**(-0.5)
    def forward(self,hidden_state:torch.Tensor,position_ids:torch.Tensor,attion_mask:torch.Tensor) ->torch.Tensor:
        B,L,_ = hidden_state.shape
        q=self.q_proj(hidden_state)
        k=self.k_proj(hidden_state)
        v=self.v_proj(hidden_state)
        #启用rope
        if self.use_rope and position_ids is not None:
            q = apply_rope(q, position_ids, base=self.rope_base)
            k = apply_rope(k, position_ids, base=self.rope_base)
        # 改形状
        q=q.view(B,L,self.num_head,self.head_dim)
        k=k.unsqueeze(2)
        v=v.unsqueeze(2)

        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)

        # 计算注意力机制
        atten_weight=torch.matmul(q,k.transpose(-1,-2))*self.scale
        # 掩码
        if attion_mask is not None:
            atten_weight=atten_weight + atten_weight
        
        atten_weight=F.softmax(atten_weight, dim=-1, dtype=torch.float32).to(q.dtype)
        atten_output=torch.matmul(atten_weight,v)
        #合并多头
        atten_output=atten_output.transpose(1,2).contiguous()
        atten_output=atten_output.reshape(B,L,self.num_head*self.head_dim)
        return self.o_proj(atten_output)


        

class MultiHeadAttention(nn.Module):

    def __init__(self,hidden_size:int,num_head:int,head_dim:int,rope:int=10000,use_rope:bool=True,dropout:float=0.0):

        self.hidden_size=hidden_size
        self.num_head=num_head
        self.head_dim=head_dim
        self.rope=rope
        self.use_rope=use_rope
        self.dropout=dropout
        super().__init__()
        #投影层
        self.q_proj=nn.Linear(self.hidden_size,self.num_head*self.head_dim,bias=False)
        self.k_proj=nn.Linear(self.hidden_size,self.num_head*self.head_dim,bias=False)
        self.v_proj=nn.Linear(self.hidden_size,self.num_head*self.head_dim,bias=False)
        self.o_proj=nn.Linear(self.num_head*self.head_dim,self.hidden_size,bias=False)
        #缩放因子根号d
        self.scale=self.head_dim**(-0.5)
    def forward(self,x:torch.Tensor,position_id:torch.Tensor,atten_mask:torch.Tensor) ->torch.Tensor:

        B,L,_ = x.shape
        # 向量投影
        q=self.q_proj(x).view(B,L,self.num_head,self.head_dim)
        k=self.k_proj(x).view(B,L,self.num_head,self.head_dim)
        v=self.v_proj(x).view(B,L,self.num_head,self.head_dim)

        #启用rope
        if self.use_rope and position_id is not None:
            q=apply_rope(q,position_ids)
            k=apply_rope(q,position_ids)
        #改变形状
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        #注意力机制
        atten_weight=torch.matmul(q,k.transpose(-1,-2))*self.scale

        if atten_mask is not None:
            atten_weight=atten_weight+atten_mask
        
        atten_weight=F.softmax(atten_weight,dim=-1,dtype=torch.float32).to(q.dtype)
        atten_output=torch.matmul(atten_weight,v)

        # 合并多头
        atten_output=atten_output.transpose(1,2).reshape(B,L,self.head_dim*self.num_head).contiguous()
        return self.o_proj(atten_output)



class GroupQueryAttention(nn.Model):
    def __init__(self,hidden_size:int,num_head:int,head_dim:int,num_key_value_head:int,rope_base:int=10000,use_rope:bool=True):
        self.hidden_size=hidden_size
        self.num_head=num_head
        self.head_dim=head_dim
        self.num_key_value_head=num_key_value_head
        self.rope_base=rope_base
        self.use_rope=use_rope
        self.group_size=self.num_head//self.num_key_value_head
        super().__init__()
        self.q_proj=nn.Linear(self.hidden_size,self.head_dim*self.num_head,bias=False)
        self.k_proj=nn.Linear(self.hidden_size,self.head_dim*self.num_key_value_head,bias=False)
        self.v_proj=nn.Linear(self.hidden_size,self.head_dim*self.num_key_value_head,bias=False)
        self.o_proj=nn.Linear(self.num_head*self.head_dim,self.hidden_size)
        self.scale=self.head_dim**(-0.5)
    def forward(self,x:torch.Tensor,position_ids:torch.Tensor,atten_mask:torch.Tensor) ->torch.Tensor:
        B,L,_ = x.shape
        #投影
        q=self.q_proj(x).view(B,L,self.num_head,self.head_dim)
        k=self.k_proj(x).view(B,L,self.num_key_value_head,self.head_dim)
        v=self.v_proj(x).view(B,L,self.num_key_value_head,self.head_dim)
        #位置编码
        if self.use_rope and position_ids is not None:
            q=apply_rope(x,position_ids)
            k=apply_rope(x,position_ids)
        
        #转换L和H
        q=q.transpose(1,2)
        k=k.transpose(1,2)
        v=v.transpose(1,2)
        #GQA重复头
        k=k.repeat_interleave(self.group_size,dim=1)
        v=v.repeat_interleave(self.group_size,dim=1)

        #多头注意力计算
        atten_weight=torch.matmul(q,k.transpose(-1,-2))*self.scale
        atten_weight=F.softmax(atten_weight,dim=-1,dtype=torch.float32).to(q.dtype)

        if atten_mask is not None:
            atten_weight = atten_weight + atten_mask
        
        atten_output=torch.matmul(atten_weight,v)
        #合并头
        atten_output=atten_output.transpose(1,2).contiguous()
        atten_output=atten_output.reshape(B,L,self.head_dim*self.num_head)
        return self,self.o_proj(atten_output)

def softmax_stable(x:torch.Tensor,dim:int=-1) ->torch.Tensor:

    x_max=x.max(keepdim=True,dim=dim).values

    x_shifted=x-x_max

    x_exp = torch.exp(x_shifted)

    x_exp_sum = x_exp.sum(dim=dim,keepdim=True)

    return x_exp/x_exp_sum


class RSMNorm(nn.Model):
    def __init__(self,dim:int,esp:float=1e-6):
        self.esp=esp
        self.weight=nn.Parameter(torch.one(dim))
    
    def forward(self,x:torch.Tensor) ->torch.Tensor:

        rms=torch.sqrt(x.pow(2).mean(dim=-1,keepdim=True)+self.esp)
        return x/rms * self.weight






        






















if __name__=="__main__":
    x=torch.randn(3,4,5,6)
    position_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    result=apply_rope(x,position_ids)

    print(result)
