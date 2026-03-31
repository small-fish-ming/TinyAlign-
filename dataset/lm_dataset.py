"""
SFTDataset: 监督微调(Supervised Fine-Tuning)专用数据集类
================================================================
核心功能：
1. 从JSONL文件加载对话数据
2. 使用tokenizer的chat template标准化对话格式
3. 智能生成训练标签：仅对assistant回复部分计算损失（用户/系统消息部分设为-100）
4. 自动处理工具调用(function calling)场景
5. 统一序列长度并生成PyTorch张量

设计亮点：
✅ 动态标签生成：通过匹配BOS/EOS标记精准定位assistant回复区间
✅ 工具调用支持：自动提取并传递functions参数给chat template
✅ 损失聚焦：避免模型在用户输入部分过拟合，提升训练效率
✅ 与Hugging Face生态无缝集成（datasets + tokenizer）
"""

from torch.utils.data import Dataset
import torch
import os
import random
from datasets import load_dataset

# 设置tokenizers不并行加速，避免报错
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def pre_processing_chat(conversations, add_system_ratio=0.2):
    SYSTEM_PROMPTS = [
        "你是一个知识丰富的AI，尽力为用户提供准确的信息。",
        "你是minimind，一个小巧但有用的语言模型。",
        "你是一个专业的AI助手，请提供有价值的回答。",
        "你是minimind，请尽力帮助用户解决问题。",
        "你是一个可靠的AI，请给出准确的回答。",
        "You are a helpful AI assistant.",
        "You are minimind, a lightweight intelligent assistant.",
        "You are a friendly chatbot. Please answer the user's questions carefully.",
        "You are a knowledgeable AI. Try your best to provide accurate information.",
        "You are minimind, a small but useful language model."
    ]
    if conversations and conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations

def post_processing_chat(prompt_content, empty_think_ratio=0.05):
    if '<think>\n\n</think>\n\n' in prompt_content and random.random() > empty_think_ratio:
        prompt_content = prompt_content.replace('<think>\n\n</think>\n\n', '')
    return prompt_content



# 定义dataset类，继承自torch.utils.data.Dataset
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples=self.load_data(data_path)
        
    def load_data(self, data_path):
        samples=[]
        with open(data_path,'r',encoding='utf-8') as f:
            for line_num,line in enumerate(f,1):
                data=json.loads(line.strip())
                samples.append(data)
        return samples


    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        sample=self.samples[index]

        encoding=self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        #(max_length,)
        input_ids=encoding['input_ids'].squeeze(0)

        # [1,1,1,0,0]
        loss_mask=input_ids!=self.tokenizer.pad_token_id

        # 自回归
        X=torch.tensor(input_ids[:-1],dtype=torch.long)
        Y=torch.tensor(input_ids[1:],dtype=torch.long)

        loss_mask=torch.tensor(loss_mask[:-1],dtype=torch.long)

        return X,Y,loss_mask

        
class SFTDataset(torch.utils.data.Dataset):
    """
    监督微调数据集实现类
    
    关键设计思想：
    - 标签生成策略：仅让模型学习"生成assistant回复"，而非复现整个对话历史
    - 特殊标记处理：通过预提取BOS/EOS token ID实现精准区间匹配
    - 安全填充：使用tokenizer原生pad_token_id保证兼容性
    """
    
    def __init__(self, jsonl_path: str, tokenizer, max_length: int = 1024):
        """
        初始化数据集
        
        Args:
            jsonl_path: JSONL格式数据文件路径（每行一个对话样本）
            tokenizer: Hugging Face tokenizer（需支持apply_chat_template）
            max_length: 序列最大长度（含填充），超过则截断
        `
        关键初始化细节：
        - self.bos_id: 提取"assistant回复开始"的token序列（含换行符）
          示例：若bos_token="<|start_header_id|>"，则编码"<|start_header_id|>assistant\n"
        - self.eos_id: 提取"回复结束"的token序列（含换行符）
          示例：编码"<|eot_id|>\n"（具体取决于tokenizer配置）
        - 为何用add_special_tokens=False：避免tokenizer二次添加BOS/EOS，确保匹配准确性
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 使用datasets库高效加载JSONL（支持流式、分片等高级特性）
        self.samples = load_dataset('json', data_files=jsonl_path, split='train')
        
        # 预提取关键标记的token ID序列（用于后续精准定位assistant回复区间）
        # 注意：字符串包含角色标识"assistant"和换行符，与chat template输出格式严格对齐
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids

    def __len__(self) -> int:
        """返回数据集样本总数"""
        return len(self.samples)

    def create_chat_prompt(self, conversations: List[Dict]) -> str:
        """
        将原始对话转换为模型可识别的标准化提示文本
        
        Args:
            conversations: 对话列表，格式示例：
                [
                    {"role": "system", "content": "...", "functions": [...]}, 
                    {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}
                ]
        
        Returns:
            格式化后的完整对话字符串（含特殊标记）
        
        特殊处理：
        - 自动检测system消息中的functions字段（用于工具调用场景）
        - tools参数会传递给tokenizer.apply_chat_template，生成包含工具描述的提示
        - tokenize=False：仅返回文本，便于后续统一tokenize和调试
        """
        messages = conversations.copy()
        # 提取工具定义（仅当首条为system消息且含functions时）
        tools = (
            conversations[0].get("functions") 
            if (conversations and conversations[0]["role"] == "system" and "functions" in conversations[0]) 
            else None
        )
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,          # 返回字符串而非token IDs
            add_generation_prompt=False,  # 不添加生成起始标记（训练时不需要）
            tools=tools              # 传递工具定义（若存在）
        )

    def generate_labels(self, input_ids: List[int]) -> List[int]:
        """
        生成训练标签：仅保留assistant回复部分的token ID，其余设为-100
        
        核心逻辑：
        1. 初始化全-100标签（PyTorch CrossEntropyLoss会忽略-100）
        2. 扫描input_ids，定位所有"assistant回复区间"：
           [BOS标记] → [回复内容] → [EOS标记]
        3. 将区间内（含EOS标记）的token ID复制到labels对应位置
        
        为何包含EOS标记？
        - 让模型学会何时结束回复（关键训练信号）
        - 避免生成无限长文本
        
        边界处理：
        - 区间结束位置取min(end + len(eos_id), max_length)防止越界
        - 严格匹配token序列（避免单token误匹配）
        """
        labels = [-100] * len(input_ids)  # -100在损失计算中被忽略
        i = 0
        while i < len(input_ids):
            # 检测是否到达assistant回复起始位置（匹配BOS序列）
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)  # 回复内容起始索引
                end = start
                
                # 向后扫描寻找EOS标记
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                
                # 标记有效回复区间（含EOS标记）
                reply_end = min(end + len(self.eos_id), self.max_length)
                for j in range(start, reply_end):
                    labels[j] = input_ids[j]  # 保留需预测的token
                
                # 跳过已处理区间，继续扫描
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1  # 未匹配到BOS，继续移动
        
        return labels

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个训练样本（input_ids + labels）
        
        处理流程：
        1. 加载原始样本 → 2. 对话预处理 → 3. 应用chat template → 
        4. 后处理 → 5. Tokenize → 6. 截断/填充 → 7. 生成标签
        
        关键细节：
        - pre_processing_chat/post_processing_chat：外部定义的清洗函数（处理特殊字符、格式等）
        - 填充策略：右侧填充（right-padding），使用tokenizer原生pad_token_id
        - 标签同步填充：labels在填充位置保持-100（避免影响损失计算）
        
        返回：
            (input_ids_tensor, labels_tensor) 
            shape: (max_length,), dtype: torch.long
        """
        # 1. 加载原始样本（含conversations字段）
        sample = self.samples[index]
        
        # 2. 对话预处理（清洗、标准化等，函数需外部实现）
        conversations = pre_processing_chat(sample['conversations'])
        
        # 3. 生成标准化对话提示文本
        prompt = self.create_chat_prompt(conversations)
        
        # 4. 后处理（如移除多余空格、特殊字符转义等）
        prompt = post_processing_chat(prompt)
        
        # 5. Tokenize并截断至max_length
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        
        # 6. 右侧填充至统一长度（labels填充位置自动为-100）
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids += [self.tokenizer.pad_token_id] * padding_length
        
        # 7. 生成训练标签（仅assistant回复部分有效）
        labels = self.generate_labels(input_ids)
        
        # === 调试辅助（使用时取消注释）===
        # print(f"\n--- Sample {index} 标签生成示例 ---")
        # for i, (tok_id, label) in enumerate(zip(input_ids, labels)):
        #     token_str = self.tokenizer.decode([tok_id], skip_special_tokens=False)
        #     print(f"{i:3d}: Token={token_str!r:15s} | Label={'[IGNORED]' if label == -100 else self.tokenizer.decode([label])}")
        # ==================================
        
        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long)
        )
