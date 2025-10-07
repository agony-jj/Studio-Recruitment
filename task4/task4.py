import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask=None):
     # --- 请你完成这部分代码 ---
    y = torch.matmul(q, k.transpose(-2,-1))/ math.sqrt(k.size(-1))
    if mask is not None:
        y = y.masked_fill(mask == 0, float('-inf'))
    attention_weights = torch.softmax(y, dim=-1)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights

class SimpleSelfAttention(nn.Module):
    def __init__(self, embed_dim, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(embed_dim, head_dim)
        self.k_proj = nn.Linear(embed_dim, head_dim)
        self.v_proj = nn.Linear(embed_dim, head_dim)
    
    def forward(self, x):
        # --- 请你完成这部分代码 ---
        # 1. 将 x 传入线性层得到 q, k, v
        # 2. 调用 scaled_dot_product_attention
        # 3. 返回输出结果
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        output, attention_weights = scaled_dot_product_attention(q, k, v)
        return output, attention_weights
    
    
# --- 验证部分 ---
# 创建实例和输入数据，并检查输出形状
embed_dim = 512
head_dim = 64
attention = SimpleSelfAttention(embed_dim, head_dim)
x = torch.rand(10, 20, embed_dim)  # (batch_size, seq_len, embed_dim)
output, attn_weights = attention(x)
print(output.shape)  # 应该是 (10, 20, head_dim)
print(attn_weights.shape)  # 应该是 (10, 20, 20) 