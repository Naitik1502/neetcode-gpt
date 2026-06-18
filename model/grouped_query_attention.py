import torch
import torch.nn as nn
from torchtyping import TensorType

class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.num_groups = self.num_heads//self.num_kv_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        B, T, D = x.shape

        q = self.q_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).permute(0,2,3,1)
        v = self.v_proj(x).reshape(B, T, self.num_kv_heads, self.head_dim).permute(0,2,1,3)
        
        k = torch.repeat_interleave(k, self.num_groups, dim = 1)
        v = torch.repeat_interleave(v, self.num_groups, dim = 1)
        
        attn_score = (q@k)/math.sqrt(self.head_dim)
        

        mask = (torch.ones((T,T))*float('-inf')).triu(diagonal=1)
        attn_score = attn_score[:,::, ::] + mask
        attn_score = nn.functional.softmax(attn_score, dim=-1)

        final_attn = (attn_score@v).permute(0,2,1,3).reshape(B, T, -1)
        
        return self.output_proj(final_attn)
