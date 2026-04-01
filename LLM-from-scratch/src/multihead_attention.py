##################################################
#
# A WRAPPER CLASS TO IMPLEMENT MULTIHEAD ATTENTION
#
##################################################


import torch
import torch.nn as nn

from self_attention import CausalAttention

torch.manual_seed(1337)

INPUT = torch.tensor(
            [[0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55]]
        )

X_2 = INPUT[1]
D_IN = INPUT.shape[1]
D_OUT = 2
BATCH = torch.stack((INPUT, INPUT), dim=0)


class MultiHeadAttentionWrapper(nn.Module):

    def __init__(self, dropout, num_heads, qkv_bias=False):

        super().__init__()
        self.context_length = BATCH.shape[1]
        self.heads = nn.ModuleList([CausalAttention(dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)


class MultiHeadAttention(nn.Module):

    def __init__(self, dropout, num_heads, qkv_bias=False, d_in=None, d_out=None, context_length=None):

        super().__init__()

        self.context_length = context_length or BATCH.shape[1]
        self.d_in = d_in or D_IN
        self.d_out = d_out or D_OUT
        self.num_heads = num_heads
        self.head_dim = self.d_out // num_heads
        self.W_query = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_key = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_value = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(self.d_in, self.d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1))

    def forward(self, x):

        b, num_tokens, _d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2)

        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


def simple_mha_example():

    a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573], [0.8993, 0.0390, 0.9268, 0.7388], [0.7179, 0.7058, 0.9156, 0.4340]], [[0.0772, 0.3565, 0.1479, 0.5331], [0.4066, 0.2318, 0.4545, 0.9737], [0.4606, 0.5159, 0.4220, 0.5786]]]])

    first_head = a[0, 0, :, :]
    first_res = first_head @ first_head.T
    print(f"✅ FIRST HEAD: {first_res}")
    second_head = a[0, 1, :, :]
    second_res = second_head @ second_head.T
    print(f"✅ SECOND HEAD: {second_res}")
