#############################################################
#
# A SIMPLE SELF-ATTENTION MECHANISM WITHOUT TRAINABLE WEIGHTS
#
#############################################################

import torch
import torch.nn as nn

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


class SelfAttentionManual:

    def softmax_naive(self, x):
        return torch.exp(x) / torch.exp(x).sum(dim=0)

    def simple_example(self):
        query = INPUT[1]
        attn_scores = torch.empty(INPUT.shape[0])

        for i, x in enumerate(INPUT):
            attn_scores[i] = torch.dot(x, query)
            print(f'✅ ATTENTION SCORE FOR {x} --> {attn_scores}')

        attn_weights = attn_scores / attn_scores.sum()
        print(f'\n✅ ATTENTION WEIGHTS: {attn_weights}')
        print(f'✅ ATTENTION SUM: {attn_weights.sum()}\n')

        attn_weights_2_naive = self.softmax_naive(attn_scores)
        print(f'✅ ATTENTION WEIGHTS NAIVE: {attn_weights_2_naive}')
        print(f'✅ ATTENTION SUM: {attn_weights_2_naive.sum()}\n')

        attn_weights_2 = torch.softmax(attn_scores, dim=0)
        print(f'✅ ATTENTION WEIGHTS NAIVE: {attn_weights_2}')
        print(f'✅ ATTENTION SUM: {attn_weights_2.sum()}\n')

        # calculate the context vector z: by multiplying the embedded input tokens, x(i),
        # with the attention weights - then summing the resulting vectors
        # (the weighted sum of all input vectors)
        query = INPUT[1]
        context_vec_2 = torch.zeros(query.shape)
        for i, x_i in enumerate(INPUT):
                context_vec_2 += attn_weights_2[i] * x_i
                print(f'✅ CONTEXT VECTOR FOR {x_i} --> {context_vec_2}')


        # compute the attention weights for all input tokens
        attn_scores = torch.empty(6, 6)
        for i, x_i in enumerate(INPUT):
            for j, x_j in enumerate(INPUT):
                attn_scores[i, j] = torch.dot(x_i, x_j)
        print(f'\n✅ ATTENTION VECTOR: {attn_scores}')

        # same results using matrix multiplication
        attn_scores = INPUT @ INPUT.T
        print(f'\n✅ ATTENTION VECTOR (MAT MULT): {attn_scores}')

        # normalize each row so that the values in each row sum to 1
        attn_weights = torch.softmax(attn_scores, dim =- 1)
        print(f'\n✅ ATTENTION WEIGHTS: {attn_weights}')

        # verify normalization
        _row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
        print(f'\n✅ EVERYTHING NORMALIZED: {attn_weights.sum(dim=-1)}')

        # use these attention weights to compute all
        # context vectors via matrix multiplication
        all_context_vecs = attn_weights @ INPUT
        print(f'\n✅ ALL CONTEXT VECTORS: {all_context_vecs}\n')


    def trainable_weights_example(self):
        '''
        self-attention mechanism used in the original transformer architecture,
        the GPT models, and most other popular LLMs. this self-attention mechanism
        is also called scaled dot-product attention.

        implement the self-attention mechanism step by step by introducing the three trainable
        weight matrices Wq, Wk, and Wv. these three matrices are used to project the embedded
        input tokens, x(i), into query, key, and value vectors.
        '''

        w_query = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)
        w_key = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)
        w_value = torch.nn.Parameter(torch.rand(D_IN, D_OUT), requires_grad=False)

        query_2 = X_2 @ w_query
        _key_2 = X_2 @ w_key
        _value_2 = X_2 @ w_value
        print(f'✅ QUERY MATRIX FOR 2ND TOKEN: {query_2}')

        keys = INPUT @ w_key
        values = INPUT @ w_value
        print(f'✅ KEY MATRIX SHAPE: {keys.shape}')
        print(f'✅ VALUE MATRIX SHAPE: {values.shape}')

        keys_2 = keys[1]
        attn_score = query_2.dot(keys_2)
        print(f'✅ UNNORMALIZED ATTENTION SCORE FOR 2ND TOKEN: {attn_score}')

        # generalize this computation to all attention scores via matrix multiplication
        attn_scores = query_2 @ keys.T
        print(f'✅ ALL ATTENTION SCORES FOR GIVEN QUERY MATRIX: {attn_scores}')

        d_k = keys.shape[-1]
        attn_weights = torch.softmax(attn_scores / d_k**0.5, dim=-1)
        print(f'✅ ATTENTION WEIGHTS: {attn_weights}')

        context_vec = attn_weights @ values
        print(f'✅ CONTEXT VECTOR: {context_vec}\n')


class SelfAttention(nn.Module):

    def __init__(self, qkv_bias=False):

        super().__init__()
        self.W_query = nn.Linear(D_IN, D_OUT, bias=qkv_bias)
        self.W_key = nn.Linear(D_IN, D_OUT, bias=qkv_bias)
        self.W_value = nn.Linear(D_IN, D_OUT, bias=qkv_bias)

    def forward(self, x):

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1)
        context_vec = attn_weights @ values

        return context_vec


def print_self_attention_nn(self_attention):
    print(self_attention(INPUT))
    print()



class CausalAttention(nn.Module):

    def __init__(self, dropout, qkv_bias=False):

        super().__init__()
        context_length = BATCH.shape[1]

        self.d_out = D_OUT
        self.W_query = nn.Linear(D_IN, D_OUT, bias=qkv_bias)
        self.W_key = nn.Linear(D_IN, D_OUT, bias=qkv_bias)
        self.W_value = nn.Linear(D_IN, D_OUT, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):

        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2)
        attn_scores.masked_fill_(
        self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(
        attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        attn_weights = self.dropout(attn_weights)
        return attn_weights @ values


def get_context_vector(causal_attention):
    print(f' ✅ CONTEXT VECTOR --> {causal_attention(BATCH).shape}\n')


