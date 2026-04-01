###############################
#
# PRETRAINING ON UNLABELED DATA
#
###############################


import torch

from gpt_model import GPT_CONFIG_124M, generate_text_simple
from utils import gpt2_tokenizer


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def run_pretraining_example(model):
    start_context = "Every effort moves you"
    tokenizer = gpt2_tokenizer()
    token_ids = generate_text_simple(
        model=model,
        idx=text_to_token_ids(start_context, tokenizer),
        max_new_tokens=10,
        context_size=GPT_CONFIG_124M['context_length']
    )
    print(f"✅ DECODED: {token_ids_to_text(token_ids, tokenizer)}")

