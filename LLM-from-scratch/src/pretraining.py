###############################
#
# PRETRAINING ON UNLABELED DATA
#
###############################


import torch

from gpt_dataloader import create_dataloader
from gpt_model import GPT_CONFIG_124M, generate_text_simple
from utils import gpt2_tokenizer


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())


def softmax(model, inputs):
    probas, logits = None, None
    with torch.no_grad():
        logits = model(inputs)
        probas = torch.softmax(logits, dim=-1)
        print(f'✅ Probability shape: {probas.shape}')
    return probas, logits


def argmax(probas):
    token_ids = torch.argmax(probas, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)
    return token_ids


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


def run_text_generation_loss_example(model, _tokenizer, raw_text):
    # Hardcoded batch uses GPT-2 BPE ids; decode/count tokens with tiktoken, not a local word vocab.
    gpt_tok = gpt2_tokenizer()
    inputs = torch.tensor([[16833, 3626, 6100], # ["every effort moves",
                          [40, 1107, 588]]) # "I really like"]
    targets = torch.tensor([[3626, 6100, 345 ], # [" effort moves you",
                           [1107, 588, 11311]]) # " really like chocolate"]

    probas, logits = softmax(model, inputs)
    argmax(probas)

    # print an initial softmax prob scores to the target tokens
    text_idx = 0
    target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"✅ PROBABILITY TEXT 1: {target_probas_1}")
    text_idx = 1
    target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
    print(f"✅ PROBABILITY TEXT 2: {target_probas_2}")

    log_probas = torch.log(torch.cat((target_probas_1, target_probas_2))) * -1
    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    perplexity = torch.exp(loss)
    total_characters = len(raw_text)
    total_tokens = len(gpt_tok.encode(raw_text))

    print(f"✅ LOSS OF PROBABILITY SCORE: {log_probas}")
    print(f"✅ LOGITS SHAPE (BATCH SIZE, # OF TOKENS, VOCAB SIZE): {logits.shape}")
    print(f"✅ TARGET SHAPE (BATCH SIZE, # OF TOKENS): {targets.shape}")
    print(f"✅ FLATTENED LOGITS: {logits_flat.shape}")
    print(f"✅ FLATTENED TARGET: {targets_flat.shape}")
    print(f"✅ LOSS: {loss}")
    print(f"✅ PERPLEXITY {perplexity}")
    print(f"✅ CHARACTERS {total_characters}")
    print(f"✅ TOKENS: {total_tokens}")


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
            input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def train_and_validation_loss_example(raw_text, model):

    train_ratio = 0.90
    split_idx = int(train_ratio * len(raw_text))
    train_data = raw_text[:split_idx]
    val_data = raw_text[split_idx:]

    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(val_loader, model, device)
    print(f"✅ TRAINING LOSS: {train_loss}")
    print(f"✅ VALIDATION LOSS: {val_loss}")

    return train_loader, val_loader
