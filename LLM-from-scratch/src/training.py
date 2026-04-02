#######################
#
#   Training a LLM
#
#######################


import torch

from gpt_model import generate_text_simple
from pretraining import (
    calc_loss_batch,
    calc_loss_loader,
    text_to_token_ids,
    token_ids_to_text,
)
from utils import gpt2_tokenizer


def run_training_example(model, train_loader, val_loader, tokenizer):

    optimizer  = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    num_epochs = 10
    eval_freq  = 5
    eval_iter  = 5
    start_context = "Every effort moves you"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses, val_losses, track_tokens_seen = train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer)

    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

    generate_and_print_sample(model, tokenizer, device, start_context)


def train_model_simple(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            # resets gradients from the previous iteration
            optimizer.zero_grad()

            loss = calc_loss_batch(
            input_batch, target_batch, model, device
            )
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                f"Train loss {train_loss:.3f}, "
                f"Val loss {val_loss:.3f}"
                )
        generate_and_print_sample(model, tokenizer, device, start_context)
    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()

    with torch.no_grad():
        train_loss = calc_loss_loader(
        train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
        val_loader, model, device, num_batches=eval_iter
        )
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, _tokenizer, device, start_context):
    model.eval()
    # GPTModel uses GPT-2 vocab (50257); local SimpleTokenizer cannot encode/decode those ids.
    gpt_tok = gpt2_tokenizer()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, gpt_tok).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
        model=model, idx=encoded,
        max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, gpt_tok)
    print(decoded_text.replace("\n", " "))
    model.train()
