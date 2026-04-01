####################################
#
#   Creating token embeddings
#   Embedding layers in PyTorch function as a lookup operation, retrieving vectors corresponding to token IDs
#   (rows from the embedding layer's weight matrix via a token ID;
#   IT CAN SEE AS A NN layer that can be optimized via backpropagation)
#
###################################

import torch


def get_embedding(vocab_size, output_dim):
    return torch.nn.Embedding(vocab_size, output_dim)


def print_embedding_example(dataloader, vocab_size, output_dim):

    embedding_layer = get_embedding(vocab_size, output_dim)

    print(f'✅ EMBEDDING LAYER: {embedding_layer}')
    print(f'✅ EMBEDDING LAYER WEIGHT:\n{embedding_layer.weight}\n')
    print(f'✅ EMBEDDING LAYER FOR TENSOR [3] :\n{embedding_layer(torch.tensor([3]))}\n')
    # Each row in this output matrix is obtained via a lookup operation from the embedding weight matrix
    print(f'✅ EMBEDDING LAYER FOR TENSOR [2, 3, 5, 1] :\n{embedding_layer(torch.tensor([2, 3, 5, 1]))}\n')

    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    print(f"✅  TOKEN IDs: {inputs}\n")
    print(f"✅  INPUT SHAPE: {inputs.shape}")

    token_embeddings = embedding_layer(inputs)
    print(f"✅  TOKEN EMBEDDING SHAPE: {token_embeddings.shape}\n")
