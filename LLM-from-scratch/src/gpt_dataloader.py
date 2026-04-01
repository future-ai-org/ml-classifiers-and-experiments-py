#######################################################
#
# DATALOADER TO GENERATE BATCHES WITH INPUT WITH PAIRS
#
########################################################


import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataloader(Dataset):

    def __init__(self, text, tokenizer, max_length, stride):

        self.text = text
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride

        self.input_ids = []
        self.target_ids = []

        self._set_ids()

    def _set_ids(self):
        token_ids = self.tokenizer.encode(self.text)
        for i in range(0, len(token_ids) - self.max_length, self.stride):
            input_chunk = token_ids[i:i + self.max_length]
            target_chunk = token_ids[i + 1: i + self.max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
                text,
                batch_size=4,
                max_length=256,
                stride=128,
                shuffle=True,
                drop_last=True,
                num_workers=0):

        tokenizer = tiktoken.get_encoding("gpt2")
        dataset = GPTDataloader(text, tokenizer, max_length, stride)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers
        )

def print_batch_example(dataloader):

    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(f"✅ FIRST BATCH: {first_batch}")
    second_batch = next(data_iter)
    print(f"✅ SECOND BATCH: {second_batch}\n")

