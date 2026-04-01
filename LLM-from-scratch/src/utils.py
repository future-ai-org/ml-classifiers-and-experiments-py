###################################################
#
#   Several util methods used accross the project
#
###################################################

import re

import tiktoken


def open_txt(filename):
    try:
        with open(filename, encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        print(f"\n❌ ERROR OPENING {filename}: {e}\n")


def process_text(text):
    preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    return [i.strip() for i in preprocessed if i.strip()]


def sort_and_remove_dups(text):
    return sorted(list(set(text)))


def create_vocabulary(words):
    return {t: i for i, t in enumerate(words)}


def int_to_str(vocab):
    return  {i: s for s, i in vocab.items()}


def create_ids(vocab, words):
    return [vocab[s] for s in words]


def decode_text(text):
    return re.sub(r'\s+([,.?!"()\'])', r'\1', text)


def gpt2_tokenizer():
    return tiktoken.get_encoding("gpt2")


def bte(text):
    tokenizer = gpt2_tokenizer()
    ids = tokenizer.encode(text)
    return ids
