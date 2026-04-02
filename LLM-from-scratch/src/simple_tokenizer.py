###############################
#
# FIRST SIMPLE TOKENIZER CLASS
#
###############################

from utils import create_ids, decode_text, int_to_str, process_text


class SimpleTokenizer:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = int_to_str(vocab)

    def encode(self, text, allowed_special=None):
        preprocessed = process_text(text)
        return create_ids(self.str_to_int, preprocessed)

    def decode(self, ids):
        text = " ".join(self.int_to_str.get(i, f"<{i}>") for i in ids)
        return decode_text(text)

    def example(self):
        example = """"It's the last he painted, you know,"
        Mrs. Gisburn said with pardonable pride."""
        ids = self.encode(example)
        decoded = self.decode(ids)

        print(f'\n✅ TEXT: {example}')
        print(f'✅ IDS: {ids}')
        print(f'✅ DECODER: {decoded}\n')
