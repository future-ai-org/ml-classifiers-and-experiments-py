# Adapted from the book "Reinforcement Learning from Human Feedback" by N. Lambert (2026)

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class BradleyTerryRewardModel(nn.Module):
    """
    Standard scalar reward model for Bradley-Terry preference learning.
    Usage (pairwise BT loss):
        rewards_chosen = model(**inputs_chosen)  # (batch,)
        rewards_rejected = model(**inputs_rejected)  # (batch,)
        loss = -F.logsigmoid(rewards_chosen - rewards_rejected).mean()
    """

    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def _sequence_rep(self, hidden, attention_mask):
        """
        Get a single vector per sequence to score.
        Default: last non-padding token (EOS token); if no mask, last token.

        hidden: (batch, seq_len, hidden_size)
        attention_mask: (batch, seq_len)
        """
        # Index of last non-pad token in each sequence
        # attention_mask is 1 for real tokens, 0 for padding
        lengths = attention_mask.sum(dim=1) - 1  # (batch,)
        batch_idx = torch.arange(hidden.size(0), device=hidden.device)
        return hidden[batch_idx, lengths]  # (batch, hidden_size)

    def forward(self, input_ids, attention_mask):
        """
        A forward pass designed to show inference structure of a standard reward model.
        To train one, this function will need to be modified to compute rewards from
        chosen and rejected inputs, applying the loss above.
        """
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        # Final hidden states: (batch, seq_len, hidden_size)
        hidden = outputs.hidden_states[-1]
        # One scalar reward per sequence: (batch,)
        seq_repr = self._sequence_rep(hidden, attention_mask)
        rewards = self.head(seq_repr).squeeze(-1)
        return rewards


class _MockCausalLM(nn.Module):
    """Tiny LM matching the HF-style call pattern (no transformers dependency)."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=(h,))


class TestBradleyTerryRewardModel(unittest.TestCase):
    def test_forward_output_shape(self):
        torch.manual_seed(0)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = BradleyTerryRewardModel(base)
        b, t = 2, 5
        input_ids = torch.randint(0, 16, (b, t))
        attention_mask = torch.ones(b, t, dtype=torch.long)
        rewards = model(input_ids, attention_mask)
        self.assertEqual(rewards.shape, (b,))
        self.assertTrue(torch.isfinite(rewards).all())

    def test_sequence_rep_last_non_padding_token(self):
        base = _MockCausalLM(hidden_size=4, vocab_size=8)
        model = BradleyTerryRewardModel(base)
        hidden = torch.zeros(1, 5, 4)
        hidden[0, 2, :] = 1.0
        mask = torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.long)
        seq_repr = model._sequence_rep(hidden, mask)
        self.assertTrue(torch.allclose(seq_repr, hidden[0, 2]))

    def test_pairwise_bt_loss_example(self):
        """Docstring-style BT loss on chosen vs rejected batches."""
        torch.manual_seed(1)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = BradleyTerryRewardModel(base)
        b, t = 3, 6
        chosen_ids = torch.randint(0, 16, (b, t))
        rejected_ids = torch.randint(0, 16, (b, t))
        mask_c = torch.ones(b, t, dtype=torch.long)
        mask_r = torch.ones(b, t, dtype=torch.long)
        r_chosen = model(chosen_ids, mask_c)
        r_rejected = model(rejected_ids, mask_r)
        loss = -F.logsigmoid(r_chosen - r_rejected).mean()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Example: single forward + BT-style loss on random pairs
    torch.manual_seed(42)
    base_lm = _MockCausalLM(hidden_size=16, vocab_size=64)
    rm = BradleyTerryRewardModel(base_lm)
    batch, seq = 4, 8
    ids_a = torch.randint(0, 64, (batch, seq))
    ids_b = torch.randint(0, 64, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    ra, rb = rm(ids_a, attn), rm(ids_b, attn)
    example_loss = -F.logsigmoid(ra - rb).mean()
    print("example rewards_a:", ra.detach().tolist())
    print("example rewards_b:", rb.detach().tolist())
    print("example BT loss:", float(example_loss.detach()))
