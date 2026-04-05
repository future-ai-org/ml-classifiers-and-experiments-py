# Adapted from the book "Reinforcement Learning from Human Feedback" by N. Lambert (2026)

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class OutcomeRewardModel(nn.Module):
    """
    Per-token logits for outcome (correctness) supervision on completions.

    Forward passes `**inputs` to the base LM (e.g. `input_ids`, `attention_mask`).
    `labels` align with sequence length: -100 ignores prompt/padding; 0/1 mark
    incorrect/correct completion tokens. Training loss is BCE-with-logits on
    positions where `labels != -100`.
    """

    def __init__(self, base_lm):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.head = nn.Linear(self.lm.config.hidden_size, 1)

    def forward(self, labels=None, **inputs):
        """
        Pass LM kwargs via `**inputs` (e.g. `input_ids`, `attention_mask`).
        Optional `labels` for training; not forwarded to `self.lm`.

        Returns:
            (loss, logits_per_token): `loss` is None if `labels` is None or every
            position is ignored (-100). `logits_per_token` is (batch, seq_len).
        """
        # Same idea as: hidden = model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
        # `return_dict=True` is needed for `.hidden_states` on HuggingFace model outputs.
        hidden = self.lm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        logits_per_token = self.head(hidden).squeeze(-1)  # (batch, seq_len)
        # Other codebases sometimes fold this into a single `model.forward()` helper.

        loss = None
        if labels is not None:
            # Binary labels: 1=correct, 0=incorrect (prompt tokens masked as -100)
            mask = labels != -100
            if mask.any():
                loss = F.binary_cross_entropy_with_logits(
                    logits_per_token[mask],
                    labels[mask].float(),
                )

        return loss, logits_per_token


class _MockCausalLM(nn.Module):
    """Tiny LM matching the HF-style call pattern (no transformers dependency)."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=(h,))


class TestOutcomeRewardModel(unittest.TestCase):
    def test_logits_shape(self):
        torch.manual_seed(0)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = OutcomeRewardModel(base)
        b, t = 2, 7
        input_ids = torch.randint(0, 16, (b, t))
        attn = torch.ones(b, t, dtype=torch.long)
        loss, logits = model(input_ids=input_ids, attention_mask=attn, labels=None)
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (b, t))
        self.assertTrue(torch.isfinite(logits).all())

    def test_loss_on_completion_tokens_only(self):
        torch.manual_seed(1)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = OutcomeRewardModel(base)
        b, t = 2, 6
        input_ids = torch.randint(0, 16, (b, t))
        attn = torch.ones(b, t, dtype=torch.long)
        labels = torch.full((b, t), -100, dtype=torch.long)
        labels[:, 3:] = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.long)
        loss, logits = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.shape, (b, t))

    def test_no_supervised_tokens_yields_no_loss(self):
        base = _MockCausalLM(hidden_size=4, vocab_size=8)
        model = OutcomeRewardModel(base)
        b, t = 1, 4
        input_ids = torch.randint(0, 8, (b, t))
        labels = torch.full((b, t), -100, dtype=torch.long)
        loss, _ = model(input_ids=input_ids, labels=labels)
        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)

    torch.manual_seed(42)
    base_lm = _MockCausalLM(hidden_size=16, vocab_size=64)
    orm = OutcomeRewardModel(base_lm)
    batch, seq = 2, 10
    input_ids = torch.randint(0, 64, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    # pretend last 4 tokens are completion: some correct (1), some not (0)
    labels[:, 6:] = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.long)

    print("training (labels on completion tokens):")
    loss, logits = orm(
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
    )
    print("  logits shape:", tuple(logits.shape))
    print("  loss:", float(loss.detach()))

    print("inference (omit labels — same logits, loss not computed):")
    _, logits_inf = orm(input_ids=input_ids, attention_mask=attn, labels=None)
    print("  logits shape:", tuple(logits_inf.shape))
