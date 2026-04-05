# Adapted from the book "Reinforcement Learning from Human Feedback" by N. Lambert (2026)

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessRewardModel(nn.Module):
    """
    Per-token multi-class logits for process supervision (e.g. step quality at boundaries).

    Tokenized prompts and completions; the end of a "reasoning step" is marked in the
    data (e.g. separator token), and labels are provided only at those positions.
    Use class indices in ``[0, num_classes - 1]`` at step boundaries and ``-100`` elsewhere
    (same masking convention as outcome RM; loss is only cross-entropy on ``labels != -100``).
    """

    def __init__(self, base_lm, num_classes: int = 3):
        super().__init__()
        self.lm = base_lm  # e.g., AutoModelForCausalLM
        self.num_classes = num_classes
        self.head = nn.Linear(self.lm.config.hidden_size, num_classes)

    def forward(self, labels=None, **inputs):
        """
        Pass LM kwargs via ``**inputs`` (e.g. ``input_ids``, ``attention_mask``).
        Optional ``labels`` (long, same length as sequence) for training; not sent to ``self.lm``.

        Returns:
            (loss, logits): ``loss`` is None if ``labels`` is None or no supervised positions.
            ``logits`` has shape (batch, seq_len, num_classes).
        """
        # Same idea as: model.lm(**inputs, output_hidden_states=True).hidden_states[-1]
        hidden = self.lm(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]
        logits = self.head(hidden)  # (batch, seq_len, num_classes)

        loss = None
        if labels is not None:
            mask = labels != -100
            if mask.any():
                loss = F.cross_entropy(logits[mask], labels[mask])

        return loss, logits


class _MockCausalLM(nn.Module):
    """Tiny LM matching the HF-style call pattern (no transformers dependency)."""

    def __init__(self, hidden_size: int = 8, vocab_size: int = 32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size)

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False, return_dict=False):
        h = self.embed(input_ids)
        return SimpleNamespace(hidden_states=(h,))


class TestProcessRewardModel(unittest.TestCase):
    def test_logits_shape(self):
        torch.manual_seed(0)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = ProcessRewardModel(base, num_classes=3)
        b, t = 2, 7
        input_ids = torch.randint(0, 16, (b, t))
        attn = torch.ones(b, t, dtype=torch.long)
        loss, logits = model(input_ids=input_ids, attention_mask=attn, labels=None)
        self.assertIsNone(loss)
        self.assertEqual(logits.shape, (b, t, 3))
        self.assertTrue(torch.isfinite(logits).all())

    def test_loss_at_step_boundaries(self):
        torch.manual_seed(1)
        base = _MockCausalLM(hidden_size=8, vocab_size=16)
        model = ProcessRewardModel(base, num_classes=3)
        b, t = 2, 8
        input_ids = torch.randint(0, 16, (b, t))
        attn = torch.ones(b, t, dtype=torch.long)
        labels = torch.full((b, t), -100, dtype=torch.long)
        labels[:, 2] = 0
        labels[:, 5] = 2
        loss, logits = model(input_ids=input_ids, attention_mask=attn, labels=labels)
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(logits.shape, (b, t, 3))

    def test_no_supervised_tokens_yields_no_loss(self):
        base = _MockCausalLM(hidden_size=4, vocab_size=8)
        model = ProcessRewardModel(base, num_classes=3)
        b, t = 1, 4
        input_ids = torch.randint(0, 8, (b, t))
        labels = torch.full((b, t), -100, dtype=torch.long)
        loss, _ = model(input_ids=input_ids, labels=labels)
        self.assertIsNone(loss)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)

    torch.manual_seed(42)
    base_lm = _MockCausalLM(hidden_size=16, vocab_size=64)
    prm = ProcessRewardModel(base_lm, num_classes=3)
    batch, seq = 2, 12
    input_ids = torch.randint(0, 64, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)
    labels = torch.full((batch, seq), -100, dtype=torch.long)
    # e.g. labels at a few "step boundary" positions (class 0, 1, or 2)
    labels[0, 3] = 0
    labels[0, 7] = 1
    labels[1, 4] = 2
    labels[1, 9] = 0

    print("training (labels only at step boundaries):")
    loss, logits = prm(
        input_ids=input_ids,
        attention_mask=attn,
        labels=labels,
    )
    print("  logits shape:", tuple(logits.shape))
    print("  loss:", float(loss.detach()))

    print("inference (omit labels — same logits, loss not computed):")
    _, logits_inf = prm(input_ids=input_ids, attention_mask=attn, labels=None)
    print("  logits shape:", tuple(logits_inf.shape))
