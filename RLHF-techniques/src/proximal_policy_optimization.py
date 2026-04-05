# Adapted from the book "Reinforcement Learning from Human Feedback" by N. Lambert (2026)

"""
Basic PPO-style critic targets and value loss (no GAE).

Shapes use batch size B and completion length L.
"""

import unittest

import torch


def ppo_value_loss_and_advantages(
    rewards,
    done_mask,
    completion_mask,
    values,
    old_values=None,
    *,
    gamma: float = 1.0,
    epsilon_v=None,
):
    """
    Monte Carlo returns per token (reset at terminals), optional PPO value clipping,
    masked mean squared error for the critic, and TD-free advantages A_t = G_t - V(s_t).

    Args:
        rewards: (B, L) per-token rewards (e.g. post-KL); terminal position can include
            outcome bonus.
        done_mask: (B, L) 1.0 at terminal token (EOS / truncated end), else 0.0.
        completion_mask: (B, L) 1.0 on response tokens to supervise (ignore prompt).
        values: (B, L) current critic V_theta(s_t).
        old_values: (B, L) critic at rollout time V_{theta_old}(s_t); required for clipping.
        gamma: Discount (often 1.0 for LM RLHF).
        epsilon_v: Value clip radius (e.g. 0.2). If None or ``old_values`` is None,
            clipping is disabled (pure MSE to targets).

    Returns:
        value_loss: scalar.
        advantages: (B, L), detached; for policy gradient / PPO policy loss.
        returns: (B, L) Monte Carlo targets G_t (same as ``targets`` in the book snippet).
    """
    b, seq_len = rewards.shape

    returns = torch.zeros_like(rewards)
    running = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for t in reversed(range(seq_len)):
        running = rewards[:, t] + gamma * (1.0 - done_mask[:, t]) * running
        returns[:, t] = running

    targets = returns
    v_pred = values
    vf_unclipped = 0.5 * (v_pred - targets) ** 2

    if old_values is not None and epsilon_v is not None:
        v_old = old_values
        v_clip = torch.clamp(v_pred, v_old - epsilon_v, v_old + epsilon_v)
        vf_clipped = 0.5 * (v_clip - targets) ** 2
        vf_loss_tok = torch.max(vf_unclipped, vf_clipped)
    else:
        vf_loss_tok = vf_unclipped

    denom = completion_mask.sum(dim=1).clamp_min(1.0)
    value_loss = ((vf_loss_tok * completion_mask).sum(dim=1) / denom).mean()
    advantages = (targets - v_pred).detach()

    return value_loss, advantages, returns


class TestPPOValueLoss(unittest.TestCase):
    def test_mc_returns_no_terminal_gamma_one(self):
        rewards = torch.tensor([[1.0, 2.0, 3.0]])
        done = torch.zeros_like(rewards)
        comp = torch.ones_like(rewards)
        values = torch.zeros_like(rewards)
        vl, adv, ret = ppo_value_loss_and_advantages(
            rewards, done, comp, values, gamma=1.0
        )
        torch.testing.assert_close(ret, torch.tensor([[6.0, 5.0, 3.0]]))
        torch.testing.assert_close(adv, ret)
        self.assertTrue(torch.isfinite(vl))

    def test_mc_returns_resets_at_done(self):
        rewards = torch.tensor([[1.0, 10.0, 100.0]])
        done = torch.tensor([[0.0, 1.0, 0.0]])
        comp = torch.ones_like(rewards)
        values = torch.zeros_like(rewards)
        _, _, ret = ppo_value_loss_and_advantages(rewards, done, comp, values, gamma=1.0)
        # t=2: 100; t=1: 10 (terminal, no carry); t=0: 1 + 10 = 11
        torch.testing.assert_close(ret, torch.tensor([[11.0, 10.0, 100.0]]))

    def test_value_clipping_changes_loss(self):
        rewards = torch.ones(1, 4)
        done = torch.zeros_like(rewards)
        comp = torch.ones_like(rewards)
        values = torch.full_like(rewards, 5.0)
        old_values = torch.zeros_like(rewards)
        vl_clip, _, _ = ppo_value_loss_and_advantages(
            rewards,
            done,
            comp,
            values,
            old_values,
            gamma=1.0,
            epsilon_v=0.2,
        )
        vl_no_clip, _, _ = ppo_value_loss_and_advantages(
            rewards, done, comp, values, None, gamma=1.0, epsilon_v=None
        )
        self.assertNotEqual(float(vl_clip), float(vl_no_clip))

    def test_completion_mask_weights_denominator(self):
        rewards = torch.ones(2, 3)
        done = torch.zeros_like(rewards)
        comp = torch.tensor([[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        values = torch.zeros_like(rewards)
        vl, _, ret = ppo_value_loss_and_advantages(rewards, done, comp, values, gamma=1.0)
        row0_mean = (0.5 * ret[0] ** 2 * comp[0]).sum() / comp[0].sum()
        row1_mean = (0.5 * ret[1] ** 2 * comp[1]).sum() / comp[1].sum()
        expected = 0.5 * (row0_mean + row1_mean)
        torch.testing.assert_close(vl, expected)


if __name__ == "__main__":
    unittest.main(argv=[""], exit=False, verbosity=2)

    # Single step: return G=2, V=1, V_old=0 → clipped V is 0.2, pessimistic max uses clipped MSE
    rewards = torch.tensor([[2.0]])
    done_mask = torch.tensor([[1.0]])
    completion_mask = torch.tensor([[1.0]])
    values = torch.tensor([[1.0]])
    old_values = torch.tensor([[0.0]])

    print("toy (B=1, L=1): return=2, V=1, V_old=0, eps_v=0.2 → clip V to [−0.2, 0.2]")
    vl, adv, ret = ppo_value_loss_and_advantages(
        rewards,
        done_mask,
        completion_mask,
        values,
        old_values,
        gamma=1.0,
        epsilon_v=0.2,
    )
    print("  value_loss (max of clipped vs unclipped MSE):", float(vl))
    print("  return:", float(ret[0, 0]), " advantage:", float(adv[0, 0]))

    vl2, _, _ = ppo_value_loss_and_advantages(
        rewards,
        done_mask,
        completion_mask,
        values,
        gamma=1.0,
    )
    print("same tensors, no clipping (MSE to return only):", float(vl2))
    print("  (training often uses total_loss = policy_loss + vf_coef * value_loss)")
