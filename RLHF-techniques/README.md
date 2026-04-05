## RLHF techniques

<br>

> *self-contained pytorch snippets adapted from [reinforcement learning from human feedback (n. lambert, 2026)](https://rlhfbook.com/course)*

<br>

---

### `src/`

<br>

| file | what it is |
|------|------------|
| [`src/bt_reward_model.py`](src/bt_reward_model.py) | bradley–terry scalar reward model: one score per sequence from the last real token’s hidden state; pairwise training uses `logsigmoid(r_chosen − r_rejected)`. |
| [`src/outcome_reward_model.py`](src/outcome_reward_model.py) | outcome / correctness head: one logit per token on the completion; binary labels with `-100` masking; training loss is bce-with-logits on supervised positions. lm args are passed as `**inputs`. |
| [`src/process_reward_model.py`](src/process_reward_model.py) | process head: `num_classes` logits per token (default 3); labels only at step-boundary positions, else `-100`; cross-entropy on the masked positions. same `**inputs` + optional `labels` pattern as the outcome rm. |
| [`src/proximal_policy_optimization.py`](src/proximal_policy_optimization.py) | ppo critic helper (no gae): monte carlo returns with terminal resets, optional value clipping vs rollout values, masked mse value loss, and detached advantages `g_t − v(s_t)` for the policy term. |


<br>

---

### try it

<br>

with pytorch installed, `cd` to this directory, then:

```bash
python3 src/bt_reward_model.py
python3 src/outcome_reward_model.py
python3 src/process_reward_model.py
python3 src/proximal_policy_optimization.py
```

each command runs the built-in tests, then prints a short example.
