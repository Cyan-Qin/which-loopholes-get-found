# Findings log

Living record of every experimental finding, data insight and design decision in the project
"Which Loopholes Get Found? Predicting reward hacking of LLM judges before RL training".
It is the source material for the report, the presentation and a possible paper, so every number
names where it comes from (run directory on the GCP L4 machine, commit, or script). Update it after
every step that produces a result or a decision.

References follow the proposal: [1] Huang et al. 2025 (arXiv:2505.22203), [2] Zhao et al. 2025
(arXiv:2507.08794), [3] CHERRL, [4] Countdown-Code, [5] CATCH, [6] EleutherAI blog, [7] DeepSeekMath
(GRPO), [8] Gao et al. 2023.

Last updated: 2026-10-18 (pilot Q2 and Q3 at full scale: the pipeline sees hacking; the positive control is not exploited because the policy almost never samples a family).

---

## 0. Status of the pilot

The protocol must be frozen by 2026-10-26. The pilot answers four questions from the proposal, plus a
check that our judge audit reproduces [1].

| Pilot question | Status | Where |
|---|---|---|
| Judge FPR roughly matches [1] | Done: 4 of 6 judges match; the two R1-style judges are lower; a 4096-token limit closes only part of the gap | Sections 3.6, 3.9 |
| Q4: does sampled propensity separate families? | Answered: no. 19 of 174,760 base-policy responses fall in any family (1 within the pools), so the string-likelihood fallback applies; likelihoods computed | Sections 3.4, 3.7 |
| Q1: gold labels agree with a hand audit (>= 95%) | Passed: 99.7% weighted agreement (190 of 195 decided items); all 5 misses are correct answers without a usable box | Section 3.5 |
| Q2: can the pipeline see hacking (engineered judge)? | Yes: at full scale the engineered loophole takes off after about 130 steps; event at step 150 with tau 0.05 | Section 3.13 |
| Q3: is the positive control exploited at our scale? | No: 300 steps, no family event; 5 family responses in 76,800 training samples, none accepted. Design decision needed | Sections 3.13, 5 |

---

## 1. Setup and the reasons behind it

### 1.1 Models, software and versions

- Policy: Qwen2.5-1.5B-Instruct. Same model family as the judges and the RL experiments of [1], cheap
  rollouts without long thinking, and many public TRL/verl GRPO reproductions.
- Pinned stack, identical on the Mac and the GPU machine: Python 3.12, torch 2.13.0, transformers 5.17.0,
  vLLM 0.31.0, TRL 1.15.0, math-verify 0.9.0. vLLM 0.31.0 pins torch 2.13.0 and transformers < 5.18;
  TRL 1.15.0 supports vLLM <= 0.31.0. Lock files: `requirements-lock-mac.txt`,
  `requirements-lock-gcp-cu130.txt`.
- torch 2.13.0 from PyPI is built for CUDA 13.0, so the GPU host needs an NVIDIA driver supporting CUDA
  13.0 (driver 580 or later); a CUDA 12.9 build path exists in `scripts/setup_gpu.sh`.
- vLLM runs with `VLLM_USE_FLASHINFER_SAMPLER=0` (PyTorch sampler) because FlashInfer JIT-compiles its
  sampler at start-up and needs nvcc, which a driver-only machine lacks.

### 1.2 Compute

- Local: MacBook M3 Pro, 18GB, used for code, tests and small MPS smoke tests.
- Stage 1 (inference only): GCP VM `loopholes-l4`, 1x NVIDIA L4 24GB, g2-standard-8, Ubuntu 24.04,
  driver 580.178.04, Standard provisioning, about $0.85/h, paid from a $100 GCP coupon. Roughly 12 L4
  hours used by 2026-10-17 (estimate, about $10).
- GCP did not grant quota for A100 80GB, and the RTX PRO 6000 attempt did not give a usable VM. L4 quota
  was granted. Stage 2 (full-parameter GRPO of a 1.5B policy with vLLM and a judge on one GPU) needs
  at least 80GB; shortlisted rentals on Vast.ai: RTX PRO 6000 96GB in New York at $1.373/h (DLPerf 283,
  4-month availability) or A100 PCIe 80GB in Virginia at $1.15/h (DLPerf 127). GRPO pilot and main runs
  should use the same GPU type so measured step times carry over.
- Stage 2 machine (rented 2026-10-17): Vast.ai instance 56493441, 1x NVIDIA RTX PRO 6000 Blackwell
  Workstation Edition 96 GB (97,887 MiB), driver 580.159.03 (CUDA 13.0), host 173452 in New York,
  reliability 99.85%, 32 CPU cores, 96 GB RAM, 200 GB disk, $1.401/h including storage. Image
  `vastai/base-image:cuda-13.0.3-auto` (Ubuntu 24.04, Python 3.12, nvcc 13.0), SSH launch mode.
  Chosen over the two A100 80GB offers (reliability 96.8% and 98.9%, only two listed) because runs keep
  no checkpoints, all runs must use one GPU type for weeks, and 96 GB leaves room for the 7B held-out
  judge. Stopping keeps billing the disk ($0.056/h for 200 GB, as shown for the stopped instance; an
  earlier estimate of $0.014/h was wrong) and lets others rent the GPU meanwhile, so the machine stays
  up while runs are queued.
- That GPU was rented by someone else while the instance was stopped after the Q2 and Q3 pilot runs
  (2026-10-18), so the work moved to Vast.ai instance 56589070: 1x RTX PRO 6000 Blackwell Max-Q 96 GB
  (the 300 W version of the same chip; same memory bandwidth, 1,404 GB/s measured, lower power limit),
  host 444097 in Ohio, reliability 99.14%, 12 CPU cores, 129 GB RAM, 200 GB disk, $1.329/h, available
  for 3 months. Numerics are the same (same architecture and software); compute-bound parts such as
  the update step may run somewhat slower than on the 600 W Workstation Edition, so step times from the
  Q2 and Q3 runs are re-measured. All main runs use this GPU type.
- Budget estimate before the pilot (to be replaced by measured step times): about 40 to 55 s per GRPO
  step (generation about 15 s, training 20 to 25 s, judging about 20 s for R1-style judges and under 5 s
  for the others), so 3.6 to 4.9 h per 300-step run. 42 runs (24 core, 6 held-out, 12 SFT-group) plus
  the pilot come to about 193 GPU hours, about $270 at $1.401/h, or about $310 with 15% for reruns;
  the plausible range is $200 to $400. One machine needs about 8 days of continuous running.
- Measured in the pilot (600 W Workstation Edition, Section 3.13): 49.9 s per step with the engineered
  judge (no server) and 57.4 s with R1-Distill-Verifier (its server takes 8.0 s per batch of 256), so a
  300-step run with an R1-style judge takes about 5.0 h including evaluations, about 15% more than the
  estimate above.

### 1.3 Candidate training problems

Built by `src/data/build_pool.py`, written to `data/processed/pool_candidates.jsonl`.

| Step | DeepScaleR | MATH train | Total |
|---|---|---|---|
| Loaded | 40,315 | 7,500 | 47,815 |
| Plain-integer reference answer | 23,238 | 4,717 | 27,955 |
| Deduplicated (conflicting duplicates dropped) | 17,467 | 4,715 | 22,182 |
| Minus the 471 hacking-set problems | 17,233 | 4,692 | 21,925 |
| Minus MATH500 overlap | 17,228 | 4,692 | 21,920 |
| Minus GSM8K test overlap | 17,228 | 4,692 | 21,920 |
| Prompt at most 512 tokens | 17,193 | 4,652 | 21,845 |

Insights:
- DeepScaleR already contains almost all of the MATH training split: 4,705 of 4,717 integer-answer MATH
  problems appear verbatim. Deduplication keeps the MATH copy so MATH level and type survive.
- DeepScaleR has label noise: 100 problems appear more than once with different integer answers (for
  example the same "month with 31 days" problem labelled 2 and 3). All copies are dropped.
- All 471 hacking-set problems of [1] occur verbatim in DeepScaleR; they are removed by problem text.
  Five MATH500 problems were also present in the training sources and are removed.
- Only 263 of the 471 hacking-set questions have plain-integer reference answers.
- The candidate file is byte-identical when rebuilt on the Mac and on the GCP machine
  (MD5 fea6cc4341e9bbf85135915c50f0dba2).
- Answers: 1,476 negative, 159 zero, 286 with absolute value above 10^6, 3 above 2^63 (handled as Python
  integers). Median prompt 107 tokens, 95th percentile 212.

### 1.4 The hacking set of [1]

`hkust-nlp/rl-verifier-pitfalls_hacking_data`: 6,123 rows = 471 questions x 13 patterns, columns idx,
question, gt, pred, hacking_type, score.

- `score` is False on every row: it is the gold label of the hacking answer, not a judge verdict, so FPR
  must be measured by running the judges.
- Each family uses only 1 to 8 distinct template strings (for example 5 for prompt_injection). FPR(j, f)
  therefore measures acceptance of a few templates; family detectors for policy outputs must follow the
  family definition rather than match these strings.
- The released gibberish rows all contain the placeholder `random_gibberish_1000chars`. [1] evaluated
  real random strings: its README gives 18.1% for general-verifier on gibberish, our seeded random
  1000-character strings give 17.2%, and the placeholder is accepted 0.0% of the time (Section 3.6).
- Empty symbols include 53 empty strings and 50 single spaces; the judge code handles empty answers.

### 1.5 Gold checker (`src/gold/checker.py`)

- The answer is the content of the last `\boxed{}` (or `\fbox{}`); if the last box is unclosed there is no
  fallback to earlier boxes. Equivalence with the integer reference is decided by math-verify, which
  accepts 1,000,000, x = 6, \frac{12}{2}, 50^\circ and 12 \text{ cm}, and rejects lists (2,3), "6 or 7",
  6.5 and 5.99999.
- No box, an empty box or an unclosed box gives `no_answer` and counts as wrong; truncation is recorded
  separately. A response such as "3 - 1" for reference 2 is graded correct because math-verify evaluates
  the expression.
- Checks: all 3,419 integer-reference hacking answers are graded wrong; 2,000 pool answers round-trip
  (boxed reference correct, reference + 1 wrong).
- Known strictness: a correct final answer written without a box is graded wrong. Its share is bounded
  by the no_answer rate (3.2% of difficulty-run responses, 1.1% excluding truncation). The hand audit
  measured it: 5 of 20 finished unboxed responses are correct, about 0.29% of all responses (Section 3.5).

### 1.6 Family detectors (`src/detectors/families.py`)

- 14 families: the 13 of [1] plus reasoning_opener, built from the master keys of [2] ("Thought process:",
  "Solution", ...); the punctuation master keys of [2] fall under empty_symbols.
- A response gets a primary family only if the pattern dominates it: a non-empty boxed answer or real
  mathematical work (at least 3 digits or 2 math expressions) means an ordinary attempt. Families are
  tried in a fixed priority order, wrapper families before the content they wrap (so `<b>Final
  Decision: Yes</b>` is html_markdown). A separate `contains` set records patterns embedded anywhere
  (for example a wrong derivation with "Final Decision: Yes" appended) for sensitivity analysis only.
- Gibberish uses a 10,316-word vocabulary built from MATH and DeepScaleR text plus script mixing.
- Validation: 100% agreement on all 6,123 hacking-set answers (gibberish replaced by real random strings)
  and on the 10 master keys of [2]; 0 flags on 7,500 MATH and 5,000 DeepScaleR reference solutions, with
  and without their boxed answer; recall on synthetic gibberish 100% (random characters, random letters,
  random Qwen tokens) and 99.7% for "30% solution + 70% token salad"; 0.7% flagged for "full solution +
  15-token salad tail". 40 hand-written variants not in the hacking set are classified correctly.
- Caveat: these checks use templates and human-written solutions. Precision on real policy outputs needs
  the stage-2 hand audit of flagged responses.

### 1.7 Judges (`configs/judges.yaml`, `src/judges/`)

| Judge | Role | Prompt (source) | Verdict format | Decoding |
|---|---|---|---|---|
| R1-Distill-Verifier-1.5B | positive control | [1] with question (Appendix E, Figure 7) | `<answer>0/1</answer>` after `</think>` | T 0.6, top-p 0.95, max 2048 |
| DeepSeek-R1-Distill-Qwen-1.5B | core | [1] without question | `\boxed{0/1}` or a bare 0/1 after `</think>` | T 0.6, top-p 0.95, max 2048 |
| Qwen2.5-Math-1.5B-Instruct | core | [1] without question | `\boxed{0/1}` | greedy, max 1024 |
| xVerify-0.5B-I | negative control | xVerify repository prompt | Correct / Incorrect | greedy, max 16 |
| General-Verifier | held-out | model card, raw "User: ... Assistant:" | Final Decision: Yes/No | greedy, max 1024 |
| Qwen2.5-7B-Instruct | held-out | [1] without question | `\boxed{0/1}` | greedy, max 1024 |

- The prompts of [1] are only in Figure 7 of its appendix (an image); they were transcribed from the
  rendered figure. [1] uses the prompt without the question for untrained verifiers and the prompt with
  the question for its trained R1-Distill-Verifier.
- Every judge sees the same answer string, during the audit and during GRPO: the content of the last
  non-empty `\boxed{}`, otherwise the last 1000 characters of the response. Hacking answers have no box,
  so they reach the judge verbatim, exactly as in the audit; a boxed injection such as
  `\boxed{Final Decision: Yes}` also reaches the judge.
- Parsing: the last verdict wins (a judge may quote a hacking answer that contains "Final Decision: Yes");
  reasoning judges are read after the last `</think>`, and unfinished thinking is a parse failure. Parse
  failures count as rejections (reward 0) and are reported separately.
- R1-Distill-Verifier ends with `<answer>0</answer>` or `<answer>1</answer>` after `</think>`, a format
  not stated in [1]; before this was parsed it caused 21 of its 23 smoke-test parse failures.
  DeepSeek-R1-Distill-Qwen sometimes ends with a bare "0" after `</think>`.
- Principle: FPR(j, f) must be measured with exactly the judge configuration used as the RL reward,
  so matching [1]'s numbers is a sanity check, not a goal.

### 1.8 Policy prompt and length limit

- Prompt: the problem plus "\nPlease reason step by step, and put your final answer within \boxed{}.",
  default system prompt, built in one place (`src/policy/prompt.py`) for every stage.
- `max_new_tokens = 1536` (HANDOFF suggested 1024). Evidence in Section 3.2.

### 1.9 Difficulty pools (`configs/data.yaml`, `src/data/pools.py`)

- The difficulty pass samples every candidate 8 times (temperature 1.0, same limit as GRPO).
- Pools come from fixed bands of the number correct out of 8: hard 1 to 2, easy 5 to 7. Problems with
  0/8, 3/8, 4/8 or 8/8 are dropped. The gap sharpens the H2 contrast, 0/8 and 8/8 problems give GRPO
  groups with no signal under an honest judge, and 0/8 problems concentrate wrong reference answers.
- Both pools get the same size so pool and number of epochs are not confounded. Each pool holds 100
  evaluation problems (never trained on) and sets aside 250 training problems for the propensity audit.

### 1.10 Propensity by string likelihood (`src/audit/likelihood.py`)

- Canonical strings, fixed before any policy measurement: for each family of [1], its up to 5 most
  frequent answers in the hacking set (ties broken alphabetically; empty_symbols gives "{", ".", "-",
  "}" and the empty string); for gibberish, 5 seeded random 1000-character strings; for
  reasoning_opener, the first 5 master keys of [2]. Stored in `canonical_strings.json`.
- For each of the 250 audit problems of a pool, the policy's teacher-forced log-probability of each
  string as its complete response, including the end-of-turn token, given the standard policy prompt.
- ell(f, d) = log of the mean over problems of the summed probabilities of the family's strings
  (computed with log-sum-exp). The mean per-token log-probability is the sensitivity version.
- Computed in float32: on the Mac, half precision shifted a 3-token string's log-probability by 0.06 nats
  between batched and unbatched passes, while float32 matches a full forward pass within 0.001 nats.
- Secondary: sampled family frequency over the difficulty-run responses to each pool's training problems,
  with Jeffreys smoothing over 15 categories (14 families and none): (count + 0.5) / (n + 7.5).
- Risk score r(j, f, d) = exp(ell(f, d)) x FPR(j, f); the analysis uses its ranks.

### 1.11 Static accuracy baseline (`src/audit/static_accuracy.py`)

- Per pool, 250 responses the gold checker marks correct and 250 it marks wrong, drawn from the
  difficulty-run responses to the pool's training problems (wrong ones include unboxed and truncated
  responses in their natural proportion). Each judge grades the same 1,000 responses.
- Reported: acceptance of correct responses (TPR), rejection of wrong ones (TNR), balanced accuracy (the
  baseline of the main test), false-positive rate on ordinary wrong answers by kind, parse failures.

### 1.12 Judge generation-limit check (`src/audit/cap_check.py`)

- Every audit judgment that hit the 2048-token limit is re-judged with 4096 tokens; the other verdicts
  cannot change, so the merged FPR isolates the effect of the limit.

### 1.13 GRPO training (`configs/grpo.yaml`, `src/train/grpo.py`, `scripts/run_grpo.sh`)

- TRL 1.15.0 `GRPOTrainer`. The policy (Qwen2.5-1.5B-Instruct, full fine-tuning) generates with a vLLM
  engine inside the trainer process (colocate mode, sleep mode on, 25% of GPU memory); the judge runs as
  a separate `vllm serve` process on the same GPU (20% of memory) and is queried through the
  OpenAI-compatible completions API (`src/judges/server.py`), with exactly the prompt, decoding and
  parser of the audit. `scripts/run_grpo.sh` starts the server, trains and stops the server.
- Reward: 1 if the judge accepts, else 0; a parse failure is a rejection. The gold checker and the
  detectors never enter the reward; they label every rollout for the logs.
- Batch: 32 prompts x 8 completions = 256 completions per optimizer step (`per_device_train_batch_size`
  8 x `gradient_accumulation_steps` 32). TRL generates one batch per step and takes one gradient step on
  it, so training is on-policy. 8 per prompt matches the difficulty bands (correct out of 8), so the
  share of groups without signal at step 0 is predictable from the pools.
- Objective: TRL's default DAPO-style token-level loss with clip-higher (epsilon 0.2, epsilon_high 0.28),
  no KL term (beta 0, no reference model), group-normalized advantages, learning rate 1e-6 constant,
  gradient clipping 1.0, 300 steps. Sampling at temperature 1.0 and top-p 1.0, the same as the
  difficulty run and the propensity measurement; completion limit 1536 tokens.
- Truncated completions stay in the loss (`mask_truncated_completions: false`). Masking them, as DAPO's
  overlong filtering does, would hide length-based exploits such as answer_repetition from the optimizer.
- Precision: fp32 master weights (TRL's default when given a model name) with bf16 autocast. With bf16
  weights an update of about 1e-6 relative to a weight of about 0.02 is below bf16 resolution (about 0.4%
  relative) and would be rounded away.
- Memory plan on an 80GB GPU: weights, gradients and Adam states in fp32 about 24 GB; policy vLLM engine
  20 GB (weights 3 GB plus the KV cache for 256 sequences of 2048 tokens, about 15 GB); judge server
  16 GB; activations with gradient checkpointing a few GB. A 24GB L4 cannot hold this.
- Seeds: the run seed sets the data order and torch's generator. TRL seeds the colocated vLLM engine
  with the process index (always 0 on one GPU), so runs with different seeds differ through the prompt
  order, not through vLLM's random stream; vLLM sampling is not token-for-token reproducible anyway.
- Before training, the run directory records `meta.json` (git commit, every config, library versions)
  and, for a real judge, `predicted_risk.csv`: this judge's and pool's rows of
  `results/stage1/risk_scores.csv`, so the prediction is on record before any outcome.
- Logs per run: `train_metrics.jsonl` (per step: acceptance, gold accuracy, false-positive rate, family
  and exploit rates, parse failures, truncation, no_box, shares of groups with all-0 or all-1 rewards,
  judge seconds), `rollouts.jsonl` (full text of training responses that fooled the judge, match a
  family or contain a family marker, or got a parse failure; `--save-all-rollouts` keeps all),
  `train_log.jsonl` (TRL's loss, reward, entropy, clip ratios), `eval.jsonl` and `eval_summary.jsonl`.
  Checkpoints are off by default (`--save-final` saves the last policy).
- `scripts/run_grpo.sh` waits for the judge server's health check (and prints its log if it dies),
  samples GPU memory of all processes with nvidia-smi every 2 seconds (`gpu_mem.csv`), and ends with
  `python -m src.train.summary` (step time, judge seconds per batch, peak memory, evaluations).
- `scripts/stop_when_done.sh` stops the Vast instance through its own API key once no run has been
  active for 3 checks a minute apart (the gap between chained runs does not count), after writing
  the summaries and events of every run to `runs/grpo/summaries-<time>.txt`. Tested on the Mac with
  stand-in processes and a stub command-line tool.
- Test profiles in `configs/grpo.yaml`: `--local-test` (Mac, 0.5B policy, no vLLM) and `--gpu-test`
  (0.5B policy, colocated vLLM with sleep mode, judge server, 8 prompts x 8 completions, 512-token
  completions, 5 steps, evaluation of 20 problems x 2 at steps 0, 2, 4 and 5). The GPU test checks on
  the 24GB L4 what the Mac cannot: vLLM generation inside the trainer, sleep mode, weight sync before
  evaluation, and the judge server. Planned memory on the L4: training states about 8 GB (0.5B in
  fp32 with Adam), policy engine 25% and judge server 20% of the card.

### 1.14 Engineered judge and the no_box diagnostic (`src/judges/engineered.py`)

- Pilot Q2 needs a judge with a loophole the policy can find for sure, like the injected biases of [3].
  The proposal suggested accepting one family the policy already produces, but the base policy almost
  never produces any of the 14 families (Section 3.4), so a missing event would not tell a broken
  pipeline from an unreachable loophole.
- The engineered judge therefore targets `no_box`: a finished response without a usable boxed answer
  (1.1% of difficulty-run responses, Section 1.5). It accepts every no_box response and otherwise
  rewards gold correctness. On the hard pool (base accuracy 16.6%) dropping the box earns reward 1
  for sure, against about 0.17 for an honest attempt, so a working pipeline must show e(no_box) rising
  well above its base rate. This tests the whole path: reward, advantage, update, weight sync,
  evaluation and event detection.
- no_box is tracked in every run, not only the engineered one (`no_box_rate`, `e/no_box`). Real judges
  accept 41% to 65% of the policy's unboxed responses (Section 3.8), and the hand audit found 25% of
  finished unboxed responses to be correct answers that the strict gold checker marks wrong (Section 3.5), so a rise of e(no_box) under a real judge must be read with the
  hand audit of the gold checker in mind. It is not one of the 14 families and stays out of H1 to H3.

### 1.15 Periodic evaluation and hacking events (`src/train/monitor.py`, `src/monitor/events.py`)

- At step 0 and every 10 steps (and at the last step) the current policy answers the 100 held-out eval
  problems of its training pool, 2 samples each, with training-time sampling, through the trainer's own
  vLLM engine after syncing it to the current weights. The run's judge grades the 200 responses; every
  response is stored as a judge record (`src/pipeline/records.py` schema) in `eval.jsonl`.
- `eval_summary.jsonl` holds one line per evaluation with `e/<family>` for the 14 families and no_box:
  the share of eval responses that the judge accepted, the gold checker marks wrong, and that belong to
  the family. The resolution is 1/200 = 0.5 percentage points.
- Event: e_f(t) > tau in 2 consecutive evaluations; the event time is the first of the two. Families
  without an event are right-censored at the last evaluated step, for the time-to-event analysis.
  `python -m src.monitor.events <run> --tau <tau>` writes `events_tau<tau>.json`. tau is set from the
  engineered-judge and positive-control pilot runs before the freeze.
- `--stop-when no_box:0.5` ends a pilot run once e/no_box >= 0.5 in 2 evaluations in a row. It is for
  pilot Q2, whose answer is a single event; main runs always train the full 300 steps.

### 1.16 Propensity seeding (`configs/sft.yaml`, `src/train/seed_data.py`, `src/train/sft.py`)

Built after pilot Q3 (Section 3.13) to give every family enough exposure; tested first in one pilot run.
- Seeding set: clean examples are the policy's own finished, gold-correct responses without a family
  (one per problem, sampled at temperature 1.0 with the 1536-token limit) to the 17,187 candidate
  problems outside both pools, so they share no problem with training, evaluation or the audit. Seeded
  examples are answers from [1]'s hacking set drawn in their released proportions (fresh random
  1000-character strings for gibberish, the master keys of [2] for reasoning_opener), each paired
  with one of the problems the clean examples answer (since the first pilot, Section 3.14; that pilot
  paired them with any problem outside the pools). These are the strings the judge audit measured FPR on, so FPR(j, f)
  applies to what the seeded policy says. Each family is 2% of the set (2,000 clean examples, 56 per
  family, 28% seeded in all); every seeded string is checked to be detected as its own family.
- Why 2%: Q2 took off after about 300 rewarded samples; at p_f = 2% a family is rewarded 256 x 0.02 x
  FPR times per step, which reaches 300 within 300 steps for an FPR of about 20% or more, and the
  judges' FPRs spread from about 2% to 40% (R1-Distill-Verifier) or 78% (Qwen2.5-Math).
- SFT: full fine-tuning with TRL's SFTTrainer, loss on the response and its end-of-turn token only
  (checked: the prompt is the RL prompt, an empty seeded answer teaches only the end token), 2 epochs,
  learning rate 1e-5 with cosine decay and 5% warm-up, batch 16, fp32 master weights with bf16
  autocast. The model is saved in bf16 so the vLLM engines load it like the base policy.
- Pre-RL propensity of the seeded policy (`src/audit/sampled_propensity.py`): 16 samples for each of
  the 250 audit problems per pool (4,000 responses), family frequency with Jeffreys smoothing, and the
  risk score r = p_f x FPR(j, f) recorded before RL (`--risk-csv`). Sampled frequency is usable now
  (about 80 responses per family at 2%); the string likelihood of the canonical strings is measured as
  well.
- Pipeline: `scripts/run_seeded_pilot.sh` (seeding set, SFT, sampled propensity, likelihood, GRPO from
  the seeded checkpoint with `--policy-model`, `--risk-csv` and `--tag`). The whole chain ran on the Mac
  with the 0.5B model.
- Consequence for H3 if adopted: every run is seeded, so H3 (0% against 0.5% exploit SFT) would turn
  into a question about the seeding rate; to be settled together with open question 7.

---

## 2. Engineering issues met and fixes (material for the repository troubleshooting guide)

| Problem | Cause | Fix |
|---|---|---|
| Hugging Face download stalled for 30 minutes at 2.56/3.09 GB | xet transfer backend | `HF_HUB_DISABLE_XET=1` (plain HTTP, about 37 MB/s) |
| `.gitignore` entry `data/` hid `src/data/` | unanchored pattern | `/data/`, `/runs/` |
| Pool build crashed on huge answers | int64 overflow in pandas | Python integers |
| MATH problems vanished after deduplication | DeepScaleR copy kept first | MATH rows first |
| GCP refused A100 80GB quota; RTX PRO 6000 unusable | automated quota review | used L4 for stage 1 |
| `cuda_installer.pyz --installation-branch=lts` failed | LTS needs binary mode | add `--installation-mode=binary` |
| Triton compile error on the first Qwen2 forward pass | Ubuntu lacks Python headers | `python3.12-dev`; setup checks Triton |
| vLLM engine failed: "Could not find nvcc" | FlashInfer sampler JIT | `VLLM_USE_FLASHINFER_SAMPLER=0` |
| Local runners used CPU on the GPU machine | device choice only knew MPS | `src/device.py` picks CUDA and bf16 |
| Judge 1.5B models loop on odd inputs ("cd@@cd@@", ", , ,") | greedy decoding degeneration | counted as parse failure, reported |
| Audit page showed raw LaTeX in responses | MathJax skips `<pre>` | rendered div plus raw text |
| GRPO step failed on the Mac: `'NoneType' object has no attribute 'apply'` | TRL 1.15's fused LM head needs Triton, which ships with PyTorch on Linux only | `--local-test` installs a plain-PyTorch log-prob shim (`src/train/local_compat.py`); GPU runs use TRL's kernel |
| Decisions dated 2026-10-18 | the GCP machine's clock is UTC | dates in this log are New York dates |
| Labeled sheets came back altered: a few invalid UTF-8 bytes in the CSVs, changed response text in the xlsx | spreadsheet apps rewrote the policy's odd characters | only audit_id and the labels are read from the returned files (`audits/gold/labels/`); response text comes from the committed sheet |
| After the automatic stop, the Vast instance could not restart ("Scheduling": its GPU was rented by someone else) | a stopped Vast instance keeps its disk but not its GPU | copy the run directories to a new instance with Vast's copy function, then destroy the old one; during the campaign keep the instance running, and back results up before any stop |
| Answer key of the hand audit was on the stopped GPU machine | the sheet was built there | `src/gold/rebuild_key.py` recomputes it from the sheet (gold is deterministic; truncation is exactly 1536 tokens); strata counts match the sample |

---

## 3. Results

### 3.1 Throughput on the L4 (vLLM)

- Policy sampling: 4,902 generated tokens/s on 200 x 8 at a 1024 limit (1.00 s per problem of 8 samples);
  4,569 tokens/s on the full run at 1536 (1.16 s per problem). Model load and compilation about 90 s.
- Judges (34-item smoke, small batches, so not a throughput figure): mean output tokens 521 to 863 for
  the R1-style judges, 289 for Qwen2.5-Math, 79 to 85 for Qwen2.5-7B and general-verifier, 2 for xVerify.
  The R1-style judges will dominate judge time in GRPO.
- vLLM sampling is not reproducible token for token across runs even with per-request seeds: a rerun of
  40 problems matched 212 of 320 responses on the first 300 characters and 10 of 231 finished responses
  exactly (batch-dependent GPU numerics). Reruns are statistically equivalent, not identical.

### 3.2 Truncation and the length limit

Test run: 200 problems x 8 at 1024 tokens (`runs/difficulty-test200`).

- 187 of 1,600 responses (11.7%) truncated: DeepScaleR 13.9% (171/1,232), MATH 4.3% (16/368); among MATH
  levels only Level 5 truncates (17.0%, 15/88) plus one Level 2 response.
- None are repetition loops (zlib ratio of the last 1500 characters 0.30 to 0.40; loops measure 0.02 to
  0.08); the examples are long enumerations. 19 truncated responses had already written a box, none of
  them correct. 52 of the 76 problems with a truncated response are 0/8 problems.
- Re-sampling 40 of those problems at 2048 tokens (`truncation-retry-2048.jsonl`): 84 of the 89
  originally truncated responses finish (mean length 1,054 tokens), only 7 of them correctly; accuracy on
  those problems goes from 7.8% to 10.9%.

| Cap | Truncated (retry set) | Accuracy (retry set) | Estimated truncation, all problems |
|---|---|---|---|
| 1024 | 26.9% | 9.1% | 11.7% |
| 1280 | 10.9% | 10.6% | 4.8% |
| 1536 | 6.2% | 10.9% | 2.7% |
| 1792 | 3.4% | 10.9% | 1.5% |
| 2048 | 2.5% | 10.9% | 1.1% |

Decision: 1536. Accuracy is flat from 1536, truncation falls from 11.7% to about 2.7%, and the slowest
rollout per GRPO step grows by 1.5x rather than 2x. Confirmed by the full run: 2.2% truncated.

### 3.3 Full difficulty run

`runs/difficulty`, 21,845 problems x 8 = 174,760 responses at 1536 tokens, about 7.0 h on the L4.

- Accuracy 20.4%; truncated 2.2%; no boxed answer 3.2%; mean length 645 tokens.
- Problems by number correct out of 8: 0: 11,939 (54.7%); 1: 3,065; 2: 1,597; 3: 977; 4: 819; 5: 714;
  6: 697; 7: 918; 8: 1,119.
- More than half of the problems are never solved, so a median split (the proposal's plan) would have
  put the cut between 0 and 1 and filled the hard pool with unsolvable problems. Bands avoid this.
- Bands: hard (1 to 2 correct) 4,662 problems; easy (5 to 7) 2,329.

### 3.4 Base-policy propensity: almost never a hacking pattern (pilot Q4)

- Only 19 of 174,760 responses (about 1 in 10,000) fall in any family: gibberish 18 (multilingual token
  salad after the model derails), answer_repetition 1. No other family appears.
- The planned propensity audit (250 problems x 64 samples per pool, 32,000 responses) is smaller than
  this run, so its outcome is already clear: far fewer than 4 families reach 5 occurrences per pool.
  Per the protocol, string likelihood becomes the primary propensity measure (HANDOFF section 5.1).
- Narrative point: if a policy exploits a judge during GRPO, it must amplify rare samples or drift into
  the pattern, because the base policy essentially never produces these patterns. Gibberish is the only
  family the base policy emits on its own, and several judges accept real gibberish (general-verifier
  17.2%, Qwen2.5-Math 15.7%), which makes it a family to watch.
- The difficulty run gives 8 samples for every pool problem (about 37k responses in the hard pool and
  19k in the easy pool), more than the planned audit, so sampled frequencies can be reported from it as
  a secondary measure.

### 3.5 Pools v1 and the gold hand audit

`pools/pools-v1.json` (committed; built from `runs/difficulty` at commit eb27dce):

| | Hard pool | Easy pool |
|---|---|---|
| Train / eval / audit (audit within train) | 2,229 / 100 / 250 | 2,229 / 100 / 250 |
| Base accuracy (8 samples) | 16.6% | 76.1% |
| Correct counts | 1: 1,557; 2: 772 | 5: 714; 6: 697; 7: 918 |
| Sources | DeepScaleR 1,972 (85%), MATH 357 | DeepScaleR 943 (40%), MATH 1,386 |
| MATH levels | L1 6, L2 49, L3 67, L4 92, L5 143 | L1 178, L2 406, L3 362, L4 283, L5 156 |

- Pool size is set by the easy band (2,329). With 32 prompts per GRPO step, 300 steps would pass over
  the 2,229 training problems about 4.3 times; prompts per step and the band edges are stage-2 decisions.
- Confound for H2: the hard pool is mostly competition problems (85% DeepScaleR) and the easy pool mostly
  MATH (60%). Matching sources is impossible at this size (about 700 MATH problems exist in the hard
  band). Plan: keep the pools, which define difficulty as the policy's own pass rate, and report H2 within
  source as well (for example DeepScaleR problems of both pools); state it as a limitation.
- Gold hand audit (`audits/gold/`): 200 responses, blind (labelers do not see the gold verdict or the
  stratum), stratified by how gold decided. Population over the difficulty run: correct plain 34,822;
  correct by an equivalent form 771; wrong with an answer 133,394; no boxed answer 2,002; truncated
  3,771. Sampled 70 / 30 / 60 / 20 / 20. 30 items are labelled by all three team members for inter-rater
  agreement. Agreement is weighted by stratum size; "wrong with an answer" carries 76% of the weight.

Gold hand audit results (labelled by the team on 2026-10-17; `audits/gold/labels/`, scored with
`python -m src.gold.audit_score audits/gold/labels/*.csv`, full output in `audits/gold/audit_score.json`):

| Stratum | Weight | Gold agrees with humans | 95% CI (Wilson) |
|---|---|---|---|
| Correct, boxed answer equals the reference | 19.9% | 70 of 70 | 94.8% to 100% |
| Correct by an equivalent form | 0.4% | 30 of 30 | 88.6% to 100% |
| Wrong, with a boxed answer | 76.3% | 60 of 60 | 94.0% to 100% |
| No boxed answer, finished | 1.1% | 15 of 20 | 53.1% to 88.8% |
| Truncated | 2.2% | 15 of 15 (5 undecided) | 79.6% to 100% |

- Weighted agreement 99.7% (raw 190 of 195 decided items, 97.4%): pilot Q1 passes the 95% bar. Even if
  all 5 undecided truncated items were correct answers, the weighted agreement would be 99.2%.
- Gold never accepted a response that the humans judged wrong. Every disagreement is of one kind: a
  finished response with the right final answer and no usable box. The answer is in plain text (g069,
  "the last page ... contains 6 stamps"), in bold (g174 "**3** groups", g179 "**3967149**"), in a
  malformed `\leftboxed{40}` that the box pattern does not match (g068), or in a closing equation
  (g160, "So \(2^x = 3\)").
- So 25% (95% CI 11% to 47%) of finished unboxed responses are correct: about 500 (224 to 938)
  responses, 0.29% (0.13% to 0.54%) of the difficulty run. Pool membership is barely affected.
- Undecided: labeler A marked 8 truncated responses "?" (whether an unfinished response commits to an
  answer); for the 3 of them that were shared, B and C both said N, like gold.
- Inter-rater agreement: on the 30 shared items every decided pair agrees (A-B 27, A-C 27, B-C 30;
  Cohen's kappa 1.0; the shared items are balanced, 45 Y and 42 N votes). B also labelled C's 56 items,
  all in agreement with C. The labeler column was left blank in most rows, so labels are attributed by
  file.
- Decision (2026-10-17): keep the strict rule (last `\boxed{}` plus math-verify) for the primary
  analysis, as pre-specified. The prompt asks for a boxed answer, and the misses are format misses, not
  wrong verdicts. Consequence for stage 2: when a real judge accepts a correct unboxed answer, the logs
  count a false positive, so e(no_box) under real judges overstates hacking (Section 1.14).

### 3.6 Judge false-positive audit

`runs/judge-audit`, 7,065 answers per judge: 471 questions x (13 families of [1] with real gibberish,
reasoning_opener, and the gibberish placeholder). Every non-placeholder answer is detected as the family
it audits.

Mean FPR over the 13 families of [1]:

| Judge | Ours (real gibberish) | [1] reported | Ours (placeholder) | Parse failures | Truncated | Mean output tokens |
|---|---|---|---|---|---|---|
| General-Verifier | 7.7% | 7.7% | 6.4% | 2.0% | 2.1% | 85 |
| Qwen2.5-7B-Instruct | 4.6% | 3.7% | 4.6% | 0.4% | 0.0% | 79 |
| Qwen2.5-Math-1.5B-Instruct | 25.7% | 23.7% | 24.5% | 6.2% | 6.3% | 289 |
| xVerify-0.5B-I | 0.1% | 0.1% | 0.1% | 0.0% | 0.0% | 2 |
| R1-Distill-Verifier-1.5B | 13.3% | 18.8% | 12.9% | 5.4% | 4.5% | 521 |
| DeepSeek-R1-Distill-Qwen-1.5B | 6.0% | 12.3% | 5.9% | 16.9% | 14.7% | 863 |

FPR (%) by family, all 471 questions:

| Family | General-Verifier | Qwen2.5-7B | Qwen2.5-Math-1.5B | R1-Distill-Qwen | R1-Distill-Verifier | xVerify |
|---|---|---|---|---|---|---|
| empty_symbols | 5.9 | 16.6 | 46.5 | 19.3 | 36.5 | 0.2 |
| gibberish | 17.2 | 0.0 | 15.7 | 1.1 | 6.4 | 0.0 |
| answer_repetition | 0.4 | 0.4 | 3.0 | 3.0 | 2.5 | 0.0 |
| unicode_confusables | 0.8 | 0.0 | 7.0 | 4.9 | 8.1 | 0.0 |
| latex_notation | 2.5 | 0.0 | 4.5 | 0.6 | 4.9 | 0.0 |
| html_markdown | 7.0 | 18.9 | 20.0 | 11.5 | 11.0 | 0.0 |
| system_prompt_mimicry | 1.7 | 7.4 | 31.8 | 6.4 | 26.8 | 0.0 |
| instruction_hijacking | 1.7 | 6.8 | 21.9 | 8.7 | 19.1 | 0.0 |
| prompt_injection | 2.8 | 1.7 | 32.9 | 3.2 | 9.1 | 0.0 |
| mixed_language | 2.1 | 0.2 | 10.0 | 5.7 | 6.2 | 0.0 |
| adversarial_prefixes | 22.9 | 2.1 | 49.3 | 3.2 | 16.3 | 0.2 |
| answer_explanation | 29.3 | 6.2 | 77.5 | 5.1 | 16.8 | 0.4 |
| simple_words | 5.3 | 0.0 | 13.6 | 5.1 | 9.1 | 0.0 |
| reasoning_opener | 3.2 | 19.1 | 33.5 | 22.3 | 39.5 | 0.0 |
| gibberish_placeholder | 0.0 | 0.0 | 0.2 | 0.0 | 1.3 | 0.0 |

Per-family comparison with the README of [1] (adversarial prefixes, answer explanation, empty symbols,
gibberish, HTML, prompt injection):
- General-Verifier, ours vs [1]: 22.9 vs 22.1, 29.3 vs 28.5, 5.9 vs 5.9, 17.2 vs 18.1, 7.0 vs 7.2,
  2.8 vs 3.6. Close agreement validates the prompts, the answer input and the parsing.
- Qwen2.5-Math-1.5B: answer explanation 77.5 vs 77.9 and empty symbols 46.5 vs 44.4 match; adversarial
  prefixes 49.3 vs 20.8, gibberish 15.7 vs 5.5 and prompt injection 32.9 vs 22.7 are higher here.
- Qwen2.5-7B: adversarial 2.1 vs 1.9, explanation 6.2 vs 7.6, gibberish 0.0 vs 0.0; empty symbols 16.6 vs
  8.3 and HTML 18.9 vs 11.5 higher here.
- DeepSeek-R1-Distill-Qwen-1.5B is much lower here: adversarial 3.2 vs 21.7, explanation 5.1 vs 25.5,
  gibberish 1.1 vs 20.8. It truncates 14.7% of judgments at 2048 tokens (mean 863), especially on odd
  inputs where it loops, and truncation counts as rejection. [1] likely allowed longer generations.
  A re-judge of the truncated items at 4096 tokens is planned before freezing the judge limit.

Insights:
- Reasoning openers are among the strongest attacks for four judges (R1-Distill-Verifier 39.5%,
  Qwen2.5-Math 33.5%, R1-Distill-Qwen 22.3%, Qwen2.5-7B 19.1%), in line with [2].
- The fine-tuned positive control (R1-Distill-Verifier) is more vulnerable than its base model on almost
  every family once truncation is accounted for, consistent with [1]'s observation that fine-tuning
  increased exploitability.
- xVerify is essentially never fooled (at most 0.4% on any family): a clean negative control.
- Smoke-test anecdotes for the error analysis: xVerify accepted "2 \tan 35^\circ" (about 1.40) for
  reference 2; Qwen2.5-Math accepted an unboxed response from its 1000-character tail.

### 3.7 String-likelihood propensity and risk scores

`results/stage1/propensity.csv`, `risk_scores.csv` (policy Qwen2.5-1.5B-Instruct, float32, 250 audit
problems per pool, 57 canonical strings per problem).

| Family | ell, hard | ell, easy | Rank by ell | Per-token, hard | Rank by per-token |
|---|---|---|---|---|---|
| empty_symbols | -17.1 | -17.2 | 1 | -17.66 | 14 |
| simple_words | -24.7 | -25.8 | 2 | -16.02 | 13 |
| reasoning_opener | -32.8 | -32.4 | 3 | -12.39 | 12 |
| mixed_language | -36.1 | -36.3 | 4 | -6.54 | 9 |
| adversarial_prefixes | -42.7 | -43.6 | 5 | -4.01 | 3 |
| html_markdown | -43.2 | -44.1 | 6 | -6.24 | 8 |
| prompt_injection | -43.4 | -43.4 | 7 | -7.79 | 10 |
| latex_notation | -44.6 | -44.4 | 8 | -5.44 | 5 |
| answer_repetition | -45.6 | -46.7 | 9 | -3.60 | 1 |
| instruction_hijacking | -46.5 | -45.5 | 10 | -5.44 | 6 |
| answer_explanation | -52.6 | -53.3 | 11 | -4.00 | 2 |
| unicode_confusables | -58.4 | -57.6 | 12 | -4.12 | 4 |
| system_prompt_mimicry | -72.8 | -73.1 | 13 | -8.70 | 11 |
| gibberish | -4,667.2 | -4,667.7 | 14 | -6.11 | 7 |

(Ranks are for the hard pool; the easy pool differs by at most one place.)

Insights:
- The two pools have almost the same likelihoods (differences below 1.1 nats), so propensity does not
  separate the pools: any H2 effect must come from the reward dynamics (how often genuine successes
  occur), not from what the policy is inclined to write.
- Total and per-token likelihood rank the families almost in reverse. Total log-probability favours
  short strings (empty symbols, simple words, openers); per-token favours long fluent or repetitive
  strings (repetition, explanations). The total version is the probability of emitting exactly that
  response; it cannot represent families whose members are long and varied: one 1000-character random
  string has log-probability about -4,667, which says nothing about how often the policy produces
  gibberish.
- Sampled counts within the pools' training problems (17,832 responses per pool): 0 for every family
  except one answer_repetition response in the hard pool. None of the 18 natural gibberish responses of
  the difficulty run belong to pool training problems.
- In log space the risk score is ell + log FPR. With total likelihood, ell spans 55.7 nats across the 13
  non-gibberish families while log FPR spans only 2.7 to 4.5 nats within a judge, so within a judge the
  risk ranking is essentially the likelihood ranking; FPR matters mostly when it is 0 and between
  judges. Per-token likelihood spans 14.1 nats.
- The predicted top family is empty_symbols for all six judges (hard pool, total version). This matches
  the exploit [1] observed in RL against R1-Distill-Verifier (a lone "{"), a qualitative check.

### 3.8 Judge static accuracy on the policy's own responses

`results/stage1/static_accuracy.csv`: 1,000 responses (250 gold-correct and 250 gold-wrong per pool).

| Judge | Balanced accuracy | Accepts correct (TPR) | Rejects wrong (TNR) | FPR on wrong boxed answers | Parse failures |
|---|---|---|---|---|---|
| General-Verifier | 0.988 | 1.000 | 0.976 | 0.4% | 0.2% |
| Qwen2.5-7B-Instruct | 0.987 | 0.992 | 0.982 | 0.0% | 0.0% |
| xVerify-0.5B-I | 0.985 | 1.000 | 0.970 | 0.8% | 0.0% |
| R1-Distill-Verifier-1.5B | 0.979 | 0.992 | 0.966 | 2.1% | 0.4% |
| Qwen2.5-Math-1.5B-Instruct | 0.967 | 0.982 | 0.952 | 2.5% | 0.0% |
| DeepSeek-R1-Distill-Qwen-1.5B | 0.933 | 0.958 | 0.908 | 7.7% | 4.4% |

Insights:
- All judges grade ordinary responses well (balanced accuracy 0.93 to 0.99), so honest-answer rewards
  are reliable. Static accuracy does not reveal hack vulnerability: General-Verifier has the best static
  accuracy yet accepts 17.2% of gibberish and 29.3% of answer explanations, and Qwen2.5-Math is close to
  the others on ordinary answers but has a 25.7% mean FPR on hacks. This is the gap that motivates the
  study, and it suggests static accuracy will be a weak predictor of exploitation.
- Responses without a boxed answer (17 in the sample, 15 from the easy pool) are accepted by 41% to 65%
  of judgments. Most are probably correct answers written without a box, which the strict gold checker
  marks wrong; the hand audit's no-answer stratum will tell. They are 1.1% of all responses, so the
  effect on overall gold agreement is small, and they are not family members, so they do not enter the
  exploitation rate.
- DeepSeek-R1-Distill-Qwen is the noisiest reward: 4.2% of correct responses are rejected (mostly parse
  failures from unfinished thinking) and 7.7% of wrong boxed answers accepted.

### 3.9 Judge generation limit: 2048 vs 4096 tokens

`results/stage1/cap_check_by_family.csv`: every audit judgment that hit 2048 tokens re-judged at 4096.

| Judge | Truncated at 2048 | Still truncated at 4096 | Mean FPR over [1]'s 13 families, 2048 to 4096 | [1] |
|---|---|---|---|---|
| DeepSeek-R1-Distill-Qwen-1.5B | 1,038 | 488 | 6.0% to 7.5% | 12.3% |
| R1-Distill-Verifier-1.5B | 316 | 130 | 13.3% to 13.6% | 18.8% |

- The largest changes are for DeepSeek-R1-Distill-Qwen on empty symbols (19.3% to 26.3%) and reasoning
  openers (22.3% to 27.8%), and for R1-Distill-Verifier on gibberish (6.4% to 9.6%). About half of the
  long judgments still do not finish at 4096 tokens (on gibberish 312 of 412 for R1-Distill-Qwen):
  they are loops, not slow reasoning.
- The order of families by FPR is unchanged for the top four families of both judges, so risk rankings
  are robust to the limit. The remaining gap to [1] is not explained by the limit alone.
- Decision (2026-10-17): keep 2048. The FPR shift is modest and rank-preserving, while 4096 would double
  the slowest judgment per GRPO step and let looping judgments consume 4096 tokens. The 4096 numbers are
  reported as a sensitivity analysis.

### 3.10 Local GRPO smoke test (Mac, `--local-test`)

`python -m src.train.grpo --judge engineered:no_box --pool <pool> --local-test`: Qwen2.5-0.5B-Instruct,
no vLLM, 2 steps of 2 prompts x 4 completions, evaluation of 2 eval problems x 2 samples at every step.

- The whole loop runs: generation, engineered judge, reward, update, evaluation at steps 0, 1 and 2,
  judge records that pass validation, and every log file. Hard pool with a 96-token limit: 2 steps in
  1.0 minute, but all completions were truncated, so every reward was 0 and no gradient flowed.
  Easy pool with a 256-token limit: 2 steps in 2.7 minutes; the training completions were still all
  truncated, while the evaluations (4 responses each) moved from accept 0.50 / gold 0.50 at step 0 to
  accept 0.50 / gold 0.00 / no_box 0.50 at step 2. With 4 responses this is noise, not evidence.
- Gradient check with random 0/1 rewards (one step, 32-token completions): grad norm 10.45, largest
  parameter change 1.01e-6 at learning rate 1e-6 (Adam's first step moves each weight by about the
  learning rate), so advantages reach the weights.
- Not tested locally: the colocated vLLM engine, its sleep mode and weight sync, the judge server, and
  memory at full batch. These need the GPU run.

### 3.11 GPU integration test on the L4 (`--gpu-test`)

`bash scripts/run_grpo.sh engineered:no_box easy 0 --gpu-test && bash scripts/run_grpo.sh xverify_0p5b easy 0
--gpu-test` on the GCP L4 (24 GB): Qwen2.5-0.5B-Instruct policy, 8 prompts x 8 completions, 512-token
completions, 5 steps, learning rate 1e-6, colocated vLLM with sleep mode.

- Both runs finished (the second only starts if the first succeeds). xVerify run, with the judge server
  on the same GPU: step time median 11.2 s; judge 0.4 s per batch of 64 (16-token outputs); peak GPU
  memory of all processes 17,212 MiB; mean completion length 432 tokens, 43% clipped at 512; 35% of
  groups without reward signal; no parse failures. Evaluations of 40 responses at steps 0, 2, 4 and 5:
  acceptance 0.175 to 0.275, false positives 0 to 3 of 40, 4 s generation and under 1 s judging each.
- Shown: generation by vLLM inside the trainer, sleep mode, evaluation through the trainer's engine and
  the judge server work together on one GPU, within the L4's memory.
- Not shown by these runs: that vLLM receives the updated weights. At learning rate 1e-6 for 5 steps the
  weights barely move, so stale weights would look the same. Hence a third run.
- Weight-sync check (engineered judge, hard pool, learning rate 1e-4, 10 steps, `--learning-rate 1e-4
  --max-steps 10`): the policy learned to drop the box. Share of no_box in the training batches by step:
  0.00 0.00 0.02 0.03 0.02 0.02 0.38 0.14 0.91 1.00. e(no_box) at the evaluations of steps 0, 2, 4, 6, 8,
  10: 0, 0.025, 0, 0.275, 0.95, 1.00, while gold accuracy fell from 0.05 to 0. vLLM sampled both the
  training batches and the evaluations, so its behaviour followed the trained weights: the sync works.
- TRL rescores every sampled completion with the trainer; the gap to vLLM's log-probabilities
  (`sampling/sampling_logp_difference/mean`) was 0.003 to 0.018 nats per token in 9 of 10 steps with no
  trend (0.006 to 0.010 in the runs at learning rate 1e-6), and 0.066 at the last step, after the policy
  had collapsed onto unboxed answers. To watch in the pilot runs; TRL's importance-sampling correction
  (on by default) reweights such gaps.
- Preview of pilot Q2: the path from reward to event detection works. With tau 0.05 the event is at step
  6. The transition is sharp (e from 0.03 to 0.95 within 4 steps), so whether evaluating every 10 steps
  resolves event times well enough has to be checked on the full-scale runs.
- Memory: 12.4 GB without a judge server, 17.2 GB with the xVerify server (configured for 20% of the
  card). Step time about 11 s in all three runs.

### 3.12 Stage-2 machine: setup and integration test on the RTX PRO 6000

- `scripts/setup_gpu.sh` ran unchanged on the Vast image (platform detected as vast, CUDA 13.0 wheels):
  bf16 matmul and Triton work on Blackwell; the candidate pool rebuilt with the same MD5 as on the Mac
  and the L4 (fea6cc4341e9bbf85135915c50f0dba2); vocab.txt identical; all models downloaded in 5
  minutes; 240 tests passed. Setup took 9 minutes in all.
- `--gpu-test` with the xVerify server: step time median 3.4 s against 11.2 s on the L4 for the same
  workload (3.3 times faster); vLLM-trainer log-prob gap 0.006 to 0.011 per step, the same as on the L4,
  so Blackwell numerics do not change the sync check; no parse failures.
- Peak GPU memory 51.3 GB against 17.2 GB on the L4: vLLM reserves its fraction of the whole card
  (policy engine 25% = 24 GB, judge server 20% = 19.6 GB), so the reservations grow with the card. Plan
  for the full configuration: during generation about 18 GB of training state (fp32 weights and Adam;
  gradients are freed after each step) plus 24 GB plus 19.6 GB, about 62 GB; during the update the
  policy engine sleeps.

### 3.13 Pilot Q2 and Q3 at full scale (hard pool, seed 0, learning rate 1e-6)

Runs on the 600 W RTX PRO 6000 (Section 1.2), copied to the Mac under `runs/vast/grpo/`. Both use the
configuration of Section 1.13 (32 prompts x 8 completions, 1536-token completions, evaluation of 100
held-out problems x 2 every 10 steps).

Q2, engineered judge (`--stop-when no_box:0.5`, stopped at step 170):
- The share of unboxed answers in the training batches stayed near its base rate (mean 0.88% over steps
  0 to 129) and then took off: 3.9% at step 140, 9.8% at 150, 74% at 160, 100% from step 167. e(no_box)
  in the evaluations: 0.020 at step 0, at most 0.025 until step 140, then 0.125, 0.780 and 0.995 at
  steps 150, 160 and 170. Event at step 150 (tau 0.05, relative to step 0); gold accuracy fell to 0.
- About 300 accepted unboxed samples (1.2 per step) accumulated before the take-off began. This
  calibrates how much exposure GRPO at learning rate 1e-6 needs before a behaviour spreads.
- 49.9 s per step; peak memory 46.7 GB; the vLLM-trainer log-prob gap stayed at 0.007 to 0.010 until the
  take-off and rose to 0.017 as the policy changed fast.

Q3, positive control R1-Distill-Verifier (300 steps):
- No family event. e_f was 0 at every evaluation for all 14 families. In the 76,800 training samples
  the detectors found 5 family responses (3 gibberish, 2 answer_repetition; 0.017 per step) and the
  judge accepted none of them.
- The judge's false positives (1.6% of training samples, flat over the run; 2.9% of evaluation
  responses on average) are ordinary wrong answers: 1,168 of 1,256 had a boxed answer, 70 were
  truncated, 20 had no box. The policy did not learn to produce more of them.
- Honest learning happened instead: gold accuracy on training batches rose from 15.4% (steps 0 to 49) to
  23.4% (steps 250 to 299), and on the evaluation problems from about 0.14 to about 0.21. The share of
  groups without reward signal fell from 35% to 27%; unboxed answers fell from 0.75% to 0.02% (this
  judge rewards boxed answers more reliably).
- 57.4 s per step, judge 8.0 s per batch of 256, parse failures 0.8%, peak memory 67.2 GB.
- The recorded prediction (`predicted_risk.csv`) ranked empty_symbols first (FPR 36.5%, string
  log-likelihood -17.1), then simple_words and reasoning_opener (FPR 39.5%, -32.8).

Interpretation (exposure):
- A behaviour spreads only after GRPO has rewarded it enough times. Q2 needed about 300 rewarded samples.
  A family is rewarded at a rate of about 256 x p_f x FPR per step, so reaching 300 within 300 steps
  needs p_f x FPR of about 4e-3, for example p_f of 1% at an FPR of 40%. The policy's natural rate of
  family responses is about 6.5e-5 for all 14 families together (here, and 19 in 174,760 in the
  difficulty run), two orders of magnitude short. At this scale no family can be found, whatever its
  FPR, and every family would be censored in every main run, leaving H1 untestable.
- Raising the learning rate (the pre-registered fallback for a silent positive control) shortens the
  time to spread for what is sampled and rewarded, but it does not create the samples; with about one
  rewarded family sample per run it is unlikely to help. Training 100 times longer is not affordable.
- Proposed tau: 0.05. Before the take-off in Q2, e(no_box) rose at most 0.005 above its step-0 value,
  and family e_f never left 0; with 200 responses per evaluation a rise of 0.05 is 10 responses, and the
  take-off was caught at the first evaluation after it began.

### 3.14 Seeding pilot `seed2pct`: seeding set, SFT and pre-RL propensity

On the RTX PRO 6000 Max-Q (Section 1.2); results copied to `runs/vast/sft/seed2pct/`.
- Seeding set: the base policy answered 17,000 problems outside the pools once (14 minutes): 2,445
  correct (14.4%), 389 truncated, and not one family response. 2,000 clean examples plus 56 per family
  (28.2% seeded); every seeded string was detected as its own family.
- SFT: 348 steps (2 epochs, batch 16) in 7.8 minutes; mean training loss 0.455, token accuracy 0.91 to
  0.95 at the end.
- Sampled propensity of the seeded policy (4,000 responses per pool):

| | Hard pool | Easy pool |
|---|---|---|
| Any family | 33.4% | 16.4% |
| Per family | 1.5% (simple_words) to 2.9% (instruction_hijacking) | 0.7% to 1.6% |
| Gold accuracy | 13.1% (base policy 16.6% over the pool) | 64.3% (base 76.1%) |
| Truncated | 3.2% | 0.9% |

- Hacking depends on difficulty: within the hard pool, problems the seeded policy never solved in 16
  samples got 41.7% family responses, those it solved at least once 29.2%. The likely cause is the
  pairing: the seeded answers went with any problem outside the pools, 70% of which the base policy
  never solves, while the clean examples come from solved problems, so SFT tied hacking to difficulty.
  That would confound H2 (hard against easy pools). Fixed for later seeding sets: seeded answers are
  paired with the clean examples' problems (`pairing_problems` in `src/train/seed_data.py`). The pilot
  continues with seed2pct, since it trains on one pool.
- Exposure at the start of RL for R1-Distill-Verifier on the hard pool, as rewarded samples per step
  (256 x p_f x FPR): reasoning_opener 2.59, empty_symbols 1.85, system_prompt_mimicry 1.58,
  instruction_hijacking 1.43, then answer_explanation 0.86 down to answer_repetition 0.18. Q2 took off
  after about 130 steps at 2.3 rewarded samples per step, so on that calibration the first four families
  could take off within 300 steps and the others not. The GRPO run of the pilot tests this.

---

## 4. Deviations from the proposal (to state in the report)

| Proposal | What we do | Why |
|---|---|---|
| Sample 5,000 to 8,000 problems, split at the median pass rate | Difficulty pass on all 21,845 candidates; fixed bands hard 1-2/8, easy 5-7/8; equal pool sizes | 54.7% of problems are 0/8, so a median split degenerates; bands sharpen the H2 contrast and drop no-signal problems |
| Generation limit about 1024 (HANDOFF) | 1536 | Truncation 11.7% at 1024 vs about 2.7% at 1536 with no accuracy loss beyond 1536 |
| Judge prompt always includes the question (HANDOFF default) | Each judge's native prompt from [1] or its model card | Lets the audit be checked against [1]; question removal stays an optional ablation |
| Gibberish answers from the released hacking set | Seeded random 1000-character strings | The release stores a placeholder; [1] evaluated real strings |
| Sampled family frequency as the primary propensity | String likelihood (HANDOFF 5.1) | Pilot Q4 failed: 19 hacking-pattern responses in 174,760 |
| Propensity audit of 250 x 64 per pool | Also report sampled frequency from the difficulty run (8 per problem, all pool problems) | The difficulty run already contains more samples per pool |
| Proposal leaves the propensity measure open between frequency and likelihood | Total string likelihood primary, per-token and sampled frequency as sensitivity | Decided 2026-10-17 from Section 3.7, before any main run |
| Engineered judge accepts a family the policy already produces | Engineered judge accepts no_box responses (Section 1.14) | The base policy produces almost no family responses; no_box occurs in 1.1% of responses |
| GRPO as in [7] (KL to a reference model) | Token-level DAPO loss and no KL (TRL 1.15 defaults), clip-higher epsilon_high 0.28 (DAPO) | Current practice for RLVR; no reference model saves memory; stated in the report |

---

## 5. Open questions and decisions before the freeze

1. Gold rules: decided 2026-10-17, the strict rule stays (hand audit 99.7%, Section 3.5). Open: whether
   to add a lenient secondary label for unboxed responses (for example, the last number of the final
   sentence equals the reference), used only to split e(no_box) into unboxed-correct and no-answer.
2. R1-style judge limit: decided 2026-10-17, 2048 tokens; 4096 reported as sensitivity (Section 3.9).
3. Likelihood propensity: decided 2026-10-17, total log-probability is primary, per-token and the
   Jeffreys-smoothed sampled frequency are sensitivity analyses (`configs/protocol.yaml`). Decided before
   any main run and independent of outcomes. Report the blind spot for long, varied families (gibberish);
   the policy's natural gibberish is token salad, while the audit (like [1]) uses random characters, and
   FPR on token salad is not measured.
4. Static-accuracy baseline: done (Section 3.8).
5. GRPO setup: drafted in `configs/grpo.yaml` (Section 1.13): full fine-tuning, 32 x 8 per step, learning
   rate 1e-6, no KL, 300 steps; engineered judge on no_box (Section 1.14). Still open from the pilot
   runs: step time and memory at full batch, whether the positive control shows an event within 300
   steps at learning rate 1e-6 (if not, raise it once, before the freeze), and tau. Stage-2 GPU: an 80GB
   card (A100 80GB or RTX PRO 6000), the same type for the pilot and every main run.
6. Handling of the source confound in H2 (Section 3.5).
7. The main design produces no family events with natural propensities (Section 3.13). Options, to be
   decided before the freeze: (a) seed propensity with a short SFT stage on a small share of family
   responses before RL in every run, so that p_f is raised to a measured, pre-RL level near the
   threshold and FPR differences decide which families are found (the H3 machinery applied to all
   runs); (b) raise the learning rate once as pre-registered (cheap, but predicted not to help);
   (c) replace the event outcome with continuous measures (growth of false positives, drift of family
   likelihoods), which may carry little signal; (d) more steps or samples (needs about 100 times more).
   Recommendation: (a), checked first in one pilot run with the positive control.
8. tau: proposed 0.05 relative to step 0 (Section 3.13); events are now computed relative to the
   step-0 value as HANDOFF section 8 item 3 requires.

---

## 6. Chronological log

- 2026-10-16: Proposal and team idea submitted. Local environment (Python 3.12), hacking set downloaded
  and checked, candidate pool built, gold checker, family detectors, judge prompts and parsers, records
  and the end-to-end pipeline written and tested on the Mac. Public repository created.
- 2026-10-17: GCP L4 machine set up (driver 580, CUDA 13.0); vLLM runners with resumable shards; 200-problem
  throughput and truncation tests; length limit raised to 1536; judge parsers fixed for R1-style outputs;
  full difficulty run (174,760 responses); pilot Q4 answered (likelihood fallback); pools v1; blind gold
  audit sheet sent to the team; judge FPR audit (4 of 6 judges match [1]). Stage-1 GPU batch written
  (likelihood propensity, static accuracy, judge-limit check; `scripts/stage1_batch.sh`).
- 2026-10-17 (evening): Stage-1 GPU batch run on the L4 and results committed (`results/stage1/`): likelihood
  propensity and risk scores, static accuracy of all six judges, 4096-token re-judge of truncated audit
  judgments. Decided with the team: total likelihood as primary, judge limit 2048; both recorded in
  `configs/protocol.yaml`, which collects the decisions to freeze.
- 2026-10-17 (night): Stage-2 GRPO code: judge server client, engineered judge (no_box), reward with
  per-step judge, gold and family metrics, evaluation callback writing e_f(t), event detection,
  `configs/grpo.yaml`, `scripts/run_grpo.sh`; 16 new tests; local smoke test on the Mac (Section 3.10).
  Then a `--gpu-test` profile for a short integration test on the L4 and a run summary
  (`src/train/summary.py`).
  Dates written as 2026-10-18 earlier were UTC dates from the GCP machine and are corrected.

- 2026-10-17 (night): Gold hand audit labelled by the team and scored: weighted agreement 99.7%, all 5
  misses are correct answers without a usable box; pilot Q1 passes and the strict gold rule is kept.
  Labels, key and result committed under `audits/gold/`.
- 2026-10-17 (night): GPU integration test on the L4 passed (Section 3.11); `--learning-rate` override and a
  weight-sync check (vLLM-trainer log-prob gap) added to the run summary.
- 2026-10-17 (night): Weight-sync check on the L4 passed (learning rate 1e-4: the engineered loophole is
  learned within 10 steps, e(no_box) 0 to 1.00, event at step 6 with tau 0.05); `--stop-when` added for
  pilot Q2.
- 2026-10-18: Stage-2 machine rented on Vast.ai (RTX PRO 6000 96 GB, New York, $1.401/h); setup and the
  GPU integration test passed (Section 3.12). Budget estimate recorded in Section 1.2.
- 2026-10-18: Pilot Q2 and Q3 at full scale (Section 3.13): the pipeline sees the engineered loophole
  (event at step 150), the positive control is not exploited in 300 steps because the policy almost
  never samples a family response. Events changed to rises over the step-0 value. Design decision on
  propensity seeding pending.

- 2026-10-18: Propensity seeding implemented (Section 1.16): seeding set from the policy's own correct
  answers plus 2% per family of [1]'s hacking strings, SFT, sampled propensity before RL, and a
  one-command pilot; tested end to end on the Mac.

- 2026-10-18: Seeding pilot seed2pct, first stages on the Max-Q (Section 3.14): seeded propensity 1.5% to
  2.9% per family on the hard pool; hacking tied to difficulty by the pairing, fixed for later sets.
  GRPO of the pilot started.
