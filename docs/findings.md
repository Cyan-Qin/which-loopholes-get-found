# Findings log

Living record of every experimental finding, data insight and design decision in the project
"Which Loopholes Get Found? Predicting reward hacking of LLM judges before RL training".
It is the source material for the report, the presentation and a possible paper, so every number
names where it comes from (run directory on the GCP L4 machine, commit, or script). Update it after
every step that produces a result or a decision.

References follow the proposal: [1] Huang et al. 2025 (arXiv:2505.22203), [2] Zhao et al. 2025
(arXiv:2507.08794), [3] CHERRL, [4] Countdown-Code, [5] CATCH, [6] EleutherAI blog, [7] DeepSeekMath
(GRPO), [8] Gao et al. 2023.

Last updated: 2026-10-17 (after the pools, the gold audit sheet and the judge FPR audit).

---

## 0. Status of the pilot

The protocol must be frozen by 2026-10-26. The pilot answers four questions from the proposal, plus a
check that our judge audit reproduces [1].

| Pilot question | Status | Where |
|---|---|---|
| Judge FPR roughly matches [1] | Done: 4 of 6 judges match; the two R1-style judges are lower, traced to the 2048-token judge limit (sensitivity check at 4096 pending) | Section 3.6 |
| Q4: does sampled propensity separate families? | Answered: no. 19 of 174,760 base-policy responses fall in any family, so the string-likelihood fallback applies | Section 3.4 |
| Q1: gold labels agree with a hand audit (>= 95%) | In progress: blind sheet of 200 responses sent to the team | Section 3.5 |
| Q2: can the pipeline see hacking (engineered judge)? | Not started (stage 2, needs an 80GB GPU) | |
| Q3: is the positive control exploited at our scale? | Not started (stage 2) | |

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
  by the no_answer rate (3.2% of difficulty-run responses, 1.1% excluding truncation); the hand audit
  measures it.

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
  Results pending (target 2026-10-19).

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

---

## 5. Open questions and decisions before the freeze

1. Gold rules: depends on the hand audit (target 2026-10-19). If agreement < 95%, revise the rules,
   re-grade all difficulty responses (CPU only) and rebuild the pools if counts change.
2. R1-style judge limit: 2048 or 4096 tokens, from the truncated-item re-judge and GRPO cost.
3. Likelihood propensity details: canonical strings per family (up to 5), total log-probability as the
   primary version, per-token average as sensitivity.
4. Static-accuracy baseline: judge accuracy on the policy's own correct and wrong responses.
5. GRPO hyperparameters (group size, prompts per step, learning rate, KL, full vs LoRA), the engineered
   judge, the threshold tau, and the stage-2 GPU.
6. Handling of the source confound in H2 (Section 3.5).

---

## 6. Chronological log

- 2026-10-16: Proposal and team idea submitted. Local environment (Python 3.12), hacking set downloaded
  and checked, candidate pool built, gold checker, family detectors, judge prompts and parsers, records
  and the end-to-end pipeline written and tested on the Mac. Public repository created.
- 2026-10-17: GCP L4 machine set up (driver 580, CUDA 13.0); vLLM runners with resumable shards; 200-problem
  throughput and truncation tests; length limit raised to 1536; judge parsers fixed for R1-style outputs;
  full difficulty run (174,760 responses); pilot Q4 answered (likelihood fallback); pools v1; blind gold
  audit sheet sent to the team; judge FPR audit (4 of 6 judges match [1]).
