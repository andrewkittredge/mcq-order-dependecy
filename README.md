# Order Dependency in LLM Answers to Multiple-Choice Questions

Large Language Models (LLMs) exhibit order dependency bias when answering multiple choice questions (MCQs), see Zheng et al. https://openreview.net/pdf?id=shr9PXz7T0. Order dependency is a bias in LLM response based on the order or inputs.

In the context of answering multiple choice questions, order dependency manifests as a bias towards selecting an answer based on index of answer in the list of answers, for instance tending to select the first answer.

This project demonstrates the order dependecy bias by prompting LLM models with MCQs multiple times, with the correct answer in a different position in the list of answers each time, the 'answer-moving attack' in Zheng and infering the bias from the results.


## Example

Run every command from the repository root. A small example with local inference using the Qwen2.5-0.5B model:

```
uv run --group hf bin/order-dependency.py --backend hf --model Qwen/Qwen2.5-0.5B -q mmlu --per-subject 20 -o mmlu-qwen05b-subset
```

The output is written to `results/<run name>/`. One report, `results/apple-fy2025-qwen05b/report.md`, is checked into this repository as an example.

The commands below generate the rest of the data reported. The Claude runs need `ANTHROPIC_API_KEY`.

```
uv run bin/order-dependency.py --model claude-opus-5 --thinking disabled --effort low -q mmlu --per-subject 20 -o mmlu-opus5-no-thinking
uv run bin/order-dependency.py --model claude-haiku-4-5 --thinking omit --effort none -q mmlu --per-subject 20 -o mmlu-haiku45
uv run bin/order-dependency.py --model claude-opus-4-5 --thinking omit --effort low -q mmlu --per-subject 5 -o mmlu-opus45
uv run bin/order-dependency.py --model claude-sonnet-4-5 --thinking omit --effort none -q mmlu --per-subject 5 -o mmlu-sonnet45
```

## Implementation

1. Build the list of questions, either the MMLU test set downloaded from Hugging Face or a custom set passed by path (see the examples in `data/`).
2. For each question, build one ordering per option position with the correct answer placed there.
3. Build a prompt from the question and the permuted options and send it to the model.
4. Parse the model's response.
5. Pass the responses to the `analysis.metrics` module, which computes the ODS for each question and aggregates the results for the experiment.
6. `analysis.report` writes the analysis to the `results/` directory.

## Evaluation Methodology

**Data**

The MMLU test set (Hendrycks et al.), 20 questions from each of its 57 subjects, 1,140 in total, for
Claude Opus 5 and Claude Haiku 4.5; 5 per subject, 285 in total, for Claude Opus 4.5 and Claude Sonnet 4.5.

**Models**

- Claude Opus 5 (thinking disabled)
- Claude Opus 4.5 (no thinking parameter; effort low)
- Claude Haiku 4.5
- Claude Sonnet 4.5

None of the runs use extended thinking, so each answer is the model's first impression of the
prompt. Claude Sonnet 4.5 and Claude Opus 4.5 are the oldest models still served on the Claude API
(Sonnet 4.5 retires on 30 November 2026).

**Metrics** 

- **ODS (Order Dependency Score)** : how much the model's choice of option moves when only the
  ordering changes, on a `[0, 1]` scale. `P_p(o)` is the probability that the model picked option `o`
  when shown ordering `p`. ODS sums the variance of `P_p(o)` across orderings over the `K` options,
  scaled so the worst case is 1:

  ```
  ODS = sum_o Var_p[ P_p(o) ] / (1 - 1/K)
  ```

  A model that picks the same option under every ordering scores 0. A model that always answers "A"
  picks a different option under each ordering and scores 1.
- **Questions with any change**: the number of questions whose chosen option differed under at
  least one ordering.
- **Mean consistency**: for each question, the share of orderings on which the model chose the
  question's modal (most common) option, averaged over questions. 100% means every ordering gave the
  same answer; a model that picks a different option under each of four orderings scores 25%.
- **Accuracy per prompt**: the share of all prompts answered correctly, averaged over orderings, so
  a question is only fully credited when it is right under every ordering.


## Results

**Headline metrics** from each run's `report.md`.

| Metric | Claude Opus 5 | Claude Opus 4.5 | Claude Haiku 4.5 | Claude Sonnet 4.5 |
|---|---|---|---|---|
| Mean ODS (0 = order-invariant, 1 = fully order-determined) | 0.029 | 0.069 | 0.140 | 0.278 |
| Questions with at least one order-induced answer change | 57 / 1140 | 35 / 285 | 273 / 1140 | 141 / 285 |
| Mean consistency (agreement with modal answer) | 98.3% | 96.1% | 91.8% | 84.0% |
| Accuracy per prompt (averaged over orderings) | 93.7% | 90.2% | 83.4% | 67.2% |
| Share of picks on position A | 25.1% | 24.0% | 30.3% | 48.0% |
| Recall std across correct-answer positions (RStd, points) | 0.3 | 1.1 | 2.8 | 11.8 |

The data indicates order dependency is fading with newer and more capable models. Claude Sonnet 4.5
(September 2025) changes its answer on 49% of questions and puts 48% of its picks on position A:
moving the correct option to A lifts its accuracy from 67% to 87%, moving it to D drops it to 60%.
Claude Haiku 4.5 (October 2025) changes on 24% and still favours A, Claude Opus 4.5 (November 2025)
on 12% with a flat position profile, and Claude Opus 5 (2026) on 5%.

The gaps are far larger than the sampling error. Opus 5 and Haiku 4.5 saw the same 1,140 questions,
so the error on their answer-change rates is under two points; Opus 4.5 and Sonnet 4.5 saw a 285
question subset of the same subjects, giving an error of about three points. The pattern matches
Zheng et al., who report an RStd of 5.5 for gpt-3.5-turbo on 0-shot MMLU: Sonnet 4.5 is worse than
that at 11.8, Haiku 4.5 better at 2.8, and the two Opus models close to order-invariant.

## Financial Filing (Beta)

*The question set is small (18 hand-written
questions over one filing), only two models were run, and the results have not been checked to the
standard of the MMLU runs above.*

`data/apple_fy2025_mcq.json` asks 18 questions
about Apple's 2025 Q4 8-K press release, which is included in the repo in `data/`.
The four options (answers) are figures from the same line of the filing
(the answer plus another period, another statement, or the GAAP rather than non-GAAP column), so they
cannot be separated without reading the document.

```
uv run bin/order-dependency.py --model claude-haiku-4-5 --thinking omit --effort none -q data/apple_fy2025_mcq.json -o apple-fy2025-haiku
uv run --group hf bin/order-dependency.py --backend hf --model Qwen/Qwen2.5-0.5B -q data/apple_fy2025_mcq.json -o apple-fy2025-qwen05b
```

**Headline metrics**

| Metric | Claude Haiku 4.5 | Qwen2.5-0.5B |
|---|---|---|
| Mean ODS (0 = order-invariant, 1 = fully order-determined) | 0.000 | 0.906 |
| Questions with at least one order-induced answer change | 0 / 18 | 18 / 18 |
| Mean consistency (agreement with modal answer) | 100.0% | 38.7% |
| Accuracy per prompt (averaged over orderings) | 100.0% | 27.3% |
