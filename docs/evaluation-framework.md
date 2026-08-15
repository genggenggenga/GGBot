# GGBot Evaluation Framework

## Scope

The evaluation system has two complementary evaluators:

- **Code-based evaluator**: deterministic contract checks for intent, state,
  tool traces, confirmation gates, postconditions, retrieval, citations, and
  abstention.
- **LLM-as-Judge**: optional semantic quality signal for relevance, accuracy,
  completeness, helpfulness, safety, and groundedness. It is not a CI safety
  gate and must be calibrated against human review.

## Dataset Layers

| Suite | Purpose | Execution |
|---|---|---|
| `smoke` | Fast regression for core paths | Every local/CI run |
| `bad_cases` | Prevent recurrence of known failure patterns | Every CI run |
| `golden` | Versioned comparison and release evidence | Scheduled/release run |
| `exploratory` | Unreviewed investigation samples | Never a release gate |

`golden-v1-candidate` contains 280 scenario-matrix cases and still requires
independent human calibration before being treated as release-golden.

Current first-phase coverage is 90 Smoke cases (18 each for NLU, DST, Tool,
RAG, and E2E) and 50 Bad-case cases, of which 42 are high or critical risk.
The second phase expands Chunking evaluation to 80 query-evidence cases and
Golden candidates to 50 NLU, 60 DST, 60 Tool, 70 RAG, and 40 E2E cases.

## Dataset Contract

Each versioned case can declare `expected_tool_trace`, `forbidden_tools`,
`expected_postconditions`, `required_evidence_ids`, and `must_abstain` in
addition to legacy intent, slot, tool, and status fields. A case is promoted
from exploratory to bad-case or golden only after its expected behavior is
reviewed.

The deterministic corpus is versioned at
`data/eval/knowledge/corpus-v1.json`. RAG fixtures must reference its chunk
ids; no independent hard-coded policy corpus is allowed.

## Commands

```bash
.venv/bin/python -m evaluation.run --suite smoke --mode deterministic
.venv/bin/python -m evaluation.run --suite bad_cases --mode deterministic
.venv/bin/python -m evaluation.run --suite golden --mode deterministic
.venv/bin/python -m evaluation.run --suite golden --mode deterministic --judge
```

`--judge` requires `ANTHROPIC_API_KEY` and is intentionally opt-in. It receives
the candidate response, recorded tool observations, citations, and expected
behavior. Judge failures are reported explicitly and never converted into a
neutral score.

`realistic` evaluation is an explicit, environment-specific harness. It must
inject real model and retrieval dependencies and is never silently replaced by
the fake deterministic components.

## Gates

Code-based CI blocks on high-risk contract failures. In particular:

- A forbidden write tool must never execute.
- Tool traces and labelled postconditions must satisfy the fixture.
- Confirm-before-write behavior must remain intact.

LLM-as-Judge reports quality trends but does not override deterministic safety
or business-contract failures.

Metrics with no labelled applicable cases report a sample size of zero. Their
numeric value must not be interpreted as a quality score until the associated
Golden or Bad-case coverage exists.
