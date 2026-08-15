# Local Evaluation Scripts

Run scripts from any directory; each resolves the repository root and uses the
project virtual environment.

```bash
./scripts/eval/run-smoke.sh
./scripts/eval/run-bad-cases.sh
./scripts/eval/run-golden.sh
./scripts/eval/run-chunking.sh
./scripts/eval/run-all.sh
```

The first three scripts run deterministic local evaluation only. They do not
call external business APIs or LLMs.

To opt into LLM-as-Judge for the Golden suite, configure `ANTHROPIC_API_KEY`
and run:

```bash
GGBOT_EVAL_JUDGE=true ./scripts/eval/run-golden.sh
```

Additional arguments are passed to the underlying evaluator. For example:

```bash
./scripts/eval/run-golden.sh --no-write
./scripts/eval/run-chunking.sh --chunk-size 256 --chunk-overlap 32
```
