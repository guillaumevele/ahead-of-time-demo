# ahead-of-time-demo

A small forecast-then-prescribe loop, built to try the stack of the Mistral x ClickHouse
"Ahead of Time" hackathon before the event:

1. a public time series is loaded into **ClickHouse**;
2. **Timestral** (Mistral's open-weight time series model) forecasts the next 24 hours on CPU;
3. **Mistral Large** reads the forecast and proposes one action. It cannot execute it:
   a human has to approve it by name first, and even then the "execution" is a dry run;
4. each agent run is instrumented with the **Langfuse** SDK;
5. a backtest scores Timestral against seasonal-naive baselines on hours the model cannot have seen.

It is one afternoon of work on one series. The section "What was and was not run" is the
part to read before trusting anything here.

## The series and the decision

French national electricity load, 15-minute readings published by RTE (eCO2mix, Licence
Ouverte v2.0), from 2026-09-01 to 2026-11-18. A frozen extract lives in `data/` so the
numbers below can be replayed offline; see `data/README.md` for provenance.

The toy decision: an operator has one flexible load that must run for 4 consecutive hours
in the next 24. The agent proposes the start hour with the lowest forecast national demand.

## Backtest

```bash
uv sync
uv run python -m aot.backtest
```

Output of that command on 2026-11-19 (Apple M-series laptop, CPU):

```text
data      : data/eco2mix_national_tr_snapshot.csv  sha256=fcf9d14fd925010f...
            7580 quarter-hour rows -> 1896 hourly points (1 missing, kept as NaN)
model     : mistralai/Timestral-123M @ 5a147287addd  (timestral 1.0.0, torch 2.15.0, chdb 4.5.1, CPU)
protocol  : 34 daily origins, 2026-10-16 00:00 -> 2026-11-18 00:00 UTC, horizon 24 h, 815 scored hours

forecaster                  MAE (MW)    MASE      WQL
Timestral-123M (median)         1214   0.722   0.0168
seasonal naive, 168 h           3940   2.350   0.0770
seasonal naive, 24 h            2832   1.675   0.0553

Timestral q10-q90 interval covers 75.1% of scored hours (nominal 80%)
Timestral has the lower daily MAE in 31/34 windows vs weekly naive, 30/34 vs daily naive
Timestral inference time: 77.2 s for 34 forecasts
```

The same figures are stored in `results/backtest.json` (`--json` rewrites it), and a test
fails if this block and that file drift apart.

How to read it:

- **Protocol.** One forecast per day at 00:00 UTC, 24 hours ahead, zero-shot, no covariates,
  model defaults, all earlier history as context. The protocol was written before the first
  run and the table is that first run; nothing was tuned on these windows.
- **Unseen data.** The Timestral-123M weights were committed to Hugging Face on
  2026-10-15 15:22 UTC. Every scored hour is dated 2026-10-16 or later, so the targets did
  not exist when the weights were published. The context fed to the model does include
  earlier public data, as any forecast input would.
- **MASE** divides the absolute error by the in-sample error of the weekly seasonal-naive
  forecast on the context. The weekly baseline scores above 1 out of sample because the test
  period is the start of the heating season: load rises week after week, which a "same hour
  last week" rule cannot follow.
- **WQL** is the weighted quantile loss over Timestral's 21 quantile levels. For the naive
  baselines, which give a single value, it reduces to sum of absolute errors over sum of load.

Limits of this backtest:

- one series, one country, 34 days, one horizon. It says nothing general about Timestral;
- the baselines are deliberately simple. RTE publishes its own day-ahead forecast in the same
  dataset; it is certainly a stronger reference and it is not compared here;
- the period contains the end of daylight saving time (the one missing hour) and two public
  holidays (1 and 11 November). No calendar information is given to any forecaster;
- real-time eCO2mix values are provisional and may be revised by RTE;
- the 80% interval covers 75.1% of hours here, slightly narrow on this sample;
- only the 123M checkpoint was run. The 209M checkpoint was not.

## The agent and the approval gate

```bash
export MISTRAL_API_KEY=...            # see .env.example
uv run python -m aot propose           # one agent run: writes a proposal, executes nothing
uv run python -m aot review <id>       # a human types "approve <id>" in a terminal, or rejects
uv run python -m aot execute <id>      # refused unless approved; prints a dry-run line
uv run python -m aot log               # append-only action log, stored in ClickHouse
```

What stands between the model and the world:

- the agent has no tools. It returns JSON, which a guardrail validates (known hour, window
  inside the horizon). An invalid answer is logged as `invalid` and can never be approved;
- the guardrail also computes the deterministic best window and records how far the agent's
  choice is from it, so a reviewer sees the gap;
- `review` refuses to run without an interactive terminal, asks for an operator name, and
  approves only on the exact phrase `approve <id>`. Anything else is a rejection;
- `execute` refuses any action whose latest event is not `approved`. When it does run, it
  prints what it would do and logs `executed_dry_run`. There is no code in this repository
  that acts on an external system.

This is a process guard, not a security boundary: anyone who can write to the database can
append an `approved` event.

## Tracing

`propose` wraps the run in Langfuse observations: one `agent` root, two `tool` spans
(ClickHouse read, Timestral forecast), one `generation` (Mistral Large, with token usage),
one `guardrail`. The trace id is stored with the proposal. If the Langfuse keys are missing,
`propose` refuses to run unless `--allow-untraced` is passed.

## What was and was not run

Run on 2026-11-19, on one laptop:

- the backtest above, end to end, from the CSV through ClickHouse and the real Timestral-123M
  weights (SHA-256 of `model.safetensors` checked against the Hugging Face LFS id);
- one successful agent run against the Mistral API (`mistral-large-4`), with
  `--allow-untraced`. It proposed 2026-11-19 01:00 to 05:00 UTC with "high" confidence, the
  same window as the deterministic reference (excess 0 MW). The proposal was left `proposed`.
  Two earlier attempts failed on bugs in this code (a sampling parameter the API rejected,
  then chunked message content) and wrote nothing;
- `execute` on that pending proposal: refused. `review` without a terminal: refused.
  `propose` without Langfuse keys and without `--allow-untraced`: refused;
- the test suite (`uv run pytest`): 13 tests, no network, Timestral and Mistral replaced by stubs.

Not run:

- **no trace has been sent to a Langfuse server.** No Langfuse project was available on the
  machine. The instrumentation is exercised in a test with the real Langfuse SDK and an
  in-memory span exporter, which shows the five observations are emitted, not that they
  display correctly in Langfuse;
- **no human has approved a proposal through the CLI.** The approve, reject and dry-run paths
  are covered by tests that call the same functions; the interactive prompt itself was not used;
- ClickHouse runs embedded through chDB (ClickHouse engine 26.11 in-process). Nothing was
  tested against a ClickHouse server or ClickHouse Cloud;
- the agent produced one proposal. There is no evaluation of how often Mistral Large picks a good
  window, and no prompt iteration.

## Layout

```text
aot/store.py      ClickHouse tables: raw_load, action_log (append-only)
aot/forecast.py   Timestral wrapper, seasonal-naive baseline
aot/metrics.py    MAE, MASE scale, weighted quantile loss, coverage
aot/backtest.py   rolling-origin backtest (the command above)
aot/agent.py      Mistral Large prompt, guardrail, proposal
aot/approval.py   human decision, dry-run execution
aot/tracing.py    Langfuse wrapper, refusal of untraced runs
tests/            fast tests with stubs
```

## Licences and credits

- Code: MIT (see `LICENSE`).
- Data: RTE, eCO2mix, Licence Ouverte v2.0.
- Timestral weights: Mistral AI, Modified MIT License (not usable by companies above
  $20M monthly revenue without a commercial licence; read the model card). The weights are
  downloaded from Hugging Face at first run and are not part of this repository.
