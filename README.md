# ahead-of-time-demo

A small forecast-then-prescribe loop on the stack of the Mistral x ClickHouse "Ahead of
Time" hackathon, built before the event to learn where it breaks:

1. two public time series are loaded into **ClickHouse**;
2. **Timestral** (Mistral's open-weight time series model) forecasts the next 24 hours on CPU;
3. **Mistral Large** reads the forecast and proposes one action. It cannot execute it: a
   named human must approve, a deterministic gate must agree, and even then the
   "execution" is a dry run;
4. each agent run is instrumented with the **Langfuse** SDK;
5. a **preregistered backtest** scores Timestral against seasonal-naive baselines on hours
   the model cannot have seen.

It is one day of work. Read [What was not run / known limits](#what-was-not-run--known-limits)
before trusting anything here: in particular, no trace has reached a Langfuse server yet.

## Backtest

```bash
uv sync
uv run python -m aot.backtest
```

Output of that command on 2026-11-19 (Apple M-series laptop, CPU):

```text
model     : mistralai/Timestral-123M @ 5a147287addd  (timestral 1.0.0, torch 2.15.0, chdb 4.5.1, CPU)
protocol  : PREREG.md - daily origins 2026-10-16 -> 2026-11-18 at 00:00 UTC, horizon 24 h
bootstrap : paired over windows, 10000 resamples, seed 20261119

Series A - French national electricity load (MW), replication
  data: data/eco2mix_national_tr_snapshot.csv  sha256=fcf9d14fd925010f...  7580 rows -> 1896 hourly points (1 missing, kept as NaN)
  34 windows, 815 scored hours

  forecaster                      MAE    MASE      WQL   per-window MAE q1 / median / q3
  Timestral-123M (median)      1213.9   0.722   0.0168   680.0 / 1051.2 / 1568.5
  seasonal naive, 168 h        3939.8   2.350   0.0770   1699.0 / 3064.9 / 6073.1
  seasonal naive, 24 h         2831.9   1.675   0.0553   1156.9 / 2389.2 / 3900.8

  relative MAE vs seasonal naive, 168 h : 0.308  95% CI [0.235, 0.398]  (lower MAE in 31/34 windows)
  relative MAE vs seasonal naive, 24 h  : 0.429  95% CI [0.345, 0.528]  (lower MAE in 30/34 windows)
  preregistered verdict: Timestral better
  q10-q90 interval covers 75.1% of scored hours (nominal 80%)

Series B - Paris bike counter, Sebastopol S-N (bikes/h), confirmatory
  data: data/paris_bike_sebastopol_sn_snapshot.csv  sha256=f863290857115aec...  1822 rows -> 1896 hourly points (74 missing, kept as NaN)
  34 windows, 790 scored hours

  forecaster                      MAE    MASE      WQL   per-window MAE q1 / median / q3
  Timestral-123M (median)        72.0   0.895   0.1308   41.2 / 52.2 / 79.3
  seasonal naive, 168 h         101.6   1.252   0.2575   47.3 / 96.9 / 137.5
  seasonal naive, 24 h          114.0   1.404   0.2889   54.2 / 100.4 / 158.7

  relative MAE vs seasonal naive, 168 h : 0.708  95% CI [0.579, 0.840]  (lower MAE in 25/34 windows)
  relative MAE vs seasonal naive, 24 h  : 0.631  95% CI [0.514, 0.762]  (lower MAE in 26/34 windows)
  preregistered verdict: Timestral better
  q10-q90 interval covers 81.6% of scored hours (nominal 80%)
```

The same figures are stored in `results/backtest.json`. A test fails if this block and that
file drift apart, and `uv run python -m aot.backtest --check` reruns everything and compares
with the recorded values (1% tolerance, for float differences across CPUs).

### What was fixed in advance

The protocol is in [`PREREG.md`](PREREG.md). The order of the commits is the point:

| Commit | Content |
|---|---|
| `b9010c3` | `PREREG.md` and the two frozen data files, nothing else |
| `0919148` | the analysis code, no result |
| `43b4f66` | the recorded run |

Fixed in `b9010c3`: the series and how the second one was chosen, the 34 daily origins, the
model revision, the baselines, the primary metric (relative MAE), its interval (paired
bootstrap over windows, 10,000 resamples, seed fixed) and the rule that turns the interval
into a verdict.

Two things that table does not show, and that matter:

- **Series A is a replication, not a blind test.** It had already been run once, in commit
  `75f8e89`, before the preregistration existed, and its point results were known. Only
  series B was untouched. `PREREG.md` says so in its first section.
- **The first attempt at the preregistered run crashed** before producing any forecast (a
  type error in the hourly grid query). It was fixed and the run was then done once. A
  second run was made only to check that the output is identical.

### How to read the table

- **Unseen data.** The Timestral-123M weights were committed to Hugging Face on
  2026-10-15 15:22 UTC. Every scored hour is dated 2026-10-16 or later. The context given to
  the model does include earlier public data, as any forecast input would.
- **Relative MAE** is Timestral's total absolute error divided by the baseline's. Below 1,
  Timestral is more accurate. The interval comes from resampling whole days.
- **MASE** divides the absolute error by the in-context error of the weekly seasonal-naive
  forecast. For series A the weekly baseline scores above 2 out of sample because the test
  period is the start of the heating season: load rises week after week, which a "same hour
  last week" rule cannot follow. That makes series A an easy win; series B, where the weekly
  baseline is a fair opponent, is the more informative row.
- **WQL** is the weighted quantile loss over Timestral's 21 quantiles. For a baseline that
  gives a single value it reduces to total absolute error over total load.

## The agent and the approval gate

```bash
npm ci                                 # installs the pinned agent-proof-kit CLI
export MISTRAL_API_KEY=...             # see .env.example
uv run python -m aot propose           # one agent run: writes a proposal, executes nothing
uv run python -m aot review <id>       # a human types "approve <id>" in a terminal, or rejects
uv run python -m aot execute <id>      # refused unless approved; prints a dry-run line
uv run python -m aot log               # append-only action log, stored in ClickHouse
```

The toy decision: an operator has one flexible load that must run for 4 consecutive hours
in the next 24. The agent proposes the start hour with the lowest forecast national demand.

What stands between the model and the world:

- the agent has no tools. It returns JSON, which a guardrail validates (known hour, window
  inside the horizon). An invalid answer is logged as `invalid` and can never be approved;
- the guardrail also computes the deterministic best window and records how far the agent's
  choice is from it, so the reviewer sees the gap;
- `review` refuses to run without an interactive terminal, asks for an operator name, and
  approves only on the exact phrase `approve <id>`. Anything else is a rejection;
- an approval writes a receipt in `receipts/<id>/`: the run exported in the
  [agent-proof-kit](https://github.com/guillaumevele/agent-proof-kit) format, its proof
  bundle and attestation (`agent-proof bundle`, `sign-bundle`), and a hash binding the
  bundle to the run document. With `AOT_SIGNING_KEY` set to an RSA private key the
  attestation is signed; without it, it is a digest only;
- `execute` needs two independent checks to agree. The latest event in the log must be
  `approved`, **and** `agent-proof verify` must pass on the state that execution would
  create. Under `policies/aot-policy.json` a `write` action completed without a recorded
  human approval is a critical finding. If the CLI is missing, `execute` refuses;
- when `execute` does run, it prints what it would do and logs `executed_dry_run`. There is
  no code in this repository that acts on an external system.

The tests that prove the gate (`tests/test_pipeline.py`):

| Test | What it shows |
|---|---|
| `test_nothing_executes_without_human_approval` | pending and rejected proposals cannot be executed |
| `test_agent_proof_gate_fails_an_unapproved_execution_and_passes_an_approved_one` | same run document: FAIL without approval, PASS with it |
| `test_gate_refuses_an_approval_not_recorded_by_a_human` | an `approved` event appended by the agent passes the status check and is stopped by the gate |
| `test_gate_fails_closed_without_the_cli` | no gate binary, no execution |
| `test_tampered_receipt_is_detected` | changing the approver or the gate's conclusion invalidates the receipt |

Each of the two checks in `execute` was also removed in turn to confirm that tests go red
without it (results under "Run on 2026-11-19" below).

This is a process guard, not a security boundary. Anyone who can write to the database can
append a forged `human:` approval, and an unsigned receipt proves integrity, not identity.

## Tracing

`propose` wraps the run in Langfuse observations: one `agent` root, two `tool` spans
(ClickHouse read, Timestral forecast), one `generation` (Mistral Large, with token usage),
one `guardrail`. The trace id is stored with the proposal. If the Langfuse keys are missing,
`propose` refuses to run unless `--allow-untraced` is passed.

## What was not run / known limits

Not run:

- **No trace has been sent to a Langfuse server, and none is shown here.** No Langfuse
  project was available on the machine. The instrumentation is exercised in a test with the
  real Langfuse SDK and an in-memory span exporter: this shows the five observations are
  emitted, not that they display correctly in Langfuse. This is the main gap against the
  hackathon rules.
- **No human has approved a proposal through the CLI.** The one real proposal is still
  pending. The approve, reject, receipt and dry-run paths are covered by tests that call the
  same functions; the interactive prompt itself was not used.
- **The CI workflow has never run.** `.github/workflows/ci.yml` is written (locked install,
  tests, backtest replay with `--check`) but the repository is not on GitHub yet. The 1%
  tolerance of `--check` across CPU architectures is a guess until it runs on Linux.
- ClickHouse runs embedded through chDB (ClickHouse engine 26.11 in-process). Nothing was
  tested against a ClickHouse server or ClickHouse Cloud.
- Only the 123M checkpoint was run, not the 209M one.
- The agent produced one proposal. There is no evaluation of how often Mistral Large picks a
  good window, and no prompt iteration.

Known limits of the backtest:

- two series, 34 days each, one horizon. Nothing general about Timestral follows;
- the baselines are simple. RTE publishes its own day-ahead forecast for series A; it is
  certainly a stronger reference and it is not compared;
- daily windows are not independent, so the bootstrap intervals are probably too narrow;
- the period includes the end of daylight saving time, school holidays and two public
  holidays, and no forecaster receives calendar information;
- series B is one sensor with 74 missing hours; outages look like demand changes;
- real-time eCO2mix values are provisional and may be revised by RTE;
- the q10-q90 interval covers 75.1% of hours on series A, a little narrow;
- until this repository is pushed, the commit order above rests on local git history, which
  its author could rewrite.

Run on 2026-11-19, on one laptop:

- the backtest above, twice, from the CSV files through ClickHouse and the real
  Timestral-123M weights (SHA-256 of `model.safetensors` checked against the Hugging Face
  LFS id);
- one successful agent run against the Mistral API (`mistral-large-4`), with
  `--allow-untraced`. It proposed 2026-11-19 01:00 to 05:00 UTC with "high" confidence, the
  same window as the deterministic reference. Two earlier attempts failed on bugs in this
  code (a sampling parameter the API rejected, then chunked message content) and wrote nothing;
- on that pending proposal: `execute` refused, `review` without a terminal refused, and
  `propose` without Langfuse keys or `--allow-untraced` refused;
- the test suite (`uv run pytest`): 21 tests, no network, Timestral and Mistral
  replaced by stubs, agent-proof-kit called for real;
- two mutation checks on `execute`: with the `agent-proof` gate call removed, 2 tests go
  red; with the status check removed, 1 test goes red. The first time the status check was
  removed, no test failed: the gate caught every case except a second execution of an
  already executed action, which had no test. That test was added.

## Built on earlier work

- [agent-proof-kit](https://github.com/guillaumevele/agent-proof-kit) 0.7.0 is a runtime
  dependency (pinned in `package.json`): it is the gate and it produces the receipts. Its
  bytefence write firewall and its agent hooks are not used here.
- The preregistration discipline (protocol committed before the data it could be tuned on,
  seeded bootstrap, amendments instead of edits, limits stated first) follows
  [unsourced-precision](https://github.com/guillaumevele/unsourced-precision). That
  project also anchors its commits with OpenTimestamps; this one does not.

## Layout

```text
PREREG.md             backtest protocol, committed before the analysis
aot/store.py          ClickHouse tables: raw (series), action_log (append-only)
aot/forecast.py       Timestral wrapper, seasonal-naive baseline
aot/metrics.py        MAE, MASE scale, weighted quantile loss, coverage
aot/backtest.py       rolling-origin backtest, bootstrap, --check
aot/agent.py          Mistral Large prompt, guardrail, proposal
aot/approval.py       human decision, gated dry-run execution
aot/proof.py          agent-proof-kit gate and receipts
aot/tracing.py        Langfuse wrapper, refusal of untraced runs
policies/             gate policy (derived from agent-proof-kit's high-stakes policy)
tests/                fast tests with stubs
```

## Licences and credits

- Code: MIT (see `LICENSE`).
- Series A: RTE, eCO2mix, Licence Ouverte v2.0. Series B: Ville de Paris, ODbL v1.0.
  See `data/README.md`.
- Timestral weights: Mistral AI, Modified MIT License (not usable by companies above
  $20M monthly revenue without a commercial licence; read the model card). The weights are
  downloaded from Hugging Face at first run and are not part of this repository.
