# Backtest preregistration

Written and committed on 2026-11-19, before the analysis it describes was coded or run.
The commit that adds this file contains the protocol and the two frozen data files, and
nothing else. Any later deviation is recorded in a `PREREG-AMENDMENT-n.md` file, never by
editing this one.

## What is already known (disclosure)

- **Series A (French national electricity load)** has already been backtested once, in
  commit `75f8e89`, under the same windows, model and baselines as below. Its point results
  (MAE, MASE, WQL, coverage) were seen before this file was written. For series A this
  document therefore fixes only what is new: the primary metric, its interval and the
  decision rule. Series A is a replication, not a blind test.
- **Series B (one Paris bike counter)** has never been given to any forecaster. Its raw
  values were downloaded and the first and last lines printed; nothing else was looked at.
  Series B is the confirmatory test.

## Question

On hours dated after the Timestral weights were published, is a zero-shot Timestral-123M
24-hour forecast more accurate than a seasonal-naive forecast?

## Data

| | Series A | Series B |
|---|---|---|
| What | French national electricity load, MW | Bikes per hour, counter "Totem 73 boulevard de Sébastopol S-N" (`100057445-104057445`) |
| Publisher, licence | RTE (eCO2mix), Licence Ouverte v2.0 | Ville de Paris, ODbL |
| File | `data/eco2mix_national_tr_snapshot.csv` | `data/paris_bike_sebastopol_sn_snapshot.csv` |
| SHA-256 | `fcf9d14fd925010f2eb78ee73bc77dae3c876e36246c19aac033e63287611c80` | `f863290857115aece9309ae0447ae32ec59667c4a64bbe7dabb69a5bc4ee7c66` |
| Native step | 15 minutes, averaged to hourly in ClickHouse (an hour needs its 4 readings) | hourly |
| Range (UTC) | 2026-09-01 00:00 to 2026-11-18 23:45 | 2026-09-01 00:00 to 2026-11-18 22:00 |

Series B selection rule: the counter with the highest total count over September 2026, a
month that lies entirely before the test period. The ranking query used only September.

Missing hours stay missing (NaN). They are passed as NaN to Timestral, skipped by the
baselines as described below, and excluded from every score when they are targets.

## Protocol

- Forecast origins: every day at 00:00 UTC, from 2026-10-16 to 2026-11-18 inclusive (34 origins).
  The Timestral-123M weights were committed to Hugging Face on 2026-10-15 15:22 UTC, so every
  target hour postdates the weights.
- Horizon: 24 hourly steps.
- Context: all hours of the same series from 2026-09-01 00:00 UTC up to the origin. No covariates.
- Model: `mistralai/Timestral-123M`, revision `5a147287addda85f6acd8da3fb8858fe9af7e8f7`,
  library defaults, CPU. Point forecast: the 0.5 quantile.
- Baselines: seasonal naive with a 168-hour season and with a 24-hour season. If the value
  one season back is missing, the value one more season back is used.
- No parameter of the model, of the baselines or of the windows is chosen after looking at
  test errors. The analysis is run once; a second run is allowed only to check determinism.

## Primary metric and decision rule

For each series and each baseline:

    relative MAE = (sum of |error| of Timestral) / (sum of |error| of the baseline)

over all scored hours. Below 1 means Timestral is more accurate.

Dispersion: paired bootstrap over the 34 daily windows (each resample draws 34 windows with
replacement and keeps Timestral and baseline errors of the same window together), 10,000
resamples, seed 20261119, percentile 95% interval.

Verdict per series, fixed in advance:

- **Timestral better**: the upper bound of the interval is below 1 against both baselines;
- **Timestral worse**: the lower bound is above 1 against at least one baseline;
- **Inconclusive**: anything else.

There is no pooled verdict across the two series.

## Secondary, descriptive only

MAE; MASE (scaled by the in-context error of the 168-hour seasonal naive); weighted quantile
loss over the 21 Timestral quantiles; empirical coverage of the q10-q90 interval; quartiles of
the per-window MAE; number of windows where Timestral has the lower MAE.

## Known weaknesses, stated before the run

- Two series, 34 days each, one horizon: no general claim about Timestral can follow.
- The baselines are simple. Operational forecasts exist for series A (RTE publishes one) and
  are not compared.
- The test period includes the end of daylight saving time, school holidays and two public
  holidays. No forecaster receives calendar information. Daily windows start at 00:00 UTC,
  which is 01:00 or 02:00 local time.
- Series B is a single sensor: outages and roadworks look like demand changes.
- Daily windows are not independent, so the bootstrap interval is probably too narrow.
- Until this repository is pushed, the only evidence that this file predates the results is
  the local git history, which its author could rewrite.
