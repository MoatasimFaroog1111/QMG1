# QMG1 — Precious Metals Forecasting Foundation

QMG1 is a modular forecasting foundation for precious-metal prices normalized to **USD per kilogram**. It separates market-data acquisition, unit normalization, technical feature engineering, target construction, out-of-sample validation, model persistence, training, and inference.

## Forecast horizons

One persisted model is trained per metal and horizon:

- 2 hours
- 4 hours
- 8 hours
- 12 hours
- 24 hours
- 72 hours
- 7 days = 168 hours
- 15 days = 360 hours
- 30 days = 720 hours

## Metals and historical sources

The default pipeline uses a **hybrid provider** behind one `HistoricalM1Provider` boundary:

- Gold — XAU/USD — HistData Generic ASCII M1 bulk archives — requested from 2009-05-01
- Silver — XAG/USD — HistData Generic ASCII M1 bulk archives — requested from 2009-05-01
- Palladium — XPD.CMD/USD — direct Dukascopy M1 BI5 — available from 2021-07-04
- Platinum — XPT.CMD/USD — direct Dukascopy M1 BI5 — available from 2021-11-01

Both verified paths use Bid OHLC. HistData timestamps are fixed EST without DST and are converted to UTC before normalization. Dukascopy BI5 files are downloaded directly in Python, LZMA-decoded, parsed as big-endian M1 records, and cached per day.

Gold and Silver are downloaded as HistData annual/monthly bulk archives. Palladium and Platinum use bounded yearly Dukascopy ranges with resumable day-level BI5 caches. This keeps the bulk path fast while retaining a direct source for metals not covered by HistData.

The configured end is August 2026. Before 2026-09-01, the pipeline stops at the latest completed UTC day rather than pretending the unfinished month is complete.

## Price unit

Source metal prices are interpreted as USD per troy ounce and normalized as:

```text
1 troy ounce = 31.1034768 grams
1 kg = 32.15074656862798... troy ounces
USD/kg = USD/troy_ounce × 32.15074656862798...
```

The final M1 files contain `open_usd_per_kg`, `high_usd_per_kg`, `low_usd_per_kg`, and `close_usd_per_kg`, plus provenance metadata.

## Architecture

```text
scripts/
  download_all_metals_m1_usd_per_kg.py  -> acquisition + normalization CLI
  train_models.py                        -> train once + persist, resume-safe
  predict.py                             -> load persisted model + predict only
  live_smoke_test.py                     -> real Dukascopy BI5 source smoke
  histdata_smoke_test.py                 -> real HistData bulk source smoke

src/qmg1/
  config.py
  features.py                            -> causal technical-analysis features
  data/
    provider.py                          -> historical M1 provider contract
    metals.py                            -> metal catalog / availability
    histdata.py                          -> XAU/XAG bulk provider
    dukascopy_direct.py                  -> direct BI5 provider + cache
    hybrid.py                            -> per-metal provider routing
    normalizer.py                        -> provider-neutral USD/troy oz -> USD/kg
    pipeline.py                          -> acquisition orchestration
  ml/
    model_factory.py                     -> replaceable regressor factory
    targets.py                           -> real elapsed-time targets
    dataset.py                           -> reusable feature base + targets
    evaluation.py                        -> walk-forward OOS evaluation + purge
    artifacts.py                         -> model persistence boundary
    trainer.py                           -> Train Once -> Persist
    predictor.py                         -> Load -> Predict only
```

The older `dukascopy.py` CLI adapter is retained as an optional compatibility adapter, but it is not the default historical path.

## Technical features

The modeling layer resamples M1 to hourly OHLC and builds causal features including multi-scale log momentum, moving-average ratios, rolling z-scores, realized volatility, rolling range/position, EMA 12/26/50/200, MACD, RSI, ATR, ADX/+DI/-DI, stochastic oscillator, Bollinger position/width, candle body/wicks/range, source-volume transforms, minute coverage, and hour/day cyclical features.

All feature calculations use information available at or before each feature timestamp. Target/future columns are explicitly excluded from the model feature list. Large normalized CSV files are loaded with only the six columns required by modeling to reduce peak memory.

## Leakage-safe targets and validation

Forecast targets use **real elapsed UTC time**, not `shift(N rows)`. If a requested target timestamp lands during a market closure, the first available quote at or after that requested time is used within a bounded tolerance.

Model validation uses expanding walk-forward splits. Before each validation fold, training samples whose future target timestamp reaches the validation period are purged. The model factory disables sklearn's internal random early-stopping split so time-series validation remains under QMG1 control.

Metrics include:

- MAE in USD/kg
- RMSE in USD/kg
- sMAPE
- directional accuracy
- persistence/no-change baseline MAE
- improvement versus persistence
- empirical 10th/90th percentile out-of-sample residuals

The residual percentiles are used to provide an empirical 80% prediction interval. It is an uncertainty estimate, not a guarantee.

## Train Once -> Persist -> Load & Predict

Training and inference are intentionally separate. `predict.py` never retrains a model.

Persisted artifacts contain the trained estimator, exact feature schema, horizon, training range, training configuration, validation method, and out-of-sample metrics.

Feature engineering is performed once per metal and reused across all requested horizons. Training is resume-safe: an existing horizon artifact is skipped unless `--force` is supplied.

## GitHub Codespaces

Python 3.12 is the tested runtime. The default data pipeline does **not** require Node.js.

Install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download and normalize all configured metals:

```bash
python scripts/download_all_metals_m1_usd_per_kg.py
```

Download only Silver:

```bash
python scripts/download_all_metals_m1_usd_per_kg.py --metal silver
```

Use an explicit UTC range when needed; `--end` is exclusive:

```bash
python scripts/download_all_metals_m1_usd_per_kg.py \
  --metal silver \
  --start 2026-07-01 \
  --end 2026-08-01
```

Generated datasets are written under:

```text
metals_m1_usd_per_kg/
  raw/
  final/
  download_report.json
```

Large datasets are intentionally ignored by Git.

Train every metal and every requested horizon:

```bash
python scripts/train_models.py --metal all
```

Train or resume Silver only:

```bash
python scripts/train_models.py --metal silver
```

Train a subset of Silver horizons:

```bash
python scripts/train_models.py --metal silver --horizon 2 4 8 12 24
```

Force retraining when deliberately replacing persisted artifacts:

```bash
python scripts/train_models.py --metal silver --force
```

Persisted model artifacts are stored under `models/<metal>/` and are ignored by Git.

Predict Silver two hours ahead:

```bash
python scripts/predict.py --metal silver --horizon 2
```

Predict Gold 15 days ahead:

```bash
python scripts/predict.py --metal gold --horizon 360
```

Predict Platinum 30 days ahead:

```bash
python scripts/predict.py --metal platinum --horizon 720
```

## Verification

Static and integration checks:

```bash
pip install -r requirements-dev.txt
ruff check src scripts tests
python -m compileall -q src scripts tests
pytest -q
```

GitHub Actions also runs live-source smoke tests against both verified data paths:

- direct Dukascopy BI5 -> provider CSV -> USD/kg
- HistData bulk ZIP -> provider CSV -> USD/kg

The test suite additionally exercises Train -> Persist -> Load -> Predict without requiring an external market source.

## Important modeling note

A backtest metric is not evidence that a future market price is certain. QMG1 benchmarks against a persistence baseline and exposes uncertainty information so model quality can be evaluated before any forecast is treated as operationally useful.
