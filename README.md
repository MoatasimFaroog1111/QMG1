# QMG1 — Precious Metals Forecasting Foundation

QMG1 is a modular forecasting foundation for precious-metal prices normalized to **USD per kilogram**. It separates market-data acquisition, unit normalization, technical feature engineering, model training, out-of-sample validation, artifact persistence, and inference.

## Forecast horizons

The project currently trains one persisted model per metal and horizon:

- 2 hours
- 4 hours
- 8 hours
- 12 hours
- 24 hours
- 72 hours
- 7 days = 168 hours
- 15 days = 360 hours
- 30 days = 720 hours

## Metals and data coverage

The data pipeline uses Dukascopy through `dukascopy-node@1.50.0` and requests M1 Bid OHLC plus source volume.

- Gold — XAU/USD — requested from 2009-05-01
- Silver — XAG/USD — requested from 2009-05-01
- Palladium — XPD.CMD/USD — configured from its available M1 history beginning 2021-07-04
- Platinum — XPT.CMD/USD — configured from its available M1 history beginning 2021-11-01

The requested end is August 2026. When run before 2026-09-01, the downloader stops at the latest completed UTC day. Re-running resumes existing yearly chunks rather than downloading valid files again.

## Price unit

Source metal prices are treated as USD per troy ounce and normalized as:

```text
1 troy ounce = 31.1034768 grams
1 kg = 32.15074656862798... troy ounces
USD/kg = USD/troy_ounce × 32.15074656862798...
```

The final M1 CSV files contain `open_usd_per_kg`, `high_usd_per_kg`, `low_usd_per_kg`, and `close_usd_per_kg`.

## Architecture

```text
scripts/
  download_all_metals_m1_usd_per_kg.py  -> thin data-pipeline entrypoint
  train_models.py                        -> train once + persist
  predict.py                             -> load persisted model + predict

src/qmg1/
  config.py
  features.py                            -> causal technical-analysis features
  data/
    metals.py                            -> metal catalog / coverage
    dukascopy.py                         -> data-source adapter
    normalizer.py                        -> USD/troy oz -> USD/kg
    pipeline.py                          -> acquisition orchestration
  ml/
    model_factory.py                     -> replaceable regressor factory
    targets.py                           -> real elapsed-time targets
    dataset.py                           -> dataset assembly
    evaluation.py                        -> walk-forward OOS evaluation + purge
    artifacts.py                         -> model persistence boundary
    trainer.py                           -> Train Once -> Persist
    predictor.py                         -> Load -> Predict only
```

This keeps responsibilities separate so the data source, regressor, persistence layer, or feature set can be replaced without rewriting unrelated components.

## Technical features

The hourly modeling layer is derived from M1 data and currently includes multi-scale log momentum, moving-average ratios, rolling z-scores, realized volatility, rolling range/position, EMA 12/26/50/200, MACD, RSI, ATR, ADX/+DI/-DI, stochastic oscillator, Bollinger position/width, candle body/wicks/range, source-volume transforms, minute coverage, and hour/day cyclical features.

All feature calculations use information available at or before each feature timestamp. Target/future columns are explicitly excluded from the model feature list.

## Leakage-safe targets and validation

Forecast targets use **real elapsed UTC time**, not `shift(N rows)`. If a requested target timestamp lands during a market closure, the first available quote at or after that requested time is used within a bounded tolerance.

Model validation uses expanding walk-forward splits. Before each validation fold, training samples whose future target timestamp reaches the validation period are purged. Metrics include:

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

Persisted artifacts contain the trained estimator, exact feature column list, horizon, training range, training configuration, and out-of-sample metrics.

## GitHub Codespaces

From the repository root:

```bash
python --version
node --version
npm --version
```

Python 3.10+ is recommended and Node.js 18+ is required by the selected downloader.

Install Python dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Download and normalize the data:

```bash
python scripts/download_all_metals_m1_usd_per_kg.py
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

Or train a single metal:

```bash
python scripts/train_models.py --metal silver
```

Persisted model artifacts are stored under `models/<metal>/` and are also ignored by Git.

Predict silver two hours ahead:

```bash
python scripts/predict.py --metal silver --horizon 2
```

Predict gold 15 days ahead:

```bash
python scripts/predict.py --metal gold --horizon 360
```

Predict platinum 30 days ahead:

```bash
python scripts/predict.py --metal platinum --horizon 720
```

## Quality checks

```bash
pip install -r requirements-dev.txt
ruff check src scripts tests
python -m compileall -q src scripts tests
pytest -q
```

## Production operation

The service exposes separate operational endpoints:

- `/livez` for process liveness
- `/readyz` for traffic admission after required, checksum-valid artifacts are available
- `/health` for detailed runtime status
- `/metrics` for request counters

Production deployments must configure `QMG1_API_KEY`. Clients pass it in `X-API-Key` when
calling `/predict`. Configure the deployed contract with `QMG1_REQUIRED_METALS` and
`QMG1_REQUIRED_HORIZONS`; `/api/meta` reports only artifacts actually available at runtime.
See `docs/operations.md` for deployment, monitoring, and rollback procedures.

The same checks run in GitHub Actions.

## Important modeling note

A backtest metric is not evidence that a future market price is certain. QMG1 records performance against a persistence baseline and exposes uncertainty information so model quality can be evaluated before any forecast is used operationally.
