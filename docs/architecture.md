# QMG1 Architecture

A modular forecasting foundation for precious-metal prices normalized to **USD per kilogram**.
This document captures the **component boundaries**, **runtime data flow**, and the
**architectural decisions** that explain why the system is shaped the way it is.

For deployment, monitoring, and rollback procedures, see [`operations.md`](./operations.md).
For the public forecast contract, see the top-level [`README.md`](../README.md).

---

## 1. Architectural principles

| Principle | Expression in QMG1 |
|---|---|
| **Train once, predict forever** | `scripts/train_models.py` writes a verified artifact; `scripts/predict.py` and the API never retrain. The same artifact is committed as the source of truth for serving. |
| **Strict separation of concerns** | `data/` owns acquisition, `ml/` owns training + the persisted artifact contract, `serving/` owns live market adapters, `api/` owns HTTP, `web/` owns the browser presentation. Cross-layer imports only flow downward. |
| **Composition over configuration** | Hot-swappable `RegressorFactory` and `LivePriceProvider` Protocols; new metals, challengers, or upstream feeds plug in without touching call sites. |
| **Immutability at rest** | Persisted artifacts and serving configs are frozen dataclasses with adjacent `.sha256` checksums. The `ModelArtifactRepository` refuses to load any artifact whose checksum does not match. |
| **Honest uncertainty** | The system reports an empirical 80% prediction interval whether the champion is persistence or a trained challenger. The README disclaims that a backtest metric is not a forward guarantee. |
| **No knowledge leakage** | Walk-forward splits purge training samples whose future target timestamp reaches the validation period. `CrossFittedSelectiveRegressor` recalibrates abstention from inner OOF predictions only. |

---

## 2. Component map

```
                ┌─────────────────────────────────────────────────────────┐
                │                       Browser                          │
                │   ┌─────────────────────────────────────────────┐       │
                │   │  src/qmg1/web/templates/dashboard.html      │       │
                │   │  src/qmg1/web/static/{css,js}/*             │       │
                │   └─────────────────────────────────────────────┘       │
                │                          │ same-origin POST             │
                │                          ▼ /web/predict                  │
                └─────────────────────────────────────────────────────────┘
                                            │
┌────────────────────────────────────────────┼─────────────────────────────────────────────┐
│                                            │                                             │
│                          ┌─────────────────┴──────────────────┐                          │
│                          │           HTTP API                 │                          │
│                          │  src/qmg1/api/{app,routes,service} │                          │
│                          └─────────────────┬──────────────────┘                          │
│                                            │  middleware: auth + rate limit + metrics   │
│                                            ▼                                             │
│                          ┌─────────────────────────────────────┐                          │
│                          │  Application Service                 │                          │
│                          │  ForecastApiService                  │                          │
│                          │  + RuntimeSettings (frozen)          │                          │
│                          └──────┬────────────────────────┬──────┘                          │
│                                 │                        │                                 │
│                                 ▼                        ▼                                 │
│              ┌────────────────────────┐   ┌──────────────────────────────────┐            │
│              │   Forecast Predictor   │   │   Live Price Provider             │            │
│              │   src/qmg1/ml/predictor │  │   src/qmg1/serving/live_price.py  │            │
│              └────────────┬───────────┘   │     ├─ BullionVault (primary)     │            │
│                           │               │     ├─ Dukascopy (secondary)      │            │
│                           ▼               │     └─ Resilient wrapper          │            │
│      ┌─────────────────────────────────┐  └──────────────┬───────────────────┘            │
│      │  Model Artifact Repository       │                 │                                │
│      │  src/qmg1/ml/artifacts.py        │                 │ HTTPS                          │
│      │  load_trained() w/ checksum      │                 ▼                                │
│      └────────────┬────────────────────┘   ┌────────────────────────────────┐            │
│                   │                         │  External market data          │            │
│                   ▼                         │  BullionVault / Dukascopy      │            │
│      ┌─────────────────────────────────┐    └────────────────────────────────┘            │
│      │  Persisted artifact             │                                                    │
│      │  serving_artifacts/models/      │  ┌──────────────────────────────────┐           │
│      │    <metal>_<horizon>h.joblib    │  │  Persisted feature data           │           │
│      │    + adjacent .sha256           │  │  metals_m1_usd_per_kg/final/      │           │
│      └─────────────────────────────────┘  │  + hourly context (UDX, SPX, …)   │           │
│                                            └──────────────────────────────────┘            │
│                                                                                          │
│                            src/qmg1/api/                                                  │
└──────────────────────────────────────────────────────────────────────────────────────────┘

                ┌─────────────────────────────────────────────────────────┐
                │       OFFLINE TRAINING PIPELINE (out-of-band)           │
                │                                                           │
                │  scripts/download_all_metals_m1_usd_per_kg.py             │
                │       │                                                   │
                │       ▼                                                   │
                │  src/qmg1/data/{pipelines,downloaders}                    │
                │  ├─ HistData (bulk yearly)                                │
                │  ├─ Dukascopy direct (BI5 daily)                          │
                │  └─ HybridPreciousMetalsM1Provider (routes per metal)     │
                │       ▼                                                   │
                │  metals_m1_usd_per_kg/final/<metal>_<range>.csv           │
                │       ▼                                                   │
                │  scripts/train_models.py                                  │
                │       ▼                                                   │
                │  src/qmg1/ml/{datasets,target,features,evaluation,         │
                │                  model_factory,selection,trainer}         │
                │       ▼                                                   │
                │  models/<metal>/<metal>_<horizon>h.joblib + .sha256       │
                │       ▼  promote                                          │
                │  serving_artifacts/models/<metal>/  ← committed artifact  │
                │                                                           │
                └─────────────────────────────────────────────────────────┘
```

### 2.1 Component responsibilities

| Component | Owns | Does NOT own |
|---|---|---|
| `data/` | Provider adapters, normalization (USD/oz → USD/kg), version-stamped CSV writing, chunked retries, offline data quality checks. | Knowing which column is a target. Re-deriving features. Holding metadata about ML champions. |
| `ml/dataset.py` + `targets.py` | The contract that says "an hourly closed bar at time `t` has a real `target_t+24h` at the first quote at or after `t + 24h` within a bounded tolerance." | Network I/O. Loading HDF5 or third-party data. |
| `ml/model_factory.py` | The `RegressorFactory` Protocol and concrete challengers (`MedianReturn`, `SelectiveHGB`, `CrossFittedSelectiveHGB`). | Any knowledge of which horizon, metal, dataset, or fold it serves. |
| `ml/evaluation.py` | Walk-forward cross-validation with target-time purge and empirical residual quantiles. Persists a `HorizonMetrics` value object. | The decision about which champion wins. |
| `ml/selection.py` | The `ChampionChallengerSelector` that compares challengers on the development set, verifies on the holdout, and enforces the dual promotion gate. | The final fit; that lives in `trainer.py`. |
| `ml/trainer.py` | Orchestrates: load feature base → for each horizon, select → final fit → persist artifact → write training report. | Reading network endpoints. Knowing how the API surfaces predictions. |
| `ml/artifacts.py` | The `ModelArtifactRepository` boundary. Verifies checksums, supports three on-disk encodings (direct `.joblib`, raw zip with members, base64-chunked zip). | Re-encoding, schema migration. |
| `ml/predictor.py` | Loads a persisted artifact and produces a prediction dict given the latest features. | Network endpoints. Live price quoting. |
| `serving/live_price.py` | `LivePriceProvider` Protocol, concrete providers (`BullionVault`, `Dukascopy`), `ResilientLivePriceProvider` with circuit breaker + persistent cache. | Knowing about persistence champions. |
| `api/service.py` | `ForecastApiService` — composition root for one request: load artifact → choose inference path (persistence vs. model) → quote live price → shape response. | Knowing the request came from FastAPI. Could be reused by any transport. |
| `api/routes.py` | FastAPI HTTP adapters. Maps JSON ↔ Pydantic schemas. | Domain logic. |
| `api/operations.py` | Sliding-window rate limiter, request Prometheus counters, request-id propagation, security headers. | Knowing which prediction logic runs. |
| `web/router.py` | Mounts the dashboard HTML, CSS, JS. Adds a same-origin `/web/predict` route that the dashboard calls without an API key. | Doing predictions. Just delegates. |
| `config.py` | `HORIZONS_HOURS` and the frozen `TrainingConfig`, `ProjectPaths`, `RuntimeSettings` dataclasses. | Reading env vars at import time. |

### 2.2 Dependency direction

```
config.py ← every module
     ↑
     │
data/  ←── ml/  ←── api/  ←── web/
                ↑        ↑
                │        │
                └────────┴── serving/
```

No upward imports, no circular dependencies. `ml/` does not import `api/`. `api/` orchestrates collaborators from `ml/`, `serving/`, and `data/` only.

---

## 3. Runtime data flow (request → response)

### 3.1 Happy path: `/predict` for a persistence champion

```
Client → POST /predict
  ↓ OperationalMiddleware:
  ├─ generate / propagate X-Request-ID
  ├─ enforce X-API-Key (if api_key configured)
  ├─ enforce per-client rate limit
  └─ record Prometheus counter
  ↓ routes.predict(PredictionRequest)
  ↓ ForecastApiService.predict(request)
      ├─ repository.load_trained(metal, horizon_hours)
      │     ├─ check <metal>/<metal>_<h>h.joblib exists
      │     ├─ verify_checksum() ← HMAC compare_digest
      │     ├─ joblib.load()
      │     └─ if missing → fall back to encoded chunks
      │                  └─ check <metal>/<bundle>.b64.part*
      │                  └─ reassemble + base64.b64decode(validate=True)
      │                  └─ verify SHA-256
      │                  └─ open as zip, pick <metal>_<h>h.joblib member
      ├─ branching on artifact["active_strategy"]:
      │   "persistence" → predictor.predict_live_persistence_from_artifact(
      │                     quote = LivePriceProvider.latest_quote(metal))
      │   "model"      → predictor.predict_latest_from_artifact(
      │                     csv_path = ServingDataLocator.target_csv(metal))
      └─ return prediction dict
  ↓ route → JSONResponse with X-Request-ID echo
Client ← 200 OK
```

### 3.2 Failure modes (HTTP mapping)

| Condition | HTTP | Why |
|---|---|---|
| No `X-API-Key` and a key is configured | 401 | `OperationalMiddleware` returns JSONResponse before the request reaches the router. |
| Client exceeds `QMG1_PREDICT_REQUESTS_PER_MINUTE` | 429 | Same middleware, returns JSONResponse with `Retry-After: 60`. |
| Horizon is not in `HORIZONS_HOURS` | 503 | `ForecastApiService._validate_horizon` raises `PredictionUnavailableError` to prevent serving requests that no training run was scoped for. |
| Artifact missing, corrupt, or checksum mismatch | 503 | The repository's load path raises `ValueError` or `FileNotFoundError`; the service maps them to `PredictionUnavailableError`. |
| Live price providers all fail and the persistent quote exceeds the stale TTL | 503 | `ResilientLivePriceProvider.latest_quote` raises `LivePriceUnavailableError` after exhausting providers + cache. |
| Liveness | `/livez` → 200 always | Process is up. |
| Readiness | `/readyz` → 200 or 503 | Returns 503 until every `required_metals × required_horizons` artifact is checksum-valid. |

The system deliberately maps almost all serving failures to **503 Prediction is temporarily unavailable** rather than leaking which layer failed. The middleware always records the actual exception in `LOGGER.warning(..., exc_info=True)` with the `request_id`, so an operator can correlate to logs and the training report.

---

## 4. SOLID compliance per component

| File | SRP | OCP | LSP | ISP | DIP | Notes |
|---|---|---|---|---|---|---|
| `config.py` | ✅ | n/a | n/a | n/a | n/a | Frozen dataclasses; read once. |
| `ml/model_factory.py` | ✅ | ✅ | ✅ | ✅ | ✅ | `RegressorFactory` Protocol + 5 implementations + `candidate_factories()`. The cleanest file in the codebase. |
| `ml/selective.py` | ✅ | ✅ | ✅ | ✅ | ✅ | `SelectiveShrinkageRegressor` and `CrossFittedSelectiveRegressor` are sklearn-compatible. |
| `ml/evaluation.py` | ✅ | ✅ | ✅ | ✅ | ✅ | Pure functions for scoring; `WalkForwardEvaluator` orchestrates. |
| `ml/dataset.py`, `targets.py` | ✅ | ✅ | n/a | ✅ | ✅ | Feature-base reuse across horizons by design. |
| `ml/predictor.py` | ✅ | ✅ | ✅ | ⚠️ | ✅ | Uses `dict[str, object]` for the artifact shape; tightening to `TypedDict` is a follow-up. |
| `ml/artifacts.py` | ✅ | ✅ | n/a | ✅ | ✅ | Three on-disk encodings behind one Protocol-shaped load surface. |
| `serving/live_price.py` | ✅ | ✅ | ✅ | ✅ | ✅ | Multiple providers behind one Protocol. Module is long (326 lines) and a future refactor splits `bullionvault`/`dukascopy_provider`/`resilient` into submodules. |
| `data/*` | ✅ | ✅ | ✅ | ✅ | ✅ | Each data source is its own class; `HybridPreciousMetalsM1Provider` composes them per metal. |
| `api/service.py` | ✅ | ✅ | ✅ | ✅ | ✅ | Composition root for one prediction. All collaborators injected. |
| `api/routes.py` | ✅ | ✅ | ✅ | n/a | ✅ | Thin HTTP adapters; no domain logic. |
| `api/operations.py` | ✅ | ✅ | ✅ | ✅ | ✅ | Rate limit, metrics, security middleware in independent classes. |
| `web/router.py` | ✅ | ✅ | n/a | n/a | ✅ | Delegates everything to `ForecastApiService`. |

Areas that need future attention (without being blockers):
- `ml/exogenous.py` is 371 lines with four providers that share most of their structure. Extracting a `BaseCausalCrossMarketProvider` would shrink each provider to ~40 lines and make adding a fifth provider a 5-line task. See **ADR-0005**.
- `selective.py::CrossFittedSelectiveRegressor.fit()` is a 73-line method. Splitting into `_run_oof_predictions`, `_calibrate_quantile_and_shrinkage`, `_refit_and_set_threshold` would document the algorithm and improve unit-testability. See **ADR-0006**.

---

## 5. Architecture Decision Records (ADRs)

### ADR-0001 — Train Once, Persist, Load & Predict

**Status:** Accepted.

**Context.** Backtests do not validate future prices. Online retraining under live request load creates dependency on training data, on expensive training, and on live market connectivity for read-only requests.

**Decision.** A training run produces a fully self-contained artifact (estimator + feature columns + metrics + training window metadata + dual gate verdict) and writes it to disk. The runtime loads the artifact and never retrains.

**Consequences.**
- ✅ A request handler cannot accidentally couple to training data freshness.
- ✅ Rollback of a model is `git revert` of the artifact, not a re-run.
- ⚠️ If the market regime shifts, a new artifact must be produced by a new commit. This is desired, not a limitation.

### ADR-0002 — Champion/Challenger selection with dual promotion gate

**Status:** Accepted.

**Context.** A challenger may look strong on the development walk-forward set yet fail on the untouched holdout. Promoting on development only overfits the available data.

**Decision.** A challenger must clear `TrainingConfig.min_promotion_improvement_pct` (default **0.5%**) on the development walk-forward MAE **and** on the untouched 20% holdout MAE. Otherwise `active_strategy` stays at `persistence`.

**Consequences.**
- ✅ The system reports `persistence` honestly when no challenger crosses the bar.
- ✅ Operators can lower the threshold for explicit experiments without affecting routine training.
- ✅ Every artifact carries a `selection.to_dict()` audit trail that explains why a challenger did or did not win.

### ADR-0003 — Three on-disk artifact encodings

**Status:** Accepted.

**Context.** `joblib` files are sensitive to the scikit-learn version of the trainer. Tracking both a `joblib` file and its originating dependency versions is necessary.

**Decision.** The repository accepts:
1. **Direct `<metal>_<horizon>h.joblib` + `.sha256`** — fast path inside a container.
2. **`trained_models_*.zip`** with each artifact as a member — single checksum covers everything.
3. **`trained_models_*.zip.b64.part*`** with adjacent `.sha256` — base64 chunked for repositories that block binary attachments.

The repository tries (1), then (2), then (3) in that order and refuses to load any artifact whose checksum does not match.

**Consequences.**
- ✅ The artifact can ride Git without binary corruption.
- ✅ The runtime refuses to serve a tampered or version-drifted artifact.
- ⚠️ The base64 chunk representation is ~33% larger. Acceptable for serving, not acceptable for bulk distribution.

### ADR-0004 — `RuntimeSettings` is a frozen dataclass, not a class hierarchy

**Status:** Accepted.

**Context.** Configuration loaded from environment variables is the kind of state that benefits from immutability. A class hierarchy invites implicit overrides and init order bugs.

**Decision.** `RuntimeSettings` is `@dataclass(frozen=True)` with sensible defaults, populated by `RuntimeSettings.from_environment()`. The composition root reads it once and passes the instance into `ForecastApiService`.

**Consequences.**
- ✅ Services cannot accidentally mutate runtime configuration.
- ✅ Test code can construct an instance directly without environment variable plumbing.
- ⚠️ Hot-reload is not supported. Restarts are required for configuration changes — acceptable for an inference service.

### ADR-0005 — `ExogenousFeatureProvider` Protocol (planned: base class extraction)

**Status:** Partially accepted — protocol in place, base class extraction is in progress.

**Context.** Silver currently consumes gold (`GoldSilverFeatureProvider`), the U.S. dollar index (`UsdIndexFeatureProvider`), the S&P 500 (`SpxFeatureProvider`), and WTI (`WtiFeatureProvider`). Four classes, each with `from_hourly_csv`, `_normalize_hourly`, `_align_backward`, and an `augment` body that shares most of its structure.

**Decision (current).** All four implement the `ExogenousFeatureProvider` Protocol. Each is independently testable and independently replaceable.

**Decision (planned).** Factor the common boilerplate into a `_BaseCrossMarketProvider` so each new provider overrides only `_build_features()` and `metadata()`. The `_BaseCrossMarketProvider` is not exposed as part of the public Protocol; providers continue to advertise themselves via the protocol name.

**Consequences.**
- ✅ Adding a new cross-market context becomes ~40 lines of new code instead of ~120.
- ✅ Drift between the four providers' date-alignment logic is impossible by construction.

### ADR-0006 — `CrossFittedSelectiveRegressor.fit()` decomposition (planned)

**Status:** Proposed.

**Context.** The inner loop performs four distinct steps: OOF prediction, calibration grid search, final refit, threshold setting. Splitting them improves readability and unit testability.

**Decision.** Three private methods:
- `_run_oof_predictions(X, y)` → returns the OOF prediction array.
- `_calibrate_quantile_and_shrinkage(X_calib, y_calib, raw_calib)` → returns `(best_quantile, best_shrinkage)`.
- `_refit_and_set_threshold(X, y, best_quantile)` → sets `estimator_` and `activation_threshold_`.

`fit()` becomes a short orchestration method that calls each in order.

**Consequences.**
- ✅ Each step can be unit-tested in isolation.
- ✅ The control flow reads top-down.

### ADR-0007 — `LivePriceProvider` resilience wrapper

**Status:** Accepted.

**Context.** Public market data endpoints are best-effort. A prediction must continue to function during upstream outages without substituting an undocumented fixed price.

**Decision.** The runtime composes providers into a `ResilientLivePriceProvider` that:
1. Holds a per-provider circuit breaker (`circuit_breaker_seconds`).
2. Falls over to the next provider on `LivePriceUnavailableError`.
3. After all providers are exhausted, accepts the last persisted quote if it is no older than `stale_ttl_seconds`.
4. If all of the above fail, raises `LivePriceUnavailableError`. There is no synthetic price fallback.

**Consequences.**
- ✅ The system survives short upstream outages.
- ✅ Operators can detect persistence-cache service from the `market_data_source` field.
- ✅ Hard failures remain hard; the system never quietly invents a price.

---

## 6. Why persistence beats ML on this dataset

After a complete 17-year walk-forward evaluation across 9 horizons per metal, the answer is that it does, on average, with rare and small wins that do not clear the 0.5% promotion gate. The truthful statement is that the persisted price is the most defensible champion. The architecture is built so that this conclusion is **observable and auditable** rather than hidden:

- Every artifact records the selected challenger, the dev MAE, the holdout MAE, and the gate verdict.
- `/health` returns `models_available=true` whenever at least one trained champion is present, but does not promise "ML is winning" — operators can run their own analysis.
- The README disclaims: "A backtest metric is not evidence that a future market price is certain."

Future work that may move the champion off persistence is exogenous context (DXY, yields, Fed funds, CPI) and interval-aware objectives rather than point estimates. Both require extending the dataset, not the architecture.

---

## 7. Anti-patterns the project avoids

| Anti-pattern | Avoided how |
|---|---|
| **Implicit provider globals** | `LivePriceProvider` is constructed and injected; `RuntimeSettings` is the only config object. |
| **Train-on-request** | `train_models.py` is offline; `ForecastApiService.predict()` rejects any path that retrains. |
| **Generic exception swallowing** | Each `except` clause is narrow; middleware logs `exc_info` but propagates a stable 503 to the client. |
| **Mutable runtime config** | `RuntimeSettings` is `@dataclass(frozen=True)`. |
| **Checksum by-pass** | `ModelArtifactRepository._verify_checksum` uses `hmac.compare_digest`, regenerating the file hash from disk. A mismatch is an integrity incident, not a warning. |
| **Estimator version drift** | `requirements.txt` pins the exact sklearn version; an `InconsistentVersionWarning` at load is treated as a release blocker until the artifact is rebuilt. |
| **Synthetic fallback prices** | When live and persistent both fail, the API returns 503. There is no "best estimate" that could mislead the caller. |

---

## 8. Where the system ends and the operator begins

| Concern | Belongs in code | Belongs in operations |
|---|---|---|
| Rejecting an obviously bad artifact | `ModelArtifactRepository._verify_checksum` | n/a |
| Choosing which `required_metals` to serve | n/a | `QMG1_REQUIRED_METALS` env var |
| Choosing `predict_requests_per_minute` | n/a (default 30) | `QMG1_PREDICT_REQUESTS_PER_MINUTE` env var |
| Bulk-downloading or refreshing data | `scripts/download_all_metals_m1_usd_per_kg.py` | n/a |
| Retraining on a new regime | `scripts/train_models.py --force` | Scheduling, approvals, artifact promotion |
| Capacity behind the API | n/a (uvicorn `--workers`, `--limit-concurrency`) | Platform / proxy configuration |
| Secret rotation | n/a (env var only) | Platform secret store |
| Rollback | n/a (`git revert`) | Deployment platform + traffic shift |

The architecture is intentionally narrow: predict from what was trained, persist with checksums, fail loudly when something is off. The rest is operations.
