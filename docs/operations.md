# QMG1 Operations Runbook

## Required production configuration

- Set `QMG1_ENVIRONMENT=production`; production is also the default when unset.
- Set `QMG1_API_KEY` and deliver it only through the platform secret store.
- Set `QMG1_REQUIRED_METALS` and `QMG1_REQUIRED_HORIZONS` to the deployed contract.
- Keep `QMG1_PREDICT_REQUESTS_PER_MINUTE` conservative until load testing establishes capacity.
- Mount immutable serving artifacts with their adjacent `.sha256` files.

## Health and observability

- `/livez` proves that the process is running.
- `/readyz` returns 503 until every required artifact is present and checksum-valid.
- `/health` is informational and must not be used for traffic admission.
- `/metrics` exposes request totals by route and status. Alert on readiness failures, 5xx rate,
  429 rate, and prediction latency at the platform proxy.
- Every response includes `X-Request-ID`; use it to correlate platform and application logs.

## Deployment

1. Build from a reviewed commit whose CI matrix and dependency audit pass.
2. Verify artifact checksums before starting the application.
3. Deploy without shifting traffic until `/readyz` returns 200.
4. Run the `Deployment smoke` workflow against the candidate URL.
5. Shift traffic gradually and watch 5xx, 429, latency, and upstream errors.

## Repository protection

Configure the `main` branch in GitHub to require pull requests, one approving review, resolved
conversations, and the complete CI matrix. Block force pushes and branch deletion. This setting
is external to the repository and must be verified in GitHub before a release is approved.

## Rollback

1. Keep the previous application image and artifact directory immutable.
2. On readiness failure or SLO regression, route traffic back to the previous release.
3. Restore application and artifacts as one versioned unit; never mix generations.
4. Confirm `/readyz`, run the deployment smoke, and record the failed release SHA.

## Artifact recovery

- Treat `joblib` files as executable input and accept them only from the trusted training workflow.
- Retain training reports, checksums, source commit, dependency versions, and datasets used.
- A checksum mismatch is an integrity incident; do not bypass it by regenerating the checksum.
