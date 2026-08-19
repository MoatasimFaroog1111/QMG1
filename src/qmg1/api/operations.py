from __future__ import annotations

import hmac
import logging
import threading
import time
import uuid
from collections import Counter, defaultdict, deque
from collections.abc import Callable

from fastapi import Request
from fastapi.responses import JSONResponse, Response


LOGGER = logging.getLogger("qmg1.http")

EXTERNAL_PREDICT_PATH = "/predict"
DASHBOARD_PREDICT_PATH = "/web/predict"
RATE_LIMITED_PREDICTION_PATHS = frozenset(
    {EXTERNAL_PREDICT_PATH, DASHBOARD_PREDICT_PATH}
)


class SlidingWindowRateLimiter:
    def __init__(self, requests: int, window_seconds: float = 60.0) -> None:
        if requests < 1 or window_seconds <= 0:
            raise ValueError("rate-limit values must be positive")
        self.requests = requests
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str, now: float | None = None) -> bool:
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.requests:
                return False
            events.append(current)
            return True


class RuntimeMetrics:
    def __init__(self) -> None:
        self._counts: Counter[tuple[str, str, int]] = Counter()
        self._lock = threading.Lock()

    def observe(self, method: str, path: str, status_code: int) -> None:
        with self._lock:
            self._counts[(method, path, status_code)] += 1

    def render_prometheus(self) -> str:
        lines = [
            "# HELP qmg1_http_requests_total Total HTTP requests.",
            "# TYPE qmg1_http_requests_total counter",
        ]
        with self._lock:
            items = sorted(self._counts.items())
        for (method, path, status_code), count in items:
            lines.append(
                "qmg1_http_requests_total"
                f'{{method="{method}",path="{path}",status="{status_code}"}} {count}'
            )
        return "\n".join(lines) + "\n"


def install_operational_middleware(
    application,
    *,
    api_key: str | None,
    predict_requests_per_minute: int,
) -> RuntimeMetrics:
    metrics = RuntimeMetrics()
    limiter = SlidingWindowRateLimiter(predict_requests_per_minute)

    @application.middleware("http")
    async def operational_middleware(request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
        client_host = request.client.host if request.client else "unknown"
        path = request.url.path

        if path in RATE_LIMITED_PREDICTION_PATHS:
            supplied_key = request.headers.get("x-api-key", "")
            if (
                path == EXTERNAL_PREDICT_PATH
                and api_key
                and not hmac.compare_digest(supplied_key, api_key)
            ):
                response = JSONResponse(
                    status_code=401,
                    content={"detail": "Authentication required.", "code": "unauthorized"},
                )
            elif not limiter.allow(client_host):
                response = JSONResponse(
                    status_code=429,
                    content={"detail": "Prediction rate limit exceeded.", "code": "rate_limited"},
                    headers={"Retry-After": "60"},
                )
            else:
                response = await call_next(request)
        else:
            response = await call_next(request)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        metrics.observe(request.method, path, response.status_code)
        LOGGER.info(
            "request_complete method=%s path=%s status=%s request_id=%s client=%s",
            request.method,
            path,
            response.status_code,
            request_id,
            client_host,
        )
        return response

    return metrics
