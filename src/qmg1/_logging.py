"""Process-wide logging helpers.

Defines a ``logging.Filter`` that pulls the current ``request_id`` from a
``contextvars.ContextVar`` so every log record emitted during a request carries
the same identifier. Production access logs and prediction-decision logs end up
correlated on one line without requiring each call site to thread the id
through manually.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar

# Default value (None) is used outside of any request scope (e.g. CLI scripts,
# background workers). Filters and formatters should treat ``None`` as "no
# request context".
_request_id_var: ContextVar[str | None] = ContextVar("qmg1_request_id", default=None)


def current_request_id() -> str | None:
    """Return the request_id for the current execution context, or ``None``.

    The value is set by the HTTP middleware when a request enters the
    application and cleared when the request finishes, so that log lines from
    concurrent requests never share an id.
    """

    return _request_id_var.get()


def bind_request_id(request_id: str):
    """Bind ``request_id`` for the current asyncio task / thread.

    Returns a ``Token`` that the caller must pass to :func:`reset_request_id`
    when the request is fully done. The token-based API is the documented
    ``contextvars`` pattern and avoids leaks across requests on the same
    worker thread.
    """

    return _request_id_var.set(request_id)


def reset_request_id(token) -> None:
    """Restore the previous request_id binding using the token from :func:`bind_request_id`."""

    _request_id_var.reset(token)


class RequestIdFilter(logging.Filter):
    """Annotate every ``LogRecord`` with the current ``request_id``, if any.

    The filter is idempotent: it never overwrites an explicit ``request_id``
    that a caller already attached via ``extra={...}``. This lets manual logs
    continue to advertise their own identifier for non-HTTP scopes.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        existing = getattr(record, "request_id", None)
        if existing:
            return True
        active = current_request_id()
        if active is not None:
            record.request_id = active
        return True


def configure_logging(level: int = logging.INFO) -> None:
    """Attach :class:`RequestIdFilter` to the root logger and every existing handler.

    Idempotent: repeated calls do not stack filters or handlers. The function
    *replaces* the root handlers list with one ``StreamHandler`` carrying the
    QMG1 request-id format, because ``logging.basicConfig`` (used by uvicorn
    and many libraries) may have installed a handler with a different format
    that would silently drop the ``request_id`` field. Filters are also
    attached to each handler so the request-id propagates through.
    """

    root = logging.getLogger()
    root.setLevel(level)

    # Replace existing handlers so a previous ``basicConfig`` does not leave a
    # handler with a format string that ignores ``request_id``.
    configured_handler = logging.StreamHandler()
    configured_handler.setLevel(level)
    configured_handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] [request_id=%(request_id)s] %(message)s",
            defaults={"request_id": "-"},
        )
    )
    configured_handler.addFilter(RequestIdFilter())
    root.handlers = [configured_handler]

    # Logger-level filter (covers propagation through the hierarchy).
    if not any(isinstance(item, RequestIdFilter) for item in root.filters):
        root.addFilter(RequestIdFilter())
