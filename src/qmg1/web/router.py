from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse

WEB_ROOT = Path(__file__).resolve().parent
STATIC_DIR = WEB_ROOT / "static"
DASHBOARD_FILE = WEB_ROOT / "templates" / "dashboard.html"


def build_web_router() -> APIRouter:
    """Build the user-facing dashboard routes without coupling them to inference logic."""

    router = APIRouter(include_in_schema=False)

    @router.get("/")
    def dashboard() -> FileResponse:
        return FileResponse(DASHBOARD_FILE, media_type="text/html")

    return router
