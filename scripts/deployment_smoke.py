from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def request_json(
    base_url: str,
    path: str,
    *,
    api_key: str | None = None,
    payload: dict | None = None,
) -> dict:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    body = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = Request(f"{base_url.rstrip('/')}{path}", headers=headers, data=body)
    try:
        with urlopen(request, timeout=20) as response:  # noqa: S310 - operator URL
            if response.status != 200:
                raise RuntimeError(f"{path} returned HTTP {response.status}")
            return json.load(response)
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"deployment smoke failed for {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    args = parser.parse_args()
    api_key = os.getenv("QMG1_API_KEY")

    live = request_json(args.base_url, "/livez")
    ready = request_json(args.base_url, "/readyz")
    metadata = request_json(args.base_url, "/api/meta")

    if live.get("status") != "ok" or ready.get("ready") is not True:
        raise SystemExit("deployment is not live and ready")
    if not metadata.get("available_models"):
        raise SystemExit("deployment exposes no available models")

    metal = sorted(metadata["available_models"])[0]
    horizon = min(metadata["available_models"][metal])
    prediction = request_json(
        args.base_url,
        "/predict",
        api_key=api_key,
        payload={"metal": metal, "horizon_hours": horizon},
    )
    if prediction.get("metal") != metal or prediction.get("horizon_hours") != horizon:
        raise SystemExit("deployment prediction contract mismatch")

    print(
        json.dumps(
            {
                "status": "ok",
                "available_models": metadata["available_models"],
                "api_key_configured": bool(api_key),
                "prediction_smoke": {"metal": metal, "horizon_hours": horizon},
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
