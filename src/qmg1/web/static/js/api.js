export class ApiError extends Error {
  constructor(message, status = 0, payload = null) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.payload = payload;
  }
}

export class QmgApiClient {
  constructor(baseUrl = "") {
    this.baseUrl = baseUrl.replace(/\/$/, "");
  }

  async health() {
    return this.#request("/health");
  }

  async metadata() {
    return this.#request("/api/meta");
  }

  async predict(metal, horizonHours) {
    return this.#request("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ metal, horizon_hours: Number(horizonHours) }),
    });
  }

  async #request(path, options = {}) {
    const response = await fetch(`${this.baseUrl}${path}`, {
      cache: "no-store",
      ...options,
    });

    const contentType = response.headers.get("content-type") || "";
    const payload = contentType.includes("application/json")
      ? await response.json()
      : await response.text();

    if (!response.ok) {
      const detail = payload && typeof payload === "object" ? payload.detail : payload;
      throw new ApiError(
        typeof detail === "string" && detail ? detail : `Request failed (${response.status})`,
        response.status,
        payload,
      );
    }

    return payload;
  }
}
