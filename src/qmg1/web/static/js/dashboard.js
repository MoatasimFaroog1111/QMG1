import {
  directionFor,
  formatDateTime,
  formatPercent,
  formatPrice,
  horizonLabel,
  metalInfo,
} from "./formatters.js";

const HISTORY_KEY = "qmg1.predictionHistory.v1";
const MAX_HISTORY = 6;

function get(id) {
  const element = document.getElementById(id);
  if (!element) throw new Error(`Missing dashboard element: #${id}`);
  return element;
}

function setText(id, value) {
  get(id).textContent = value;
}

function setIndicator(id, state) {
  const element = get(id);
  element.classList.remove("pending", "ok", "warn", "error");
  element.classList.add(state);
}

function setHealthDot(id, state) {
  const element = get(id);
  element.classList.remove("ok", "warn", "error");
  element.classList.add(state);
}

export class DashboardController {
  constructor(apiClient) {
    this.api = apiClient;
    this.form = get("prediction-form");
    this.metalSelect = get("metal-select");
    this.horizonSelect = get("horizon-select");
    this.predictButton = get("predict-button");
    this.errorBox = get("prediction-error");
    this.healthRefresh = get("health-refresh");
    this.themeToggle = get("theme-toggle");
    this.clearHistory = get("clear-history");
    this.historyList = get("history-list");
    this.historyEmpty = get("history-empty");
    this.latestHealth = null;
  }

  async init() {
    this.#restoreTheme();
    this.#bindEvents();
    this.#renderHistory();
    await Promise.allSettled([this.refreshHealth(), this.#loadMetadata()]);
  }

  #bindEvents() {
    this.form.addEventListener("submit", (event) => {
      event.preventDefault();
      void this.runPrediction();
    });

    this.horizonSelect.addEventListener("change", () => {
      this.#syncHorizonButtons(this.horizonSelect.value);
    });

    document.querySelectorAll("[data-horizon]").forEach((button) => {
      button.addEventListener("click", () => {
        this.horizonSelect.value = button.dataset.horizon;
        this.#syncHorizonButtons(button.dataset.horizon);
      });
    });

    this.healthRefresh.addEventListener("click", () => void this.refreshHealth());
    this.themeToggle.addEventListener("click", () => this.#toggleTheme());
    this.clearHistory.addEventListener("click", () => {
      localStorage.removeItem(HISTORY_KEY);
      this.#renderHistory();
    });
  }

  async #loadMetadata() {
    try {
      const metadata = await this.api.metadata();
      const allowed = new Set((metadata.forecast_horizons_hours || []).map(Number));
      document.querySelectorAll("[data-horizon]").forEach((button) => {
        button.disabled = !allowed.has(Number(button.dataset.horizon));
      });
    } catch {
      // Health rendering already provides the primary connectivity signal.
    }
  }

  async refreshHealth() {
    this.healthRefresh.classList.add("loading");
    try {
      const health = await this.api.health();
      this.latestHealth = health;
      this.#renderHealth(health);
    } catch (error) {
      this.latestHealth = null;
      this.#renderHealthFailure(error);
    } finally {
      this.healthRefresh.classList.remove("loading");
    }
  }

  #renderHealth(health) {
    const models = Boolean(health.models_available);
    const persistedData = Boolean(health.target_data_available);
    const liveMarket = Boolean(health.live_market_data_enabled);
    const servingData = persistedData || liveMarket;
    const context = Boolean(health.hourly_context_available);

    setText("status-api", "Online");
    setText("status-models", models ? "Ready" : "غير محمّلة");
    setText(
      "status-data",
      liveMarket ? "Live Feed" : persistedData ? "Ready" : "غير متاحة",
    );
    setText("status-context", context ? "Ready" : "غير متاح");

    setIndicator("indicator-api", "ok");
    setIndicator("indicator-models", models ? "ok" : "warn");
    setIndicator("indicator-data", servingData ? "ok" : "warn");
    setIndicator("indicator-context", context ? "ok" : "warn");

    setText("health-api", "Online");
    setText("health-models", models ? "Ready" : "Missing");
    setText(
      "health-data",
      liveMarket ? "Live Feed" : persistedData ? "Ready" : "Missing",
    );
    setText("health-context", context ? "Ready" : "Optional");

    setHealthDot("health-dot-api", "ok");
    setHealthDot("health-dot-models", models ? "ok" : "warn");
    setHealthDot("health-dot-data", servingData ? "ok" : "warn");
    setHealthDot("health-dot-context", context ? "ok" : "warn");

    const livePill = get("api-live-pill");
    livePill.classList.remove("offline");
    livePill.classList.add("online");
    setText("api-live-text", "API Online");

    const heroHealth = get("hero-health-label");
    heroHealth.parentElement.classList.add("online");
    heroHealth.textContent = models && servingData
      ? "Inference runtime ready"
      : "API ready · artifacts pending";
    setText("health-checked-at", new Intl.DateTimeFormat("ar-SA", {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }).format(new Date()));
  }

  #renderHealthFailure(error) {
    setText("status-api", "Offline");
    setText("status-models", "Unknown");
    setText("status-data", "Unknown");
    setText("status-context", "Unknown");

    ["api", "models", "data", "context"].forEach((key) => {
      setIndicator(`indicator-${key}`, "error");
    });

    setText("health-api", "Offline");
    setText("health-models", "Unknown");
    setText("health-data", "Unknown");
    setText("health-context", "Unknown");
    ["api", "models", "data", "context"].forEach((key) => {
      setHealthDot(`health-dot-${key}`, "error");
    });

    const livePill = get("api-live-pill");
    livePill.classList.remove("online");
    livePill.classList.add("offline");
    setText("api-live-text", "API Offline");
    setText("hero-health-label", error?.message || "Unable to reach API");
    setText("health-checked-at", "فشل الاتصال");
  }

  async runPrediction() {
    this.#clearError();
    this.#setLoading(true);

    try {
      const result = await this.api.predict(
        this.metalSelect.value,
        Number(this.horizonSelect.value),
      );
      this.#renderResult(result);
      this.#storeHistory(result);
      this.#renderHistory();
    } catch (error) {
      this.#showError(this.#friendlyError(error));
    } finally {
      this.#setLoading(false);
    }
  }

  #renderResult(result) {
    const metal = metalInfo(result.metal);
    const direction = directionFor(result.predicted_change_pct);
    const directionalApplicable = result.active_strategy !== "persistence";

    get("result-empty").hidden = true;
    get("result-content").hidden = false;

    setText("result-symbol", metal.symbol);
    setText("result-metal", metal.label);
    setText("result-price", formatPrice(result.predicted_usd_per_kg));
    setText("result-current", formatPrice(result.current_usd_per_kg));
    setText("result-change", formatPercent(result.predicted_change_pct));
    setText("result-horizon", horizonLabel(result.horizon_hours));
    setText(
      "result-range-text",
      `${formatPrice(result.prediction_interval_80_low_usd_per_kg)} — ${formatPrice(result.prediction_interval_80_high_usd_per_kg)}`,
    );
    setText("result-strategy", result.active_strategy || "—");
    setText("result-challenger", result.selected_challenger || "—");
    setText(
      "result-directional",
      directionalApplicable
        ? formatPercent(result.validation_directional_accuracy_pct)
        : "غير منطبق",
    );
    setText(
      "result-improvement",
      formatPercent(result.validation_improvement_vs_persistence_pct),
    );
    setText("result-mae", `${formatPrice(result.validation_mae_usd_per_kg)} USD/kg`);
    setText("result-target-time", formatDateTime(result.target_timestamp_utc));

    const directionBadge = get("result-direction");
    directionBadge.textContent = direction.label;
    directionBadge.classList.remove("up", "down", "neutral");
    directionBadge.classList.add(direction.key);
  }

  #friendlyError(error) {
    const message = String(error?.message || "تعذر تنفيذ التنبؤ.");
    if (error?.status === 503) {
      if (message.includes("temporarily unavailable")) {
        return `تعذر الوصول إلى مصدر السوق الحي مؤقتًا. ${message}`;
      }
      return `الخدمة تعمل، لكن مورد التنبؤ المطلوب غير جاهز. ${message}`;
    }
    if (error?.status === 422) {
      return "المدخلات غير مقبولة. تحقق من المعدن والأفق الزمني.";
    }
    return message;
  }

  #showError(message) {
    this.errorBox.textContent = message;
    this.errorBox.hidden = false;
  }

  #clearError() {
    this.errorBox.hidden = true;
    this.errorBox.textContent = "";
  }

  #setLoading(loading) {
    this.predictButton.disabled = loading;
    this.predictButton.classList.toggle("loading", loading);
    const label = this.predictButton.querySelector(".button-label");
    if (label) label.textContent = loading ? "جاري التنبؤ…" : "تشغيل التنبؤ";
  }

  #syncHorizonButtons(value) {
    document.querySelectorAll("[data-horizon]").forEach((button) => {
      button.classList.toggle("active", button.dataset.horizon === String(value));
    });
  }

  #storeHistory(result) {
    const history = this.#readHistory();
    history.unshift({
      metal: result.metal,
      horizon_hours: result.horizon_hours,
      predicted_usd_per_kg: result.predicted_usd_per_kg,
      current_usd_per_kg: result.current_usd_per_kg,
      predicted_change_pct: result.predicted_change_pct,
      active_strategy: result.active_strategy,
      target_timestamp_utc: result.target_timestamp_utc,
      saved_at: new Date().toISOString(),
    });
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, MAX_HISTORY)));
  }

  #readHistory() {
    try {
      const value = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(value) ? value : [];
    } catch {
      return [];
    }
  }

  #renderHistory() {
    const history = this.#readHistory();
    this.historyList.querySelectorAll(".history-item").forEach((item) => item.remove());
    this.historyEmpty.hidden = history.length > 0;

    history.forEach((item) => {
      const metal = metalInfo(item.metal);
      const direction = directionFor(item.predicted_change_pct);
      const row = document.createElement("article");
      row.className = "history-item";

      const cells = [
        ["المعدن", `${metal.label} · ${metal.symbol}`],
        ["السعر المتوقع", `${formatPrice(item.predicted_usd_per_kg)} USD/kg`],
        ["التغير", formatPercent(item.predicted_change_pct)],
        ["الأفق / الهدف", `${horizonLabel(item.horizon_hours)} · ${formatDateTime(item.target_timestamp_utc)}`],
      ];

      cells.forEach(([label, value]) => {
        const cell = document.createElement("div");
        cell.className = "history-cell";
        const small = document.createElement("small");
        small.textContent = label;
        const strong = document.createElement("strong");
        strong.textContent = value;
        cell.append(small, strong);
        row.append(cell);
      });

      const directionElement = document.createElement("span");
      directionElement.className = `history-direction ${direction.key}`;
      directionElement.textContent = direction.label;
      row.append(directionElement);
      this.historyList.append(row);
    });
  }

  #restoreTheme() {
    const saved = localStorage.getItem("qmg1.theme");
    if (saved === "light" || saved === "dark") {
      document.documentElement.dataset.theme = saved;
      return;
    }
    const prefersLight = window.matchMedia?.("(prefers-color-scheme: light)").matches;
    document.documentElement.dataset.theme = prefersLight ? "light" : "dark";
  }

  #toggleTheme() {
    const next = document.documentElement.dataset.theme === "light" ? "dark" : "light";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("qmg1.theme", next);
  }
}
