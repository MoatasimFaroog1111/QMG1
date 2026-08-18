const METALS = {
  silver: { label: "الفضة", symbol: "XAG" },
  gold: { label: "الذهب", symbol: "XAU" },
  platinum: { label: "البلاتينيوم", symbol: "XPT" },
  palladium: { label: "البلاديوم", symbol: "XPD" },
};

const HORIZONS = {
  2: "2h",
  4: "4h",
  8: "8h",
  12: "12h",
  24: "24h",
  72: "72h",
  168: "7d",
  360: "15d",
  720: "30d",
};

export function metalInfo(name) {
  return METALS[name] || { label: name, symbol: String(name || "").toUpperCase() };
}

export function horizonLabel(hours) {
  return HORIZONS[Number(hours)] || `${hours}h`;
}

export function formatPrice(value, maximumFractionDigits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  return new Intl.NumberFormat("en-US", {
    minimumFractionDigits: 2,
    maximumFractionDigits,
  }).format(number);
}

export function formatPercent(value, digits = 2) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "—";
  const sign = number > 0 ? "+" : "";
  return `${sign}${number.toFixed(digits)}%`;
}

export function formatDateTime(value) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat("ar-SA", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  }).format(date);
}

export function directionFor(changePct) {
  const value = Number(changePct);
  if (!Number.isFinite(value) || Math.abs(value) < 0.005) {
    return { key: "neutral", label: "محايد" };
  }
  return value > 0
    ? { key: "up", label: "صعود" }
    : { key: "down", label: "هبوط" };
}
