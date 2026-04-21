import axios from "axios";

const BASE = import.meta.env.VITE_API_URL?.trim() || "";

export const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
});

export async function uploadComtrade(cfg: File, dat: File) {
  const form = new FormData();
  form.append("cfg_file", cfg);
  form.append("dat_file", dat);
  const { data } = await api.post("/api/upload", form);
  return data;
}

export async function recalculateRatio(comtrade: unknown, ratios: unknown[]) {
  const { data } = await api.post("/api/recalculate-ratio", { comtrade, ratios });
  return data;
}

export async function computeLocus(comtrade: unknown, zones: unknown[], loop: string) {
  const { data } = await api.post("/api/analyze/21/locus", { comtrade, zones, loop });
  return data;
}

export async function aiFaultAnalysis21(features: unknown) {
  const { data } = await api.post("/api/analyze/21/ai-analysis", features);
  return data;
}

export async function diffRestraint87L(comtrade: unknown, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/diff-restraint", { comtrade, params, relay_type: "87L" });
  return data;
}

export async function aiFaultAnalysis87L(comtrade: unknown, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/ai-analysis", { comtrade, params, relay_type: "87L" });
  return data;
}

export async function diffRestraint87T(comtrade: unknown, params: unknown) {
  const { data } = await api.post("/api/analyze/87t/diff-restraint", { comtrade, params, relay_type: "87T" });
  return data;
}

export async function overCurrentCharacteristic(comtrade: unknown, curve_type: string, is_pickup_a: number, tms: number) {
  const { data } = await api.post("/api/analyze/ocr/characteristic", { comtrade, curve_type, is_pickup_a, tms });
  return data;
}
