import axios from "axios";
import type { ComtradeData } from "../context/AnalysisContext";

const BASE = import.meta.env.VITE_API_URL?.trim() || "";

export const api = axios.create({
  baseURL: BASE,
  timeout: 30000,
});

export interface UploadedAnalysis {
  analysis_id: string;
  station_name: string;
  rec_dev_id: string;
  total_samples: number;
  analog_channel_count: number;
  status_channel_count: number;
}

export async function uploadComtrade(cfg: File, dat: File) {
  const form = new FormData();
  form.append("cfg_file", cfg);
  form.append("dat_file", dat);
  const { data } = await api.post<UploadedAnalysis>("/api/upload", form);
  return data;
}

export async function fetchAnalysis(analysisId: string) {
  const { data } = await api.get<ComtradeData>(`/api/analysis/${analysisId}`);
  return data;
}

export async function recalculateRatio(analysisId: string, ratios: unknown[]) {
  const { data } = await api.post("/api/recalculate-ratio", { analysis_id: analysisId, ratios });
  return data;
}

export async function computeLocus(analysisId: string, zones: unknown[], loop: string) {
  const { data } = await api.post("/api/analyze/21/locus", { analysis_id: analysisId, zones, loop });
  return data;
}

export async function aiFaultAnalysis21(features: unknown) {
  const { data } = await api.post("/api/analyze/21/ai-analysis", features);
  return data;
}

export async function diffRestraint87L(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/diff-restraint", { analysis_id: analysisId, params, relay_type: "87L" });
  return data;
}

export async function aiFaultAnalysis87L(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87l/ai-analysis", { analysis_id: analysisId, params, relay_type: "87L" });
  return data;
}

export async function diffRestraint87T(analysisId: string, params: unknown) {
  const { data } = await api.post("/api/analyze/87t/diff-restraint", { analysis_id: analysisId, params, relay_type: "87T" });
  return data;
}

export async function overCurrentCharacteristic(analysisId: string, curve_type: string, is_pickup_a: number, tms: number) {
  const { data } = await api.post("/api/analyze/ocr/characteristic", { analysis_id: analysisId, curve_type, is_pickup_a, tms });
  return data;
}
