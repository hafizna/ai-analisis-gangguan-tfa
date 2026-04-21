import { useState } from "react";
import Plot from "react-plotly.js";
import type { ComtradeData } from "../../../context/AnalysisContext";
import { computeLocus } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Zone {
  label: string;
  shape: "mho" | "quad";
  center_r: number;
  center_x: number;
  radius: number;
  rf_fwd: number;
  rf_rev: number;
  xf: number;
  xr: number;
  line_angle_deg: number;
  color: string;
}

const DEFAULT_ZONES: Zone[] = [
  { label: "Z1", shape: "mho", center_r: 0, center_x: 5, radius: 5, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#22c55e" },
  { label: "Z2", shape: "mho", center_r: 0, center_x: 8, radius: 8, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#f59e0b" },
  { label: "Z3", shape: "mho", center_r: 0, center_x: 12, radius: 12, rf_fwd: 0, rf_rev: 0, xf: 0, xr: 0, line_angle_deg: 75, color: "#ef4444" },
];

const LOOPS = ["ZA", "ZB", "ZC", "ZAB", "ZBC", "ZCA"];

interface LocusPoint { t: number; r: number; x: number; }

function mhoCircleTrace(zone: Zone): Partial<Plotly.ScatterData> {
  const theta = Array.from({ length: 361 }, (_, i) => (i * Math.PI) / 180);
  return {
    x: theta.map((t) => zone.center_r + zone.radius * Math.cos(t)),
    y: theta.map((t) => zone.center_x + zone.radius * Math.sin(t)),
    type: "scatter",
    mode: "lines",
    name: zone.label,
    line: { color: zone.color, width: 1.5, dash: "dot" },
    showlegend: true,
  };
}

function quadTrace(zone: Zone): Partial<Plotly.ScatterData> {
  const ang = (zone.line_angle_deg * Math.PI) / 180;
  const cos = Math.cos(ang), sin = Math.sin(ang);
  // 4-vertex quadrilateral: RF_FWD, XF, RF_REV, XR — projected onto R-X plane
  const pts = [
    [zone.rf_fwd * cos, zone.rf_fwd * sin],
    [-zone.rf_rev * cos, -zone.rf_rev * sin],
    [-zone.rf_rev * cos + zone.xr * sin, zone.xr * cos - zone.rf_rev * sin],
    [zone.rf_fwd * cos + zone.xf * sin, zone.xf * cos + zone.rf_fwd * sin],
    [zone.rf_fwd * cos, zone.rf_fwd * sin],
  ];
  return {
    x: pts.map((p) => p[0]),
    y: pts.map((p) => p[1]),
    type: "scatter",
    mode: "lines",
    name: zone.label,
    line: { color: zone.color, width: 1.5, dash: "dot" },
    showlegend: true,
  };
}

export default function ImpedanceLocus({ comtrade }: { comtrade: ComtradeData }) {
  const [zones, setZones] = useState<Zone[]>(DEFAULT_ZONES);
  const [loop, setLoop] = useState("ZA");
  const [points, setPoints] = useState<LocusPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function fetchLocus() {
    setLoading(true);
    setError(null);
    try {
      const res = await computeLocus(comtrade, zones, loop);
      setPoints(res.points ?? []);
    } catch {
      setError("Failed to compute impedance locus. Check that current/voltage channels are present.");
    } finally {
      setLoading(false);
    }
  }

  function updateZone(idx: number, field: keyof Zone, val: string | number) {
    setZones((prev) => prev.map((z, i) => (i === idx ? { ...z, [field]: val } : z)));
  }

  // Locus trace — color by time (use gradient marker colors)
  const locusTrace: Partial<Plotly.ScatterData> = points.length > 0 ? {
    x: points.map((p) => p.r),
    y: points.map((p) => p.x),
    text: points.map((p) => `t=${(p.t * 1000).toFixed(1)}ms R=${p.r.toFixed(2)}Ω X=${p.x.toFixed(2)}Ω`),
    hovertemplate: "%{text}<extra></extra>",
    type: "scatter",
    mode: "lines+markers",
    name: `Locus ${loop}`,
    marker: {
      size: 4,
      color: points.map((_, i) => i / points.length),
      colorscale: [[0, "#22c55e"], [0.5, "#f59e0b"], [1, "#ef4444"]],
      showscale: true,
      colorbar: { title: { text: "Time →" }, thickness: 10, len: 0.5, tickfont: { size: 9 } },
    },
    line: { width: 1, color: "#94a3b8" },
  } : { x: [], y: [], type: "scatter", mode: "markers", name: "No data" };

  const zoneTraces = zones.map((z) =>
    z.shape === "mho" ? mhoCircleTrace(z) : quadTrace(z)
  );

  // Axes cross-hairs
  const axisShapes: Partial<Plotly.Shape>[] = [
    { type: "line", x0: -50, x1: 50, y0: 0, y1: 0, line: { color: "#cbd5e1", width: 1 } },
    { type: "line", x0: 0, x1: 0, y0: -50, y1: 50, line: { color: "#cbd5e1", width: 1 } },
  ];

  const layout: Partial<Plotly.Layout> = {
    height: 480,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: { title: { text: "R (Ω secondary)" }, zeroline: false, tickfont: { size: 10 }, range: [-20, 20] },
    yaxis: { title: { text: "X (Ω secondary)" }, zeroline: false, tickfont: { size: 10 }, scaleanchor: "x", range: [-10, 30] },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    shapes: axisShapes as Plotly.Shape[],
    legend: { orientation: "h", y: -0.12 },
  };

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Impedance Locus Diagram</h2>
        <div className={styles.controls}>
          <label className={styles.label}>Loop</label>
          <select className={styles.selectField} value={loop} onChange={(e) => setLoop(e.target.value)}>
            {LOOPS.map((l) => <option key={l}>{l}</option>)}
          </select>
          <button className={styles.applyBtn} onClick={fetchLocus} disabled={loading}>
            {loading ? "Computing…" : "Compute Locus"}
          </button>
        </div>
      </div>

      {error && <div className={styles.warning} style={{ marginBottom: 12 }}>⚠ {error}</div>}

      <Plot
        data={[locusTrace, ...zoneTraces] as Plotly.Data[]}
        layout={layout}
        config={{ displayModeBar: true, responsive: true }}
        style={{ width: "100%" }}
      />

      {/* Zone polygon editor */}
      <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 10px" }}>Zone Settings</h3>
      {zones.map((z, i) => (
        <div key={z.label} style={{ marginBottom: 12 }}>
          <div className={styles.row}>
            <span className={styles.label} style={{ fontWeight: 600, width: 28 }}>{z.label}</span>
            <select
              className={styles.selectField}
              value={z.shape}
              onChange={(e) => updateZone(i, "shape", e.target.value)}
            >
              <option value="mho">Mho (Circle)</option>
              <option value="quad">Quadrilateral</option>
            </select>
            <input
              type="color"
              value={z.color}
              onChange={(e) => updateZone(i, "color", e.target.value)}
              style={{ width: 32, height: 28, border: "none", cursor: "pointer" }}
            />
          </div>
          {z.shape === "mho" ? (
            <div className={styles.zoneEditorRow}>
              <label className={styles.zoneLabel}>
                Center R (Ω)
                <input className={styles.inputField} type="number" value={z.center_r} onChange={(e) => updateZone(i, "center_r", parseFloat(e.target.value))} />
              </label>
              <label className={styles.zoneLabel}>
                Center X (Ω)
                <input className={styles.inputField} type="number" value={z.center_x} onChange={(e) => updateZone(i, "center_x", parseFloat(e.target.value))} />
              </label>
              <label className={styles.zoneLabel}>
                Radius (Ω)
                <input className={styles.inputField} type="number" value={z.radius} onChange={(e) => updateZone(i, "radius", parseFloat(e.target.value))} />
              </label>
            </div>
          ) : (
            <div className={styles.zoneEditorRow}>
              <label className={styles.zoneLabel}>Rf Forward (Ω)<input className={styles.inputField} type="number" value={z.rf_fwd} onChange={(e) => updateZone(i, "rf_fwd", parseFloat(e.target.value))} /></label>
              <label className={styles.zoneLabel}>Rf Reverse (Ω)<input className={styles.inputField} type="number" value={z.rf_rev} onChange={(e) => updateZone(i, "rf_rev", parseFloat(e.target.value))} /></label>
              <label className={styles.zoneLabel}>X Forward (Ω)<input className={styles.inputField} type="number" value={z.xf} onChange={(e) => updateZone(i, "xf", parseFloat(e.target.value))} /></label>
              <label className={styles.zoneLabel}>X Reverse (Ω)<input className={styles.inputField} type="number" value={z.xr} onChange={(e) => updateZone(i, "xr", parseFloat(e.target.value))} /></label>
              <label className={styles.zoneLabel}>Line Angle (°)<input className={styles.inputField} type="number" value={z.line_angle_deg} onChange={(e) => updateZone(i, "line_angle_deg", parseFloat(e.target.value))} /></label>
            </div>
          )}
        </div>
      ))}
    </div>
  );
}
