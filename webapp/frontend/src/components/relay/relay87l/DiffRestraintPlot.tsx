import { useState } from "react";
import Plot from "react-plotly.js";
import { diffRestraint87L, diffRestraint87T } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props {
  analysisId: string;
  relayType: "87L" | "87T";
}

interface DiffParams {
  device_type: "SP5" | "SP4";
  idiff_pickup: number;
  slope1: number;
  intersection1: number;
  slope2: number;
  intersection2: number;
  idiff_fast: number;
}

const SP5_DEFAULTS: DiffParams = {
  device_type: "SP5",
  idiff_pickup: 0.20,
  slope1: 0.30,
  intersection1: 0.30,
  slope2: 0.70,
  intersection2: 2.50,
  idiff_fast: 7.50,
};

const SP4_DEFAULTS: DiffParams = {
  device_type: "SP4",
  idiff_pickup: 0.20,
  slope1: 0.25,
  intersection1: 0.0,
  slope2: 0.50,
  intersection2: 2.50,
  idiff_fast: 7.50,
};

interface Sample { t: number; i_diff: number; i_rest: number; phase: string; }

const PHASE_COLORS: Record<string, string> = {
  L1: "#f59e0b",
  L2: "#22c55e",
  L3: "#3b82f6",
};

function buildCharacteristic(p: DiffParams) {
  const maxRest = 10;
  const points: { x: number; y: number }[] = [];

  // Pickup line (flat from 0 to intersection1)
  points.push({ x: 0, y: p.idiff_pickup });
  points.push({ x: p.intersection1, y: p.idiff_pickup });

  // Slope 1
  const y_at_int2 = p.idiff_pickup + p.slope1 * (p.intersection2 - p.intersection1);
  points.push({ x: p.intersection2, y: y_at_int2 });

  // Slope 2
  const y_end = y_at_int2 + p.slope2 * (maxRest - p.intersection2);
  points.push({ x: maxRest, y: y_end });

  return points;
}

export default function DiffRestraintPlot({ analysisId, relayType }: Props) {
  const [params, setParams] = useState<DiffParams>(SP5_DEFAULTS);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [status, setStatus] = useState<string | null>(null);
  const [operatedPhases, setOperatedPhases] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  function updateParam(field: keyof DiffParams, val: number | string) {
    setParams((prev) => ({ ...prev, [field]: val }));
  }

  function setDeviceType(t: "SP5" | "SP4") {
    setParams(t === "SP5" ? SP5_DEFAULTS : SP4_DEFAULTS);
  }

  async function fetchPlot() {
    setLoading(true);
    try {
      const fn = relayType === "87T" ? diffRestraint87T : diffRestraint87L;
      const res = await fn(analysisId, params);
      setSamples(res.samples ?? []);
      setStatus(res.operated_status);
      setOperatedPhases(res.operated_phases ?? []);
    } finally {
      setLoading(false);
    }
  }

  const charPts = buildCharacteristic(params);

  const charTrace: Partial<Plotly.ScatterData> = {
    x: charPts.map((p) => p.x),
    y: charPts.map((p) => p.y),
    type: "scatter",
    mode: "lines",
    name: "I-DIFF> Characteristic",
    line: { color: "#f59e0b", width: 2 },
  };

  const fastLine: Partial<Plotly.ScatterData> = {
    x: [0, 10],
    y: [params.idiff_fast, params.idiff_fast],
    type: "scatter",
    mode: "lines",
    name: "I-DIFF>>",
    line: { color: "#ef4444", width: 1.5, dash: "dash" },
  };

  const phases = ["L1", "L2", "L3"];
  const phaseTraces: Partial<Plotly.ScatterData>[] = phases.map((ph) => ({
    x: samples.filter((s) => s.phase === ph).map((s) => s.i_rest),
    y: samples.filter((s) => s.phase === ph).map((s) => s.i_diff),
    type: "scatter",
    mode: "markers",
    name: ph,
    marker: { color: PHASE_COLORS[ph], size: 4, opacity: 0.8 },
  }));

  const layout: Partial<Plotly.Layout> = {
    height: 400,
    margin: { t: 20, b: 50, l: 60, r: 20 },
    xaxis: { title: { text: "I Restraint (p.u.)" }, tickfont: { size: 10 }, range: [0, 10] },
    yaxis: { title: { text: "I Differential (p.u.)" }, tickfont: { size: 10 }, range: [0, params.idiff_fast * 1.1] },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    legend: { orientation: "h", y: -0.15 },
    shapes: [
      {
        type: "rect",
        x0: 0, x1: 10,
        y0: charPts[0].y, y1: params.idiff_fast,
        fillcolor: "#fef2f2",
        opacity: 0.3,
        line: { width: 0 },
        layer: "below",
      } as Plotly.Shape,
    ],
  };

  const statusClass = status === "NOT_OPERATED" ? styles.statusNot : status === "IDIFF_FAST_OPERATED" ? styles.statusFast : styles.statusOperated;
  const statusLabel =
    status === "NOT_OPERATED" ? "NOT OPERATED"
    : status === "IDIFF_FAST_OPERATED" ? "I-DIFF FAST OPERATED"
    : "IDIFF OPERATED";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Differential / Restraint Characteristic</h2>
        <div className={styles.controls}>
          <select className={styles.selectField} value={params.device_type} onChange={(e) => setDeviceType(e.target.value as "SP5" | "SP4")}>
            <option value="SP5">SIPROTEC 5</option>
            <option value="SP4">SIPROTEC 4</option>
          </select>
          <button className={styles.applyBtn} onClick={fetchPlot} disabled={loading}>
            {loading ? "Computing…" : "Compute"}
          </button>
        </div>
      </div>

      {status && (
        <div className={`${styles.statusBadge} ${statusClass}`} style={{ marginBottom: 12 }}>
          {statusLabel}
          {operatedPhases.length > 0 && ` — Phase ${operatedPhases.join(", ")}`}
        </div>
      )}

      <Plot
        data={[charTrace, fastLine, ...phaseTraces] as Plotly.Data[]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />

      {/* Parameter editor */}
      <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 10px" }}>Parameters</h3>
      <div className={styles.zoneEditorRow}>
        <label className={styles.zoneLabel}>
          I-DIFF&gt; Pickup (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.idiff_pickup} onChange={(e) => updateParam("idiff_pickup", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Slope 1
          <input className={styles.inputField} type="number" step={0.01} value={params.slope1} onChange={(e) => updateParam("slope1", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Intersection 1 (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.intersection1} onChange={(e) => updateParam("intersection1", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Slope 2
          <input className={styles.inputField} type="number" step={0.01} value={params.slope2} onChange={(e) => updateParam("slope2", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Intersection 2 (p.u.)
          <input className={styles.inputField} type="number" step={0.01} value={params.intersection2} onChange={(e) => updateParam("intersection2", parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          I-DIFF&gt;&gt; Fast (p.u.)
          <input className={styles.inputField} type="number" step={0.1} value={params.idiff_fast} onChange={(e) => updateParam("idiff_fast", parseFloat(e.target.value))} />
        </label>
      </div>
    </div>
  );
}
