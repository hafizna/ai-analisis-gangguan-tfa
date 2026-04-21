import { useState } from "react";
import type { ComtradeData } from "../../../context/AnalysisContext";
import { aiFaultAnalysis21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props { comtrade: ComtradeData; }

interface AIResult {
  cause_ranking: { cause: string; label: string; confidence: number }[];
  fault_type: string;
  overall_confidence: number;
  evidence: string[];
}

export default function AIFaultAnalysis21({ comtrade }: Props) {
  const [fia, setFia] = useState(0);
  const [dur, setDur] = useState(100);
  const [prefault, setPrefault] = useState(0);
  const [impedance, setImpedance] = useState(0);
  const [asym, setAsym] = useState(0);
  const [dc, setDc] = useState(0);
  const [arResult, setArResult] = useState<"" | "successful" | "failed">("");
  const [result, setResult] = useState<AIResult | null>(null);
  const [loading, setLoading] = useState(false);

  // Suppress unused comtrade warning — available for future auto-extraction
  void comtrade;

  async function run() {
    setLoading(true);
    try {
      const res = await aiFaultAnalysis21({
        fault_inception_angle_deg: fia,
        fault_duration_ms: dur,
        prefault_load_a: prefault,
        impedance_at_trip_ohm: impedance,
        waveform_asymmetry: asym,
        dc_offset: dc,
        ar_result: arResult || null,
      });
      setResult(res);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>AI Fault Analysis — Distance (21)</h2>
      </div>

      <div className={styles.row} style={{ flexWrap: "wrap", gap: 14 }}>
        <label className={styles.zoneLabel}>
          FIA (°)
          <input className={styles.inputField} type="number" value={fia} onChange={(e) => setFia(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Fault Duration (ms)
          <input className={styles.inputField} type="number" value={dur} onChange={(e) => setDur(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Pre-fault Load (A)
          <input className={styles.inputField} type="number" value={prefault} onChange={(e) => setPrefault(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          |Z| at Trip (Ω)
          <input className={styles.inputField} type="number" value={impedance} onChange={(e) => setImpedance(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Waveform Asymmetry
          <input className={styles.inputField} type="number" step={0.01} value={asym} onChange={(e) => setAsym(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          DC Offset
          <input className={styles.inputField} type="number" step={0.01} value={dc} onChange={(e) => setDc(parseFloat(e.target.value))} />
        </label>
        <label className={styles.zoneLabel}>
          Auto-Reclose Result
          <select className={styles.selectField} value={arResult} onChange={(e) => setArResult(e.target.value as typeof arResult)}>
            <option value="">Unknown / Not present</option>
            <option value="successful">Successful (transient)</option>
            <option value="failed">Failed (permanent)</option>
          </select>
        </label>
      </div>

      <button className={styles.applyBtn} onClick={run} disabled={loading} style={{ marginBottom: 20 }}>
        {loading ? "Analysing…" : "Run AI Analysis"}
      </button>

      {result && (
        <>
          <div className={styles.row}>
            <span className={`${styles.statusBadge} ${result.fault_type === "permanent" ? styles.statusOperated : styles.statusNot}`}>
              {result.fault_type === "permanent" ? "Permanent Fault" : "Transient Fault"}
            </span>
            <span className={styles.badge}>
              Confidence: {(result.overall_confidence * 100).toFixed(0)}%
            </span>
          </div>

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "12px 0 8px" }}>Cause Ranking</h3>
          {result.cause_ranking.map((r) => (
            <div key={r.cause} className={styles.rankingBar}>
              <span className={styles.rankLabel}>{r.label}</span>
              <div style={{ flex: 1, background: "#f1f5f9", height: 10, borderRadius: 5, overflow: "hidden" }}>
                <div className={styles.rankFill} style={{ width: `${r.confidence * 100}%` }} />
              </div>
              <span className={styles.rankPct}>{(r.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "16px 0 8px" }}>Evidence</h3>
          <ul className={styles.evidenceList}>
            {result.evidence.map((e, i) => (
              <li key={i} className={styles.evidenceItem}>{e}</li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
}
