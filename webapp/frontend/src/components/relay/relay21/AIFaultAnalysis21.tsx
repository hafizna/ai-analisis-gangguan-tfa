import { useEffect, useState } from "react";
import { extractFeatures21, aiFaultAnalysis21 } from "../../../api/client";
import styles from "../../panels/Panel.module.css";

interface Props { analysisId: string; }

interface Features {
  fault_inception_angle_deg: number;
  fault_duration_ms: number;
  prefault_load_a: number;
  impedance_at_trip_ohm: number;
  waveform_asymmetry: number;
  dc_offset: number;
  ar_result: "successful" | "failed" | null | "";
}

interface AIResult {
  cause_ranking: { cause: string; label: string; confidence: number }[];
  fault_type: string;
  overall_confidence: number;
  evidence: string[];
}

const EMPTY_FEATURES: Features = {
  fault_inception_angle_deg: 0,
  fault_duration_ms: 0,
  prefault_load_a: 0,
  impedance_at_trip_ohm: 0,
  waveform_asymmetry: 0,
  dc_offset: 0,
  ar_result: null,
};

export default function AIFaultAnalysis21({ analysisId }: Props) {
  const [features, setFeatures] = useState<Features>(EMPTY_FEATURES);
  const [extracting, setExtracting] = useState(true);
  const [result, setResult] = useState<AIResult | null>(null);
  const [running, setRunning] = useState(false);

  useEffect(() => {
    setExtracting(true);
    extractFeatures21(analysisId)
      .then((f) => setFeatures({ ...f, ar_result: f.ar_result ?? "" }))
      .catch(() => {/* leave defaults */})
      .finally(() => setExtracting(false));
  }, [analysisId]);

  function updateField(field: keyof Features, val: string | number) {
    setFeatures((prev) => ({ ...prev, [field]: val }));
  }

  async function run() {
    setRunning(true);
    try {
      const res = await aiFaultAnalysis21(analysisId, {
        ...features,
        ar_result: features.ar_result || null,
      });
      setResult(res);
    } finally {
      setRunning(false);
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>AI Fault Analysis — Distance (21)</h2>
        {extracting && <span className={styles.badge}>Extracting features…</span>}
        {!extracting && <span className={styles.badge} style={{ background: "#f0fdf4", color: "#16a34a" }}>Auto-extracted from COMTRADE</span>}
      </div>

      <p style={{ fontSize: "0.82rem", color: "#64748b", marginBottom: 14 }}>
        Features below were extracted automatically. Review and adjust if needed before running analysis.
      </p>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))", gap: 12, marginBottom: 16 }}>
        <FeatureField label="Fault Inception Angle (°)" value={features.fault_inception_angle_deg} step={1}
          onChange={(v) => updateField("fault_inception_angle_deg", v)} loading={extracting} />
        <FeatureField label="Fault Duration (ms)" value={features.fault_duration_ms} step={1}
          onChange={(v) => updateField("fault_duration_ms", v)} loading={extracting} />
        <FeatureField label="Pre-fault Load (A)" value={features.prefault_load_a} step={0.1}
          onChange={(v) => updateField("prefault_load_a", v)} loading={extracting} />
        <FeatureField label="|Z| at Trip (Ω)" value={features.impedance_at_trip_ohm} step={0.01}
          onChange={(v) => updateField("impedance_at_trip_ohm", v)} loading={extracting} />
        <FeatureField label="Waveform Asymmetry" value={features.waveform_asymmetry} step={0.01}
          onChange={(v) => updateField("waveform_asymmetry", v)} loading={extracting} />
        <FeatureField label="DC Offset" value={features.dc_offset} step={0.01}
          onChange={(v) => updateField("dc_offset", v)} loading={extracting} />

        <label className={styles.zoneLabel}>
          Auto-Reclose Result
          <select
            className={styles.selectField}
            value={features.ar_result ?? ""}
            onChange={(e) => updateField("ar_result", e.target.value)}
            disabled={extracting}
          >
            <option value="">Unknown / Not present</option>
            <option value="successful">Successful (transient)</option>
            <option value="failed">Failed (permanent)</option>
          </select>
        </label>
      </div>

      <button className={styles.applyBtn} onClick={run} disabled={running || extracting} style={{ marginBottom: 20 }}>
        {running ? "Analysing…" : "Run AI Analysis"}
      </button>

      {result && (
        <>
          <div className={styles.row}>
            <span className={`${styles.statusBadge} ${result.fault_type === "permanent" ? styles.statusOperated : styles.statusNot}`}>
              {result.fault_type === "permanent" ? "Permanent Fault" : "Transient Fault"}
            </span>
            <span className={styles.badge}>Confidence: {(result.overall_confidence * 100).toFixed(0)}%</span>
          </div>

          <h3 style={{ fontSize: "0.85rem", color: "#475569", margin: "14px 0 8px" }}>Cause Ranking</h3>
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

function FeatureField({
  label, value, step, onChange, loading,
}: {
  label: string; value: number; step: number; onChange: (v: number) => void; loading: boolean;
}) {
  return (
    <label className={styles.zoneLabel}>
      {label}
      <input
        className={styles.inputField}
        style={{ width: "100%" }}
        type="number"
        step={step}
        value={value}
        disabled={loading}
        onChange={(e) => { const n = parseFloat(e.target.value); if (!isNaN(n)) onChange(n); }}
      />
    </label>
  );
}
