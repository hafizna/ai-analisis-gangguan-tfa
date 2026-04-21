import { useState } from "react";
import type { ComtradeData } from "../../context/AnalysisContext";
import { recalculateRatio } from "../../api/client";
import styles from "./Panel.module.css";

interface Props {
  analysisId: string;
  comtrade: ComtradeData;
  onUpdate: (updated: ComtradeData) => void;
}

interface RatioRow {
  channel_id: string;
  label: string;
  type: "CT" | "VT";
  cfg_primary: number;
  cfg_secondary: number;
  new_primary: number;
  new_secondary: number;
}

function cfgRatioLabel(pri: number, sec: number) {
  if (sec === 0 || pri === 0) return "—";
  return `${pri} / ${sec} (×${(pri / sec).toFixed(1)})`;
}

function correctionFactor(row: RatioRow) {
  const oldR = row.cfg_secondary !== 0 ? row.cfg_primary / row.cfg_secondary : 1;
  const newR = row.new_secondary !== 0 ? row.new_primary / row.new_secondary : 1;
  return oldR !== 0 ? newR / oldR : 1;
}

export default function CTVTRatioCorrection({ analysisId, comtrade, onUpdate }: Props) {
  const [rows, setRows] = useState<RatioRow[]>(() =>
    comtrade.analog_channels.map((ch) => ({
      channel_id: ch.id,
      label: ch.canonical_name || ch.name,
      type: ch.measurement === "voltage" ? "VT" : "CT",
      cfg_primary: ch.ct_primary,
      cfg_secondary: ch.ct_secondary,
      new_primary: ch.ct_primary,
      new_secondary: ch.ct_secondary,
    }))
  );
  const [loading, setLoading] = useState(false);
  const [applied, setApplied] = useState(false);

  function update(idx: number, field: "new_primary" | "new_secondary", raw: string) {
    const n = parseFloat(raw);
    if (!isNaN(n) && n > 0) {
      setRows((prev) => prev.map((r, i) => (i === idx ? { ...r, [field]: n } : r)));
      setApplied(false);
    }
  }

  async function apply() {
    setLoading(true);
    try {
      const ratios = rows.map((r) => ({
        channel_id: r.channel_id,
        primary: r.new_primary,
        secondary: r.new_secondary,
      }));
      const updated = await recalculateRatio(analysisId, ratios);
      onUpdate(updated);
      setApplied(true);
    } finally {
      setLoading(false);
    }
  }

  const hasChanges = rows.some((r) => Math.abs(correctionFactor(r) - 1.0) > 0.001);

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>CT / VT Ratio Correction</h2>
        {applied && <span className={styles.badge} style={{ background: "#f0fdf4", color: "#16a34a" }}>Applied</span>}
      </div>

      <p style={{ fontSize: "0.82rem", color: "#64748b", marginBottom: 14 }}>
        The values below come from your .cfg file. If the relay CT/VT ratios are different from what the recorder used,
        enter the correct ratio — the waveforms will be rescaled accordingly.
      </p>

      <div style={{ overflowX: "auto" }}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>Channel</th>
              <th>Type</th>
              <th>From .cfg</th>
              <th>Correct Primary</th>
              <th>Correct Secondary</th>
              <th>Correction</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => {
              const factor = correctionFactor(r);
              const changed = Math.abs(factor - 1.0) > 0.001;
              return (
                <tr key={r.channel_id}>
                  <td style={{ fontWeight: 600 }}>{r.label}</td>
                  <td>
                    <span className={styles.badge} style={r.type === "CT" ? { background: "#eff6ff", color: "#3b82f6" } : { background: "#faf5ff", color: "#7c3aed" }}>
                      {r.type}
                    </span>
                  </td>
                  <td style={{ color: "#94a3b8", fontSize: "0.8rem" }}>{cfgRatioLabel(r.cfg_primary, r.cfg_secondary)}</td>
                  <td>
                    <input
                      className={styles.ratioInput}
                      type="number"
                      min={0.001}
                      step="any"
                      value={r.new_primary}
                      onChange={(e) => update(i, "new_primary", e.target.value)}
                    />
                  </td>
                  <td>
                    <input
                      className={styles.ratioInput}
                      type="number"
                      min={0.001}
                      step="any"
                      value={r.new_secondary}
                      onChange={(e) => update(i, "new_secondary", e.target.value)}
                    />
                  </td>
                  <td style={{ fontWeight: 600, color: changed ? "#dc2626" : "#16a34a" }}>
                    {changed ? `×${factor.toFixed(3)}` : "No change"}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div style={{ marginTop: 14, display: "flex", alignItems: "center", gap: 12 }}>
        <button className={styles.applyBtn} onClick={apply} disabled={loading || !hasChanges}>
          {loading ? "Recalculating…" : "Apply Ratio Corrections"}
        </button>
        {!hasChanges && <span style={{ fontSize: "0.8rem", color: "#94a3b8" }}>No changes — ratios match .cfg values</span>}
      </div>
    </div>
  );
}
