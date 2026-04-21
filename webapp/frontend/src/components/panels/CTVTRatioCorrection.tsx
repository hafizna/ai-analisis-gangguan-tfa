import { useState } from "react";
import type { ComtradeData } from "../../context/AnalysisContext";
import { recalculateRatio } from "../../api/client";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
  onUpdate: (updated: ComtradeData) => void;
}

interface RatioEntry {
  channel_id: string;
  name: string;
  primary: number;
  secondary: number;
}

export default function CTVTRatioCorrection({ comtrade, onUpdate }: Props) {
  const [ratios, setRatios] = useState<RatioEntry[]>(
    comtrade.analog_channels.map((ch) => ({
      channel_id: ch.id,
      name: ch.canonical_name || ch.name,
      primary: ch.ct_primary,
      secondary: ch.ct_secondary,
    }))
  );
  const [loading, setLoading] = useState(false);

  function updateRatio(idx: number, field: "primary" | "secondary", val: string) {
    const n = parseFloat(val);
    if (isNaN(n)) return;
    setRatios((prev) => prev.map((r, i) => (i === idx ? { ...r, [field]: n } : r)));
  }

  async function apply() {
    setLoading(true);
    try {
      const updated = await recalculateRatio(comtrade, ratios);
      onUpdate(updated);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>CT / VT Ratio Correction</h2>
        <span className={styles.badge}>Per-channel override</span>
      </div>

      <div className={styles.ratioGrid}>
        {ratios.map((r, i) => (
          <div key={r.channel_id} className={styles.ratioRow}>
            <span className={styles.ratioName}>{r.name}</span>
            <input
              className={styles.ratioInput}
              type="number"
              value={r.primary}
              min={0}
              onChange={(e) => updateRatio(i, "primary", e.target.value)}
              title="Primary"
            />
            <span style={{ color: "#94a3b8", fontSize: "0.8rem" }}>:</span>
            <input
              className={styles.ratioInput}
              type="number"
              value={r.secondary}
              min={0.001}
              onChange={(e) => updateRatio(i, "secondary", e.target.value)}
              title="Secondary"
            />
          </div>
        ))}
      </div>

      <button className={styles.applyBtn} onClick={apply} disabled={loading}>
        {loading ? "Recalculating…" : "Apply Ratios"}
      </button>
    </div>
  );
}
