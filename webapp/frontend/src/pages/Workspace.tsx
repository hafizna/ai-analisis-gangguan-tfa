import { useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import type { ComtradeData } from "../context/AnalysisContext";

import AnalogSignalRecap from "../components/panels/AnalogSignalRecap";
import COMTRADEExplorer from "../components/panels/COMTRADEExplorer";
import CTVTRatioCorrection from "../components/panels/CTVTRatioCorrection";
import SOETimeline from "../components/panels/SOETimeline";
import ImpedanceLocus from "../components/relay/relay21/ImpedanceLocus";
import AIFaultAnalysis21 from "../components/relay/relay21/AIFaultAnalysis21";
import DiffRestraintPlot from "../components/relay/relay87l/DiffRestraintPlot";
import AIFaultAnalysis87L from "../components/relay/relay87l/AIFaultAnalysis87L";
import OvercurrentOverlay from "../components/relay/relay_ocr/OvercurrentOverlay";

import styles from "./Workspace.module.css";

const RELAY_LABELS: Record<string, string> = {
  "21": "21 — Distance",
  "87L": "87L — Differential Line",
  "87T": "87T — Differential Transformer",
  "OCR": "50/51 — Overcurrent",
  "REF": "REF / GFR / SBEF",
};

type Tab = "waveforms" | "explorer" | "soe" | "ratio" | "locus" | "diff" | "ocr" | "ai";

function tabsForRelay(relayType: string): { id: Tab; label: string }[] {
  const base: { id: Tab; label: string }[] = [
    { id: "waveforms", label: "Analog Signals" },
    { id: "explorer", label: "COMTRADE Explorer" },
    { id: "soe", label: "SOE" },
    { id: "ratio", label: "CT/VT Ratio" },
  ];
  if (relayType === "21") {
    base.push({ id: "locus", label: "Impedance Locus" });
    base.push({ id: "ai", label: "AI Fault Analysis" });
  }
  if (relayType === "87L") {
    base.push({ id: "diff", label: "Diff / Restraint" });
    base.push({ id: "ai", label: "AI Fault Analysis" });
  }
  if (relayType === "87T") {
    base.push({ id: "diff", label: "Diff / Restraint" });
  }
  if (relayType === "OCR" || relayType === "REF") {
    if (relayType === "OCR") base.push({ id: "ocr", label: "Overcurrent Curve" });
  }
  return base;
}

export default function Workspace() {
  const { relayType: urlType } = useParams<{ relayType: string }>();
  const { comtrade: ctxComtrade, relayType: ctxRelayType, reset } = useAnalysis();
  const navigate = useNavigate();

  const relayType = urlType ?? ctxRelayType ?? "21";
  const [comtrade, setComtrade] = useState<ComtradeData | null>(ctxComtrade);

  const tabs = tabsForRelay(relayType);
  const [activeTab, setActiveTab] = useState<Tab>(tabs[0].id);

  if (!comtrade) {
    navigate("/");
    return null;
  }

  function handleReset() {
    reset();
    navigate("/");
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.homeBtn} onClick={handleReset}>← New Analysis</button>
          <div>
            <span className={styles.relayBadge}>{RELAY_LABELS[relayType] ?? relayType}</span>
            <span className={styles.stationName}>{comtrade.station_name}</span>
            <span className={styles.deviceId}>{comtrade.rec_dev_id}</span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.meta}>{comtrade.total_samples} samples</span>
          <span className={styles.meta}>{comtrade.sampling_rates[0]?.[0]} Hz</span>
          <span className={styles.meta}>{comtrade.frequency} Hz nominal</span>
        </div>
      </header>

      <nav className={styles.tabs}>
        {tabs.map((t) => (
          <button
            key={t.id}
            className={`${styles.tab} ${activeTab === t.id ? styles.tabActive : ""}`}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <main className={styles.content}>
        {activeTab === "waveforms" && (
          <AnalogSignalRecap comtrade={comtrade} relayType={relayType} />
        )}
        {activeTab === "explorer" && (
          <COMTRADEExplorer comtrade={comtrade} />
        )}
        {activeTab === "soe" && (
          <SOETimeline comtrade={comtrade} />
        )}
        {activeTab === "ratio" && (
          <CTVTRatioCorrection comtrade={comtrade} onUpdate={setComtrade} />
        )}
        {activeTab === "locus" && relayType === "21" && (
          <ImpedanceLocus comtrade={comtrade} />
        )}
        {activeTab === "ai" && relayType === "21" && (
          <AIFaultAnalysis21 comtrade={comtrade} />
        )}
        {activeTab === "diff" && (relayType === "87L" || relayType === "87T") && (
          <DiffRestraintPlot comtrade={comtrade} relayType={relayType as "87L" | "87T"} />
        )}
        {activeTab === "ai" && relayType === "87L" && (
          <AIFaultAnalysis87L comtrade={comtrade} />
        )}
        {activeTab === "ocr" && relayType === "OCR" && (
          <OvercurrentOverlay comtrade={comtrade} />
        )}
        {relayType === "87T" && activeTab === "diff" && (
          <div style={{ fontSize: "0.85rem", color: "#64748b", marginTop: 12, padding: "10px 16px", background: "#f8fafc", borderRadius: 8 }}>
            AI fault cause analysis for transformer differential is pending — relay coordination evidence required.
          </div>
        )}
      </main>
    </div>
  );
}
