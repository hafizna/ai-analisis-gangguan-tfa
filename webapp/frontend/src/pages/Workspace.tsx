import { useEffect, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import type { ComtradeData } from "../context/AnalysisContext";
import { fetchAnalysis } from "../api/client";

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
  "21": "21 - Distance",
  "87L": "87L - Differential Line",
  "87T": "87T / REF",
  OCR: "50/51 / GFR",
  REF: "REF",
  SBEF: "SBEF",
};

type Tab = "waveforms" | "explorer" | "soe" | "ratio" | "locus" | "diff" | "ocr" | "ai";

function initialTabForRelay(relayType: string): Tab {
  if (relayType === "21" || relayType === "87L" || relayType === "87T" || relayType === "REF") {
    return "explorer";
  }
  return "explorer";
}

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
  if (relayType === "87T" || relayType === "REF") {
    base.push({ id: "diff", label: "Diff / Restraint" });
  }
  if ((relayType === "OCR" || relayType === "SBEF") && !base.find((t) => t.id === "ocr")) {
    base.push({ id: "ocr", label: "Overcurrent Curve" });
  }

  return base;
}

export default function Workspace() {
  const { relayType: urlType, analysisId } = useParams<{ relayType: string; analysisId: string }>();
  const { relayType: ctxRelayType, reset } = useAnalysis();
  const navigate = useNavigate();

  const relayType = urlType ?? ctxRelayType ?? "21";
  const [comtrade, setComtrade] = useState<ComtradeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function loadAnalysis() {
      if (!analysisId) {
        if (!cancelled) {
          setLoading(false);
          setError("Missing analysis session id.");
        }
        return;
      }

      setLoading(true);
      setError(null);

      try {
        const data = await fetchAnalysis(analysisId);
        if (!cancelled) {
          setComtrade(data);
        }
      } catch (err: unknown) {
        const response = (err as { response?: { data?: { detail?: string } } }).response;
        if (!cancelled) {
          setError(response?.data?.detail ?? "Failed to load the analysis session.");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadAnalysis();
    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  const tabs = tabsForRelay(relayType);
  const [activeTab, setActiveTab] = useState<Tab>(initialTabForRelay(relayType));

  useEffect(() => {
    setActiveTab(initialTabForRelay(relayType));
  }, [relayType]);

  if (!analysisId) {
    return <Navigate to={ctxRelayType ? "/upload" : "/"} replace />;
  }

  if (loading) {
    return (
      <div className={styles.page}>
        <main className={styles.content}>
          <div style={{ padding: "48px 0", color: "#475569", fontSize: "0.95rem" }}>
            Loading analysis session...
          </div>
        </main>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.page}>
        <main className={styles.content}>
          <div style={{ padding: "24px", background: "#fff7ed", border: "1px solid #fdba74", borderRadius: 12, color: "#9a3412" }}>
            {error}
          </div>
        </main>
      </div>
    );
  }

  if (!comtrade) {
    return <Navigate to={ctxRelayType ? "/upload" : "/"} replace />;
  }

  function handleReset() {
    reset();
    navigate("/");
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <button className={styles.homeBtn} onClick={handleReset} type="button">New Analysis</button>
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
            type="button"
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
        {activeTab === "diff" && (relayType === "87L" || relayType === "87T" || relayType === "REF") && (
          <DiffRestraintPlot comtrade={comtrade} relayType={relayType === "87L" ? "87L" : "87T"} />
        )}
        {activeTab === "ai" && relayType === "87L" && (
          <AIFaultAnalysis87L comtrade={comtrade} />
        )}
        {activeTab === "ocr" && (relayType === "OCR" || relayType === "SBEF") && (
          <OvercurrentOverlay comtrade={comtrade} relayType={relayType} />
        )}
        {(relayType === "87T" || relayType === "REF") && activeTab === "diff" && (
          <div style={{ fontSize: "0.85rem", color: "#64748b", marginTop: 12, padding: "10px 16px", background: "#f8fafc", borderRadius: 8 }}>
            AI fault cause analysis for transformer differential is pending - relay coordination evidence required.
          </div>
        )}
      </main>
    </div>
  );
}
