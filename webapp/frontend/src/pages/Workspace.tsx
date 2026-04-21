import { useEffect, useState } from "react";
import { Navigate, useNavigate, useParams } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import type { ComtradeData } from "../context/AnalysisContext";
import { fetchAnalysis, fetchAnalysisSummary } from "../api/client";
import type { AnalysisSummary } from "../api/client";

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
const DETAIL_TABS: Set<Tab> = new Set(["waveforms", "explorer", "soe", "ratio"]);

function initialTabForRelay(relayType: string): Tab {
  if (relayType === "21") return "ai";
  if (relayType === "87L" || relayType === "87T" || relayType === "REF") return "diff";
  if (relayType === "OCR" || relayType === "SBEF") return "ocr";
  return "explorer";
}

function tabsForRelay(relayType: string): { id: Tab; label: string }[] {
  const detailTabs: { id: Tab; label: string }[] = [
    { id: "waveforms", label: "Analog Signals" },
    { id: "explorer", label: "COMTRADE Explorer" },
    { id: "soe", label: "SOE" },
    { id: "ratio", label: "CT/VT Ratio" },
  ];

  const primaryTabs: { id: Tab; label: string }[] = [];

  if (relayType === "21") {
    primaryTabs.push({ id: "ai", label: "AI Fault Analysis" });
    primaryTabs.push({ id: "locus", label: "Impedance Locus" });
  }
  if (relayType === "87L") {
    primaryTabs.push({ id: "diff", label: "Diff / Restraint" });
    primaryTabs.push({ id: "ai", label: "AI Fault Analysis" });
  }
  if (relayType === "87T" || relayType === "REF") {
    primaryTabs.push({ id: "diff", label: "Diff / Restraint" });
  }
  if ((relayType === "OCR" || relayType === "SBEF") && !primaryTabs.find((t) => t.id === "ocr")) {
    primaryTabs.push({ id: "ocr", label: "Overcurrent Curve" });
  }

  return [...primaryTabs, ...detailTabs];
}

export default function Workspace() {
  const { relayType: urlType, analysisId } = useParams<{ relayType: string; analysisId: string }>();
  const { relayType: ctxRelayType, reset } = useAnalysis();
  const navigate = useNavigate();

  const relayType = urlType ?? ctxRelayType ?? "21";
  const [summary, setSummary] = useState<AnalysisSummary | null>(null);
  const [comtrade, setComtrade] = useState<ComtradeData | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const tabs = tabsForRelay(relayType);
  const [activeTab, setActiveTab] = useState<Tab>(initialTabForRelay(relayType));

  useEffect(() => {
    let cancelled = false;

    async function loadSummary() {
      if (!analysisId) {
        if (!cancelled) {
          setSummaryLoading(false);
          setError("Missing analysis session id.");
        }
        return;
      }

      setSummaryLoading(true);
      setError(null);

      try {
        const data = await fetchAnalysisSummary(analysisId);
        if (!cancelled) {
          setSummary(data);
        }
      } catch (err: unknown) {
        const response = (err as { response?: { data?: { detail?: string } } }).response;
        if (!cancelled) {
          setError(response?.data?.detail ?? "Failed to load the analysis session.");
        }
      } finally {
        if (!cancelled) {
          setSummaryLoading(false);
        }
      }
    }

    void loadSummary();
    return () => {
      cancelled = true;
    };
  }, [analysisId]);

  useEffect(() => {
    setActiveTab(initialTabForRelay(relayType));
  }, [relayType]);

  useEffect(() => {
    let cancelled = false;

    async function loadDetail() {
      if (!analysisId || comtrade || !DETAIL_TABS.has(activeTab)) {
        return;
      }

      setDetailLoading(true);
      try {
        const data = await fetchAnalysis(analysisId);
        if (!cancelled) {
          setComtrade(data);
        }
      } catch (err: unknown) {
        const response = (err as { response?: { data?: { detail?: string } } }).response;
        if (!cancelled) {
          setError(response?.data?.detail ?? "Failed to load waveform detail for this analysis.");
        }
      } finally {
        if (!cancelled) {
          setDetailLoading(false);
        }
      }
    }

    void loadDetail();
    return () => {
      cancelled = true;
    };
  }, [activeTab, analysisId, comtrade]);

  if (!analysisId) {
    return <Navigate to={ctxRelayType ? "/upload" : "/"} replace />;
  }

  if (summaryLoading) {
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

  if (!summary) {
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
            <span className={styles.stationName}>{summary.station_name}</span>
            <span className={styles.deviceId}>{summary.rec_dev_id}</span>
          </div>
        </div>
        <div className={styles.headerRight}>
          <span className={styles.meta}>{summary.total_samples} samples</span>
          <span className={styles.meta}>{summary.sampling_rates[0]?.[0]} Hz</span>
          <span className={styles.meta}>{summary.frequency} Hz nominal</span>
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
        {activeTab === "ai" && relayType === "21" && (
          <AIFaultAnalysis21 comtrade={comtrade ?? emptyComtradeFromSummary(summary)} />
        )}
        {activeTab === "waveforms" && comtrade && (
          <AnalogSignalRecap comtrade={comtrade} relayType={relayType} />
        )}
        {activeTab === "explorer" && comtrade && (
          <COMTRADEExplorer comtrade={comtrade} />
        )}
        {activeTab === "soe" && comtrade && (
          <SOETimeline comtrade={comtrade} />
        )}
        {activeTab === "ratio" && comtrade && (
          <CTVTRatioCorrection analysisId={analysisId} comtrade={comtrade} onUpdate={setComtrade} />
        )}
        {activeTab === "locus" && relayType === "21" && (
          <ImpedanceLocus analysisId={analysisId} />
        )}
        {activeTab === "diff" && (relayType === "87L" || relayType === "87T" || relayType === "REF") && (
          <DiffRestraintPlot analysisId={analysisId} relayType={relayType === "87L" ? "87L" : "87T"} />
        )}
        {activeTab === "ai" && relayType === "87L" && (
          <AIFaultAnalysis87L analysisId={analysisId} />
        )}
        {activeTab === "ocr" && (relayType === "OCR" || relayType === "SBEF") && (
          <OvercurrentOverlay analysisId={analysisId} relayType={relayType} />
        )}
        {(relayType === "87T" || relayType === "REF") && activeTab === "diff" && (
          <div style={{ fontSize: "0.85rem", color: "#64748b", marginTop: 12, padding: "10px 16px", background: "#f8fafc", borderRadius: 8 }}>
            AI fault cause analysis for transformer differential is pending - relay coordination evidence required.
          </div>
        )}
        {DETAIL_TABS.has(activeTab) && detailLoading && (
          <div className={styles.placeholderCard}>
            <div className={styles.placeholderTitle}>Loading waveform detail...</div>
            <p className={styles.placeholderText}>
              The workspace shell is already ready. This tab is now fetching only the raw samples it needs.
            </p>
          </div>
        )}
        {!detailLoading && !comtrade && DETAIL_TABS.has(activeTab) && (
          <div className={styles.placeholderCard}>
            <div className={styles.placeholderTitle}>Analysis Overview</div>
            <div className={styles.summaryGrid}>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Duration</span>
                <strong>{summary.duration_ms.toFixed(1)} ms</strong>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Analog Channels</span>
                <strong>{summary.analog_channels.length}</strong>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Status Channels</span>
                <strong>{summary.status_channels.length}</strong>
              </div>
              <div className={styles.summaryItem}>
                <span className={styles.summaryLabel}>Revision</span>
                <strong>{summary.rev_year || "-"}</strong>
              </div>
            </div>
            <p className={styles.placeholderText}>
              Waveform-heavy tabs are loaded lazily now, so the workspace can open before the full COMTRADE arrays arrive.
            </p>
          </div>
        )}
      </main>
    </div>
  );
}

function emptyComtradeFromSummary(summary: AnalysisSummary): ComtradeData {
  return {
    station_name: summary.station_name,
    rec_dev_id: summary.rec_dev_id,
    rev_year: summary.rev_year,
    sampling_rates: summary.sampling_rates,
    trigger_time: summary.trigger_time,
    total_samples: summary.total_samples,
    frequency: summary.frequency,
    time: [],
    analog_channels: [],
    status_channels: [],
    warnings: summary.warnings,
  };
}
