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
  "87T": "87T / Transformer Differential",
  OCR: "50/51 - Overcurrent",
  REF: "REF / GFR / SBEF",
  SBEF: "SBEF",
};

export default function Workspace() {
  const { relayType: urlType, analysisId } = useParams<{ relayType: string; analysisId: string }>();
  const { relayType: ctxRelayType, reset } = useAnalysis();
  const navigate = useNavigate();

  const relayType = urlType ?? ctxRelayType ?? "21";
  const [comtrade, setComtrade] = useState<ComtradeData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!analysisId) return;
    setLoading(true);
    setError(null);
    fetchAnalysis(analysisId)
      .then(setComtrade)
      .catch(() => setError("Failed to load analysis data. The session may have expired."))
      .finally(() => setLoading(false));
  }, [analysisId]);

  if (!analysisId) {
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
          <button className={styles.homeBtn} onClick={handleReset} type="button">← New Analysis</button>
          <div>
            <span className={styles.relayBadge}>{RELAY_LABELS[relayType] ?? relayType}</span>
            {comtrade && (
              <>
                <span className={styles.stationName}>{comtrade.station_name}</span>
                <span className={styles.deviceId}>{comtrade.rec_dev_id}</span>
              </>
            )}
          </div>
        </div>
        {comtrade && (
          <div className={styles.headerRight}>
            <span className={styles.meta}>{comtrade.total_samples} samples</span>
            <span className={styles.meta}>{comtrade.sampling_rates[0]?.[0]} Hz</span>
            <span className={styles.meta}>{comtrade.frequency} Hz nominal</span>
          </div>
        )}
      </header>

      <main className={styles.content}>
        {loading && (
          <div className={styles.loadingState}>Loading waveforms…</div>
        )}

        {error && (
          <div className={styles.errorState}>{error}</div>
        )}

        {!loading && !error && comtrade && (
          <>
            {/* 1. Waveforms — always first */}
            <AnalogSignalRecap comtrade={comtrade} relayType={relayType} />

            {/* 2. Relay-specific analysis panels */}
            {relayType === "21" && (
              <>
                <ImpedanceLocus analysisId={analysisId} />
                <AIFaultAnalysis21 analysisId={analysisId} />
              </>
            )}

            {(relayType === "87L") && (
              <>
                <DiffRestraintPlot analysisId={analysisId} relayType="87L" />
                <AIFaultAnalysis87L analysisId={analysisId} />
              </>
            )}

            {(relayType === "87T" || relayType === "REF") && (
              <>
                <DiffRestraintPlot analysisId={analysisId} relayType="87T" />
                <div className={styles.pendingNote}>
                  AI fault cause analysis for transformer differential is pending — relay coordination evidence required.
                </div>
              </>
            )}

            {(relayType === "OCR" || relayType === "SBEF") && (
              <OvercurrentOverlay analysisId={analysisId} relayType={relayType} />
            )}

            {/* 3. SOE Timeline */}
            <SOETimeline comtrade={comtrade} />

            {/* 4. CT/VT Ratio Correction */}
            <CTVTRatioCorrection
              analysisId={analysisId}
              comtrade={comtrade}
              onUpdate={setComtrade}
            />

            {/* 5. COMTRADE Explorer — full metadata at bottom */}
            <COMTRADEExplorer comtrade={comtrade} />
          </>
        )}
      </main>
    </div>
  );
}
