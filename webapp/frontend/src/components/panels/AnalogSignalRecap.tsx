import { useState } from "react";
import Plot from "react-plotly.js";
import type { ComtradeData } from "../../context/AnalysisContext";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
  relayType: string;
}

type ViewMode = "primary" | "secondary";

export default function AnalogSignalRecap({ comtrade, relayType }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("primary");
  const [selectedPhases, setSelectedPhases] = useState<Set<string>>(
    new Set(comtrade.analog_channels.map((c) => c.id))
  );

  const time = comtrade.time;

  function toggleChannel(id: string) {
    setSelectedPhases((prev) => {
      const next = new Set(prev);
      next.has(id) ? next.delete(id) : next.add(id);
      return next;
    });
  }

  const voltageChannels = comtrade.analog_channels.filter((c) => c.measurement === "voltage");
  const currentChannels = comtrade.analog_channels.filter((c) => c.measurement === "current");

  function scaleSamples(ch: ComtradeData["analog_channels"][number]) {
    if (viewMode === "secondary" && ch.ct_primary > 0 && ch.ct_secondary > 0) {
      const factor = ch.ct_secondary / ch.ct_primary;
      return ch.samples.map((s) => s * factor);
    }
    return ch.samples;
  }

  function makeTraces(channels: ComtradeData["analog_channels"]) {
    return channels.filter((c) => selectedPhases.has(c.id)).map((ch) => ({
      x: time,
      y: scaleSamples(ch),
      type: "scatter" as const,
      mode: "lines" as const,
      name: ch.canonical_name || ch.name,
      line: { width: 1.5 },
    }));
  }

  const vTraces = makeTraces(voltageChannels);
  const iTraces = makeTraces(currentChannels);

  const plotLayout = (title: string, yUnit: string) => ({
    title: { text: title, font: { size: 13 } },
    height: 220,
    margin: { t: 36, b: 40, l: 55, r: 12 },
    xaxis: { title: { text: "Time (s)" }, tickfont: { size: 10 } },
    yaxis: { title: { text: yUnit }, tickfont: { size: 10 } },
    legend: { orientation: "h" as const, y: -0.25 },
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
  });

  const config = { displayModeBar: false, responsive: true };

  // Channel label for display based on relay type hint
  const channelHint =
    relayType === "87T"
      ? "6 CTs (HV + LV)"
      : relayType === "OCR" || relayType === "REF"
      ? "Phase Currents"
      : "3I + 3V";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Analog Signal Recap</h2>
        <div className={styles.controls}>
          <span className={styles.badge}>{channelHint}</span>
          <label className={styles.toggle}>
            <input
              type="radio"
              name="viewmode"
              value="primary"
              checked={viewMode === "primary"}
              onChange={() => setViewMode("primary")}
            />{" "}
            Primary
          </label>
          <label className={styles.toggle}>
            <input
              type="radio"
              name="viewmode"
              value="secondary"
              checked={viewMode === "secondary"}
              onChange={() => setViewMode("secondary")}
            />{" "}
            Secondary
          </label>
        </div>
      </div>

      <div className={styles.channelToggles}>
        {comtrade.analog_channels.map((ch) => (
          <button
            key={ch.id}
            className={`${styles.chBtn} ${selectedPhases.has(ch.id) ? styles.chBtnOn : ""}`}
            onClick={() => toggleChannel(ch.id)}
          >
            {ch.canonical_name || ch.name}
          </button>
        ))}
      </div>

      {vTraces.length > 0 && (
        <Plot
          data={vTraces}
          layout={plotLayout("Voltage Channels", `V (${viewMode === "primary" ? "kV" : "V"})`)}
          config={config}
          style={{ width: "100%" }}
        />
      )}

      {iTraces.length > 0 && (
        <Plot
          data={iTraces}
          layout={plotLayout("Current Channels", `I (A)`)}
          config={config}
          style={{ width: "100%" }}
        />
      )}

      {comtrade.warnings.length > 0 && (
        <div className={styles.warnings}>
          {comtrade.warnings.map((w, i) => (
            <div key={i} className={styles.warning}>⚠ {w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
