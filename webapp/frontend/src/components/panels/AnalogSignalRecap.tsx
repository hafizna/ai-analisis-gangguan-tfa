import { useMemo, useState } from "react";
import Plot from "react-plotly.js";
import type { ComtradeData } from "../../context/AnalysisContext";
import styles from "./Panel.module.css";

interface Props {
  comtrade: ComtradeData;
  relayType: string;
}

type ViewMode = "primary" | "secondary";

const MAX_PLOT_POINTS = 4000;

function preferredChannels(comtrade: ComtradeData, relayType: string) {
  const preferredNames =
    relayType === "87T" || relayType === "REF"
      ? ["IA_HV", "IB_HV", "IC_HV", "IA_LV", "IB_LV", "IC_LV", "3I0", "IN", "IE"]
      : relayType === "OCR" || relayType === "SBEF"
      ? ["IA", "IB", "IC", "IN", "I0", "IE"]
      : ["VA", "VB", "VC", "IA", "IB", "IC", "IN", "I0"];

  const matching = comtrade.analog_channels
    .filter((ch) => preferredNames.includes(ch.canonical_name))
    .map((ch) => ch.id);

  if (matching.length > 0) {
    return new Set(matching);
  }

  return new Set(comtrade.analog_channels.slice(0, Math.min(6, comtrade.analog_channels.length)).map((c) => c.id));
}

function downsample<T>(values: T[], maxPoints: number) {
  if (values.length <= maxPoints) {
    return values;
  }

  const step = Math.ceil(values.length / maxPoints);
  const sampled: T[] = [];

  for (let i = 0; i < values.length; i += step) {
    sampled.push(values[i]);
  }

  if (sampled[sampled.length - 1] !== values[values.length - 1]) {
    sampled.push(values[values.length - 1]);
  }

  return sampled;
}

export default function AnalogSignalRecap({ comtrade, relayType }: Props) {
  const [viewMode, setViewMode] = useState<ViewMode>("primary");
  const [selectedPhases, setSelectedPhases] = useState<Set<string>>(
    () => preferredChannels(comtrade, relayType)
  );

  const time = useMemo(
    () => downsample(comtrade.time, MAX_PLOT_POINTS),
    [comtrade.time]
  );

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
    const raw = downsample(ch.samples, MAX_PLOT_POINTS);
    if (viewMode === "secondary" && ch.ct_primary > 0 && ch.ct_secondary > 0) {
      const factor = ch.ct_secondary / ch.ct_primary;
      return raw.map((s) => s * factor);
    }
    return raw;
  }

  function makeTraces(channels: ComtradeData["analog_channels"]) {
    return channels
      .filter((c) => selectedPhases.has(c.id))
      .map((ch) => ({
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

  const channelHint =
    relayType === "87T" || relayType === "REF"
      ? "Transformer / REF currents"
      : relayType === "OCR"
      ? "OCR / GFR currents"
      : relayType === "SBEF"
      ? "Sensitive earth-fault currents"
      : "3I + 3V";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Analog Signal Recap</h2>
        <div className={styles.controls}>
          <span className={styles.badge}>{channelHint}</span>
          <span className={styles.badge}>Showing up to {MAX_PLOT_POINTS.toLocaleString()} points per trace</span>
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
            type="button"
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
          layout={plotLayout("Current Channels", "I (A)")}
          config={config}
          style={{ width: "100%" }}
        />
      )}

      {comtrade.warnings.length > 0 && (
        <div className={styles.warnings}>
          {comtrade.warnings.map((w, i) => (
            <div key={i} className={styles.warning}>Warning: {w}</div>
          ))}
        </div>
      )}
    </div>
  );
}
