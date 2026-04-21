import Plot from "react-plotly.js";
import type { ComtradeData } from "../../context/AnalysisContext";
import styles from "./Panel.module.css";

interface Props { comtrade: ComtradeData; }

// Keywords that indicate protection operation events
const TRIP_KEYWORDS = ["TRIP", "CB", "CLOSE", "AR", "RECLOSE", "OPEN", "BLOCK", "START", "PICKUP"];

function isProtectionChannel(name: string) {
  const u = name.toUpperCase();
  return TRIP_KEYWORDS.some((k) => u.includes(k));
}

function findEdges(samples: number[], time: number[]) {
  const edges: { t: number; type: "rise" | "fall" }[] = [];
  for (let i = 1; i < samples.length; i++) {
    if (samples[i - 1] === 0 && samples[i] === 1) {
      edges.push({ t: time[i] * 1000, type: "rise" });
    } else if (samples[i - 1] === 1 && samples[i] === 0) {
      edges.push({ t: time[i] * 1000, type: "fall" });
    }
  }
  return edges;
}

export default function SOETimeline({ comtrade }: Props) {
  const time = comtrade.time;
  const tMin = time[0] * 1000;
  const tMax = time[time.length - 1] * 1000;

  const channels = comtrade.status_channels;

  // Sort: protection channels first
  const sorted = [...channels].sort((a, b) => {
    const ap = isProtectionChannel(a.name) ? 0 : 1;
    const bp = isProtectionChannel(b.name) ? 0 : 1;
    return ap - bp;
  });

  if (sorted.length === 0) {
    return (
      <div className={styles.panel}>
        <div className={styles.panelHeader}>
          <h2 className={styles.panelTitle}>Sequence of Events (SOE)</h2>
        </div>
        <p style={{ fontSize: "0.85rem", color: "#94a3b8" }}>No binary/status channels in this record.</p>
      </div>
    );
  }

  // Build Gantt-style shapes + annotations
  const shapes: Plotly.Shape[] = [];
  const annotations: Partial<Plotly.Annotations>[] = [];

  sorted.forEach((ch, rowIdx) => {
    const y = sorted.length - 1 - rowIdx;
    const runs = buildRuns(ch.samples, time);
    runs.filter((r) => r.val === 1).forEach((r) => {
      shapes.push({
        type: "rect",
        x0: r.tStart * 1000,
        x1: r.tEnd * 1000,
        y0: y - 0.4,
        y1: y + 0.4,
        fillcolor: isProtectionChannel(ch.name) ? "#ef4444" : "#3b82f6",
        opacity: 0.7,
        line: { width: 0 },
      } as Plotly.Shape);
    });

    // Annotate first rise edge
    const edges = findEdges(ch.samples, time);
    const firstRise = edges.find((e) => e.type === "rise");
    if (firstRise) {
      annotations.push({
        x: firstRise.t,
        y,
        text: `+${(firstRise.t - tMin).toFixed(1)}ms`,
        showarrow: false,
        font: { size: 9, color: "#475569" },
        xanchor: "left",
        yanchor: "middle",
      } as Partial<Plotly.Annotations>);
    }
  });

  const layout: Partial<Plotly.Layout> = {
    height: Math.max(150, sorted.length * 30 + 60),
    margin: { t: 20, b: 40, l: 170, r: 20 },
    xaxis: {
      title: { text: "Time (ms)" },
      range: [tMin, tMax],
      tickfont: { size: 10 },
    },
    yaxis: {
      tickvals: sorted.map((_, i) => sorted.length - 1 - i),
      ticktext: sorted.map((ch) => ch.name),
      tickfont: { size: 10 },
      range: [-0.6, sorted.length - 0.4],
    },
    shapes,
    annotations,
    plot_bgcolor: "#ffffff",
    paper_bgcolor: "#ffffff",
    showlegend: false,
  };

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>Sequence of Events (SOE)</h2>
        <span className={styles.badge}>{sorted.length} channels</span>
      </div>
      <Plot
        data={[{ type: "scatter", x: [], y: [], mode: "markers" }]}
        layout={layout}
        config={{ displayModeBar: false, responsive: true }}
        style={{ width: "100%" }}
      />
    </div>
  );
}

function buildRuns(samples: number[], time: number[]) {
  const runs: { val: number; tStart: number; tEnd: number }[] = [];
  if (!samples.length) return runs;
  let cur = { val: samples[0], tStart: time[0], tEnd: time[0] };
  for (let i = 1; i < samples.length; i++) {
    if (samples[i] === cur.val) {
      cur.tEnd = time[i];
    } else {
      runs.push(cur);
      cur = { val: samples[i], tStart: time[i], tEnd: time[i] };
    }
  }
  runs.push(cur);
  return runs;
}
