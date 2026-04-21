import type { ComtradeData } from "../../context/AnalysisContext";
import styles from "./Panel.module.css";

interface Props { comtrade: ComtradeData; }

export default function COMTRADEExplorer({ comtrade }: Props) {
  const sr = comtrade.sampling_rates[0]?.[0] ?? "—";
  const duration = comtrade.time.length > 1
    ? ((comtrade.time[comtrade.time.length - 1] - comtrade.time[0]) * 1000).toFixed(1)
    : "—";

  return (
    <div className={styles.panel}>
      <div className={styles.panelHeader}>
        <h2 className={styles.panelTitle}>COMTRADE Explorer</h2>
      </div>

      {/* Metadata */}
      <table className={styles.table} style={{ marginBottom: 20 }}>
        <tbody>
          <tr><td className={styles.table}>Station</td><td>{comtrade.station_name}</td></tr>
          <tr><td>Device ID</td><td>{comtrade.rec_dev_id}</td></tr>
          <tr><td>Revision</td><td>{comtrade.rev_year}</td></tr>
          <tr><td>Sample Rate</td><td>{sr} Hz</td></tr>
          <tr><td>Duration</td><td>{duration} ms</td></tr>
          <tr><td>Total Samples</td><td>{comtrade.total_samples}</td></tr>
          <tr><td>Frequency</td><td>{comtrade.frequency} Hz</td></tr>
          <tr><td>Analog Channels</td><td>{comtrade.analog_channels.length}</td></tr>
          <tr><td>Status Channels</td><td>{comtrade.status_channels.length}</td></tr>
        </tbody>
      </table>

      {/* Analog channel table */}
      <h3 style={{ fontSize: "0.85rem", color: "#475569", marginBottom: 8 }}>Analog Channels</h3>
      <div style={{ overflowX: "auto", marginBottom: 20 }}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th>#</th>
              <th>Name</th>
              <th>Canonical</th>
              <th>Unit</th>
              <th>Phase</th>
              <th>CT Pri</th>
              <th>CT Sec</th>
              <th>P/S</th>
            </tr>
          </thead>
          <tbody>
            {comtrade.analog_channels.map((ch, i) => (
              <tr key={ch.id}>
                <td>{i + 1}</td>
                <td>{ch.name}</td>
                <td>{ch.canonical_name}</td>
                <td>{ch.unit}</td>
                <td>{ch.phase ?? "—"}</td>
                <td>{ch.ct_primary}</td>
                <td>{ch.ct_secondary}</td>
                <td>{ch.pors}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Status channels as logic timeline */}
      {comtrade.status_channels.length > 0 && (
        <>
          <h3 style={{ fontSize: "0.85rem", color: "#475569", marginBottom: 10 }}>
            Binary / Status Channels
          </h3>
          {comtrade.status_channels.map((ch) => {
            const total = ch.samples.length;
            const runs = buildRuns(ch.samples);
            return (
              <div key={ch.id} className={styles.soeRow}>
                <span className={styles.soeLabel} title={ch.name}>{ch.name}</span>
                <div className={styles.soeBar}>
                  {runs.filter((r) => r.val === 1).map((r, i) => (
                    <div
                      key={i}
                      className={styles.soeSegment}
                      style={{
                        left: `${(r.start / total) * 100}%`,
                        width: `${((r.end - r.start) / total) * 100}%`,
                      }}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </>
      )}
    </div>
  );
}

function buildRuns(samples: number[]) {
  const runs: { val: number; start: number; end: number }[] = [];
  if (!samples.length) return runs;
  let cur = { val: samples[0], start: 0, end: 1 };
  for (let i = 1; i < samples.length; i++) {
    if (samples[i] === cur.val) {
      cur.end = i + 1;
    } else {
      runs.push(cur);
      cur = { val: samples[i], start: i, end: i + 1 };
    }
  }
  runs.push(cur);
  return runs;
}
