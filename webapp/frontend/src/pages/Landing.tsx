import { useNavigate } from "react-router-dom";
import { useAnalysis } from "../context/AnalysisContext";
import type { RelayType } from "../context/AnalysisContext";
import styles from "./Landing.module.css";

interface RelayOption {
  id: RelayType;
  label: string;
  subtitle: string;
  tooltip: string;
  icon: string;
}

const RELAY_OPTIONS: RelayOption[] = [
  {
    id: "21",
    label: "21 - Distance",
    subtitle: "Distance protection and fault cause triage",
    tooltip:
      "Impedance locus diagram, zone polygon editor, phase and earth loop selector, plus AI-based fault cause ranking.",
    icon: "21",
  },
  {
    id: "87L",
    label: "87L - Differential Line",
    subtitle: "Line differential diagnostics",
    tooltip:
      "Differential vs restraint characteristic plot, SIPROTEC-style parameter editor, and internal vs external fault analysis.",
    icon: "87L",
  },
  {
    id: "87T",
    label: "87T - Differential Transformer",
    subtitle: "Transformer differential review",
    tooltip:
      "HV and LV differential-restraint plots per phase with operated, not operated, and IDIFF FAST state checks.",
    icon: "87T",
  },
  {
    id: "OCR",
    label: "50/51 - OCR",
    subtitle: "Overcurrent timing and pickup review",
    tooltip:
      "Measured current against IEC and IEEE overcurrent characteristic curves with configurable pickup and TMS.",
    icon: "50/51",
  },
  {
    id: "REF",
    label: "REF / GFR / SBEF",
    subtitle: "Earth-fault supporting panels",
    tooltip:
      "Waveform viewer, COMTRADE explorer, CT/VT ratio correction, and sequence-of-events timeline for earth-fault relays.",
    icon: "REF",
  },
];

export default function Landing() {
  const { setRelayType } = useAnalysis();
  const navigate = useNavigate();

  function select(type: RelayType) {
    setRelayType(type);
    navigate("/upload");
  }

  return (
    <div className={styles.page}>
      <div className={styles.orbLeft} />
      <div className={styles.orbRight} />

      <header className={styles.header}>
        <div className={styles.eyebrow}>DFR UIT JBT</div>
        <div className={styles.logo}>DFR Analyser</div>
        <p className={styles.subtitle}>
          Pick the protection family first, then upload the COMTRADE pair for analysis.
        </p>
      </header>

      <main className={styles.grid}>
        {RELAY_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            className={styles.card}
            onClick={() => select(opt.id)}
            title={opt.tooltip}
            type="button"
          >
            <span className={styles.cardIcon}>{opt.icon}</span>
            <span className={styles.cardLabel}>{opt.label}</span>
            <span className={styles.cardSub}>{opt.subtitle}</span>
            <span className={styles.cardTooltip}>{opt.tooltip}</span>
          </button>
        ))}
      </main>

      <footer className={styles.footer}>
        Upload a matching <code>.cfg</code> and <code>.dat</code> pair after selecting the relay type.
      </footer>
    </div>
  );
}
