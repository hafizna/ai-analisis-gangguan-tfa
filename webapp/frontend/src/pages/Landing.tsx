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
    label: "21 — Distance",
    subtitle: "Distance Protection",
    tooltip:
      "Impedance locus diagram (R-X plane), zone polygon editor (Z1/Z2/Z3 mho/quad), phase/earth loop selector, AI fault cause analysis.",
    icon: "⊕",
  },
  {
    id: "87L",
    label: "87L — Differential Line",
    subtitle: "Line Differential",
    tooltip:
      "Differential/Restraint characteristic plot (I-diff vs I-rest), SIPROTEC 4/5 parameter editor, internal vs external fault AI classification.",
    icon: "⇌",
  },
  {
    id: "87T",
    label: "87T — Differential Transformer",
    subtitle: "Transformer Differential",
    tooltip:
      "HV + LV winding differential/restraint plots per phase, OPERATED / NOT OPERATED / IDIFF FAST status badge.",
    icon: "⊞",
  },
  {
    id: "OCR",
    label: "50/51 — OCR",
    subtitle: "Overcurrent",
    tooltip:
      "Measured current vs IEC/IEEE overcurrent trip curve (Normal Inverse, Very Inverse, Extremely Inverse). Pickup and TMS configurable.",
    icon: "↑",
  },
  {
    id: "REF",
    label: "REF / GFR / SBEF",
    subtitle: "Earth Fault Protection",
    tooltip:
      "Base panels: waveform viewer, COMTRADE explorer, CT/VT ratio correction, Sequence of Events timeline.",
    icon: "⏚",
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
      <header className={styles.header}>
        <div className={styles.logo}>DFR Analyser</div>
        <p className={styles.subtitle}>
          Select relay protection type to begin COMTRADE fault analysis
        </p>
      </header>

      <main className={styles.grid}>
        {RELAY_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            className={styles.card}
            onClick={() => select(opt.id)}
            title={opt.tooltip}
          >
            <span className={styles.cardIcon}>{opt.icon}</span>
            <span className={styles.cardLabel}>{opt.label}</span>
            <span className={styles.cardSub}>{opt.subtitle}</span>
            <span className={styles.cardTooltip}>{opt.tooltip}</span>
          </button>
        ))}
      </main>

      <footer className={styles.footer}>
        Upload a .cfg + .dat COMTRADE pair after selecting relay type
      </footer>
    </div>
  );
}
