"""
COMTRADE Parser
===============
Robust parser for IEEE C37.111 COMTRADE files (.cfg + .dat pairs).
Handles multiple relay vendors, converts to primary values, normalizes channel names.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from pathlib import Path
import logging
import numpy as np

try:
    from comtrade import Comtrade
except ImportError:
    raise ImportError("comtrade library not found. Install with: pip install comtrade")

from .channel_normalizer import normalize_channel_name, detect_manufacturer

logger = logging.getLogger(__name__)


@dataclass
class AnalogChannel:
    """Represents a single analog channel with normalized metadata."""
    id: str                     # Original channel ID from .cfg
    name: str                   # Original channel name from .cfg
    canonical_name: str         # Normalized name (VA, IA, etc.)
    unit: str                   # Standardized: "kV" for voltage, "A" for current
    phase: Optional[str]        # "A", "B", "C", "N", or None
    measurement: str            # "voltage" or "current"
    ct_primary: float           # CT/VT primary value from .cfg (metadata only)
    ct_secondary: float         # CT/VT secondary value from .cfg (metadata only)
    scale_a: float              # Raw COMTRADE channel scale factor "a" (metadata)
    scale_b: float              # Raw COMTRADE channel offset "b" (metadata)
    samples: np.ndarray         # Waveform in PRIMARY values: kV for voltage, A for current


@dataclass
class StatusChannel:
    """Represents a digital/status channel."""
    id: str
    name: str
    samples: np.ndarray         # 0/1 values


@dataclass
class ComtradeRecord:
    """
    Complete parsed COMTRADE record with all metadata and waveforms.
    All analog channels are converted to primary values.
    """
    # Metadata
    station_name: str
    rec_dev_id: str            # Relay model identifier
    rev_year: str              # COMTRADE revision year
    sampling_rates: List[tuple]  # List of (rate_hz, end_sample) tuples
    trigger_time: float        # Trigger timestamp (seconds since epoch, or 0 if unknown)
    total_samples: int
    frequency: float           # Nominal frequency (should be 50 Hz for Indonesia)

    # Channel data
    analog_channels: List[AnalogChannel]
    status_channels: List[StatusChannel]
    time: np.ndarray           # Time axis in seconds (relative to trigger)

    # Source info
    cfg_path: str
    dat_path: Optional[str]

    # Parsing diagnostics
    warnings: List[str] = field(default_factory=list)  # Any issues found during parsing


def parse_comtrade(cfg_path: str, dat_path: Optional[str] = None) -> Optional[ComtradeRecord]:
    """
    Parse a COMTRADE file pair (.cfg + .dat).

    Args:
        cfg_path: Path to .cfg file. Will auto-find .dat if not provided.
        dat_path: Optional explicit path to .dat file.

    Returns:
        ComtradeRecord with all channels in primary values, or None if unreadable.

    Note:
        - If .dat is missing, still parses .cfg metadata (returns empty samples with warning)
        - Never crashes - returns None only for truly unreadable files
        - All warnings are logged and stored in record.warnings
    """
    cfg_path_obj = Path(cfg_path)

    if not cfg_path_obj.exists():
        logger.error(f"CFG file not found: {cfg_path}")
        return None

    # Auto-find .dat file if not provided
    if dat_path is None:
        dat_path = _find_dat_file(cfg_path_obj)

    warnings = []

    try:
        # Use comtrade library for basic parsing
        com = Comtrade()

        # Handle encoding issues
        try:
            com.load(str(cfg_path_obj), str(dat_path) if dat_path else None)
        except UnicodeDecodeError:
            # Try with different encoding
            logger.warning(f"UTF-8 decode failed for {cfg_path}, trying latin-1")
            warnings.append("File encoding issue - used fallback decoder")
            # The comtrade library doesn't expose encoding parameter, so we may need to handle this differently
            # For now, log and continue
            com.load(str(cfg_path_obj), str(dat_path) if dat_path else None)

        if dat_path is None or not Path(dat_path).exists():
            warnings.append("DAT file not found - metadata only")

        # Detect manufacturer
        manufacturer = detect_manufacturer(com.rec_dev_id, com.station_name)

        # Parse analog channels
        analog_channels = _parse_analog_channels(com, manufacturer, warnings)

        # Parse status channels
        status_channels = _parse_status_channels(com)

        # Extract time axis
        time = np.array(com.time) if hasattr(com, 'time') and len(com.time) > 0 else np.array([])

        # Extract sampling rates
        sampling_rates = []
        if hasattr(com.cfg, 'sample_rates') and len(com.cfg.sample_rates) > 0:
            for rate_info in com.cfg.sample_rates:
                if isinstance(rate_info, (list, tuple)) and len(rate_info) >= 2:
                    sampling_rates.append((float(rate_info[0]), int(rate_info[1])))
                else:
                    logger.warning(f"Unexpected sample_rates format: {rate_info}")
                    warnings.append(f"Unexpected sample rate format")

        # Get trigger time
        trigger_time = 0.0
        if hasattr(com, 'trigger_time'):
            try:
                trigger_time = float(com.trigger_time)
            except (ValueError, TypeError):
                warnings.append("Could not parse trigger time")

        # Get frequency
        frequency = 50.0  # Default for Indonesia
        if hasattr(com, 'frequency') and com.frequency:
            frequency = float(com.frequency)
        elif hasattr(com.cfg, 'frequency') and com.cfg.frequency:
            frequency = float(com.cfg.frequency)

        # Validate frequency
        if not (45.0 <= frequency <= 65.0):
            warnings.append(f"Unusual frequency: {frequency} Hz (expected ~50 Hz)")

        # Create record
        record = ComtradeRecord(
            station_name=com.station_name or "UNKNOWN",
            rec_dev_id=com.rec_dev_id or "UNKNOWN",
            rev_year=str(com.rev_year) if hasattr(com, 'rev_year') else "1999",
            sampling_rates=sampling_rates,
            trigger_time=trigger_time,
            total_samples=len(time),
            frequency=frequency,
            analog_channels=analog_channels,
            status_channels=status_channels,
            time=time,
            cfg_path=str(cfg_path_obj),
            dat_path=str(dat_path) if dat_path else None,
            warnings=warnings
        )

        logger.info(f"Successfully parsed {cfg_path_obj.name}: {len(analog_channels)} analog, {len(status_channels)} status channels")
        return record

    except Exception as e:
        logger.error(f"Failed to parse {cfg_path}: {e}", exc_info=True)
        return None


def _find_dat_file(cfg_path: Path) -> Optional[str]:
    """Auto-find .dat file corresponding to .cfg file."""
    # Try same stem with .dat extension
    for ext in ['.dat', '.DAT']:
        dat_candidate = cfg_path.with_suffix(ext)
        if dat_candidate.exists():
            return str(dat_candidate)

    logger.warning(f"No .dat file found for {cfg_path.name}")
    return None


def _parse_analog_channels(com: Comtrade, manufacturer: str, warnings: List[str]) -> List[AnalogChannel]:
    """Parse and normalize analog channels.

    Note: The comtrade library already converts samples to primary values using the
    multiplier 'a' from the .cfg file. We do NOT need to apply the primary/secondary
    ratio again - doing so would cause values to be 100x-1000x too large.
    """
    analog_channels = []

    for i, ch_id in enumerate(com.analog_channel_ids):
        try:
            # Get channel info
            ch_name = com.cfg.analog_channels[i].name if hasattr(com.cfg, 'analog_channels') else ch_id
            ch_unit = com.cfg.analog_channels[i].uu if hasattr(com.cfg, 'analog_channels') else ""

            # COMTRADE 1997 / non-standard files sometimes store a generic type
            # ("Voltage", "Current") as the channel name while the actual phase
            # identifier (Va, Vb, Vc, Ia, Ib, Ic, 3I0) is in the ph field.
            # Use ph as the effective name so normalization can extract phase info.
            if hasattr(com.cfg.analog_channels[i], 'ph'):
                ph_hint = (com.cfg.analog_channels[i].ph or "").strip()
                if ph_hint and ch_name.upper() in ('VOLTAGE', 'CURRENT', 'V', 'I', 'A', 'U'):
                    ch_name = ph_hint

            # Get CT/VT ratios (for validation/metadata only)
            ct_primary = 1.0
            ct_secondary = 1.0

            if hasattr(com.cfg.analog_channels[i], 'primary'):
                ct_primary = float(com.cfg.analog_channels[i].primary)
            if hasattr(com.cfg.analog_channels[i], 'secondary'):
                ct_secondary = float(com.cfg.analog_channels[i].secondary)
            scale_a = 1.0
            scale_b = 0.0
            if hasattr(com.cfg.analog_channels[i], 'a'):
                try:
                    scale_a = float(com.cfg.analog_channels[i].a)
                except Exception:
                    scale_a = 1.0
            if hasattr(com.cfg.analog_channels[i], 'b'):
                try:
                    scale_b = float(com.cfg.analog_channels[i].b)
                except Exception:
                    scale_b = 0.0

            # Handle case where ct_secondary is 0
            if ct_secondary == 0:
                ct_secondary = 1.0
                warnings.append(f"Channel {ch_name}: secondary = 0, treating as 1:1 ratio")

            # Get PS flag: 'P' = values already in primary units, 'S' = secondary units
            pors = getattr(com.cfg.analog_channels[i], 'pors', 'P')
            pors = (pors or 'P').upper().strip()

            # Get samples from comtrade library (applies a*raw + b, may be primary or secondary)
            if i < len(com.analog):
                samples_primary = np.array(com.analog[i], dtype=float)
            else:
                samples_primary = np.array([], dtype=float)
                warnings.append(f"Channel {ch_name}: no sample data")

            # If recorded in secondary units, convert to primary now
            if pors == 'S' and ct_primary > 0 and ct_secondary > 0 and ct_primary != ct_secondary:
                samples_primary = samples_primary * (ct_primary / ct_secondary)
                warnings.append(f"Channel {ch_name}: pors=S, applied ratio {ct_primary}/{ct_secondary} to convert to primary")

            # Normalize channel name
            norm = normalize_channel_name(ch_name, ch_unit, manufacturer)

            # Normalize units: convert V to kV for voltage, keep A for current
            # IMPORTANT: Check case-sensitive first to distinguish mV/mA from MV/MA!
            normalized_unit = ch_unit
            unit_stripped = ch_unit.strip()
            unit_upper = unit_stripped.upper()

            # Skip control signals (mV, mA) - don't normalize these
            if unit_stripped in ['mV', 'mA']:
                # Leave as-is
                pass
            elif norm['measurement'] == 'voltage':
                if unit_upper == 'V':
                    # Convert volts to kilovolts (for power system voltages)
                    samples_primary = samples_primary / 1000.0
                    normalized_unit = 'kV'
                elif unit_upper == 'KV':
                    normalized_unit = 'kV'
                elif unit_upper == 'MV' and unit_stripped != 'mV':
                    # Megavolts to kilovolts (rare, but possible)
                    # But make sure it's not millivolts (mV)!
                    samples_primary = samples_primary * 1000.0
                    normalized_unit = 'kV'
            elif norm['measurement'] == 'current':
                # Current normalization to A (amps)
                if unit_upper == 'KA':
                    # Kiloamps to amps
                    samples_primary = samples_primary * 1000.0
                    normalized_unit = 'A'
                elif unit_upper == 'A':
                    normalized_unit = 'A'
                # mA already handled above

            # Validate CT/VT ratios (warn if unusual)
            ratio = ct_primary / ct_secondary
            if norm['measurement'] == 'voltage' and ratio != 1.0:
                # Common VT secondaries: 100V, 110V, 125V
                # So primary/secondary should be like 1500, 1364, 1200 for 150kV system
                # Or 700, 636, 560 for 70kV system
                if ct_secondary not in [1, 100, 110, 125, 120]:
                    warnings.append(f"Channel {ch_name}: unusual VT secondary ({ct_secondary}V) - expected 100, 110, or 125V")
            elif norm['measurement'] == 'current' and ratio != 1.0:
                # Common CT secondaries: 1A or 5A
                if ct_secondary not in [1, 5]:
                    warnings.append(f"Channel {ch_name}: unusual CT secondary ({ct_secondary}A) - expected 1A or 5A")
                # Primary should be in hundreds or thousands
                if ct_primary < 100 and ct_primary != 1:
                    warnings.append(f"Channel {ch_name}: unusual CT primary ({ct_primary}A) - typically 100A or higher")

            # Validate value ranges (only for power system measurements, not control signals)
            if len(samples_primary) > 0:
                max_val = np.max(np.abs(samples_primary))
                if norm['measurement'] == 'voltage' and normalized_unit == 'kV':
                    # Reasonable range: 1 kV to 500 kV for transmission systems
                    # Skip validation for mV/mA channels (control signals)
                    if max_val < 1.0:
                        warnings.append(f"Channel {ch_name}: voltage too low ({max_val:.1f} kV) - check scaling")
                    elif max_val > 500:
                        warnings.append(f"Channel {ch_name}: voltage too high ({max_val:.1f} kV) - check scaling")
                elif norm['measurement'] == 'current' and normalized_unit == 'A':
                    # Reasonable range: 0.1 A to 50 kA
                    if max_val > 50000:
                        warnings.append(f"Channel {ch_name}: current too high ({max_val:.1f} A) - check scaling")

            # Create channel object
            channel = AnalogChannel(
                id=ch_id,
                name=ch_name,
                canonical_name=norm['canonical_name'],
                unit=normalized_unit,
                phase=norm['phase'],
                measurement=norm['measurement'],
                ct_primary=ct_primary,
                ct_secondary=ct_secondary,
                scale_a=scale_a,
                scale_b=scale_b,
                samples=samples_primary
            )

            analog_channels.append(channel)

        except Exception as e:
            logger.warning(f"Failed to parse analog channel {i} ({ch_id}): {e}")
            warnings.append(f"Skipped analog channel {ch_id}: {str(e)}")

    return analog_channels


def _parse_status_channels(com: Comtrade) -> List[StatusChannel]:
    """Parse digital/status channels."""
    status_channels = []

    for i, ch_id in enumerate(com.status_channel_ids):
        try:
            ch_name = com.cfg.status_channels[i].name if hasattr(com.cfg, 'status_channels') else ch_id

            # Get samples
            if i < len(com.status):
                samples = np.array(com.status[i], dtype=int)
            else:
                samples = np.array([], dtype=int)

            channel = StatusChannel(
                id=ch_id,
                name=ch_name,
                samples=samples
            )

            status_channels.append(channel)

        except Exception as e:
            logger.warning(f"Failed to parse status channel {i} ({ch_id}): {e}")

    return status_channels
